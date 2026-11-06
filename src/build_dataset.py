#!/usr/bin/env python3
"""Build the labeled screening dataset from final labels + local full-text package.

Inputs
------
data/final_labels.csv
    UTF-8 (with BOM), columns include pmid,title,year,label,label_source,
    confidence,weak,design,reason,...
data/fulltext_package/manifest_fulltext.csv
    Columns include PMID,PMCID,TXT,PDF,XML,TXT_file,PDF_file,XML_file,...

Output
------
data/labeled_papers.csv
    One row per PMID with doc_id/group_id = PMID, JATS abstract, full text
    (TXT preferred, PDF fallback), label metadata and availability flags.

Usage
-----
    python src/build_dataset.py
    python src/build_dataset.py --limit 20          # quick smoke
    python src/build_dataset.py --no-pdf            # skip PyMuPDF fallback
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "final_labels.csv"
DEFAULT_PACKAGE = ROOT / "data" / "fulltext_package"
DEFAULT_OUTPUT = ROOT / "data" / "labeled_papers.csv"

OUTPUT_FIELDS = [
    "doc_id",
    "group_id",
    "pmid",
    "pmcid",
    "title",
    "year",
    "label",
    "label_source",
    "confidence",
    "weak",
    "design",
    "label_reason",
    "abstract",
    "text",
    "abstract_available",
    "full_text_available",
    "text_source",
    "text_chars",
    "abstract_chars",
    "tests",
    "methods",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def norm_pmid(value: Any) -> str:
    """Normalize a PMID cell (handles '123.0', whitespace, None)."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".", 1)[0]
    return s


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def as_float(value: Any) -> float | None:
    try:
        s = str(value).strip()
        return float(s) if s and s.lower() not in {"nan", "none"} else None
    except (TypeError, ValueError):
        return None


def clean_text(text: str) -> str:
    """Collapse NULs / excessive whitespace while keeping paragraph breaks."""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


# ---------------------------------------------------------------------------
# JATS abstract extraction
# ---------------------------------------------------------------------------
def _strip_tags(raw: str) -> str:
    raw = re.sub(r"<(script|style)\b.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    return clean_text(raw)


def extract_jats_abstract(xml_path: Path) -> str:
    """Extract the JATS <abstract> text. Returns '' when unavailable.

    Handles namespaced and malformed XML; skips graphical/teaser abstracts.
    """
    if not xml_path or not xml_path.exists():
        return ""
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(str(xml_path)).getroot()
        nodes = [el for el in root.iter() if local_name(el.tag) == "abstract"]
        texts: list[str] = []
        for node in nodes:
            atype = str(node.attrib.get("abstract-type", "")).lower()
            if atype in {"graphical", "teaser"}:
                continue
            # Prefer paragraph-level text; fall back to all descendant text.
            paras = [
                " ".join(el.itertext())
                for el in node.iter()
                if local_name(el.tag) == "p"
            ]
            body = " ".join(paras) if paras else " ".join(node.itertext())
            body = clean_text(body)
            if body:
                texts.append(body)
        if texts:
            body = "\n\n".join(texts)
            # Drop a leading generic heading such as "ABSTRACT".
            body = re.sub(r"^(abstract|summary)\s*[:\-]?\s*", "", body, flags=re.I)
            return clean_text(body)
    except Exception:
        pass

    # Regex fallback for malformed XML.
    try:
        raw = xml_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    blocks = re.findall(r"<abstract\b[^>]*>(.*?)</abstract>", raw, flags=re.S | re.I)
    if not blocks:
        return ""
    body = " ".join(_strip_tags(b) for b in blocks)
    body = re.sub(r"^(abstract|summary)\s*[:\-]?\s*", "", body, flags=re.I)
    return clean_text(body)


# ---------------------------------------------------------------------------
# Full-text extraction
# ---------------------------------------------------------------------------
def read_txt(path: Path) -> str:
    try:
        return clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return ""


def read_pdf(path: Path) -> tuple[str, str]:
    """Extract PDF text with PyMuPDF. Returns (text, error)."""
    try:
        try:
            import pymupdf as fitz  # PyMuPDF >= 1.24
        except ImportError:  # pragma: no cover - older PyMuPDF
            import fitz

        parts: list[str] = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                parts.append(page.get_text("text"))
        return clean_text("\n\n".join(parts)), ""
    except Exception as exc:  # noqa: BLE001 - report extraction failures
        return "", f"{type(exc).__name__}: {exc}"


def resolve_member(package: Path, rel: str, flag: str, subdir: str, pmid: str) -> Path | None:
    """Resolve a manifest file column, with a glob fallback by PMID."""
    rel = (rel or "").strip()
    if rel:
        p = Path(rel)
        if not p.is_absolute():
            p = package / p
        if p.exists():
            return p
        # Some archives store paths relative to the package root differently.
        p2 = package / p.name
        if p2.exists():
            return p2
    if as_bool(flag):
        # Skip macOS AppleDouble resource forks ("._name") shipped in the tar.
        matches = sorted(
            p for p in (package / subdir).glob(f"{pmid}_*") if not p.name.startswith("._")
        )
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build(
    labels_path: Path = DEFAULT_LABELS,
    package: Path = DEFAULT_PACKAGE,
    output: Path = DEFAULT_OUTPUT,
    *,
    use_pdf: bool = True,
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels not found: {labels_path}")
    if not package.exists():
        raise FileNotFoundError(f"full-text package not found: {package}")

    with labels_path.open("r", encoding="utf-8-sig", newline="") as fh:
        labels = list(csv.DictReader(fh))
    if not labels:
        raise ValueError(f"no rows in {labels_path}")

    manifest_path = package / "manifest_fulltext.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as fh:
        manifest_rows = list(csv.DictReader(fh))
    manifest = {norm_pmid(r.get("PMID")): r for r in manifest_rows if norm_pmid(r.get("PMID"))}

    if limit:
        labels = labels[:limit]

    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    pdf_errors: list[dict[str, str]] = []
    missing_manifest: list[str] = []

    for i, lab in enumerate(labels, 1):
        pmid = norm_pmid(lab.get("pmid"))
        if not pmid:
            stats["skipped_no_pmid"] += 1
            continue
        man = manifest.get(pmid)
        if man is None:
            missing_manifest.append(pmid)
            stats["missing_manifest"] += 1
            man = {}

        xml_path = resolve_member(package, man.get("XML_file", ""), man.get("XML", ""), "xml", pmid)
        txt_path = resolve_member(package, man.get("TXT_file", ""), man.get("TXT", ""), "txt", pmid)
        pdf_path = resolve_member(package, man.get("PDF_file", ""), man.get("PDF", ""), "pdfs", pmid)

        abstract = extract_jats_abstract(xml_path) if xml_path else ""
        if abstract:
            stats["with_abstract"] += 1

        text = ""
        text_source = "none"
        if txt_path:
            text = read_txt(txt_path)
            if text:
                text_source = "txt"
        if not text and use_pdf and pdf_path:
            text, err = read_pdf(pdf_path)
            if text:
                text_source = "pdf"
            elif err:
                pdf_errors.append({"pmid": pmid, "pdf": str(pdf_path), "error": err})
                stats["pdf_error"] += 1
        if text:
            stats["with_fulltext"] += 1

        title = clean_text(lab.get("title", "") or man.get("Title", "") or "")
        row = {
            "doc_id": pmid,
            "group_id": pmid,
            "pmid": pmid,
            "pmcid": (man.get("PMCID") or "").strip(),
            "title": title,
            "year": (lab.get("year") or man.get("Year") or "").strip(),
            "label": int(float(lab.get("label"))) if str(lab.get("label", "")).strip() else "",
            "label_source": (lab.get("label_source") or "").strip(),
            "confidence": lab.get("confidence", ""),
            "weak": as_bool(lab.get("weak")),
            "design": (lab.get("design") or "").strip(),
            "label_reason": (lab.get("reason") or "").strip(),
            "abstract": abstract,
            "text": text,
            "abstract_available": bool(abstract),
            "full_text_available": bool(text),
            "text_source": text_source,
            "text_chars": len(text),
            "abstract_chars": len(abstract),
            "tests": (lab.get("tests") or "").strip(),
            "methods": (lab.get("methods") or "").strip(),
        }
        rows.append(row)
        stats[f"source_{text_source}"] += 1
        if row["label"] == 1:
            stats["label_1"] += 1
        elif row["label"] == 0:
            stats["label_0"] += 1

        if verbose and (i % 100 == 0 or i == len(labels)):
            print(f"  processed {i}/{len(labels)}", file=sys.stderr, flush=True)

    if not rows:
        raise RuntimeError("no rows built")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "labels_rows": len(labels),
        "output_rows": len(rows),
        "manifest_rows": len(manifest_rows),
        "overlap_with_manifest": len({norm_pmid(r.get("pmid")) for r in labels} & set(manifest)),
        "missing_manifest": missing_manifest,
        "with_abstract": stats["with_abstract"],
        "with_fulltext": stats["with_fulltext"],
        "without_fulltext": len(rows) - stats["with_fulltext"],
        "text_source": {
            "txt": stats["source_txt"],
            "pdf": stats["source_pdf"],
            "none": stats["source_none"],
        },
        "labels": {"1": stats["label_1"], "0": stats["label_0"]},
        "mean_text_chars": round(sum(r["text_chars"] for r in rows) / len(rows), 1),
        "mean_abstract_chars": round(sum(r["abstract_chars"] for r in rows) / len(rows), 1),
        "pdf_errors": pdf_errors,
        "output": str(output),
    }
    stats_path = output.parent / "build_stats.json"
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(rows)} rows -> {output}")
    print(f"Stats -> {stats_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="only process first N labels")
    parser.add_argument("--no-pdf", action="store_true", help="disable PyMuPDF fallback")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    build(
        args.labels,
        args.package,
        args.output,
        use_pdf=not args.no_pdf,
        limit=args.limit,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
