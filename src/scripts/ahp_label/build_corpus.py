#!/usr/bin/env python3
"""Extract PDF text for the AHP wave sample into corpus.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE = ROOT / "data" / "ahp_wave1" / "sample_3000.csv"
DEFAULT_OUT = ROOT / "data" / "ahp_wave1" / "corpus.jsonl"
DEFAULT_PDF_DIR = Path("/data/yangzs/Projects/05_Tools/04_AutomatedPubMed/downloads/pdf")


def extract_pdf(path: Path, *, max_chars: int) -> tuple[str, dict]:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    meta: dict = {"n_pages": 0, "ocr_needed": False, "error": ""}
    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return "", meta
    try:
        meta["n_pages"] = doc.page_count
        parts = []
        for page in doc:
            parts.append(page.get_text("text") or "")
        text = "\n\n".join(parts).replace("\x00", " ").strip()
        if len(text) < 200 and doc.page_count > 0:
            meta["ocr_needed"] = True
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        return text, meta
    finally:
        doc.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-chars", type=int, default=22000)
    ap.add_argument("--limit", type=int, default=None, help="optional cap for pilot")
    args = ap.parse_args()

    df = pd.read_csv(args.sample, dtype=str)
    if args.limit:
        df = df.head(args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_ocr = n_err = 0
    with args.out.open("w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            pmid = str(row["PMID"])
            fname = str(row.get("File") or "")
            pdf = Path(str(row.get("pdf_path") or (args.pdf_dir / fname)))
            text, meta = extract_pdf(pdf, max_chars=args.max_chars) if pdf.is_file() else (
                "",
                {"n_pages": 0, "ocr_needed": False, "error": "missing_file"},
            )
            if meta.get("error"):
                n_err += 1
            elif meta.get("ocr_needed"):
                n_ocr += 1
            else:
                n_ok += 1
            rec = {
                "pmid": pmid,
                "doi": row.get("DOI") or "",
                "title": row.get("Title") or "",
                "journal": row.get("Journal") or "",
                "year": row.get("Year") or "",
                "pdf_path": str(pdf),
                "text": text,
                "text_chars": len(text),
                "n_pages": meta.get("n_pages"),
                "ocr_needed": bool(meta.get("ocr_needed")),
                "extract_error": meta.get("error") or "",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  extracted {i + 1}/{len(df)}", flush=True)

    print(
        f"Wrote {args.out} n={len(df)} ok={n_ok} ocr_needed={n_ocr} err={n_err}"
    )


if __name__ == "__main__":
    main()
