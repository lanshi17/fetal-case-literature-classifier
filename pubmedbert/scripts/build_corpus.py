"""Extract text for every paper in a fulltext_package (manifest_fulltext.csv) and the curated gold set.

Writes corpus.jsonl (pmid,title,year,text,source,n_chars,stub,gold) next to --out.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casefilter.fulltext import best_text, STUB_CHARS, txt_text, pdf_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="folder holding manifest_fulltext.csv, pdfs/, xml/, txt/")
    ap.add_argument("--gold-dir", help="folder with extracted/period*/ *_PMID<id>.txt of curated positives")
    ap.add_argument("--gold-pdf-dir", help="folder tree with curated gold PDFs (*_PMID<id>.pdf); used for gold papers outside the package")
    ap.add_argument("--out", default="data/corpus.jsonl")
    a = ap.parse_args()
    pkg = Path(a.package)

    gold = {}
    if a.gold_dir:
        for p in Path(a.gold_dir).rglob("*.txt"):
            m = re.search(r"PMID(\d+)", p.name)
            if m and not p.name.startswith("._"):
                gold[m.group(1)] = txt_text(p)

    gold_pdf = {}
    if a.gold_pdf_dir:
        for p in Path(a.gold_pdf_dir).rglob("*.pdf"):
            m = re.search(r"PMID(\d+)", p.name)
            if m and not p.name.startswith("._"):
                gold_pdf[m.group(1)] = p

    rows = {}
    with open(pkg / "manifest_fulltext.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            files = {k: (pkg / r[c]) if r[c] else None for k, c in (("xml", "XML_file"), ("txt", "TXT_file"), ("pdf", "PDF_file"))}
            text, src = best_text(files)
            rows[r["PMID"]] = dict(pmid=r["PMID"], title=r["Title"], year=r["Year"], text=text, source=src, in_package=True)
    # Gold papers already in the package keep the package text (same extraction pipeline as the unlabeled
    # papers); only the rest fall back to their own PDF, then to the pre-extracted txt. This avoids the
    # classifier learning "curated txt format" instead of content.
    for pmid, t in gold.items():
        if pmid in rows:
            continue
        pt = pdf_text(gold_pdf[pmid]) if pmid in gold_pdf else ""
        text, src = (pt, "pdf") if len(pt) >= STUB_CHARS else (t, "gold_txt")
        rows[pmid] = dict(pmid=pmid, title="", year="", text=text, source=src, in_package=False)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for pmid, r in rows.items():
            r.update(n_chars=len(r["text"]), stub=len(r["text"]) < STUB_CHARS, gold=pmid in gold)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    print(f"papers={n} gold={len(gold)} stub={sum(r['stub'] for r in rows.values())}")


if __name__ == "__main__":
    main()
