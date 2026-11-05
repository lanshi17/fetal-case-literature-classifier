"""Readers/writers. Input: .csv/.tsv, .jsonl, PubMed XML (.xml). Columns: pmid,title,abstract[,pub_types,fulltext]."""
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _xml_text(el):
    return "".join(el.itertext()).strip() if el is not None else ""


def _read_pubmed_xml(path):
    for _, art in ET.iterparse(path, events=("end",)):
        if art.tag not in ("PubmedArticle", "PubmedBookArticle"):
            continue
        yield {
            "pmid": _xml_text(art.find(".//PMID")),
            "title": _xml_text(art.find(".//ArticleTitle")),
            "abstract": " ".join(_xml_text(a) for a in art.findall(".//Abstract/AbstractText")),
            "pub_types": ";".join(_xml_text(p) for p in art.findall(".//PublicationType")),
        }
        art.clear()


def read_records(path):
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".xml":
        yield from _read_pubmed_xml(p)
    elif suf == ".jsonl":
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif suf in (".csv", ".tsv"):
        with p.open(encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f, delimiter="\t" if suf == ".tsv" else ",")
    else:
        raise ValueError(f"unsupported input format: {suf}")


def write_csv(path, rows):
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["pmid", "label", "score", "methods", "case_type", "reasons", "stage", "evidence"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
