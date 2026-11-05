"""Export uncertain papers (|prob-threshold|<0.15) as an annotation sheet with evidence sentences."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casefilter.rules import classify_rules  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--pred", default="out/predictions.pkl")
    ap.add_argument("--out", default="out/to_annotate.csv")
    a = ap.parse_args()
    pred = pd.read_pickle(a.pred)
    pred = pred[pred.in_package & pred.uncertain]
    text = {}
    for line in open(a.corpus, encoding="utf-8"):
        r = json.loads(line)
        text[r["pmid"]] = r["text"]
    rows = []
    for r in pred.sort_values("prob").itertuples():
        d = classify_rules(dict(pmid=r.pmid, title=r.title, abstract="", fulltext=text[r.pmid]))
        rows.append(dict(pmid=r.pmid, title=r.title, year=r.year, prob=r.prob, methods=r.methods,
                         evidence=" || ".join(d.evidence[:2]), label="", note=""))
    Path(a.out).parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8-sig")
    print(len(rows), "rows ->", a.out)


if __name__ == "__main__":
    main()
