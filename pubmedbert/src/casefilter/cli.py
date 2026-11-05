import argparse
import csv
from collections import Counter

from .io import read_records, write_csv
from .rules import classify_rules, INCLUDE, REVIEW, EXCLUDE


def run(args):
    recs = list(read_records(args.input))
    decs = [classify_rules(r) for r in recs]
    if args.llm:
        from .llm import LLMJudge
        judge = LLMJudge(model=args.model, cache=args.cache, workers=args.workers)
        idx = [i for i, d in enumerate(decs) if d.label in args.llm_on]
        for i, d in zip(idx, judge.adjudicate([(recs[i], decs[i]) for i in idx])):
            decs[i] = d
    rows = [d.to_dict() for d in decs]
    write_csv(f"{args.out}/all.csv", rows)
    for lab in (INCLUDE, REVIEW, EXCLUDE):
        write_csv(f"{args.out}/{lab.lower()}.csv", [r for r in rows if r["label"] == lab])
    c = Counter(d.label for d in decs)
    print(f"total={len(decs)}  " + "  ".join(f"{k}={c.get(k, 0)}" for k in (INCLUDE, REVIEW, EXCLUDE)))
    print(f"results written to {args.out}/")


def predict(args):
    from .predict import predict_folder
    rows = predict_folder(args.folder, args.weights, args.model, args.device, args.out)
    pos = sum(r["label"] for r in rows)
    print(f"papers={len(rows)}  label1={pos}  label0={len(rows) - pos}  -> {args.out}")


def evaluate(args):
    """Compare all.csv with a gold CSV (pmid,label); label is 1/0 or INCLUDE/EXCLUDE."""
    with open(args.pred, encoding="utf-8-sig") as f:
        pred = {r["pmid"]: r["label"] for r in csv.DictReader(f)}
    tp = fp = fn = tn = 0
    with open(args.gold, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            g = str(r["label"]).strip().upper() in ("1", "TRUE", "INCLUDE", "YES")
            p = pred.get(r["pmid"], EXCLUDE) in (INCLUDE, REVIEW)  # REVIEW counts as kept
            tp += g and p
            fn += g and not p
            fp += (not g) and p
            tn += (not g) and not p
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  recall={rec:.3f} precision={prec:.3f} "
          "(INCLUDE+REVIEW counted as positive)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="casefilter")
    sub = ap.add_subparsers(required=True)
    r = sub.add_parser("run", help="classify a literature file")
    r.add_argument("input")
    r.add_argument("--out", default="out")
    r.add_argument("--llm", action="store_true", help="adjudicate with Claude (needs ANTHROPIC_API_KEY)")
    r.add_argument("--llm-on", nargs="+", default=[REVIEW], choices=[INCLUDE, REVIEW, EXCLUDE])
    r.add_argument("--model", default="claude-haiku-5-5")
    r.add_argument("--cache", default="llm.cache.jsonl")
    r.add_argument("--workers", type=int, default=8)
    r.set_defaults(fn=run)
    pr = sub.add_parser("predict", help="classify full-text files (xml/txt/pdf) with the exported weights, no Claude")
    pr.add_argument("folder")
    pr.add_argument("--weights", default="weights")
    pr.add_argument("--model", default=None, help="local path of the PubMedBERT embedding model")
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--out", default="out/predict.csv")
    pr.set_defaults(fn=predict)
    e = sub.add_parser("eval", help="precision/recall against a labelled gold file")
    e.add_argument("pred", help="out/all.csv")
    e.add_argument("gold", help="CSV with pmid,label")
    e.set_defaults(fn=evaluate)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
