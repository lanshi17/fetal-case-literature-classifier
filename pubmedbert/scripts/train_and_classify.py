"""Train a local embedding + logistic-regression head, classify every paper, write predictions and a summary.

Training set: curated gold papers (label 1) vs papers the rule stage confidently EXCLUDEs (label 0).
Training papers get out-of-fold predictions so their reported labels never come from a model that saw them.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casefilter.embed import Embedder, pool  # noqa: E402
from casefilter.rules import classify_rules  # noqa: E402
from casefilter import lexicon as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-chunks", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([json.loads(x) for x in open(a.corpus, encoding="utf-8")])
    df["rule"] = [classify_rules(dict(pmid=r.pmid, title=r.title, abstract="", fulltext=r.text)).label
                  for r in df.itertuples()]
    df["train_label"] = np.where(df.gold, 1, np.where(df.rule == "EXCLUDE", 0, -1))

    emb_path = out / "chunk_embeddings.pkl"
    if emb_path.exists() and len(joblib.load(emb_path)) == len(df):
        chunks = joblib.load(emb_path)
    else:
        emb = Embedder(a.model, device=a.device) if a.model else Embedder(device=a.device)
        chunks = emb.embed_docs((df.title.fillna("") + ". " + df.text.fillna("")).tolist(), max_chunks=a.max_chunks)
        joblib.dump(chunks, emb_path)
    X = pool(chunks)

    tr = (df.train_label >= 0).values
    y = df.train_label[tr].values
    print(f"training docs: pos={int((y == 1).sum())} neg={int((y == 0).sum())}")

    # Source-balanced weights: every (text source, label) cell carries equal total weight, so the head cannot
    # use "this came from a PDF / PMC XML" as a shortcut for the label.
    src = df.source[tr].values
    cell = pd.Series(list(zip(src, y)))
    w = np.array(1.0 / cell.map(cell.value_counts()), dtype=float)
    for lab in (0, 1):
        w[y == lab] *= 0.5 / w[y == lab].sum()
    w = w / w.mean()

    def fit(C, idx):
        return LogisticRegression(C=C, max_iter=5000).fit(X[tr][idx], y[idx], sample_weight=w[idx])

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    strat = np.array([f"{a_}{b_}" for a_, b_ in zip(src, y)])
    best = None
    for C in (0.1, 1, 10, 100):
        p = np.zeros(len(y))
        for tri, tei in cv.split(X[tr], strat):
            p[tei] = fit(C, tri).predict_proba(X[tr][tei])[:, 1]
        auc = roc_auc_score(y, p)
        print(f"C={C}: CV AUC={auc:.3f}")
        if best is None or auc > best[0]:
            best = (auc, C, p)
    auc, C, p_oof = best
    pred = (p_oof >= a.threshold).astype(int)
    metrics = dict(C=C, auc=auc, precision=precision_score(y, pred), recall=recall_score(y, pred), f1=f1_score(y, pred),
                   n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()), threshold=a.threshold)
    for sname in sorted(set(src)):
        mk = src == sname
        if len(set(y[mk])) == 2:
            metrics[f"auc_within_{sname}"] = roc_auc_score(y[mk], p_oof[mk])
    print(metrics)

    final = fit(C, np.arange(len(y)))
    joblib.dump(final, out / "head.joblib")
    prob = final.predict_proba(X)[:, 1]
    prob[np.where(tr)[0]] = p_oof  # out-of-fold for training docs
    df["prob"] = prob.round(4)
    df["pred"] = (df.prob >= a.threshold).astype(int)
    df["uncertain"] = (df.prob - a.threshold).abs() < 0.15
    df["methods"] = [";".join(m for m, rx in L.METHODS.items() if rx.search(t)) for t in df.text]
    cols = ["pmid", "title", "year", "source", "n_chars", "in_package", "gold", "train_label", "rule", "prob", "pred",
            "uncertain", "methods"]
    df[cols].to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    df[cols].to_pickle(out / "predictions.pkl")
    print("saved", out / "predictions.csv")


if __name__ == "__main__":
    main()
