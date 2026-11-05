"""Retrain the head with curated gold (1) + reviewed Claude labels (0/1) + rule negatives (0).

30% of the Claude-labelled papers are held out for validation (stratified by label and text source).
The held-out predictions come from a model that never saw them; the final model is then refit on everything.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casefilter.embed import pool  # noqa: E402


def weights(src, y):
    cell = pd.Series(list(zip(src, y)))
    w = np.array(1.0 / cell.map(cell.value_counts()), dtype=float)
    for lab in (0, 1):
        w[y == lab] *= 0.5 / w[y == lab].sum()
    return w / w.mean()


def fit(X, y, src, C):
    return LogisticRegression(C=C, max_iter=5000).fit(X, y, sample_weight=weights(src, y))


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), n_pos=int(y.sum()), auc=float(roc_auc_score(y, p)), precision=float(precision_score(y, pred)),
                recall=float(recall_score(y, pred)), f1=float(f1_score(y, pred)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--labels", default="out/llm_labels_final.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)

    df = pd.read_pickle(out / "predictions.pkl").reset_index(drop=True)
    df["old_prob"] = df.prob
    X = pool(joblib.load(out / "chunk_embeddings.pkl"))
    cl = pd.read_csv(a.labels, dtype={"pmid": str}, encoding="utf-8-sig").set_index("pmid").label_llm
    df["claude"] = df.pmid.map(cl)
    df["y"] = np.where(df.claude.notna(), df.claude, np.where(df.gold, 1, np.where(df.rule == "EXCLUDE", 0, -1)))
    df["y_src"] = np.where(df.claude.notna(), "claude", np.where(df.gold, "gold", np.where(df.y == 0, "rule_neg", "none")))
    lab = df.y >= 0
    print(df[lab].groupby(["y_src", "y"]).size())

    # 70/30 split of the Claude-labelled papers; gold and rule negatives always train.
    cidx = df.index[df.y_src == "claude"]
    strat = df.loc[cidx, "y"].astype(int).astype(str) + df.loc[cidx, "source"]
    try:
        tr_c, va_c = train_test_split(cidx, test_size=0.3, random_state=a.seed, stratify=strat)
    except ValueError:  # a stratum too small
        tr_c, va_c = train_test_split(cidx, test_size=0.3, random_state=a.seed, stratify=df.loc[cidx, "y"])
    train_idx = np.array(sorted(set(df.index[lab]) - set(va_c)))
    va_idx = np.array(sorted(va_c))
    ytr, yva = df.y[train_idx].astype(int).values, df.y[va_idx].astype(int).values
    str_, sva = df.source[train_idx].values, df.source[va_idx].values
    print(f"train={len(train_idx)} (pos {int(ytr.sum())}) validation={len(va_idx)} (pos {int(yva.sum())})")

    best = None
    for C in (0.1, 1, 10, 100):  # choose C by CV inside the training part only
        from sklearn.model_selection import StratifiedKFold
        p = np.zeros(len(ytr))
        for t, v in StratifiedKFold(5, shuffle=True, random_state=0).split(X[train_idx], ytr):
            p[v] = fit(X[train_idx][t], ytr[t], str_[t], C).predict_proba(X[train_idx][v])[:, 1]
        auc = roc_auc_score(ytr, p)
        print(f"C={C}: train-CV AUC={auc:.3f}")
        if best is None or auc > best[0]:
            best = (auc, C)
    C = best[1]
    m70 = fit(X[train_idx], ytr, str_, C)
    pva = m70.predict_proba(X[va_idx])[:, 1]
    res = dict(C=C, threshold=a.threshold, new_model=metrics(yva, pva, a.threshold),
               old_model_same_papers=metrics(yva, df.old_prob[va_idx].values, a.threshold),
               majority_baseline_acc=float(max(yva.mean(), 1 - yva.mean())))
    print(json.dumps(res, indent=2))

    # final model on all labelled papers
    all_idx = np.array(df.index[lab])
    final = fit(X[all_idx], df.y[all_idx].astype(int).values, df.source[all_idx].values, C)
    joblib.dump(final, out / "head_v2.joblib")
    prob = final.predict_proba(X)[:, 1]
    prob[va_idx] = pva  # validation papers keep their unseen-model probability
    df["prob_v2"] = prob.round(4)
    df["pred_model"] = (df.prob_v2 >= a.threshold).astype(int)
    # Final label: human-reviewed (Claude) or curated gold labels win; everything else uses the model.
    df["final"] = np.where(df.y_src.isin(["claude", "gold"]), df.y, df.pred_model).astype(int)
    df["final_src"] = np.where(df.y_src.isin(["claude", "gold"]), df.y_src, "model")
    df["split"] = np.where(df.index.isin(va_idx), "validation", np.where(lab, "train", "unlabelled"))
    df["uncertain_v2"] = (df.final_src == "model") & ((df.prob_v2 - a.threshold).abs() < 0.15)
    df.to_pickle(out / "predictions_v2.pkl")
    cols = ["pmid", "title", "year", "source", "in_package", "split", "y_src", "old_prob", "prob_v2", "pred_model",
            "final", "final_src", "uncertain_v2", "methods"]
    df[df.in_package][cols].to_csv(out / "predictions_v2.csv", index=False, encoding="utf-8-sig")
    (out / "metrics_v2.json").write_text(json.dumps(res, indent=2))
    print("saved predictions_v2.csv")


if __name__ == "__main__":
    main()
