"""v3: retrain the head on all Claude-labelled papers (+ curated gold), 70/30 stratified validation.

Training labels: Claude labels from out/final_labels.csv (rounds 1 and 3) and curated gold papers.
Rule-derived pseudo-negatives and model-only labels are NOT used any more.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

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


def report(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=int(len(y)), n_pos=int(y.sum()), n_neg=int((1 - y).sum()), threshold=float(thr),
                auc=float(roc_auc_score(y, p)), pr_auc_pos=float(average_precision_score(y, p)),
                pr_auc_neg=float(average_precision_score(1 - y, 1 - p)),
                precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred)),
                specificity=float(tn / (tn + fp)) if tn + fp else None,
                npv=float(tn / (tn + fn)) if tn + fn else None, f1=float(f1_score(y, pred)),
                balanced_acc=float(balanced_accuracy_score(y, pred)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)

    df = pd.read_pickle(out / "predictions.pkl").reset_index(drop=True)
    X = pool(joblib.load(out / "chunk_embeddings.pkl"))
    fl = pd.read_csv(out / "final_labels.csv", dtype={"pmid": str}, encoding="utf-8-sig").set_index("pmid")
    claude = fl[fl.label_source.str.startswith("claude")]
    df["claude"] = df.pmid.map(claude.label)
    df["conf"] = df.pmid.map(claude.confidence)
    old_gold = df.gold & df.claude.isna()
    df["y"] = np.where(df.claude.notna(), df.claude, np.where(old_gold, 1, -1))
    lab = df.y >= 0
    print(df[lab].groupby(["gold", "y"]).size())

    # validation = 30% of the Claude-labelled papers; gold-only papers always train
    cidx = df.index[df.claude.notna()]
    strat = df.loc[cidx, "y"].astype(int).astype(str) + df.loc[cidx, "source"]
    tr_c, va_idx = train_test_split(cidx, test_size=0.3, random_state=a.seed, stratify=strat)
    train_idx = np.array(sorted(set(df.index[lab]) - set(va_idx)))
    va_idx = np.array(sorted(va_idx))
    ytr, yva = df.y[train_idx].astype(int).values, df.y[va_idx].astype(int).values
    str_ = df.source[train_idx].values
    print(f"train={len(train_idx)} (neg {int((ytr == 0).sum())}) validation={len(va_idx)} (neg {int((yva == 0).sum())})")

    best = None
    for C in (0.1, 1, 10, 100):
        p = np.zeros(len(ytr))
        for t, v in StratifiedKFold(5, shuffle=True, random_state=0).split(X[train_idx], ytr):
            p[v] = fit(X[train_idx][t], ytr[t], str_[t], C).predict_proba(X[train_idx][v])[:, 1]
        auc = roc_auc_score(ytr, p)
        print(f"C={C}: train-CV AUC={auc:.3f}")
        if best is None or auc > best[0]:
            best = (auc, C, p)
    _, C, p_cv = best
    m70 = fit(X[train_idx], ytr, str_, C)
    pva = m70.predict_proba(X[va_idx])[:, 1]

    # threshold choices: 0.5, and the one maximising balanced accuracy on the training-CV predictions
    grid = np.linspace(0.05, 0.95, 91)
    thr_bal = float(grid[np.argmax([balanced_accuracy_score(ytr, (p_cv >= g).astype(int)) for g in grid])])
    res = dict(C=C, n_train=len(train_idx), n_val=len(va_idx), train_cv_auc=float(best[0]),
               val_at_0_5=report(yva, pva, 0.5), val_at_balanced_thr=report(yva, pva, thr_bal),
               balanced_thr=thr_bal)
    # baseline: label everything 1
    res["baseline_all_ones"] = dict(precision=float(yva.mean()), recall=1.0, f1=float(2 * yva.mean() / (1 + yva.mean())),
                                    specificity=0.0, balanced_acc=0.5)
    # sensitivity: only validation papers whose Claude confidence >= 0.8
    hi = (df.conf[va_idx].fillna(1).values >= 0.8)
    if len(set(yva[hi])) == 2:
        res["val_confident_only_at_0_5"] = report(yva[hi], pva[hi], 0.5)
    # per-source AUC on validation
    res["val_auc_by_source"] = {s: float(roc_auc_score(yva[df.source[va_idx].values == s], pva[df.source[va_idx].values == s]))
                                for s in sorted(set(df.source[va_idx])) if len(set(yva[df.source[va_idx].values == s])) == 2}
    print(json.dumps(res, indent=2))

    # final head on everything labelled, then score every paper
    final = fit(X[lab.values], df.y[lab].astype(int).values, df.source[lab].values, C)
    joblib.dump(final, out / "head_v3.joblib")
    prob = final.predict_proba(X)[:, 1]
    prob[va_idx] = pva
    df["prob_v3"] = prob.round(4)
    df["split"] = np.where(df.index.isin(va_idx), "validation", np.where(lab, "train", "unlabelled"))
    pk = df[df.in_package].merge(fl[["label", "label_source"]], left_on="pmid", right_index=True, how="left")
    pk["pred_v3"] = (pk.prob_v3 >= 0.5).astype(int)
    pk[["pmid", "title", "year", "split", "label_source", "label", "prob_v3", "pred_v3"]].to_csv(
        out / "predictions_v3.csv", index=False, encoding="utf-8-sig")
    mo = pk[pk.label_source == "model"]
    res["model_only_papers"] = dict(n=len(mo), v2_label_ones=int(mo.label.sum()), v3_pred_ones=int(mo.pred_v3.sum()),
                                    agree=int((mo.label == mo.pred_v3).sum()))
    (out / "metrics_v3.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["model_only_papers"]))


if __name__ == "__main__":
    main()
