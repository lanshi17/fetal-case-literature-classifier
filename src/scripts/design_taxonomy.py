#!/usr/bin/env python3
"""Bootstrap hierarchical design taxonomy for the case corpus.

Maps the ``design`` column onto fine-grained classes with explicit precedence,
then trains a multinomial logistic regression on cached 1536-d document
features (frozen PubMedBERT, 5-fold CV). This is a rule-based bootstrap, not
expert re-annotation; ``unclear`` collects rows whose design was missing/other.

Outputs (``artifacts/design_taxonomy``): metrics.json, report.md, taxonomy.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "artifacts" / "features" / "pubmedbert_mc12_ct480_n974_a9ccfb8d4b.npz"
DATA = ROOT / "data" / "labeled_papers.csv"
OUT = ROOT / "artifacts" / "design_taxonomy"

RULES = [
    ("case_report", r"^case\s+report\b"),
    ("case_series", r"^case\s+series\b"),
    ("review", r"review"),
    ("method", r"method"),
    ("validation", r"validation"),
    ("screening", r"screening"),
    ("registry", r"registry"),
    ("cohort", r"cohort|retrospective|prospective|observational|metabolomics"),
]
LABELS = [k for k, _ in RULES] + ["unclear"]


def classify(design: str) -> str:
    d = str(design or "").strip().lower()
    for name, pat in RULES:
        if re.search(pat, d):
            return name
    return "unclear"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", type=Path, default=None, help="adjudication.csv with pmid/suggested_design/action")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, low_memory=False, dtype={"pmid": str})
    df["design_class"] = df["design"].map(classify)
    if args.overlay:
        ov = pd.read_csv(args.overlay, dtype={"pmid": str})
        ov = ov[ov["action"].isin(["add", "relabel"])]
        mp = dict(zip(ov["pmid"], ov["suggested_design"]))
        hit = df["pmid"].isin(mp)
        df.loc[hit, "design_class"] = df.loc[hit, "pmid"].map(mp)
        print(f"overlay applied to {int(hit.sum())} rows")
    counts = df["design_class"].value_counts().to_dict()

    z = np.load(FEATURES, allow_pickle=True)
    feat = np.concatenate([z["mean_all"], z["mean_head"]], axis=1)
    ids = pd.Index(z["doc_ids"].astype(str))
    pos = ids.get_indexer(df["pmid"].astype(str))
    if (pos < 0).any():
        raise SystemExit(f"{int((pos < 0).sum())} docs without cached features")
    X = feat[pos]
    y = df["design_class"].map({k: i for i, k in enumerate(LABELS)}).to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.empty_like(y)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])

    names = LABELS
    macro = f1_score(y, pred, average="macro", labels=list(range(len(names))), zero_division=0)
    weighted = f1_score(y, pred, average="weighted", labels=list(range(len(names))), zero_division=0)
    report = classification_report(y, pred, labels=list(range(len(names))), target_names=names, zero_division=0, output_dict=True)
    cm = confusion_matrix(y, pred, labels=list(range(len(names))))

    # Binary sanity check: case_report|case_series vs everything else.
    yb = np.isin(y, [names.index("case_report"), names.index("case_series")]).astype(int)
    pb = np.isin(pred, [names.index("case_report"), names.index("case_series")]).astype(int)
    tp = int(((pb == 1) & (yb == 1)).sum()); fp = int(((pb == 1) & (yb == 0)).sum()); fn = int(((pb == 0) & (yb == 1)).sum())
    b_p = tp / max(tp + fp, 1); b_r = tp / max(tp + fn, 1)

    metrics = {
        "n": int(len(y)),
        "class_counts": counts,
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "per_class": {k: {"precision": v["precision"], "recall": v["recall"], "f1": v["f1-score"], "support": int(v["support"])} for k, v in report.items() if k in names},
        "confusion": {"labels": names, "matrix": cm.tolist()},
        "binary_case_vs_other": {"precision": b_p, "recall": b_r, "f1": 2 * b_p * b_r / max(b_p + b_r, 1e-9)},
        "note": "rule-based design_class bootstrap, not expert re-annotation",
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    df[["pmid", "design", "design_class"]].to_csv(args.out / "taxonomy.csv", index=False)

    lines = [
        "# Design taxonomy (bootstrap)",
        "",
        f"- n = {len(y)}; macro-F1 **{macro:.3f}**, weighted-F1 **{weighted:.3f}** (5-fold, frozen 1536-d features + LR)",
        f"- binary case vs other: P {b_p:.3f} / R {b_r:.3f} / F1 {metrics['binary_case_vs_other']['f1']:.3f}",
        "",
        "| class | n | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        v = metrics["per_class"][name]
        lines.append(f"| {name} | {v['support']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
    lines += ["", "Rules: " + " > ".join(f"`{k}`" for k, _ in RULES) + " > `unclear` (missing/other)."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
