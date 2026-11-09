#!/usr/bin/env python3
"""Keep only case-type papers (case report / case series) in screening results.

The relevance model scores *subject match*, not study design: cohort studies and
screening evaluations can also be highly relevant.  This script trains a small
title-level design classifier on the existing labeled set (``design`` column) and
applies it to the full screening output.

Label mapping
-------------
positive: design in {case report, case series}
negative: design matches cohort / screening / review / method / validation / registry
excluded: missing design, "other"

Model: TF-IDF word 1-2 grams -> logistic regression (class-balanced).
Reported metrics are 5-fold cross-validated on the labeled set.

Outputs (``--out-dir``, default ``artifacts/screen_full``)
    results_with_design.csv   all screening rows + design_score / design_guess
    case_relevant.csv         decision == relevant AND case
    case_review.csv           decision == review   AND case
    case_retained.csv         decision != irrelevant AND case (both above)
    case_summary.json         counts + validation metrics
    CASE_ONLY.md              human-readable summary

Usage
-----
    .venv/bin/python src/scripts/filter_case_type.py
    .venv/bin/python src/scripts/filter_case_type.py --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = ROOT / "data" / "labeled_papers.csv"
DEFAULT_RESULTS = ROOT / "artifacts" / "screen_full" / "results.csv"

CASE_DESIGNS = {"case report", "case series"}
NEG_PATTERN = r"cohort|screening|review|method|validation|registry"


def build_training_set(labels: pd.DataFrame) -> tuple[pd.Series, np.ndarray, dict[str, int]]:
    labels = labels.copy()
    labels["design"] = labels["design"].fillna("").astype(str)
    pos = labels["design"].isin(CASE_DESIGNS)
    neg = labels["design"].str.contains(NEG_PATTERN, regex=True) & ~pos
    mask = pos | neg
    y = pos[mask].astype(int).to_numpy()
    counts = {
        "case_report": int((labels["design"] == "case report").sum()),
        "case_series": int((labels["design"] == "case series").sum()),
        "negative_designs": int(neg.sum()),
        "excluded_other_or_missing": int((~mask).sum()),
    }
    return labels.loc[mask, "title"].fillna("").astype(str), y, counts


def make_model() -> Any:
    return make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50_000,
            sublinear_tf=True,
        ),
        LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced"),
    )


def crossval_metrics(texts: pd.Series, y: np.ndarray, threshold: float) -> dict[str, Any]:
    model = make_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prob = cross_val_predict(model, texts, y, cv=cv, method="predict_proba")[:, 1]
    pred = prob >= threshold
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((1 - y).sum()),
        "auc": round(float(roc_auc_score(y, prob)), 4),
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def render_report(summary: dict[str, Any], df: pd.DataFrame) -> str:
    m = summary["validation"]
    counts = summary["counts"]
    lines = [
        "# Case 类筛选（仅保留病例报告 / 病例系列）",
        "",
        "## 口径",
        "",
        "- 正类：`design ∈ {case report, case series}`（既有 974 篇人工/LLM 设计标签）",
        "- 负类：cohort / screening / review / method / validation / registry",
        "- 分类器：标题 TF-IDF（word 1–2gram）+ 逻辑回归（class-balanced），仅用标题，不依赖全文",
        "",
        "## 5 折交叉验证（训练集）",
        "",
        f"- 样本 {m['n']}（正 {m['positive']} / 负 {m['negative']}），AUC **{m['auc']:.3f}**",
        f"- 阈值 {m['threshold']:.2f}：Precision **{m['precision']:.3f}**、"
        f"Recall **{m['recall']:.3f}**、F1 **{m['f1']:.3f}**",
        f"- 混淆矩阵：TP {m['confusion']['tp']} / FP {m['confusion']['fp']} / "
        f"FN {m['confusion']['fn']} / TN {m['confusion']['tn']}",
        "",
        "## 全量筛选结果中的 case 类",
        "",
        "| 相关性决策 | 总数 | 其中 case 类 | 占比 |",
        "|---|---:|---:|---:|",
    ]
    for decision in ("relevant", "review", "irrelevant"):
        total = counts["by_decision_total"][decision]
        case = counts["by_decision_case"][decision]
        lines.append(f"| {decision} | {total:,} | {case:,} | {case / max(total, 1):.2%} |")
    lines += [
        f"| **保留（relevant+review）** | **{counts['retained_total']:,}** | "
        f"**{counts['retained_case']:,}** | **{counts['retained_case'] / max(counts['retained_total'], 1):.2%}** |",
        "",
        "## 产出",
        "",
        f"- `case_retained.csv`：{counts['retained_case']:,} 篇（relevant + review，且判为 case 类）",
        f"- `case_relevant.csv`：{counts['by_decision_case']['relevant']:,} 篇",
        f"- `case_review.csv`：{counts['by_decision_case']['review']:,} 篇",
        "- `results_with_design.csv`：23,378 篇全量结果 + `design_score` / `design_guess`",
        "",
        "## 高置信 case 示例（前 10，按相关性分数）",
        "",
    ]
    for _, r in df.head(10).iterrows():
        lines.append(
            f"- `{r['pmid']}` ({r['year'] or '?'}) score={r['relevance_score']:.4f} "
            f"design={r['design_score']:.3f} — {r['title']}"
        )
    lines += [
        "",
        "## 注意",
        "",
        "- 仅用标题判设计；未写“case report/series”的单病例论文可能被漏掉（CV Recall≈0.86），"
        "边界条目可结合 `design_score` 和 `review` 档人工复核。",
        "- `design_score` 是设计类型概率，与相关性分数独立；最终集合 = 相关性保留 ∩ case 类。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS.parent)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    labels = pd.read_csv(args.labels, low_memory=False)
    texts, y, label_counts = build_training_set(labels)
    validation = crossval_metrics(texts, y, args.threshold)

    model = make_model()
    model.fit(texts, y)

    df = pd.read_csv(args.results, dtype={"pmid": str})
    df["design_score"] = model.predict_proba(df["title"].fillna("").astype(str))[:, 1]
    df["design_guess"] = np.where(df["design_score"] >= args.threshold, "case", "non-case")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "results_with_design.csv", index=False)

    case = df[df["design_guess"] == "case"]
    retained = case[case["decision"] != "irrelevant"].sort_values(
        "relevance_score", ascending=False
    )
    for decision, name in (("relevant", "case_relevant.csv"), ("review", "case_review.csv")):
        case[case["decision"] == decision].sort_values(
            "relevance_score", ascending=False
        ).to_csv(args.out_dir / name, index=False)
    retained.to_csv(args.out_dir / "case_retained.csv", index=False)

    by_decision_total = df["decision"].value_counts().to_dict()
    by_decision_case = case["decision"].value_counts().to_dict()
    summary = {
        "definition": {
            "positive_designs": sorted(CASE_DESIGNS),
            "negative_pattern": NEG_PATTERN,
            "features": "title only (TF-IDF word 1-2gram + logistic regression)",
            "training_label_counts": label_counts,
        },
        "validation": validation,
        "counts": {
            "total": int(len(df)),
            "by_decision_total": {
                k: int(by_decision_total.get(k, 0))
                for k in ("relevant", "review", "irrelevant")
            },
            "by_decision_case": {
                k: int(by_decision_case.get(k, 0))
                for k in ("relevant", "review", "irrelevant")
            },
            "retained_total": int((df["decision"] != "irrelevant").sum()),
            "retained_case": int(len(retained)),
        },
        "outputs": {
            "results_with_design": str(args.out_dir / "results_with_design.csv"),
            "case_retained": str(args.out_dir / "case_retained.csv"),
            "case_relevant": str(args.out_dir / "case_relevant.csv"),
            "case_review": str(args.out_dir / "case_review.csv"),
        },
    }
    (args.out_dir / "case_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "CASE_ONLY.md").write_text(
        render_report(summary, retained), encoding="utf-8"
    )
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.out_dir}/case_retained.csv ({len(retained)} rows)")


if __name__ == "__main__":
    main()
