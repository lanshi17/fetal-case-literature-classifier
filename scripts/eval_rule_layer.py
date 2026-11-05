#!/usr/bin/env python3
"""Evaluate rule layer on top of DoRA (or any score CSV).

Compares model-only vs model+rules on:
  - seed-42 default test fold
  - gold-49
  - hard-negative subset (label=0 & hard tag)

Usage
-----
    .venv/bin/python scripts/eval_rule_layer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rule_layer import apply_rules  # noqa: E402
from train import default_subset_mask, load_dataset, make_splits  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from ablation_abcd import tag_hard_negative  # noqa: E402

OUT = ROOT / "artifacts" / "rule_layer"
DORA = ROOT / "artifacts" / "finetune_fair" / "dora"


def _thr() -> float:
    p = DORA / "threshold.json"
    if p.exists():
        return float(json.loads(p.read_text())["threshold"])
    return float(json.loads((DORA / "metrics.json").read_text())["threshold"])


def decide_auto(decision: str) -> int:
    """Auto-include only."""
    return 1 if decision == "relevant" else 0


def decide_pipeline(decision: str) -> int:
    """Screening keep-set: relevant or review (human still sees it)."""
    return 1 if decision in {"relevant", "review"} else 0


def metrics_from_decisions(
    y: np.ndarray, decisions: list[str], scores: np.ndarray, thr: float, *, mode: str
) -> dict:
    from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

    fn = decide_pipeline if mode == "pipeline" else decide_auto
    pred = np.asarray([fn(d) for d in decisions], dtype=int)
    tn, fp, fn_, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "mode": mode,
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "ap_model_scores": float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn_), "tp": int(tp)},
        "n_relevant": int(sum(d == "relevant" for d in decisions)),
        "n_review": int(sum(d == "review" for d in decisions)),
        "n_irrelevant": int(sum(d == "irrelevant" for d in decisions)),
        "threshold": thr,
    }


def apply_on_frame(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        res = apply_rules(
            str(getattr(r, "title", "") or ""),
            str(getattr(r, "abstract", "") or ""),
            str(getattr(r, "text", "") or ""),
            score=float(r.score),
            threshold=thr,
            design=str(getattr(r, "design", "") or ""),
        )
        rows.append(
            {
                "pmid": str(r.pmid),
                "label": int(r.label),
                "score": float(r.score),
                "model_decision": res.model_decision,
                "decision": res.decision,
                "overridden": res.overridden,
                "protected": res.protected_by_case,
                "rules": ";".join(h.code for h in res.rules_fired),
            }
        )
    return pd.DataFrame(rows)


def hard_spec(df_out: pd.DataFrame, papers: pd.DataFrame) -> dict:
    pmap = papers.set_index("pmid")
    hard = []
    for r in df_out.itertuples(index=False):
        if int(r.label) != 0:
            continue
        if str(r.pmid) not in pmap.index:
            continue
        row = pmap.loc[str(r.pmid)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        is_h, tag = tag_hard_negative(row)
        if is_h:
            hard.append((r.decision, tag, r.pmid))
    n = len(hard)
    if n == 0:
        return {"n_hard_neg": 0}
    tn = sum(1 for d, _, _ in hard if d != "relevant")
    # specificity for exclusion: not predicted relevant
    fp = sum(1 for d, _, _ in hard if d == "relevant")
    return {
        "n_hard_neg": n,
        "tn_or_review": tn,
        "fp_relevant": fp,
        "specificity": float(tn / n),
        "fp_rate": float(fp / n),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    thr = _thr()
    papers = load_dataset(ROOT / "data" / "labeled_papers.csv")
    subset = default_subset_mask(papers).to_numpy()
    _, _, test_idx = make_splits(papers, seed=42)
    test_sub = test_idx[subset[test_idx]]

    # Prefer exported DoRA test/gold scores (already enriched with text/design)
    test_pred = pd.read_csv(DORA / "test_predictions.csv", dtype={"pmid": str})
    gold_pred = pd.read_csv(DORA / "gold_predictions.csv", dtype={"pmid": str})
    test_pred["label"] = test_pred["label"].astype(int)
    gold_pred["label"] = gold_pred["label"].astype(int)

    # gold labels from gold_standard (authoritative after review)
    gold = pd.read_csv(ROOT / "data" / "gold_standard.csv", dtype={"pmid": str})
    gold = gold[gold["gold_label"].astype(str).isin(["0", "1"])].drop_duplicates("pmid", keep="last")
    gmap = gold.set_index("pmid")["gold_label"].astype(int)
    gold_pred = gold_pred[gold_pred["pmid"].isin(gmap.index)].copy()
    gold_pred["label"] = gold_pred["pmid"].map(gmap).astype(int)

    test_pmids = set(papers.iloc[test_sub]["pmid"].astype(str))
    test_df = test_pred[test_pred["pmid"].isin(test_pmids)].copy()
    gold_df = gold_pred.copy()

    test_out = apply_on_frame(test_df, thr)
    gold_out = apply_on_frame(gold_df, thr)

    y_t = test_out["label"].to_numpy(dtype=int)
    y_g = gold_out["label"].to_numpy(dtype=int)
    sc_t = test_out["score"].to_numpy(dtype=float)
    sc_g = gold_out["score"].to_numpy(dtype=float)

    model_t = metrics_from_decisions(y_t, test_out["model_decision"].tolist(), sc_t, thr, mode="pipeline")
    rules_t = metrics_from_decisions(y_t, test_out["decision"].tolist(), sc_t, thr, mode="pipeline")
    rules_t_auto = metrics_from_decisions(y_t, test_out["decision"].tolist(), sc_t, thr, mode="auto")
    model_g = metrics_from_decisions(y_g, gold_out["model_decision"].tolist(), sc_g, thr, mode="pipeline")
    rules_g = metrics_from_decisions(y_g, gold_out["decision"].tolist(), sc_g, thr, mode="pipeline")
    rules_g_auto = metrics_from_decisions(y_g, gold_out["decision"].tolist(), sc_g, thr, mode="auto")

    hard_model = hard_spec(
        test_out.assign(decision=test_out["model_decision"]), papers
    )
    hard_rules = hard_spec(test_out, papers)

    overrides = int(test_out["overridden"].sum())
    report = {
        "threshold": thr,
        "note": "pipeline metrics count relevant|review as positive (kept for human); auto = relevant only",
        "test": {
            "model_pipeline": model_t,
            "rules_pipeline": rules_t,
            "rules_auto": rules_t_auto,
            "n_overridden": overrides,
        },
        "gold49": {
            "model_pipeline": model_g,
            "rules_pipeline": rules_g,
            "rules_auto": rules_g_auto,
        },
        "hard_negatives_test": {"model": hard_model, "model_plus_rules": hard_rules},
        "rule_code_counts_test": (
            test_out.loc[test_out["rules"] != "", "rules"]
            .str.split(";")
            .explode()
            .value_counts()
            .to_dict()
        ),
    }

    test_out.to_csv(OUT / "test_decisions.csv", index=False)
    gold_out.to_csv(OUT / "gold_decisions.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Rule layer evaluation (DoRA + rules)",
        "",
        f"Threshold (gold-recal): **{thr:.4f}**",
        "",
        "Pipeline = keep if `relevant|review` (FN only when forced irrelevant).",
        "",
        "| set | system | R | P | F1 | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    rows = [
        ("test", "model_pipeline", report["test"]["model_pipeline"]),
        ("test", "rules_pipeline", report["test"]["rules_pipeline"]),
        ("test", "rules_auto", report["test"]["rules_auto"]),
        ("gold49", "model_pipeline", report["gold49"]["model_pipeline"]),
        ("gold49", "rules_pipeline", report["gold49"]["rules_pipeline"]),
        ("gold49", "rules_auto", report["gold49"]["rules_auto"]),
    ]
    for set_name, sys_name, m in rows:
        cm = m["confusion_matrix"]
        md.append(
            f"| {set_name} | {sys_name} | {m['recall']:.3f} | {m['precision']:.3f} | "
            f"{m['f1']:.3f} | {cm['fp']} | {cm['fn']} |"
        )
    md += [
        "",
        "## Hard negatives (test, label=0 & hard tag; FP = predicted relevant)",
        f"- model Spec={hard_model.get('specificity'):.3f} FP={hard_model.get('fp_relevant')} / n={hard_model.get('n_hard_neg')}",
        f"- +rules Spec=**{hard_rules.get('specificity'):.3f}** FP={hard_rules.get('fp_relevant')} / n={hard_rules.get('n_hard_neg')}",
        "",
        f"Test overrides: {overrides} / {len(test_out)}",
        "",
        "Rule codes (test):",
    ]
    for k, v in sorted(report["rule_code_counts_test"].items(), key=lambda kv: -kv[1]):
        md.append(f"- {k}: {v}")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
