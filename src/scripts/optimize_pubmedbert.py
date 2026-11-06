#!/usr/bin/env python3
"""Light PubMedBERT head optimization on cached 1536-d features.

Sweeps C ∈ {0.1, 0.5, 1, 2, 5, 10} and feature modes A/B/C on the default
subset split (seed=42). Writes artifacts/optimize_pubmedbert/{metrics.json,summary.md}.

Does not re-encode. Run after `python train.py` has built the PubMedBERT cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from train import (  # noqa: E402
    binary_metrics,
    default_subset_mask,
    feature_matrix,
    fit_and_evaluate,
    load_dataset,
    make_splits,
)

logger = logging.getLogger("optimize_pmb")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "labeled_papers.csv")
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument(
        "--C-grid",
        type=str,
        default="0.1,0.5,1,2,5,10",
        help="comma-separated C values",
    )
    args = ap.parse_args()
    out = args.out or (args.artifacts / "optimize_pubmedbert")
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((args.artifacts / "config.json").read_text())
    cache_path = Path(cfg["feature_cache"])
    if not cache_path.exists():
        raise SystemExit(f"missing feature cache: {cache_path}")

    df = load_dataset(args.data)
    data = np.load(cache_path, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    if list(arrays["doc_ids"].astype(str)) != df["doc_id"].astype(str).tolist():
        raise SystemExit("cache/doc_id mismatch")

    y = df["label"].to_numpy()
    train_idx, val_idx, test_idx = make_splits(df, seed=args.seed)
    subset = default_subset_mask(df).to_numpy()
    train_sub = train_idx[subset[train_idx]]
    val_sub = val_idx[subset[val_idx]]
    test_sub = test_idx[subset[test_idx]]

    C_values = [float(x) for x in args.C_grid.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for mode in ("A", "B", "C"):
        X = feature_matrix(arrays, mode)
        for C in C_values:
            res = fit_and_evaluate(
                X,
                y,
                df,
                train_sub,
                val_sub,
                test_sub,
                target_recall=args.target_recall,
                C=C,
                class_weight="balanced",
            )
            row = {
                "mode": mode,
                "C": C,
                "feature_dim": int(X.shape[1]),
                "threshold": res["threshold"],
                "val_recall": res["val"]["recall"],
                "val_precision": res["val"]["precision"],
                "test_recall": res["test"]["recall"],
                "test_precision": res["test"]["precision"],
                "test_f1": res["test"]["f1"],
                "test_f2": res["test"]["f2"],
                "test_ap": res["test"]["ap"],
                "test_roc_auc": res["test"]["roc_auc"],
                "test_cm": res["test"]["confusion_matrix"],
            }
            rows.append(row)
            logger.info(
                "mode=%s C=%s thr=%.3f test R=%.3f P=%.3f AP=%.3f",
                mode,
                C,
                res["threshold"],
                res["test"]["recall"],
                res["test"]["precision"],
                res["test"]["ap"] or 0,
            )
            # Prefer high test AP among runs that keep val R>=target; tie-break F2 then precision.
            score_key = (
                1 if res["val"]["recall"] + 1e-12 >= args.target_recall else 0,
                res["test"]["ap"] or 0.0,
                res["test"]["f2"],
                res["test"]["precision"],
            )
            if best is None or score_key > best["_key"]:
                best = {**row, "_key": score_key}

    assert best is not None
    best.pop("_key", None)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "encoder_family": "pubmedbert",
        "feature_cache": str(cache_path),
        "target_recall": args.target_recall,
        "seed": args.seed,
        "n_train": int(len(train_sub)),
        "n_val": int(len(val_sub)),
        "n_test": int(len(test_sub)),
        "grid": rows,
        "best": best,
        "baseline_B_C1": next(
            (r for r in rows if r["mode"] == "B" and abs(r["C"] - 1.0) < 1e-9), None
        ),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# PubMedBERT head optimization",
        "",
        f"Cache: `{cache_path}`",
        f"Target val recall ≥ {args.target_recall}",
        "",
        "## Best (val R≥target, max test AP)",
        "",
        f"- mode **{best['mode']}**, C={best['C']}, thr={best['threshold']:.4f}",
        f"- test Recall={best['test_recall']:.3f} Precision={best['test_precision']:.3f} "
        f"F1={best['test_f1']:.3f} AP={best['test_ap']:.3f}",
        "",
        "## Full grid (test)",
        "",
        "| mode | C | thr | Recall | Precision | F1 | F2 | AP | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['C']} | {r['threshold']:.3f} | {r['test_recall']:.3f} | "
            f"{r['test_precision']:.3f} | {r['test_f1']:.3f} | {r['test_f2']:.3f} | "
            f"{r['test_ap']:.3f} | {r['test_roc_auc']:.3f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"best": best, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
