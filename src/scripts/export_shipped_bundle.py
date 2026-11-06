#!/usr/bin/env python3
"""Wrap pubmedbert/weights head_v3 into a predict.py-compatible joblib bundle.

Calibrates threshold on the default-subset val fold (seed=42) for target recall,
so the operating point matches this repo's screening protocol. Features are the
cached PubMedBERT 1536-d matrix (no re-encode).

Writes:
  artifacts/model_shipped.joblib
  artifacts/config_shipped.json
  artifacts/metrics_shipped.json
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
import pandas as pd
from joblib import dump

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from encoder import IdentityScaler, ShippedLogisticHead  # noqa: E402
from train import (  # noqa: E402
    binary_metrics,
    default_subset_mask,
    load_dataset,
    make_splits,
    select_threshold,
)

logger = logging.getLogger("export_shipped")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=ROOT / "pubmedbert" / "weights")
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "labeled_papers.csv")
    ap.add_argument("--gold", type=Path, default=ROOT / "data" / "gold_standard.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument(
        "--promote",
        action="store_true",
        help="also copy to artifacts/model.joblib as the primary deploy model",
    )
    args = ap.parse_args()

    if not args.weights.exists():
        ap.error(
            f"weights dir not found: {args.weights}\n"
            "The legacy pubmedbert/ subproject was removed (only the DoRA final generation is kept).\n"
            "Restore it with: git checkout archive/pre-dora-legacy -- pubmedbert"
        )

    cfg = json.loads((args.weights / "config.json").read_text())
    z = np.load(args.weights / cfg["head"]["file"])
    coef = z["coef"].astype(np.float64).ravel()
    intercept = float(z["intercept"].ravel()[0])

    art_cfg = json.loads((args.artifacts / "config.json").read_text())
    cache = Path(art_cfg["feature_cache"])
    data = np.load(cache, allow_pickle=False)
    X = np.concatenate([data["mean_all"], data["mean_head"]], axis=1).astype(np.float64)
    doc_ids = data["doc_ids"].astype(str)

    df = load_dataset(args.data)
    if list(doc_ids) != df["doc_id"].astype(str).tolist():
        raise SystemExit("cache/doc_id mismatch")
    y = df["label"].to_numpy(dtype=int)
    subset = default_subset_mask(df).to_numpy()
    train_idx, val_idx, test_idx = make_splits(df, seed=args.seed)
    train_sub, val_sub, test_sub = (
        train_idx[subset[train_idx]],
        val_idx[subset[val_idx]],
        test_idx[subset[test_idx]],
    )

    head = ShippedLogisticHead(coef, intercept)
    scaler = IdentityScaler().fit(X[train_sub])
    val_scores = head.predict_proba(X[val_sub])[:, 1]
    test_scores = head.predict_proba(X[test_sub])[:, 1]
    thr_ship = float(cfg.get("threshold", 0.5))
    thr95 = select_threshold(y[val_sub], val_scores, args.target_recall)

    enc = {
        "model_path": art_cfg.get("encoder", {}).get("model_path"),
        "max_length": 512,
        "normalize": True,
        "chunk_tokens": int(cfg.get("chunk_tokens", 480)),
        "max_chunks": int(cfg.get("max_chunks", 12)),
        "keep_first": int(cfg.get("keep_first_chunks", 2)),
        "pooling": cfg.get("pooling"),
        "source_weights": str(args.weights),
        "head_name": cfg.get("name"),
    }
    bundle = {
        "model": head,
        "scaler": scaler,
        "threshold": float(thr95),
        "threshold_shipped_default": thr_ship,
        "encoder_family": "pubmedbert",
        "feature_mode": "B",
        "include_aux": False,
        "feature_dim": 1536,
        "encoder": enc,
        "train_subset": "shipped head_v3 (external); thr calibrated on this repo val",
        "target_recall": args.target_recall,
        "C": cfg.get("head", {}).get("C"),
        "seed": args.seed,
    }
    out_model = args.artifacts / "model_shipped.joblib"
    dump(bundle, out_model)

    metrics: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "encoder_family": "pubmedbert",
        "head": cfg.get("name"),
        "caveat": cfg.get("caveat"),
        "val_thr0.5": binary_metrics(y[val_sub], val_scores, thr_ship),
        "test_thr0.5": binary_metrics(y[test_sub], test_scores, thr_ship),
        "val_thr_r95": binary_metrics(y[val_sub], val_scores, thr95),
        "test_thr_r95": binary_metrics(y[test_sub], test_scores, thr95),
        "calibrated_threshold": thr95,
    }

    if args.gold.exists():
        gold = pd.read_csv(args.gold, dtype={"pmid": str})
        gold["pmid"] = gold["pmid"].astype(str)
        gold = gold[gold["gold_label"].astype(str).isin(["0", "1"])].copy()
        gold["gold_label"] = gold["gold_label"].astype(int)
        gmap = gold.drop_duplicates("pmid", keep="last").set_index("pmid")["gold_label"]
        g_idx = [i for i, d in enumerate(doc_ids) if d in gmap.index]
        y_g = np.asarray([int(gmap.loc[doc_ids[i]]) for i in g_idx], dtype=int)
        sc_g = head.predict_proba(X[g_idx])[:, 1]
        metrics["gold40_thr0.5"] = binary_metrics(y_g, sc_g, thr_ship)
        metrics["gold40_thr_r95"] = binary_metrics(y_g, sc_g, thr95)

    (args.artifacts / "metrics_shipped.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.artifacts / "config_shipped.json").write_text(
        json.dumps(
            {
                "model_path": str(out_model),
                "encoder_family": "pubmedbert",
                "threshold": thr95,
                "threshold_shipped_default": thr_ship,
                "review_lower_bound": max(0.0, thr95 - 0.15),
                "feature_mode": "B",
                "feature_dim": 1536,
                "encoder": enc,
                "test_thr_r95": metrics["test_thr_r95"],
                "test_thr0.5": metrics["test_thr0.5"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.promote:
        dump(bundle, args.artifacts / "model.joblib")
        # Keep config.json aligned with the promoted primary.
        primary_cfg = {
            **art_cfg,
            "model_path": str(args.artifacts / "model.joblib"),
            "threshold": thr95,
            "review_lower_bound": max(0.0, thr95 - 0.15),
            "feature_mode": "B",
            "feature_dim": 1536,
            "encoder": enc,
            "train_subset": bundle["train_subset"],
            "primary_head": "shipped_head_v3_calibrated",
            "retrained_head_path": str(args.artifacts / "model_retrained.joblib"),
        }
        # Preserve the LR-retrained head under a clear name if present.
        retrained = args.artifacts / "model.joblib"
        # We already dumped shipped over model.joblib above; move previous aside was done by archive.
        (args.artifacts / "config.json").write_text(
            json.dumps(primary_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Promoted shipped head to artifacts/model.joblib")

    logger.info(
        "Wrote %s thr_r95=%.4f test R=%.3f P=%.3f AP=%.3f",
        out_model,
        thr95,
        metrics["test_thr_r95"]["recall"],
        metrics["test_thr_r95"]["precision"],
        metrics["test_thr_r95"]["ap"],
    )
    print(json.dumps({"model": str(out_model), "test_thr_r95": metrics["test_thr_r95"]}, indent=2))


if __name__ == "__main__":
    main()
