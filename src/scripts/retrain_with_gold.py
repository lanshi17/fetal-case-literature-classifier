#!/usr/bin/env python3
"""Merge gold_standard.csv into labels and retrain the logistic head only.

Why a special protocol
----------------------
The 40 gold PMIDs came from the baseline *test* FN/FP set. A naïve label
override + the same GroupShuffleSplit would leave them in test and leave the
training matrix unchanged. This script therefore:

1. Applies gold 0/1 overrides (and keeps those rows even if previously weak/model).
2. **Holdout model**: train/val on non-gold default-subset papers; evaluate on
   the 40 gold papers (hard-case generalization under true labels).
3. **Deploy model**: force all gold PMIDs into the training fold; split the
   remaining papers into train/val/test; save as artifacts/model_gold.joblib
   for production use.

Frozen PubMedBERT features are reused from the existing cache (no re-encode).

Usage
-----
    .venv/bin/python src/scripts/retrain_with_gold.py
    .venv/bin/python src/scripts/retrain_with_gold.py --mode B
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from train import (  # noqa: E402
    binary_metrics,
    default_subset_mask,
    feature_matrix,
    load_dataset,
    make_splits,
    select_threshold,
)

logger = logging.getLogger("retrain_gold")

DEFAULT_DATA = ROOT / "data" / "labeled_papers.csv"
DEFAULT_GOLD = ROOT / "data" / "gold_standard.csv"
DEFAULT_OUT_DATA = ROOT / "data" / "labeled_papers_gold.csv"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_CACHE = ROOT / "artifacts" / "features"


def load_gold(path: Path) -> pd.DataFrame:
    g = pd.read_csv(path, encoding="utf-8-sig", dtype={"pmid": str})
    g["pmid"] = g["pmid"].astype(str)
    g["gold_label"] = g["gold_label"].astype(str).str.strip()
    g = g[g["gold_label"].isin(["0", "1"])].copy()
    g["gold_label"] = g["gold_label"].astype(int)
    if g["pmid"].duplicated().any():
        # keep last review
        g = g.drop_duplicates("pmid", keep="last")
    return g


def apply_gold(df: pd.DataFrame, gold: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    out["label_before_gold"] = out["label"]
    out["gold_override"] = False
    out["gold_reason_code"] = ""
    # ensure assignable dtypes for columns we overwrite
    out["confidence"] = out["confidence"].astype(str)
    out["weak"] = out["weak"].astype(str)
    out["label_source"] = out["label_source"].astype(str)
    out["label_reason"] = out["label_reason"].astype(str)
    gmap = gold.set_index("pmid")
    n_flip = 0
    for i, row in out.iterrows():
        pmid = str(row["pmid"] if "pmid" in out.columns else row["doc_id"])
        if pmid not in gmap.index:
            continue
        g = gmap.loc[pmid]
        new_lab = int(g["gold_label"])
        if int(row["label"]) != new_lab:
            n_flip += 1
        out.at[i, "label"] = new_lab
        out.at[i, "label_source"] = "curated_gold"
        out.at[i, "weak"] = "False"
        out.at[i, "weak_bool"] = False
        out.at[i, "confidence"] = "1.0"
        reason = str(g.get("gold_reason_text") or g.get("gold_reason_code") or "")
        note = str(g.get("gold_note") or "").strip()
        if note:
            reason = f"{reason} | note: {note}"
        out.at[i, "label_reason"] = reason
        out.at[i, "gold_override"] = True
        out.at[i, "gold_reason_code"] = str(g.get("gold_reason_code") or "")
    stats = {
        "n_gold": int(len(gold)),
        "n_matched": int(out["gold_override"].sum()),
        "n_label_flips": int(n_flip),
        "gold_pos": int((gold["gold_label"] == 1).sum()),
        "gold_neg": int((gold["gold_label"] == 0).sum()),
    }
    return out, stats


def training_mask(df: pd.DataFrame) -> pd.Series:
    """Default subset OR any gold-overridden row."""
    return default_subset_mask(df) | df["gold_override"].astype(bool)


def fit_lr(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    target_recall: float,
    C: float,
) -> dict[str, Any]:
    scaler = StandardScaler()
    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=0,
    )
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    clf.fit(X_train, y[train_idx])
    val_scores = clf.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y[val_idx], val_scores, target_recall)
    return {
        "model": clf,
        "scaler": scaler,
        "threshold": float(threshold),
        "val": binary_metrics(y[val_idx], val_scores, threshold),
        "val_scores": val_scores,
    }


def scores_for(clf, scaler, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return clf.predict_proba(scaler.transform(X[idx]))[:, 1]


def split_non_gold(
    df: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """64/16/20 on rows whose gold_override is False."""
    non_gold = np.where(~df["gold_override"].to_numpy())[0]
    sub = df.iloc[non_gold].reset_index(drop=True)
    tr, va, te = make_splits(sub, seed=seed)
    return non_gold[tr], non_gold[va], non_gold[te]


def deploy_split(
    df: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Force gold into train; split the rest into train/val/test."""
    gold_idx = np.where(df["gold_override"].to_numpy())[0]
    rest_train, rest_val, rest_test = split_non_gold(df, seed=seed)
    train_idx = np.unique(np.concatenate([gold_idx, rest_train]))
    return train_idx, rest_val, rest_test


def bundle(
    fit: dict[str, Any],
    *,
    feature_mode: str,
    encoder_cfg: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": fit["model"],
        "scaler": fit["scaler"],
        "threshold": fit["threshold"],
        "encoder_family": "pubmedbert",
        "feature_mode": feature_mode,
        "include_aux": feature_mode == "C",
        "feature_dim": int(fit["model"].n_features_in_),
        "encoder": encoder_cfg,
        "review_band": 0.15,
        **extra,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--out-data", type=Path, default=DEFAULT_OUT_DATA)
    ap.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--mode", default="B", choices=["A", "B", "C"])
    args = ap.parse_args()

    if not args.gold.exists():
        raise SystemExit(f"missing gold file: {args.gold}")

    df0 = load_dataset(args.data)
    gold = load_gold(args.gold)
    df, gold_stats = apply_gold(df0, gold)
    args.out_data.parent.mkdir(parents=True, exist_ok=True)
    # persist gold-merged table (drop helper cols that may confuse consumers? keep them)
    df.to_csv(args.out_data, index=False)
    logger.info("Wrote %s (%s)", args.out_data, gold_stats)

    # Features from existing cache only (no re-encode). Labels come from df.
    cfg_path = args.artifacts / "config.json"
    enc_cfg: dict[str, Any] = {
        "model_path": None,
        "max_length": 512,
        "normalize": True,
        "chunk_tokens": 480,
        "max_chunks": 12,
        "keep_first": 2,
        "pooling": "concat[mean(all), mean(first 2)] L2-normed -> 1536",
    }
    cache_path = None
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        enc = cfg.get("encoder") or {}
        enc_cfg = {
            "model_path": enc.get("model_path") or enc.get("base_path"),
            "max_length": enc.get("max_length", 512),
            "normalize": enc.get("normalize", True),
            "chunk_tokens": enc.get("chunk_tokens", 480),
            "max_chunks": enc.get("max_chunks", 12),
            "keep_first": enc.get("keep_first", 2),
            "pooling": enc.get(
                "pooling",
                "concat[mean(all), mean(first 2)] L2-normed -> 1536",
            ),
        }
        cache_path = Path(cfg["feature_cache"]) if cfg.get("feature_cache") else None

    if cache_path is None or not cache_path.exists():
        hits = sorted(args.cache_dir.glob("pubmedbert_mc*_n974_*.npz"))
        if not hits:
            raise SystemExit(
                f"No PubMedBERT feature cache under {args.cache_dir}; run train.py once before gold retrain."
            )
        cache_path = hits[-1]

    logger.info("Loading feature cache %s (no re-encode)", cache_path)
    data = np.load(cache_path, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    if list(arrays["doc_ids"].astype(str)) != df["doc_id"].astype(str).tolist():
        raise SystemExit("feature cache doc_id order does not match labeled_papers.csv")
    cache_meta = {"cache_path": str(cache_path)}
    meta_sidecar = cache_path.with_suffix(".json")
    if meta_sidecar.exists():
        cache_meta.update(json.loads(meta_sidecar.read_text()))

    X_all = feature_matrix(arrays, args.mode)
    y_all = df["label"].to_numpy(dtype=np.int64)

    mask = training_mask(df).to_numpy()
    work_idx = np.where(mask)[0]
    df_w = df.iloc[work_idx].reset_index(drop=True)
    X = X_all[work_idx]
    y = y_all[work_idx]
    # remap gold_override onto work frame
    logger.info(
        "Working set %d (default|gold); gold in working set=%d",
        len(df_w),
        int(df_w["gold_override"].sum()),
    )

    out_dir = args.artifacts / "gold_retrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1) Holdout: train without gold, eval on gold -----
    non_gold_local = np.where(~df_w["gold_override"].to_numpy())[0]
    gold_local = np.where(df_w["gold_override"].to_numpy())[0]
    if len(gold_local) == 0:
        raise SystemExit("no gold rows in working set")
    sub = df_w.iloc[non_gold_local].reset_index(drop=True)
    tr_r, va_r, te_r = make_splits(sub, seed=args.seed)
    # map back to df_w indices
    tr_h = non_gold_local[tr_r]
    va_h = non_gold_local[va_r]
    # unused held-out non-gold test for secondary report
    te_h_nongold = non_gold_local[te_r]

    fit_h = fit_lr(X, y, tr_h, va_h, target_recall=args.target_recall, C=args.C)
    gold_scores = scores_for(fit_h["model"], fit_h["scaler"], X, gold_local)
    nongold_test_scores = scores_for(fit_h["model"], fit_h["scaler"], X, te_h_nongold)
    holdout = {
        "protocol": "train/val on non-gold default-subset; evaluate on curated gold PMIDs",
        "n_train": int(len(tr_h)),
        "n_val": int(len(va_h)),
        "n_gold_eval": int(len(gold_local)),
        "n_nongold_test": int(len(te_h_nongold)),
        "threshold": fit_h["threshold"],
        "val": fit_h["val"],
        "gold_eval": binary_metrics(y[gold_local], gold_scores, fit_h["threshold"]),
        "nongold_test": binary_metrics(y[te_h_nongold], nongold_test_scores, fit_h["threshold"]),
        "gold_label_counts": {
            "pos": int((y[gold_local] == 1).sum()),
            "neg": int((y[gold_local] == 0).sum()),
        },
    }
    logger.info(
        "Holdout gold_eval R=%.3f P=%.3f F1=%.3f AP=%.3f (thr=%.4f)",
        holdout["gold_eval"]["recall"],
        holdout["gold_eval"]["precision"],
        holdout["gold_eval"]["f1"],
        holdout["gold_eval"]["ap"],
        holdout["threshold"],
    )

    dump(
        bundle(
            fit_h,
            feature_mode=args.mode,
            encoder_cfg=enc_cfg,
            extra={
                "train_protocol": "holdout_exclude_gold",
                "target_recall": args.target_recall,
                "C": args.C,
            },
        ),
        out_dir / "model_holdout.joblib",
    )

    # ----- 2) Deploy: gold forced into train -----
    # Build deploy split on df_w indices
    gold_idx_w = gold_local
    rest = non_gold_local
    sub2 = df_w.iloc[rest].reset_index(drop=True)
    tr_r2, va_r2, te_r2 = make_splits(sub2, seed=args.seed + 7)
    tr_d = np.unique(np.concatenate([gold_idx_w, rest[tr_r2]]))
    va_d = rest[va_r2]
    te_d = rest[te_r2]

    fit_d = fit_lr(X, y, tr_d, va_d, target_recall=args.target_recall, C=args.C)
    test_scores = scores_for(fit_d["model"], fit_d["scaler"], X, te_d)
    # Also score gold (in-train — report as train-gold diagnostic only)
    gold_intrain_scores = scores_for(fit_d["model"], fit_d["scaler"], X, gold_idx_w)
    deploy = {
        "protocol": "gold PMIDs forced into train; val/test from remaining non-gold",
        "n_train": int(len(tr_d)),
        "n_val": int(len(va_d)),
        "n_test": int(len(te_d)),
        "n_gold_in_train": int(len(gold_idx_w)),
        "threshold": fit_d["threshold"],
        "val": fit_d["val"],
        "test": binary_metrics(y[te_d], test_scores, fit_d["threshold"]),
        "gold_in_train_diagnostic": binary_metrics(
            y[gold_idx_w], gold_intrain_scores, fit_d["threshold"]
        ),
        "note": "gold_in_train_diagnostic is optimistic (same papers seen in training)",
    }
    logger.info(
        "Deploy test R=%.3f P=%.3f F1=%.3f AP=%.3f (thr=%.4f); gold_in_train R=%.3f",
        deploy["test"]["recall"],
        deploy["test"]["precision"],
        deploy["test"]["f1"],
        deploy["test"]["ap"],
        deploy["threshold"],
        deploy["gold_in_train_diagnostic"]["recall"],
    )

    deploy_bundle = bundle(
        fit_d,
        feature_mode=args.mode,
        encoder_cfg=enc_cfg,
        extra={
            "train_protocol": "gold_forced_into_train",
            "target_recall": args.target_recall,
            "C": args.C,
            "gold_path": str(args.gold),
            "data_path": str(args.out_data),
        },
    )
    model_gold_path = args.artifacts / "model_gold.joblib"
    dump(deploy_bundle, model_gold_path)
    dump(deploy_bundle, out_dir / "model_gold.joblib")

    # Predictions on deploy test + gold for audit
    pred_rows = []
    for local_i, score in zip(te_d, test_scores):
        r = df_w.iloc[int(local_i)]
        pred_rows.append(
            {
                "doc_id": r["doc_id"],
                "title": r["title"],
                "label": int(r["label"]),
                "gold_override": bool(r["gold_override"]),
                "score": float(score),
                "threshold": deploy["threshold"],
                "decision": "relevant" if score >= deploy["threshold"] else (
                    "review" if score >= deploy["threshold"] - 0.15 else "irrelevant"
                ),
                "split": "test",
            }
        )
    for local_i, score in zip(gold_idx_w, gold_intrain_scores):
        r = df_w.iloc[int(local_i)]
        pred_rows.append(
            {
                "doc_id": r["doc_id"],
                "title": r["title"],
                "label": int(r["label"]),
                "gold_override": True,
                "score": float(score),
                "threshold": deploy["threshold"],
                "decision": "relevant" if score >= deploy["threshold"] else (
                    "review" if score >= deploy["threshold"] - 0.15 else "irrelevant"
                ),
                "split": "gold_in_train",
            }
        )
    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv", index=False)

    # Compare vs baseline model on the same gold labels
    baseline_cmp = None
    baseline_path = args.artifacts / "model.joblib"
    if baseline_path.exists():
        from joblib import load

        base = load(baseline_path)
        # map gold pmid -> row in full df / X_all
        gold_pmids = set(df.loc[df["gold_override"], "doc_id"].astype(str))
        full_ids = df["doc_id"].astype(str).tolist()
        g_full = np.array([i for i, d in enumerate(full_ids) if d in gold_pmids], dtype=int)
        Xs = base["scaler"].transform(feature_matrix(arrays, base.get("feature_mode", args.mode))[g_full])
        sc = base["model"].predict_proba(Xs)[:, 1]
        thr = float(base["threshold"])
        y_g = df.iloc[g_full]["label"].to_numpy(dtype=int)
        baseline_cmp = {
            "model": str(baseline_path),
            "threshold": thr,
            "gold_eval": binary_metrics(y_g, sc, thr),
        }

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold_stats": gold_stats,
        "feature_mode": args.mode,
        "feature_dim": int(X.shape[1]),
        "working_set_n": int(len(df_w)),
        "cache": cache_meta.get("cache_path"),
        "holdout": holdout,
        "deploy": deploy,
        "baseline_on_gold": baseline_cmp,
        "artifacts": {
            "model_gold": str(model_gold_path),
            "model_holdout": str(out_dir / "model_holdout.joblib"),
            "labeled_papers_gold": str(args.out_data),
            "predictions": str(out_dir / "predictions.csv"),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.artifacts / "metrics_gold.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # lightweight config sidecar for predict.py users
    gold_cfg = {
        "model_path": str(model_gold_path),
        "threshold": deploy["threshold"],
        "review_lower_bound": max(0.0, deploy["threshold"] - 0.15),
        "feature_mode": args.mode,
        "feature_dim": int(X.shape[1]),
        "train_protocol": "gold_forced_into_train",
        "target_recall": args.target_recall,
        "C": args.C,
        "gold_path": str(args.gold),
        "data_path": str(args.out_data),
        "holdout_gold_recall": holdout["gold_eval"]["recall"],
        "holdout_gold_precision": holdout["gold_eval"]["precision"],
        "deploy_test_recall": deploy["test"]["recall"],
        "deploy_test_precision": deploy["test"]["precision"],
        "baseline_gold_recall": (baseline_cmp or {}).get("gold_eval", {}).get("recall"),
        "baseline_gold_precision": (baseline_cmp or {}).get("gold_eval", {}).get("precision"),
    }
    (args.artifacts / "config_gold.json").write_text(json.dumps(gold_cfg, indent=2), encoding="utf-8")

    print(json.dumps({
        "gold_stats": gold_stats,
        "holdout_gold": holdout["gold_eval"],
        "deploy_test": deploy["test"],
        "baseline_on_gold": (baseline_cmp or {}).get("gold_eval"),
        "model_gold": str(model_gold_path),
        "metrics": str(out_dir / "metrics.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
