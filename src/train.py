#!/usr/bin/env python3
"""Train the PubMedBERT + LogisticRegression screening baseline.

Pipeline
--------
1. Read data/labeled_papers.csv.
2. Encode each document once with frozen NeuML PubMedBERT
   (mean-all || mean-first-2 → 1536-d, optional lexical aux) and cache under
   artifacts/features/.
3. Split groups (group_id = PMID) roughly 64/16/20 with GroupShuffleSplit.
4. Fit StandardScaler + LogisticRegression; choose the threshold on validation
   for a target recall (default 0.95).
5. Report test metrics, ablations A/B/C, a full-label control, and a
   full-text-availability stratified report.

Outputs
-------
artifacts/model.joblib     primary model bundle
artifacts/model_all.joblib full-label control bundle
artifacts/config.json
artifacts/metrics.json
artifacts/test_predictions.csv

Usage
-----
    python src/train.py
    python src/train.py --limit 64 --max-chunks 4 --artifacts artifacts/smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from encoder import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_KEEP_FIRST,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_MODEL,
    EMBED_DIM,
    document_features_batch,
    feature_dim,
)

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("train")

DEFAULT_DATA = ROOT / "data" / "labeled_papers.csv"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_TARGET_RECALL = 0.95
FEATURE_MODES = {
    "A": "mean(first 2 chunks) only — title/abstract heavy",
    "B": "mean(all) || mean(first 2) — PubMedBERT 1536-d",
    "C": "B + 2 lexical auxiliary features",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def as_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def load_dataset(path: Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False, dtype={"pmid": str, "doc_id": str})
    required = {"doc_id", "group_id", "title", "abstract", "text", "label", "weak", "label_source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["doc_id"] = df["doc_id"].astype(str)
    df["group_id"] = df["group_id"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["weak_bool"] = as_bool_series(df["weak"])
    for col in ("title", "abstract", "text", "label_source"):
        df[col] = df[col].fillna("").astype(str)
    df["full_text_available"] = (
        df["full_text_available"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
        if "full_text_available" in df.columns
        else df["text"].str.strip().ne("")
    )
    if limit:
        df = df.head(limit).reset_index(drop=True)
    return df


def default_subset_mask(df: pd.DataFrame) -> pd.Series:
    """weak=False and label_source != model."""
    return (~df["weak_bool"]) & (df["label_source"].str.strip().str.lower() != "model")


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float = DEFAULT_TARGET_RECALL,
) -> float:
    """Pick a threshold reaching >= target recall on validation, maximizing precision."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return 0.5
    if y_true.sum() == 0:
        return 0.5
    unique = np.unique(scores)
    candidates = np.unique(
        np.concatenate(([0.0], (unique[:-1] + unique[1:]) / 2.0, [float(unique[-1]) + 1e-6], [1.0]))
    )
    best_t: float | None = None
    best_precision = -1.0
    best_recall = -1.0
    for t in candidates:
        pred = scores >= t
        rec = recall_score(y_true, pred, zero_division=0)
        if rec + 1e-12 < target_recall:
            continue
        prec = precision_score(y_true, pred, zero_division=0)
        if prec > best_precision + 1e-12 or (abs(prec - best_precision) <= 1e-12 and rec > best_recall):
            best_t, best_precision, best_recall = float(t), float(prec), float(rec)
    if best_t is None:
        return float(scores.min()) - 1e-6
    return best_t


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int(len(y_true) - y_true.sum()),
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "ap": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "predicted_positive": int(pred.sum()),
        "mean_score_positive": float(scores[y_true == 1].mean()) if (y_true == 1).any() else None,
        "mean_score_negative": float(scores[y_true == 0].mean()) if (y_true == 0).any() else None,
    }
    return out


def stratified_report(df: pd.DataFrame, y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    flags = df["full_text_available"].to_numpy(dtype=bool)
    for name, mask in (
        ("full_text_available", flags),
        ("abstract_only", ~flags),
    ):
        n = int(mask.sum())
        if n == 0:
            out[name] = {"n": 0, "note": "empty stratum"}
            continue
        yy = y[mask]
        ss = scores[mask]
        if len(np.unique(yy)) < 2:
            out[name] = {
                "n": n,
                "positives": int(yy.sum()),
                "note": "single-class stratum; threshold metrics are not informative",
                "recall": float(recall_score(yy, ss >= threshold, zero_division=0)),
                "precision": float(precision_score(yy, ss >= threshold, zero_division=0)),
            }
            continue
        out[name] = binary_metrics(yy, ss, threshold)
    return out


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------
def dataset_fingerprint(df: pd.DataFrame, params: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(params, sort_keys=True).encode())
    for _, r in df.iterrows():
        h.update(str(r["doc_id"]).encode())
        h.update(str(len(str(r["title"]))).encode())
        h.update(str(len(str(r["abstract"]))).encode())
        h.update(str(len(str(r["text"]))).encode())
        h.update(hashlib.sha1(str(r["text"])[:2000].encode("utf-8", "ignore")).digest())
    return h.hexdigest()


def build_feature_cache(
    df: pd.DataFrame,
    cache_dir: Path,
    *,
    encoder_kwargs: dict[str, Any],
    feature_kwargs: dict[str, Any],
    batch_size: int,
    device: str | None,
    force: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"pubmedbert_mc{feature_kwargs['max_chunks']}"
        f"_ct{feature_kwargs['chunk_tokens']}"
        f"_n{len(df)}"
    )
    params = {
        **encoder_kwargs,
        **feature_kwargs,
        "n_docs": len(df),
        "doc_ids": df["doc_id"].tolist()[:5],
        "encoder_family": "pubmedbert",
    }
    fp = dataset_fingerprint(df, params)
    cache_path = cache_dir / f"{tag}_{fp[:10]}.npz"
    meta_path = cache_path.with_suffix(".json")

    if cache_path.exists() and not force:
        logger.info("Loading feature cache %s", cache_path)
        data = np.load(cache_path, allow_pickle=False)
        arrays = {k: data[k] for k in data.files}
        if list(arrays["doc_ids"].astype(str)) == df["doc_id"].astype(str).tolist():
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            logger.info("Cache hit: %s", {k: v.shape for k, v in arrays.items() if hasattr(v, "shape")})
            return arrays, meta
        logger.warning("Cache doc_ids mismatch; re-encoding")

    # Reuse compare-script cache when shapes match (1536-d pooled features).
    legacy = (
        ROOT
        / "artifacts"
        / "compare_pubmedbert"
        / f"pubmedbert_features_n{len(df)}_mc{feature_kwargs['max_chunks']}.npz"
    )
    if legacy.exists() and not force:
        data = np.load(legacy, allow_pickle=False)
        if list(data["doc_ids"].astype(str)) == df["doc_id"].astype(str).tolist() and data["X"].shape[1] == 1536:
            logger.info("Importing pooled features from %s", legacy)
            X = data["X"].astype(np.float32)
            # Still need aux; compute cheaply without the encoder.
            from encoder import aux_features

            aux = np.stack(
                [
                    aux_features(str(r.title), str(r.abstract), str(r.text))
                    for r in df.itertuples(index=False)
                ]
            ).astype(np.float32)
            arrays = {
                "doc_ids": np.asarray(df["doc_id"].astype(str).tolist(), dtype="U"),
                "mean_all": X[:, :EMBED_DIM].copy(),
                "mean_head": X[:, EMBED_DIM:1536].copy(),
                "aux": aux,
                "full_text_available": df["full_text_available"].to_numpy(dtype=bool),
                "labels": df["label"].to_numpy(dtype=np.int64),
            }
            meta = {
                "cache_path": str(cache_path),
                "imported_from": str(legacy),
                "fingerprint": fp,
                "n_docs": len(df),
                "feature_kwargs": feature_kwargs,
                "encoder": {"model_path": encoder_kwargs.get("model_path"), "imported": True},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            np.savez_compressed(cache_path, **arrays)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return arrays, meta

    logger.info("Encoding %d documents with PubMedBERT (cached afterwards)...", len(df))
    from encoder import load_encoder

    enc = load_encoder(
        encoder_kwargs.get("model_path"),
        device=device,
        max_length=encoder_kwargs.get("max_length", 512),
        normalize=encoder_kwargs.get("normalize", True),
        batch_size=batch_size,
    )
    t0 = time.time()
    records = df[["title", "abstract", "text"]].to_dict("records")
    feats, metas = document_features_batch(
        records,
        enc,
        include_fulltext=True,
        include_aux=True,
        chunk_tokens=feature_kwargs["chunk_tokens"],
        max_chunks=feature_kwargs["max_chunks"],
        keep_first=feature_kwargs["keep_first"],
        batch_size=batch_size,
        show_progress=True,
    )
    elapsed = time.time() - t0

    arrays = {
        "doc_ids": np.asarray(df["doc_id"].astype(str).tolist(), dtype="U"),
        "mean_all": feats[:, :EMBED_DIM].copy(),
        "mean_head": feats[:, EMBED_DIM : 2 * EMBED_DIM].copy(),
        "aux": feats[:, 2 * EMBED_DIM : 2 * EMBED_DIM + 2].copy(),
        "full_text_available": df["full_text_available"].to_numpy(dtype=bool),
        "labels": df["label"].to_numpy(dtype=np.int64),
    }
    np.savez_compressed(cache_path, **arrays)
    meta = {
        "cache_path": str(cache_path),
        "fingerprint": fp,
        "n_docs": len(df),
        "feature_kwargs": feature_kwargs,
        "encoder": {
            "model_path": enc.model_path,
            "device": enc.device,
            "max_length": enc.max_length,
            "normalize": enc.normalize,
            **enc.metadata,
        },
        "encoding_seconds": round(elapsed, 1),
        "docs_per_second": round(len(df) / max(elapsed, 1e-9), 2),
        "mean_chunks_per_doc": round(float(np.mean([m["n_chunks"] for m in metas])), 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Cached features -> %s (%.1fs, %.2f docs/s)", cache_path, elapsed, len(df) / max(elapsed, 1e-9))
    return arrays, meta


def feature_matrix(arrays: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode == "A":
        return arrays["mean_head"]
    if mode == "B":
        return np.concatenate([arrays["mean_all"], arrays["mean_head"]], axis=1)
    if mode == "C":
        return np.concatenate([arrays["mean_all"], arrays["mean_head"], arrays["aux"]], axis=1)
    raise ValueError(f"unknown feature mode: {mode}")


# ---------------------------------------------------------------------------
# Splits / training
# ---------------------------------------------------------------------------
def make_splits(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    train_frac: float = 0.64,
    val_frac: float = 0.16,
    test_frac: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = df["group_id"].to_numpy()
    y = df["label"].to_numpy()
    n = len(df)
    target = (train_frac, val_frac, test_frac)
    for attempt in range(60):
        rs = seed + attempt
        gss = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=rs)
        train_idx, temp_idx = next(gss.split(np.zeros(n), y, groups))
        val_share = val_frac / (val_frac + test_frac)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=val_share, random_state=rs + 10_000)
        val_rel, test_rel = next(
            gss2.split(np.zeros(len(temp_idx)), y[temp_idx], groups[temp_idx])
        )
        val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]
        ok = True
        for idx in (train_idx, val_idx, test_idx):
            if idx.size == 0 or len(np.unique(y[idx])) < 2:
                ok = False
                break
        if ok:
            logger.info(
                "Split seed=%d sizes train/val/test=%d/%d/%d (targets %.0f/%.0f/%.0f%%)",
                rs,
                len(train_idx),
                len(val_idx),
                len(test_idx),
                *(100 * x for x in target),
            )
            return train_idx, val_idx, test_idx
    raise RuntimeError("could not build a group split with both classes in every split")


def fit_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    target_recall: float,
    C: float,
    class_weight: str | None,
) -> dict[str, Any]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])
    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=3000,
        solver="lbfgs",
        random_state=0,
    )
    clf.fit(X_train, y[train_idx])
    val_scores = clf.predict_proba(X_val)[:, 1]
    test_scores = clf.predict_proba(X_test)[:, 1]
    threshold = select_threshold(y[val_idx], val_scores, target_recall)
    return {
        "threshold": float(threshold),
        "val": binary_metrics(y[val_idx], val_scores, threshold),
        "test": binary_metrics(y[test_idx], test_scores, threshold),
        "test_by_full_text": stratified_report(
            df.iloc[test_idx].reset_index(drop=True), y[test_idx], test_scores, threshold
        ),
        "model": clf,
        "scaler": scaler,
        "test_scores": test_scores,
        "val_scores": val_scores,
    }


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": result["threshold"],
        "val": result["val"],
        "test": result["test"],
        "test_by_full_text": result["test_by_full_text"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--keep-first", type=int, default=DEFAULT_KEEP_FIRST)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="PubMedBERT snapshot or HF id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--class-weight", type=str, default="balanced", choices=["balanced", "none"])
    parser.add_argument("--no-aux", action="store_true", help="skip ablation C; primary=B")
    parser.add_argument("--primary-mode", type=str, default=None, choices=["A", "B", "C"])
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--skip-full-control", action="store_true")
    args = parser.parse_args()

    df = load_dataset(args.data, args.limit)
    logger.info("Loaded %d rows from %s (labels: %s)", len(df), args.data, df["label"].value_counts().to_dict())
    logger.info("Default subset (weak=False, label_source!=model): %d rows", int(default_subset_mask(df).sum()))

    artifacts = args.artifacts
    cache_dir = args.cache_dir or (artifacts / "features")
    artifacts.mkdir(parents=True, exist_ok=True)

    encoder_kwargs = {
        "model_path": args.model or DEFAULT_MODEL,
        "max_length": 512,
        "normalize": True,
    }
    feature_kwargs = {
        "chunk_tokens": args.chunk_tokens,
        "max_chunks": args.max_chunks,
        "keep_first": args.keep_first,
    }
    arrays, cache_meta = build_feature_cache(
        df,
        cache_dir,
        encoder_kwargs=encoder_kwargs,
        feature_kwargs=feature_kwargs,
        batch_size=args.batch_size,
        device=args.device,
        force=args.force_reencode,
    )

    if list(arrays["doc_ids"].astype(str)) != df["doc_id"].astype(str).tolist():
        raise RuntimeError("feature cache order does not match dataset")

    y = df["label"].to_numpy()
    train_idx, val_idx, test_idx = make_splits(df, seed=args.seed)
    subset = default_subset_mask(df).to_numpy()
    train_sub = train_idx[subset[train_idx]]
    val_sub = val_idx[subset[val_idx]]
    test_sub = test_idx[subset[test_idx]]

    split_info = {
        "method": "GroupShuffleSplit (group_id=PMID), 64/16/20",
        "seed": args.seed,
        "sizes": {
            "all": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
            "default": {"train": len(train_sub), "val": len(val_sub), "test": len(test_sub)},
        },
        "positives": {
            "all": {
                "train": int(y[train_idx].sum()),
                "val": int(y[val_idx].sum()),
                "test": int(y[test_idx].sum()),
            },
            "default": {
                "train": int(y[train_sub].sum()),
                "val": int(y[val_sub].sum()),
                "test": int(y[test_sub].sum()),
            },
        },
    }
    logger.info("Split info: %s", json.dumps(split_info["sizes"]))

    modes = ["A", "B"] + ([] if args.no_aux else ["C"])
    primary_mode = args.primary_mode or ("B" if args.no_aux else "B")
    if primary_mode not in modes:
        modes.append(primary_mode)
    # Prefer B as primary (matches shipped PubMedBERT head layout); C is ablation.
    if args.primary_mode:
        primary_mode = args.primary_mode
    else:
        primary_mode = "B"

    ablations: dict[str, Any] = {}
    primary_result: dict[str, Any] | None = None
    for mode in modes:
        logger.info("Training ablation %s (%s)", mode, FEATURE_MODES[mode])
        X = feature_matrix(arrays, mode)
        t0 = time.time()
        res = fit_and_evaluate(
            X,
            y,
            df,
            train_sub,
            val_sub,
            test_sub,
            target_recall=args.target_recall,
            C=args.C,
            class_weight=None if args.class_weight == "none" else "balanced",
        )
        ablations[mode] = {
            "description": FEATURE_MODES[mode],
            "feature_dim": int(X.shape[1]),
            "fit_seconds": round(time.time() - t0, 2),
            **_public_result(res),
        }
        logger.info(
            "  %s: val recall=%.3f precision=%.3f | test recall=%.3f precision=%.3f AP=%.3f threshold=%.3f",
            mode,
            res["val"]["recall"],
            res["val"]["precision"],
            res["test"]["recall"],
            res["test"]["precision"],
            res["test"]["ap"] or 0.0,
            res["threshold"],
        )
        if mode == primary_mode:
            primary_result = res

    assert primary_result is not None
    X_primary = feature_matrix(arrays, primary_mode)

    full_control: dict[str, Any] | None = None
    if not args.skip_full_control:
        logger.info("Training full-label control (all %d rows)", len(df))
        full_control = fit_and_evaluate(
            X_primary,
            y,
            df,
            train_idx,
            val_idx,
            test_idx,
            target_recall=args.target_recall,
            C=args.C,
            class_weight=None if args.class_weight == "none" else "balanced",
        )
        logger.info(
            "  full control: test recall=%.3f precision=%.3f AP=%.3f",
            full_control["test"]["recall"],
            full_control["test"]["precision"],
            full_control["test"]["ap"] or 0.0,
        )

    common_bundle = {
        "encoder_family": "pubmedbert",
        "feature_mode": primary_mode,
        "feature_dim": int(X_primary.shape[1]),
        "include_fulltext": True,
        "include_aux": primary_mode == "C",
        "encoder": {
            "model_path": cache_meta.get("encoder", {}).get("model_path") or encoder_kwargs["model_path"],
            "max_length": 512,
            "normalize": True,
            "chunk_tokens": args.chunk_tokens,
            "max_chunks": args.max_chunks,
            "keep_first": args.keep_first,
            "pooling": "concat[mean(all), mean(first 2)] L2-normed -> 1536",
        },
        "split": split_info,
        "seed": args.seed,
        "target_recall": args.target_recall,
        "C": args.C,
        "class_weight": args.class_weight,
    }
    model_path = artifacts / "model.joblib"
    bundle = {
        "model": primary_result["model"],
        "scaler": primary_result["scaler"],
        "threshold": primary_result["threshold"],
        "train_subset": "default (weak=False, label_source!=model)",
        **common_bundle,
    }
    dump(bundle, model_path)
    logger.info("Wrote %s", model_path)

    # Keep SPECTER2 artifacts aside if present (do not overwrite silently without marker).
    legacy_marker = artifacts / "LEGACY_SPECTER2.txt"
    if not legacy_marker.exists():
        legacy_marker.write_text(
            "Primary encoder is now PubMedBERT. Older SPECTER2 metrics/models may remain "
            "under artifacts/ as historical references (metrics_specter2_* if archived).\n",
            encoding="utf-8",
        )

    model_all_path = None
    if full_control is not None:
        model_all_path = artifacts / "model_all.joblib"
        dump(
            {
                "model": full_control["model"],
                "scaler": full_control["scaler"],
                "threshold": full_control["threshold"],
                "train_subset": "all",
                **common_bundle,
            },
            model_all_path,
        )
        logger.info("Wrote %s", model_all_path)

    config = {
        "model_path": str(model_path),
        "model_all_path": str(model_all_path) if model_all_path else None,
        "encoder_family": "pubmedbert",
        "threshold": float(primary_result["threshold"]),
        "review_lower_bound": float(primary_result["threshold"] - 0.15),
        "feature_mode": primary_mode,
        "feature_dim": int(X_primary.shape[1]),
        "feature_layout": [
            "mean(all chunks) 768 L2",
            "mean(first 2 chunks) 768 L2",
        ]
        + (["case density", "variant/test density"] if primary_mode == "C" else []),
        "encoder": common_bundle["encoder"],
        "encoder_runtime": cache_meta.get("encoder", {}),
        "train_subset": "default (weak=False, label_source!=model)",
        "n_train": len(train_sub),
        "n_val": len(val_sub),
        "n_test": len(test_sub),
        "n_total": len(df),
        "split": split_info,
        "target_recall": args.target_recall,
        "C": args.C,
        "class_weight": args.class_weight,
        "seed": args.seed,
        "feature_cache": cache_meta.get("cache_path"),
    }
    (artifacts / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "encoder_family": "pubmedbert",
        "data": {
            "path": str(args.data),
            "n_rows": len(df),
            "n_default_subset": int(subset.sum()),
            "n_full_text": int(df["full_text_available"].sum()),
            "n_abstract_only": int((~df["full_text_available"]).sum()),
            "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        },
        "split": split_info,
        "primary": {
            "feature_mode": primary_mode,
            "train_subset": "default (weak=False, label_source!=model)",
            **_public_result(primary_result),
        },
        "ablations": ablations,
        "full_label_control": {
            "train_subset": "all",
            "feature_mode": primary_mode,
            **_public_result(full_control),
        }
        if full_control
        else None,
        "encoder_runtime": cache_meta.get("encoder", {}),
        "encoding": {
            "seconds": cache_meta.get("encoding_seconds"),
            "docs_per_second": cache_meta.get("docs_per_second"),
            "mean_chunks_per_doc": cache_meta.get("mean_chunks_per_doc"),
            "imported_from": cache_meta.get("imported_from"),
        },
        "notes": [
            "SPECTER2 abandoned; primary encoder is frozen NeuML/pubmedbert-base-embeddings.",
            "No PubMedBERT fine-tuning; only the logistic head is trained.",
            "Pooling matches casefilter-head-v3: concat[mean(all), mean(first 2)] -> 1536-d.",
            "Threshold selected on validation at target recall %.2f." % args.target_recall,
            "GroupShuffleSplit over PMID groups; group_id equals PMID in this dataset.",
        ],
    }
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    pred_df = df.iloc[test_sub][["doc_id", "title", "label", "full_text_available", "label_source", "weak"]].copy()
    pred_df["score"] = primary_result["test_scores"]
    pred_df["threshold"] = float(primary_result["threshold"])
    pred_df["decision"] = np.where(
        pred_df["score"] >= primary_result["threshold"],
        "relevant",
        np.where(pred_df["score"] >= primary_result["threshold"] - 0.15, "review", "irrelevant"),
    )
    pred_df.to_csv(artifacts / "test_predictions.csv", index=False)

    summary = {
        "encoder_family": "pubmedbert",
        "model": str(model_path),
        "config": str(artifacts / "config.json"),
        "metrics": str(artifacts / "metrics.json"),
        "feature_mode": primary_mode,
        "feature_dim": int(X_primary.shape[1]),
        "threshold": float(primary_result["threshold"]),
        "test": primary_result["test"],
        "ablations": {m: ablations[m]["test"] for m in ablations},
        "full_label_control_test": full_control["test"] if full_control else None,
    }
    (artifacts / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== TRAIN SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts -> {artifacts}")


if __name__ == "__main__":
    main()
