#!/usr/bin/env python3
"""Evaluation hardening for the primary screening baseline.

Adds three things the original single-split report lacks:

1. PMID-cluster percentile bootstrap CI on the existing test split.
2. GroupKFold (k=5) on the default subset, using the cached feature matrix.
3. Threshold strategies selected on validation and reported on test.

The script is cache-only: it loads ``artifacts/features/*.npz`` selected by
``artifacts/config.json -> feature_cache`` and never calls the encoder. The
existing ``artifacts/model.joblib`` is loaded read-only for the bootstrap and
threshold-strategy sections; GroupKFold refits scaler + logistic regression
inside each fold, but writes no model bundle.

Outputs
-------
artifacts/optimization/bootstrap_ci.json
artifacts/optimization/group_cv.json
artifacts/optimization/threshold_strategies.json

Usage
-----
    .venv/bin/python scripts/eval_hardening.py
    .venv/bin/python scripts/eval_hardening.py --bootstrap 5000 --jobs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    DEFAULT_TARGET_RECALL,
    binary_metrics,
    default_subset_mask,
    feature_matrix,
    load_dataset,
    make_splits,
    select_threshold,
)

logger = logging.getLogger("eval_hardening")

DEFAULT_CONFIG = ROOT / "artifacts" / "config.json"
DEFAULT_DATA = ROOT / "data" / "labeled_papers.csv"
DEFAULT_METRICS = ROOT / "artifacts" / "metrics.json"
DEFAULT_PREDICTIONS = ROOT / "artifacts" / "test_predictions.csv"
DEFAULT_OUT = ROOT / "artifacts" / "optimization"

CI_METRICS = ("recall", "precision", "f1", "f2", "ap")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def threshold_grid(scores: np.ndarray) -> np.ndarray:
    """Same candidate set used by train.select_threshold."""
    scores = np.asarray(scores, dtype=float)
    unique = np.unique(scores)
    if len(unique) == 0:
        return np.array([0.5])
    candidates = np.unique(
        np.concatenate(([0.0], (unique[:-1] + unique[1:]) / 2.0, [float(unique[-1]) + 1e-6], [1.0]))
    )
    return candidates


def threshold_pred(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)


def select_max_fbeta(y_true: np.ndarray, scores: np.ndarray, beta: float = 2.0) -> float:
    """Threshold maximizing F-beta on validation (tie-break: higher precision)."""
    scores = np.asarray(scores, dtype=float)
    best: tuple[float, float, float] | None = None
    for t in threshold_grid(scores):
        pred = threshold_pred(y_true, scores, t)
        fb = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        if best is None or fb > best[0] + 1e-12 or (
            abs(fb - best[0]) <= 1e-12 and prec > best[1]
        ):
            best = (float(fb), float(prec), float(t))
    return best[2] if best is not None else float(scores.min() - 1e-6)


def select_max_recall_at_precision(
    y_true: np.ndarray, scores: np.ndarray, min_precision: float = 0.80
) -> tuple[float, bool]:
    """Max recall subject to validation precision >= min_precision."""
    scores = np.asarray(scores, dtype=float)
    best: tuple[float, float, float] | None = None
    for t in threshold_grid(scores):
        pred = threshold_pred(y_true, scores, t)
        prec = precision_score(y_true, pred, zero_division=0)
        if prec + 1e-12 < min_precision:
            continue
        rec = recall_score(y_true, pred, zero_division=0)
        if best is None or rec > best[0] + 1e-12 or (
            abs(rec - best[0]) <= 1e-12 and prec > best[1]
        ):
            best = (float(rec), float(prec), float(t))
    if best is None:
        return float(scores.min() - 1e-6), True
    return best[2], False


def select_max_recall_at_pp_budget(
    y_true: np.ndarray, scores: np.ndarray, budget: int
) -> tuple[float, bool]:
    """Max recall subject to validation predicted-positive count <= budget."""
    scores = np.asarray(scores, dtype=float)
    best: tuple[float, float, float] | None = None
    for t in threshold_grid(scores):
        pred = threshold_pred(y_true, scores, t)
        pp = int(pred.sum())
        if pp > budget:
            continue
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        if best is None or rec > best[0] + 1e-12 or (
            abs(rec - best[0]) <= 1e-12 and prec > best[1]
        ):
            best = (float(rec), float(prec), float(t))
    if best is None:
        return float(scores.min() - 1e-6), True
    return best[2], False


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------
def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    *,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """PMID-cluster percentile bootstrap.

    Groups are resampled with replacement; every row of a sampled group is
    included. Metrics are recomputed from scratch for each replicate.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups).astype(str)

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_indices = [np.flatnonzero(inverse == g) for g in range(len(unique_groups))]
    rng = np.random.default_rng(seed)

    values: dict[str, list[float]] = {m: [] for m in CI_METRICS}
    values["predicted_positive"] = []
    values["tp"] = []
    values["fp"] = []
    values["fn"] = []
    values["tn"] = []

    for _ in range(n_bootstrap):
        draw = rng.integers(0, len(unique_groups), size=len(unique_groups))
        idx = np.concatenate([group_indices[g] for g in draw])
        y_b = y_true[idx]
        s_b = scores[idx]
        m = binary_metrics(y_b, s_b, threshold)
        for name in CI_METRICS:
            value = m.get(name)
            if value is not None and np.isfinite(value):
                values[name].append(float(value))
        values["predicted_positive"].append(float(m["predicted_positive"]))
        for name in ("tp", "fp", "fn", "tn"):
            values[name].append(float(m["confusion_matrix"][name]))

    point = binary_metrics(y_true, scores, threshold)
    out: dict[str, Any] = {}
    for name in CI_METRICS:
        arr = np.asarray(values[name], dtype=float)
        if arr.size == 0:
            out[name] = {"n_valid": 0, "mean": None, "ci_low": None, "ci_high": None}
            continue
        lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[name] = {
            "point": float(point[name]),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "n_valid": int(arr.size),
        }

    conf = {}
    for name in ("tp", "fp", "fn", "tn", "predicted_positive"):
        arr = np.asarray(values[name], dtype=float)
        lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        conf[name] = {"mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi)}

    return {
        "method": "PMID-cluster (group) percentile bootstrap",
        "n_bootstrap": n_bootstrap,
        "n_groups": int(len(unique_groups)),
        "n_rows": int(len(y_true)),
        "seed": seed,
        "confidence_level": 1 - alpha,
        "threshold": float(threshold),
        "point_estimate": point,
        "metrics": out,
        "confusion_mean_and_ci": conf,
    }


# ---------------------------------------------------------------------------
# Group cross-validation
# ---------------------------------------------------------------------------
def fit_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    *,
    C: float,
    class_weight: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaler + logistic regression once, score validation and test."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=3000,
        solver="lbfgs",
        random_state=0,
    )
    clf.fit(Xtr, y_train)
    val_scores = clf.predict_proba(scaler.transform(X_val))[:, 1]
    test_scores = clf.predict_proba(scaler.transform(X_test))[:, 1]
    return val_scores, test_scores


def group_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    target_recall: float,
    C: float,
    class_weight: str | None,
) -> dict[str, Any]:
    """GroupKFold with an inner group-aware validation split per fold.

    Protocol: within each outer training portion, an 80/20 group split selects
    the threshold at the target recall; the held-out outer fold is untouched
    until evaluation.
    """
    gkf = GroupKFold(n_splits=n_splits)
    per_fold: list[dict[str, Any]] = []
    oof_scores = np.full(len(y), np.nan, dtype=float)
    oof_thresholds = np.full(len(y), np.nan, dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        inner_ok = False
        for attempt in range(30):
            gss = GroupShuffleSplit(
                n_splits=1, test_size=0.2, random_state=seed + fold * 100 + attempt
            )
            fit_rel, val_rel = next(gss.split(X[tr_idx], y[tr_idx], groups[tr_idx]))
            fit_idx, val_idx = tr_idx[fit_rel], tr_idx[val_rel]
            if (
                len(np.unique(y[fit_idx])) == 2
                and len(np.unique(y[val_idx])) == 2
                and len(np.unique(y[te_idx])) == 2
            ):
                inner_ok = True
                break
        if not inner_ok:  # pragma: no cover - data-dependent guard
            raise RuntimeError(f"fold {fold}: could not build a two-class inner split")

        val_scores, te_scores = fit_and_score(
            X[fit_idx],
            y[fit_idx],
            X[val_idx],
            X[te_idx],
            C=C,
            class_weight=class_weight,
        )
        threshold = select_threshold(y[val_idx], val_scores, target_recall)
        metrics = binary_metrics(y[te_idx], te_scores, threshold)
        oof_scores[te_idx] = te_scores
        oof_thresholds[te_idx] = threshold

        per_fold.append(
            {
                "fold": fold,
                "n_train": int(len(fit_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(te_idx)),
                "positives_test": int(y[te_idx].sum()),
                "threshold": float(threshold),
                "val_at_threshold": binary_metrics(y[val_idx], val_scores, threshold),
                "test": metrics,
            }
        )
        logger.info(
            "fold %d: n_train=%d n_val=%d n_test=%d threshold=%.4f "
            "test recall=%.3f precision=%.3f AP=%.3f",
            fold,
            len(fit_idx),
            len(val_idx),
            len(te_idx),
            threshold,
            metrics["recall"],
            metrics["precision"],
            metrics["ap"] or 0.0,
        )

    aggregated: dict[str, Any] = {}
    for name in CI_METRICS:
        arr = np.asarray(
            [f["test"][name] for f in per_fold if f["test"].get(name) is not None], dtype=float
        )
        if arr.size == 0:
            aggregated[name] = {"mean": None, "std": None, "per_fold": []}
            continue
        aggregated[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "per_fold": [float(x) for x in arr],
        }
    thresholds = np.asarray([f["threshold"] for f in per_fold], dtype=float)
    aggregated["threshold"] = {
        "mean": float(thresholds.mean()),
        "std": float(thresholds.std(ddof=1)) if len(thresholds) > 1 else 0.0,
        "per_fold": [float(x) for x in thresholds],
    }

    # Pooled out-of-fold predictions (each row uses the threshold from the fold
    # in which it was held out; thresholds are fold-specific by design).
    pred = (oof_scores >= oof_thresholds).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    pooled = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "ap": float(average_precision_score(y, oof_scores)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "note": "fold-specific thresholds applied to each held-out row",
    }

    return {
        "method": "GroupKFold",
        "k": n_splits,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "groups": int(len(np.unique(groups))),
        "inner_protocol": "GroupShuffleSplit 80/20 inside each outer training fold; threshold selected on inner validation at target recall",
        "target_recall": target_recall,
        "C": C,
        "class_weight": class_weight,
        "seed": seed,
        "metrics": aggregated,
        "per_fold": per_fold,
        "pooled_oof": pooled,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap replicates (>=500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=5, help="GroupKFold folds")
    parser.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    parser.add_argument("--precision-floor", type=float, default=0.80)
    parser.add_argument("--pp-budget", type=int, default=120)
    args = parser.parse_args()
    if args.bootstrap < 500:
        parser.error("--bootstrap must be >= 500 (plan O3)")
    if args.jobs < 2:
        parser.error("--jobs (GroupKFold folds) must be >= 2")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t_start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    cache_path = Path(config["feature_cache"])
    if not cache_path.exists():
        raise SystemExit(f"feature cache not found: {cache_path}")
    logger.info("Loading feature cache (no encoder call): %s", cache_path)

    data = np.load(cache_path, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    df = load_dataset(args.data)
    if list(arrays["doc_ids"].astype(str)) != df["doc_id"].astype(str).tolist():
        raise RuntimeError("feature cache doc_ids do not match data/labeled_papers.csv order")

    y = df["label"].to_numpy(dtype=int)
    if not np.array_equal(arrays["labels"].astype(int), y):
        raise RuntimeError("cached labels do not match data/labeled_papers.csv labels")
    groups = df["group_id"].to_numpy(dtype=str)
    subset = default_subset_mask(df).to_numpy()
    X = feature_matrix(arrays, config["feature_mode"])
    if X.shape[1] != int(config["feature_dim"]):
        raise RuntimeError(f"feature_dim mismatch: {X.shape[1]} != {config['feature_dim']}")

    train_idx, val_idx, test_idx = make_splits(df, seed=int(config["seed"]))
    train_sub = train_idx[subset[train_idx]]
    val_sub = val_idx[subset[val_idx]]
    test_sub = test_idx[subset[test_idx]]
    expected_sizes = (config["n_train"], config["n_val"], config["n_test"])
    actual_sizes = (len(train_sub), len(val_sub), len(test_sub))
    if actual_sizes != expected_sizes:
        raise RuntimeError(f"split sizes {actual_sizes} != config {expected_sizes}")
    logger.info("Reproduced split: train/val/test=%d/%d/%d", *actual_sizes)

    bundle = joblib.load(config["model_path"])
    scaler = bundle["scaler"]
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    if abs(threshold - float(config["threshold"])) > 1e-9:
        raise RuntimeError("model bundle threshold != config threshold")

    val_scores = model.predict_proba(scaler.transform(X[val_sub]))[:, 1]
    test_scores = model.predict_proba(scaler.transform(X[test_sub]))[:, 1]

    # Reproduction check against the saved predictions and metrics.
    preds = pd.read_csv(args.predictions, keep_default_na=False, dtype={"doc_id": str})
    preds = preds.set_index("doc_id").loc[df.iloc[test_sub]["doc_id"].astype(str)].reset_index()
    if len(preds) != len(test_sub) or preds["doc_id"].duplicated().any():
        raise RuntimeError("test_predictions.csv does not map 1:1 onto the reproduced test split")
    score_diff = float(np.max(np.abs(preds["score"].to_numpy(dtype=float) - test_scores)))
    metrics_json = json.loads(args.metrics.read_text(encoding="utf-8"))
    baseline_test = metrics_json["primary"]["test"]
    recomputed = binary_metrics(y[test_sub], test_scores, threshold)
    metric_diffs = {
        k: abs(float(recomputed[k]) - float(baseline_test[k]))
        for k in ("recall", "precision", "f1", "f2", "ap")
    }
    reproduction = {
        "max_abs_score_diff_vs_test_predictions": score_diff,
        "max_abs_metric_diff_vs_metrics_json": float(max(metric_diffs.values())),
        "threshold_match": bool(abs(threshold - float(config["threshold"])) <= 1e-9),
        "confusion_match": recomputed["confusion_matrix"] == baseline_test["confusion_matrix"],
        "reproduced": bool(score_diff < 1e-5 and max(metric_diffs.values()) < 1e-9),
    }
    logger.info("Reproduction: %s", reproduction)

    # 1) Bootstrap CI.
    logger.info("Bootstrap: %d PMID-cluster replicates", args.bootstrap)
    boot = bootstrap_ci(
        y[test_sub],
        test_scores,
        groups[test_sub],
        threshold,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    boot.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "split": {"n_train": len(train_sub), "n_val": len(val_sub), "n_test": len(test_sub)},
            "feature_cache": str(cache_path),
            "feature_mode": config["feature_mode"],
            "model_path": str(config["model_path"]),
            "reproduction": reproduction,
            "notes": [
                "Groups are PMIDs; resampling is with replacement at the group level.",
                "Percentile interval; AP replicates with a single class are excluded.",
                "No encoder is called: features come from the cached npz.",
            ],
        }
    )
    boot_path = args.out / "bootstrap_ci.json"
    boot_path.write_text(json.dumps(boot, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", boot_path)

    # 2) GroupKFold on the default subset.
    X_def = X[subset]
    y_def = y[subset]
    groups_def = groups[subset]
    logger.info(
        "GroupKFold: k=%d on default subset n=%d (positives=%d)",
        args.jobs,
        len(y_def),
        int(y_def.sum()),
    )
    cv = group_cv(
        X_def,
        y_def,
        groups_def,
        n_splits=args.jobs,
        seed=args.seed,
        target_recall=args.target_recall,
        C=float(config["C"]),
        class_weight=None if config["class_weight"] == "none" else config["class_weight"],
    )
    cv.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_cache": str(cache_path),
            "feature_mode": config["feature_mode"],
            "feature_dim": int(X_def.shape[1]),
            "encoder_invoked": False,
            "no_reencoding": True,
            "fit_seconds": round(time.time() - t_start, 2),
            "notes": [
                "Uses only artifacts/features/*.npz; no SPECTER2 encoding.",
                "Outer folds are grouped by PMID; inner validation is also group-aware.",
                "Thresholds differ by fold; pooled OOF uses each row's fold threshold.",
            ],
        }
    )
    cv_path = args.out / "group_cv.json"
    cv_path.write_text(json.dumps(cv, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", cv_path)

    # 3) Threshold strategies (same val selection, same test report).
    strategies: dict[str, dict[str, Any]] = {}

    t1 = select_threshold(y[val_sub], val_scores, args.target_recall)
    strategies["max_precision_at_recall_ge_0.95"] = {
        "selection_rule": f"maximize val precision subject to val recall >= {args.target_recall}",
        "threshold": float(t1),
        "matches_deployed_threshold": bool(abs(t1 - threshold) <= 1e-9),
    }

    t2 = select_max_fbeta(y[val_sub], val_scores, beta=2.0)
    strategies["max_f2"] = {
        "selection_rule": "maximize F2 on validation (tie-break higher precision)",
        "threshold": float(t2),
    }

    t3, t3_fallback = select_max_recall_at_precision(
        y[val_sub], val_scores, args.precision_floor
    )
    strategies[f"max_recall_at_precision_ge_{args.precision_floor:.2f}"] = {
        "selection_rule": f"maximize val recall subject to val precision >= {args.precision_floor}",
        "threshold": float(t3),
        "fallback_all_positive": bool(t3_fallback),
    }

    t4, t4_fallback = select_max_recall_at_pp_budget(y[val_sub], val_scores, args.pp_budget)
    strategies[f"max_recall_at_pp_budget_{args.pp_budget}"] = {
        "selection_rule": f"maximize val recall subject to val predicted positives <= {args.pp_budget}",
        "threshold": float(t4),
        "fallback_all_positive": bool(t4_fallback),
    }

    baseline_name = "max_precision_at_recall_ge_0.95"
    baseline_test = None
    for name, spec in strategies.items():
        t = spec["threshold"]
        spec["val"] = binary_metrics(y[val_sub], val_scores, t)
        spec["test"] = binary_metrics(y[test_sub], test_scores, t)
        if name == baseline_name:
            baseline_test = spec["test"]
    assert baseline_test is not None
    for name, spec in strategies.items():
        if name == baseline_name:
            spec["test_delta_vs_deployed"] = {k: 0.0 for k in ("recall", "precision", "f1", "f2")}
        else:
            spec["test_delta_vs_deployed"] = {
                k: float(spec["test"][k] - baseline_test[k])
                for k in ("recall", "precision", "f1", "f2")
            }

    comparison = []
    for name, spec in strategies.items():
        comparison.append(
            {
                "strategy": name,
                "threshold": round(spec["threshold"], 6),
                "val_recall": round(spec["val"]["recall"], 4),
                "val_precision": round(spec["val"]["precision"], 4),
                "test_recall": round(spec["test"]["recall"], 4),
                "test_precision": round(spec["test"]["precision"], 4),
                "test_f1": round(spec["test"]["f1"], 4),
                "test_f2": round(spec["test"]["f2"], 4),
                "test_predicted_positive": spec["test"]["predicted_positive"],
                "test_confusion": spec["test"]["confusion_matrix"],
            }
        )

    thr = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": {"n_train": len(train_sub), "n_val": len(val_sub), "n_test": len(test_sub)},
        "selection_protocol": "threshold selected on the default-subset validation split; test untouched during selection",
        "deployed_threshold": threshold,
        "ap_is_threshold_free": True,
        "strategies": strategies,
        "comparison_table": comparison,
        "feature_cache": str(cache_path),
        "model_path": str(config["model_path"]),
        "reproduction": reproduction,
        "notes": [
            "All strategies share the same fitted model and the same val/test rows.",
            "AP is identical across strategies because it ignores the threshold.",
            "Fallbacks are recorded explicitly; none are silently substituted.",
        ],
    }
    thr_path = args.out / "threshold_strategies.json"
    thr_path.write_text(json.dumps(thr, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", thr_path)

    elapsed = time.time() - t_start
    print(
        json.dumps(
            {
                "reproduction": reproduction,
                "bootstrap": {
                    m: {
                        "mean": round(boot["metrics"][m]["mean"], 4),
                        "ci": [round(boot["metrics"][m]["ci_low"], 4), round(boot["metrics"][m]["ci_high"], 4)],
                    }
                    for m in ("recall", "precision", "f1", "f2", "ap")
                },
                "group_cv": {
                    m: {
                        "mean": round(cv["metrics"][m]["mean"], 4),
                        "std": round(cv["metrics"][m]["std"], 4),
                    }
                    for m in ("recall", "precision", "f1", "f2", "ap")
                },
                "threshold_comparison": comparison,
                "seconds": round(elapsed, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
