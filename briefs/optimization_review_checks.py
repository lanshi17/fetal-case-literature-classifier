#!/usr/bin/env python3
"""Reproducible check suite for the OMP review of Optimization Phase 1.

Every scenario maps to a review claim in `briefs/OPTIMIZATION_REVIEW_SPEC.md`
or a plan acceptance item (O1-O7). The suite fails closed: any exception inside
a scenario is a FAIL and the process exits nonzero.

Usage:
    .venv/bin/python briefs/optimization_review_checks.py
    .venv/bin/python briefs/optimization_review_checks.py --no-rerun
    .venv/bin/python briefs/optimization_review_checks.py --json briefs/optimization_review_checks.json

Exit codes: 0 = all scenarios pass, 1 = at least one scenario failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train  # noqa: E402  (production split/feature definitions)

ART = ROOT / "artifacts"
OPT = ART / "optimization"
CONFIG = json.loads((ART / "config.json").read_text(encoding="utf-8"))

REQUIRED_AUDIT_COLUMNS = [
    "pmid",
    "title",
    "score",
    "decision",
    "design",
    "reason",
    "text_source",
    "heuristic_tag",
    "manual_note",
]
PROTECTED = sorted(
    [ART / name for name in ("model.joblib", "model_all.joblib", "metrics.json", "test_predictions.csv", "config.json")]
    + sorted((ART / "features").glob("*.npz"))
    + [ROOT / "data" / "labeled_papers.csv", ROOT / "data" / "final_labels.csv"]
)
VOLATILE_KEYS = {"generated_at", "fit_seconds", "seconds"}


class CheckFailure(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_json(obj: Any, out_dirs: tuple[str, ...] = ()) -> Any:
    """Drop volatile fields and out-directory strings before comparing runs."""
    if isinstance(obj, dict):
        return {
            k: normalize_json(v, out_dirs)
            for k, v in obj.items()
            if k not in VOLATILE_KEYS
        }
    if isinstance(obj, list):
        return [normalize_json(v, out_dirs) for v in obj]
    if isinstance(obj, str):
        text = obj
        for out in out_dirs:
            text = text.replace(out, "<OUT>")
        return text
    return obj


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_audits() -> tuple[pd.DataFrame, pd.DataFrame]:
    dtype = {"doc_id": str, "pmid": str}
    fn = pd.read_csv(OPT / "fn_audit.csv", keep_default_na=False, dtype=dtype)
    fp = pd.read_csv(OPT / "fp_audit.csv", keep_default_na=False, dtype=dtype)
    return fn, fp


def metrics_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "ap": float(average_precision_score(y, scores)),
        "predicted_positive": int(pred.sum()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=float))
    require(len(unique) > 0, "empty score vector")
    return np.unique(
        np.concatenate(([0.0], (unique[:-1] + unique[1:]) / 2.0, [float(unique[-1]) + 1e-6], [1.0]))
    )


def select_max_precision_at_recall(y: np.ndarray, scores: np.ndarray, target: float) -> float:
    best: tuple[float, float, float] | None = None
    for t in threshold_grid(scores):
        pred = scores >= t
        rec = recall_score(y, pred, zero_division=0)
        if rec + 1e-12 < target:
            continue
        prec = precision_score(y, pred, zero_division=0)
        if best is None or prec > best[0] + 1e-12 or (abs(prec - best[0]) <= 1e-12 and rec > best[1]):
            best = (float(prec), float(rec), float(t))
    return best[2] if best is not None else float(np.min(scores) - 1e-6)


def select_max_fbeta(y: np.ndarray, scores: np.ndarray, beta: float) -> float:
    best: tuple[float, float, float] | None = None
    for t in threshold_grid(scores):
        pred = scores >= t
        fb = fbeta_score(y, pred, beta=beta, zero_division=0)
        prec = precision_score(y, pred, zero_division=0)
        if best is None or fb > best[0] + 1e-12 or (abs(fb - best[0]) <= 1e-12 and prec > best[1]):
            best = (float(fb), float(prec), float(t))
    require(best is not None, "no candidate threshold")
    return best[2]


def _val_scores() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_val, val_scores, y_test, test_scores) from cache + bundle."""
    cache = Path(CONFIG["feature_cache"])
    require(cache.exists(), f"feature cache missing: {cache}")
    data = np.load(cache, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    df = train.load_dataset(ROOT / "data" / "labeled_papers.csv")
    require(
        list(arrays["doc_ids"].astype(str)) == df["doc_id"].astype(str).tolist(),
        "feature cache row order != labeled_papers.csv",
    )
    subset = train.default_subset_mask(df).to_numpy()
    _, val_idx, test_idx = train.make_splits(df, seed=int(CONFIG["seed"]))
    val = val_idx[subset[val_idx]]
    test = test_idx[subset[test_idx]]
    X = train.feature_matrix(arrays, CONFIG["feature_mode"])
    bundle = joblib.load(CONFIG["model_path"])
    scaler, model = bundle["scaler"], bundle["model"]
    y = df["label"].to_numpy(dtype=int)
    return (
        y[val],
        model.predict_proba(scaler.transform(X[val]))[:, 1],
        y[test],
        model.predict_proba(scaler.transform(X[test]))[:, 1],
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def s01_audit_rows_match_predictions() -> dict[str, Any]:
    """FN/FP CSV row sets equal the independent score-rule selection."""
    preds = pd.read_csv(ART / "test_predictions.csv", dtype={"doc_id": str, "pmid": str})
    threshold = float(CONFIG["threshold"])
    fn_expected = set(preds.loc[(preds["label"] == 1) & (preds["score"] < threshold), "doc_id"])
    fp_expected = set(preds.loc[(preds["label"] == 0) & (preds["score"] >= threshold), "doc_id"])
    fn, fp = read_audits()
    require(set(fn["doc_id"]) == fn_expected, "fn_audit.csv doc_ids != score-rule FN")
    require(set(fp["doc_id"]) == fp_expected, "fp_audit.csv doc_ids != score-rule FP")
    require(len(fn) == len(fn_expected) == 10, f"FN rows {len(fn)} != 10")
    require(len(fp) == len(fp_expected) == 30, f"FP rows {len(fp)} != 30")
    baseline = load_json(ART / "metrics.json")["primary"]["test"]["confusion_matrix"]
    require(baseline == {"tn": 17, "fp": 30, "fn": 10, "tp": 120}, f"metrics.json cm changed: {baseline}")
    scores = preds.set_index("doc_id")["score"]
    require(
        np.allclose(fn["score"].to_numpy(float), scores.loc[fn["doc_id"]].to_numpy(float), atol=1e-9),
        "fn_audit scores != test_predictions scores",
    )
    require(
        np.allclose(fp["score"].to_numpy(float), scores.loc[fp["doc_id"]].to_numpy(float), atol=1e-9),
        "fp_audit scores != test_predictions scores",
    )
    # decision column is redundant with the score rule in the deployed build
    decision_fn = set(preds.loc[(preds["label"] == 1) & (preds["decision"] != "relevant"), "doc_id"])
    decision_fp = set(preds.loc[(preds["label"] == 0) & (preds["decision"] == "relevant"), "doc_id"])
    require(decision_fn == fn_expected and decision_fp == fp_expected, "decision column disagrees with score rule")
    return {"fn_rows": len(fn), "fp_rows": len(fp), "baseline_cm": baseline}


def s02_audit_schema_and_manual_columns() -> dict[str, Any]:
    """O1 columns present, metadata filled, manual columns blank, text_head bounded."""
    fn, fp = read_audits()
    for name, frame in (("fn", fn), ("fp", fp)):
        missing = [c for c in REQUIRED_AUDIT_COLUMNS if c not in frame.columns]
        require(not missing, f"{name}_audit.csv missing columns {missing}")
        require(frame["pmid"].str.len().gt(0).all(), f"{name}: blank pmid")
        require(frame["title"].str.len().gt(0).all(), f"{name}: blank title")
        require(frame["heuristic_tag"].str.len().gt(0).all(), f"{name}: blank heuristic_tag")
        require((frame["manual_tag"] == "").all(), f"{name}: manual_tag not blank")
        require((frame["manual_note"] == "").all(), f"{name}: manual_note not blank")
        require(frame["text_head"].str.len().max() <= 500, f"{name}: text_head longer than 500 chars")
        require(not frame["text_head"].str.contains("\n").any(), f"{name}: text_head contains newlines")
    return {"fn_columns": list(fn.columns), "fp_columns": list(fp.columns)}


def s03_audit_tags_defined_and_consistent() -> dict[str, Any]:
    """Heuristic tags match their documented definitions against source data."""
    fn, fp = read_audits()
    summary = load_json(OPT / "error_summary.json")
    defined = set(summary["tag_definitions"])
    df = pd.read_csv(
        ROOT / "data" / "labeled_papers.csv",
        keep_default_na=False,
        dtype={"doc_id": str, "pmid": str},
    ).set_index("doc_id")

    def truthy(value: Any) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}

    def split_tags(value: str) -> list[str]:
        return [t for t in str(value).split(";") if t]

    def expected_design_tags(design: str) -> set[str]:
        d = design.strip().lower()
        if not d:
            return {"design_missing"}
        if "review" in d:
            return {"design_review"}
        if "case report" in d or "case series" in d:
            return {"design_case_report"}
        if any(k in d for k in ("cohort", "screening", "validation", "registry")):
            return {"design_cohort_or_screening"}
        if d.startswith("method"):
            return {"design_method"}
        return {"design_other"}

    threshold = float(CONFIG["threshold"])
    for name, frame in (("FN", fn), ("FP", fp)):
        for _, row in frame.iterrows():
            tags = set(split_tags(row["heuristic_tag"]))
            require(tags <= defined, f"{name} {row['doc_id']}: undefined tag {tags - defined}")
            src = df.loc[row["doc_id"]]
            require(str(row["pmid"]) == str(src["pmid"]), f"{name} {row['doc_id']}: pmid != labeled_papers")
            require(str(row["title"]) == str(src["title"]), f"{name} {row['doc_id']}: title != labeled_papers")
            has_abstract = truthy(src["abstract_available"]) and str(src["abstract"]).strip() != ""
            require(("empty_abstract" in tags) == (not has_abstract), f"{name} {row['doc_id']}: empty_abstract mismatch")
            require(
                ("pdf_text_source" in tags) == (str(src["text_source"]).strip().lower() == "pdf"),
                f"{name} {row['doc_id']}: pdf_text_source mismatch",
            )
            emitted_design = {t for t in tags if t.startswith("design_")}
            require(
                emitted_design == expected_design_tags(str(src["design"])),
                f"{name} {row['doc_id']}: design tags {sorted(emitted_design)} != {sorted(expected_design_tags(str(src['design'])))}",
            )
            score = float(row["score"])
            if not truthy(src["weak"]) and str(src["label_source"]).strip().lower() != "model":
                require("weak_or_model_label" not in tags, f"{name} {row['doc_id']}: spurious weak_or_model_label")
            require(
                ("high_score_confident_error" in tags) == (abs(score - 0.5) >= 0.4),
                f"{name} {row['doc_id']}: high_score_confident_error mismatch",
            )
            if name == "FN":
                require(score < threshold, f"FN {row['doc_id']} not below threshold")
                expect_grey = score >= threshold - 0.15
                require(("low_score_band" in tags) == expect_grey, f"FN {row['doc_id']}: grey-band mismatch")
                require(("below_review_band" in tags) == (not expect_grey), f"FN {row['doc_id']}: below-band mismatch")
            else:
                require(score >= threshold, f"FP {row['doc_id']} not at/above threshold")
                if score <= threshold + 0.15:
                    require("low_score_band" in tags, f"FP {row['doc_id']}: missing low_score_band")
                elif score < 0.9:
                    require("mid_score_band" in tags, f"FP {row['doc_id']}: missing mid_score_band")

    # Recount tags from the CSVs and compare with error_summary.json.
    def counts(frame: pd.DataFrame) -> dict[str, int]:
        counter: dict[str, int] = {}
        for value in frame["heuristic_tag"]:
            for tag in split_tags(value):
                counter[tag] = counter.get(tag, 0) + 1
        return counter

    fn_tags, fp_tags = counts(fn), counts(fp)
    require(summary["heuristic_tag_counts"]["fn"] == fn_tags, "error_summary FN tag counts != CSV recount")
    require(summary["heuristic_tag_counts"]["fp"] == fp_tags, "error_summary FP tag counts != CSV recount")
    combined = dict(fn_tags)
    for tag, n in fp_tags.items():
        combined[tag] = combined.get(tag, 0) + n
    require(summary["heuristic_tag_counts"]["all"] == combined, "error_summary ALL tag counts != CSV recount")
    return {"fn_tags": fn_tags, "fp_tags": fp_tags}


def s04_summary_confusion_and_totals() -> dict[str, Any]:
    """O2: error_summary totals match metrics.json confusion matrix."""
    summary = load_json(OPT / "error_summary.json")
    baseline = load_json(ART / "metrics.json")["primary"]["test"]["confusion_matrix"]
    require(summary["computed_confusion_matrix"] == baseline, "computed cm != metrics.json cm")
    require(summary["baseline_confusion_matrix"] == baseline, "recorded baseline cm != metrics.json cm")
    require(summary["confusion_consistent"] is True, "confusion_consistent is not true")
    require(summary["checks"]["all_pass"] is True, "checks.all_pass is not true")
    require(summary["totals"]["fn"] == 10 and summary["totals"]["fp"] == 30, "totals FN/FP != 10/30")
    require(summary["totals"]["n_test"] == 177, "totals n_test != 177")
    fn, fp = read_audits()
    require(summary["totals"]["fn"] == len(fn) and summary["totals"]["fp"] == len(fp), "totals != CSV rows")
    return {"totals": summary["totals"]}


def s05_documented_reason_gap() -> dict[str, Any]:
    """The single empty reason/quote row is the PMID disclosed in DONE."""
    fn, _ = read_audits()
    empties = fn.loc[fn["reason"].str.strip() == "", "pmid"].tolist()
    require(empties == ["35791503"], f"empty-reason FN rows {empties} != documented ['35791503']")
    done = (ROOT / "briefs" / "OPTIMIZATION_PHASE1_DONE.md").read_text(encoding="utf-8")
    require("35791503" in done, "DONE does not disclose the empty-reason PMID")
    return {"empty_reason_pmids": empties}


def s06_threshold_strategies_metrics() -> dict[str, Any]:
    """O5: >=3 strategies, same val/test rows, test metrics reproducible."""
    thr_json = load_json(OPT / "threshold_strategies.json")
    strategies = thr_json["strategies"]
    require(len(strategies) >= 3, f"only {len(strategies)} strategies")
    preds = pd.read_csv(ART / "test_predictions.csv", dtype={"doc_id": str})
    y_test = preds["label"].to_numpy(int)
    scores = preds["score"].to_numpy(float)
    deployed = float(CONFIG["threshold"])
    require(abs(float(thr_json["deployed_threshold"]) - deployed) < 1e-9, "deployed threshold mismatch")
    baseline = load_json(ART / "metrics.json")["primary"]["test"]
    for name, spec in strategies.items():
        t = float(spec["threshold"])
        recomputed = metrics_at(y_test, scores, t)
        for key in ("recall", "precision", "f1", "f2"):
            require(abs(recomputed[key] - float(spec["test"][key])) <= 1e-9, f"{name}: test {key} mismatch")
        require(
            recomputed["predicted_positive"] == int(spec["test"]["predicted_positive"]),
            f"{name}: predicted_positive mismatch",
        )
        require(recomputed["confusion_matrix"] == spec["test"]["confusion_matrix"], f"{name}: confusion mismatch")
        require(spec["val"]["n"] == 140 and spec["test"]["n"] == 177, f"{name}: val/test n mismatch")
        require(abs(float(spec["test"]["ap"]) - float(baseline["ap"])) <= 1e-9, f"{name}: AP drifted")
        require(abs(float(spec["test"]["ap"]) - recomputed["ap"]) <= 1e-9, f"{name}: AP not reproducible")
    first = strategies["max_precision_at_recall_ge_0.95"]
    require(abs(float(first["threshold"]) - deployed) <= 1e-9, "baseline strategy threshold != deployed threshold")
    require(first["matches_deployed_threshold"] is True, "matches_deployed_threshold not true")
    require(abs(float(first["test"]["recall"]) - baseline["recall"]) <= 1e-12, "baseline test recall != metrics.json")
    require(abs(float(first["test"]["precision"]) - baseline["precision"]) <= 1e-12, "baseline test precision != metrics.json")
    return {name: {"threshold": spec["threshold"], "test_recall": spec["test"]["recall"], "test_precision": spec["test"]["precision"]} for name, spec in strategies.items()}


def s07_threshold_selection_on_val() -> dict[str, Any]:
    """Each strategy's threshold is re-derived from cached val scores."""
    y_val, val_scores, _, _ = _val_scores()
    thr_json = load_json(OPT / "threshold_strategies.json")
    target = float(CONFIG["target_recall"])
    expected = {
        "max_precision_at_recall_ge_0.95": select_max_precision_at_recall(y_val, val_scores, target),
        "max_f2": select_max_fbeta(y_val, val_scores, 2.0),
    }
    for name, t in expected.items():
        recorded = float(thr_json["strategies"][name]["threshold"])
        require(abs(recorded - t) <= 1e-9, f"{name}: threshold {recorded} != re-derived {t}")
    for name, spec in thr_json["strategies"].items():
        recomputed = metrics_at(y_val, val_scores, float(spec["threshold"]))
        for key in ("recall", "precision", "f1", "f2"):
            require(abs(recomputed[key] - float(spec["val"][key])) <= 1e-9, f"{name}: val {key} mismatch")
    require(
        abs(float(thr_json["strategies"]["max_precision_at_recall_ge_0.95"]["val"]["recall"]) - 0.9633) < 5e-4,
        "val recall of baseline strategy drifted",
    )
    return {name: float(thr_json["strategies"][name]["threshold"]) for name in thr_json["strategies"]}


def s08_bootstrap_structure_and_independent_ci() -> dict[str, Any]:
    """O3: PMID-group bootstrap, >=500 replicates, independent CI agrees."""
    boot = load_json(OPT / "bootstrap_ci.json")
    require(boot["n_bootstrap"] >= 500, f"n_bootstrap {boot['n_bootstrap']} < 500")
    require(boot["confidence_level"] == 0.95, "confidence level != 0.95")
    require(Path(boot["feature_cache"]).exists(), "bootstrap records a missing feature cache")
    require(boot["reproduction"]["reproduced"] is True, "bootstrap reproduction flag not true")
    require(boot["reproduction"]["max_abs_score_diff_vs_test_predictions"] < 1e-5, "score reproduction drift")
    require(boot["reproduction"]["max_abs_metric_diff_vs_metrics_json"] < 1e-9, "metric reproduction drift")
    baseline = load_json(ART / "metrics.json")["primary"]["test"]
    for metric in ("recall", "precision", "f1", "ap"):
        entry = boot["metrics"][metric]
        require(entry["n_valid"] == boot["n_bootstrap"], f"{metric}: n_valid {entry['n_valid']} != replicates")
        require(abs(float(entry["point"]) - float(baseline[metric])) <= 1e-12, f"{metric}: point != metrics.json")
        require(entry["ci_low"] <= entry["mean"] <= entry["ci_high"], f"{metric}: mean outside CI")

    # Independent percentile bootstrap (fresh implementation, different seed).
    preds = pd.read_csv(ART / "test_predictions.csv", dtype={"doc_id": str})
    labels = pd.read_csv(
        ROOT / "data" / "labeled_papers.csv", keep_default_na=False, dtype={"doc_id": str, "pmid": str}
    )[["doc_id", "pmid"]]
    merged = preds.merge(labels, on="doc_id", how="left")
    require(merged["pmid"].notna().all(), "test rows missing PMID")
    y = merged["label"].to_numpy(int)
    scores = merged["score"].to_numpy(float)
    threshold = float(CONFIG["threshold"])
    groups = merged["pmid"].astype(str).to_numpy()
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(20261103)
    draws: dict[str, list[float]] = {m: [] for m in ("recall", "precision", "f1", "ap")}
    for _ in range(2000):
        pick = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([members[i] for i in pick])
        y_b, s_b = y[idx], scores[idx]
        m = metrics_at(y_b, s_b, threshold)
        for name in ("recall", "precision", "f1", "ap"):
            if m[name] is not None and np.isfinite(m[name]):
                draws[name].append(m[name])
    report: dict[str, Any] = {}
    for name, values in draws.items():
        lo, hi = np.percentile(values, [2.5, 97.5])
        recorded = boot["metrics"][name]
        require(abs(lo - recorded["ci_low"]) <= 0.02, f"{name}: ci_low {lo:.4f} vs recorded {recorded['ci_low']:.4f}")
        require(abs(hi - recorded["ci_high"]) <= 0.02, f"{name}: ci_high {hi:.4f} vs recorded {recorded['ci_high']:.4f}")
        report[name] = {"independent_ci": [round(float(lo), 4), round(float(hi), 4)],
                        "recorded_ci": [round(float(recorded["ci_low"]), 4), round(float(recorded["ci_high"]), 4)]}
    return report


def s09_group_cv_protocol() -> dict[str, Any]:
    """O4: GroupKFold k=5 on cached features, per-fold threshold from val only."""
    cv = load_json(OPT / "group_cv.json")
    require(cv["k"] == 5, f"k={cv['k']} != 5")
    require(cv["encoder_invoked"] is False, "encoder_invoked is not false")
    require(Path(cv["feature_cache"]).exists(), "group_cv records a missing feature cache")
    require(cv["n"] == 883 and cv["positives"] == 678, f"n/positives drifted: {cv['n']}/{cv['positives']}")
    require(len(cv["per_fold"]) == 5, "per_fold count != 5")
    target = float(cv["target_recall"])
    for fold in cv["per_fold"]:
        require(fold["n_train"] + fold["n_val"] + fold["n_test"] == cv["n"], f"fold {fold['fold']}: sizes do not cover n")
        require(
            fold["val_at_threshold"]["recall"] + 1e-9 >= target,
            f"fold {fold['fold']}: val recall {fold['val_at_threshold']['recall']} < target {target}",
        )
        require(fold["threshold"] > 0, f"fold {fold['fold']}: non-positive threshold")
    for metric in ("recall", "precision", "f1", "ap"):
        values = np.asarray([f["test"][metric] for f in cv["per_fold"]], dtype=float)
        recorded = cv["metrics"][metric]
        require(abs(values.mean() - recorded["mean"]) <= 1e-9, f"{metric}: mean not reproducible from folds")
        require(abs(values.std(ddof=1) - recorded["std"]) <= 1e-9, f"{metric}: std not reproducible from folds")
        require(recorded["per_fold"] == [float(v) for v in values], f"{metric}: per_fold list mismatch")
    th = np.asarray([f["threshold"] for f in cv["per_fold"]], dtype=float)
    require(abs(th.mean() - cv["metrics"]["threshold"]["mean"]) <= 1e-9, "threshold mean mismatch")
    pooled = cv["pooled_oof"]
    cm = pooled["confusion_matrix"]
    require(
        cm["fp"] + cm["fn"] + cm["tn"] + cm["tp"] == cv["n"],
        "pooled confusion does not cover all rows",
    )
    require(abs(pooled["recall"] - cm["tp"] / (cm["tp"] + cm["fn"])) <= 1e-9, "pooled recall inconsistent")
    require(abs(pooled["precision"] - cm["tp"] / (cm["tp"] + cm["fp"])) <= 1e-9, "pooled precision inconsistent")
    require(abs(pooled["ap"] - cv["metrics"]["ap"]["mean"]) <= 0.01, "pooled AP far from mean fold AP")
    return {
        "n": cv["n"],
        "n_groups": cv["groups"],
        "thresholds": [round(float(t), 6) for t in th],
        "metrics": {m: [round(cv["metrics"][m]["mean"], 4), round(cv["metrics"][m]["std"], 4)] for m in ("recall", "precision", "f1", "ap")},
        "note": "groups == rows in this dataset (unique PMIDs), so grouped CV and grouped bootstrap reduce to row-level resampling",
    }


def s10_cached_features_only() -> dict[str, Any]:
    """No encoder path is reachable: patch encoder entry points to abort."""
    drives = {
        "eval_hardening.py": ["--out", "<OUT>", "--bootstrap", "500", "--jobs", "2"],
        "error_audit.py": ["--out", "<OUT>"],
    }
    report: dict[str, Any] = {}
    for script, argv in drives.items():
        with tempfile.TemporaryDirectory(prefix="omp-encoder-guard-") as tmp:
            out = Path(tmp) / "out"
            driver = Path(tmp) / "driver.py"
            driver.write_text(
                "import runpy, sys\n"
                "net = []\n"
                "def _audit(event, args):\n"
                "    if event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):\n"
                "        net.append(event)\n"
                "sys.addaudithook(_audit)\n"
                "sys.path.insert(0, {root!r})\n"
                "import encoder\n"
                "calls = []\n"
                "def _boom(*a, **k):\n"
                "    calls.append('called')\n"
                "    raise SystemExit('ENCODER_CALLED')\n"
                "encoder.document_features = _boom\n"
                "encoder.document_features_batch = _boom\n"
                "encoder.load_encoder = _boom\n"
                "class _StubEncoder:\n"
                "    def __init__(self, *a, **k):\n"
                "        calls.append('init')\n"
                "        raise SystemExit('ENCODER_CALLED')\n"
                "encoder.SpecterEncoder = _StubEncoder\n"
                "sys.argv = [{script!r}] + {argv!r}\n"
                "runpy.run_path({path!r}, run_name='__main__')\n"
                "print('ENCODER_CALLS=' + str(len(calls)))\n"
                "print('NETWORK_EVENTS=' + str(len(net)))\n".format(
                    root=str(ROOT),
                    script=script,
                    argv=[str(out) if a == "<OUT>" else a for a in argv],
                    path=str(ROOT / "scripts" / script),
                ),
                encoding="utf-8",
            )
            started = time.time()
            proc = subprocess.run(
                [sys.executable, str(driver)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=1800,
            )
            elapsed = time.time() - started
            combined = proc.stdout + proc.stderr
            require(proc.returncode == 0, f"{script}: patched run exited {proc.returncode}: {combined[-800:]}")
            require("ENCODER_CALLED" not in combined, f"{script}: reached an encoder entry point")
            require(re.search(r"ENCODER_CALLS=0", combined) is not None, f"{script}: call counter missing")
            require(re.search(r"NETWORK_EVENTS=0", combined) is not None, f"{script}: network events observed")
            produced = sorted(p.name for p in out.glob("*"))
            require(produced, f"{script}: patched run produced no output")
            report[script] = {"seconds": round(elapsed, 1), "outputs": produced, "encoder_calls": 0}
            if script == "eval_hardening.py":
                require(elapsed < 180, f"{script}: {elapsed:.0f}s is implausible without cached features")
    return report


def s11_rerun_determinism() -> dict[str, Any]:
    """Both scripts reproduce the committed artifacts byte/number-identically."""
    outputs = {
        "error_audit.py": ["fn_audit.csv", "fp_audit.csv", "error_summary.json"],
        "eval_hardening.py": ["bootstrap_ci.json", "group_cv.json", "threshold_strategies.json"],
    }
    report: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="omp-rerun-a-") as tmp_a, tempfile.TemporaryDirectory(
        prefix="omp-rerun-b-"
    ) as tmp_b:
        for script in outputs:
            for tmp in (tmp_a, tmp_b):
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / script), "--out", str(Path(tmp) / "out")],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                require(proc.returncode == 0, f"{script} rerun failed: {proc.stderr[-800:]}")
            dir_a, dir_b = Path(tmp_a) / "out", Path(tmp_b) / "out"
            for name in outputs[script]:
                a, b, committed = dir_a / name, dir_b / name, OPT / name
                if name.endswith(".csv"):
                    require(a.read_bytes() == b.read_bytes(), f"{name}: two reruns differ")
                    require(a.read_bytes() == committed.read_bytes(), f"{name}: rerun != committed artifact")
                else:
                    norm_a = normalize_json(load_json(a), (str(Path(tmp_a)),))
                    norm_b = normalize_json(load_json(b), (str(Path(tmp_b)),))
                    require(norm_a == norm_b, f"{name}: two reruns differ")
                    norm_c = normalize_json(load_json(committed), ())
                    if name == "error_summary.json":
                        norm_c = normalize_json(load_json(committed), (str(OPT),))
                        norm_a = normalize_json(load_json(a), (str(dir_a),))
                        norm_b = normalize_json(load_json(b), (str(dir_b),))
                    require(norm_a == norm_c, f"{name}: rerun != committed artifact")
            report[script] = "reproduced"
    return report


def s12_protected_artifacts_unchanged() -> dict[str, Any]:
    """O7: main model, cache, metrics, predictions are not rewritten."""
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in PROTECTED if p.exists()}
    require(len(hashes) == len(PROTECTED), f"missing protected files: {[p for p in PROTECTED if not p.exists()]}")
    md5s = {
        "artifacts/model.joblib": "90905b95",
        "artifacts/test_predictions.csv": "bb079439",
    }
    for rel, prefix in md5s.items():
        actual = hashlib.md5((ROOT / rel).read_bytes()).hexdigest()
        require(actual.startswith(prefix), f"{rel}: md5 {actual[:8]} != DONE-recorded {prefix}")
    model_npz = [p for p in PROTECTED if p.suffix == ".npz"]
    require(len(model_npz) == 1, "expected exactly one feature npz")
    actual_npz = hashlib.md5(model_npz[0].read_bytes()).hexdigest()
    require(actual_npz.startswith("3507745f"), f"{model_npz[0].name}: md5 {actual_npz[:8]} != DONE-recorded 3507745f")
    for rel in ("scripts/error_audit.py", "scripts/eval_hardening.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for banned in ("joblib.dump(", "np.savez(", "torch.save(", "savez_compressed"):
            require(banned not in text, f"{rel}: contains write path {banned}")
    return hashes


def s13_docs_and_phase2_matrix() -> dict[str, Any]:
    """O6: DONE report + Phase 2 matrix + README pointer exist with substance."""
    done = ROOT / "briefs" / "OPTIMIZATION_PHASE1_DONE.md"
    matrix = OPT / "phase2_experiment_matrix.md"
    require(done.exists() and done.stat().st_size > 2000, "DONE report missing or too small")
    require(matrix.exists() and matrix.stat().st_size > 1000, "phase2 matrix missing or too small")
    done_text = done.read_text(encoding="utf-8")
    for token in ("O1", "O7", "bootstrap", "GroupKFold", "max_f2"):
        require(token in done_text, f"DONE report missing {token}")
    matrix_text = matrix.read_text(encoding="utf-8")
    for token in ("困难负样本", "章节感知分块", "CUDA"):
        require(token in matrix_text, f"phase2 matrix missing {token}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    require("Optimization Phase 1" in readme, "README lacks the Optimization Phase 1 section")
    require("artifacts/optimization/" in readme, "README does not point at artifacts/optimization")
    return {"done_bytes": done.stat().st_size, "matrix_bytes": matrix.stat().st_size}


def s15_doc_numbers_match_artifacts() -> dict[str, Any]:
    """DONE/README numbers are regenerated from the artifacts, not remembered."""
    done = (ROOT / "briefs" / "OPTIMIZATION_PHASE1_DONE.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    boot = load_json(OPT / "bootstrap_ci.json")["metrics"]
    cv = load_json(OPT / "group_cv.json")
    thr = load_json(OPT / "threshold_strategies.json")["strategies"]
    mismatches: list[str] = []

    def fmt(value: float) -> str:
        # Human-facing tables round half up from the decimal repr; Python's
        # format() rounds the underlying binary value, so 0.8385 would become
        # "0.838" even though the report correctly says 0.839.
        return str(Decimal(repr(float(value))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

    def expect(container: str, text: str, value: str) -> None:
        if value not in text:
            mismatches.append(f"{container}: missing {value}")

    for metric in ("recall", "precision", "f1", "ap"):
        for bound in ("ci_low", "ci_high"):
            value = fmt(boot[metric][bound])
            expect("DONE bootstrap", done, value)
            if metric in ("recall", "precision", "ap"):
                expect("README bootstrap", readme, value)
    for metric in ("recall", "precision", "f1", "ap"):
        mean, std = fmt(cv["metrics"][metric]["mean"]), fmt(cv["metrics"][metric]["std"])
        expect("DONE group_cv", done, mean)
        expect("DONE group_cv", done, std)
        if metric in ("recall", "precision", "ap"):
            expect("README group_cv", readme, mean)
            expect("README group_cv", readme, std)
    pooled = cv["pooled_oof"]
    for key in ("recall", "precision", "ap"):
        expect("DONE pooled_oof", done, fmt(pooled[key]))
    for fold in cv["per_fold"]:
        expect("DONE fold threshold", done, fmt(fold["threshold"]))
    for name, spec in thr.items():
        for key in ("recall", "precision"):
            expect("DONE strategy val", done, fmt(spec["val"][key]))
        for key in ("recall", "precision", "f1"):
            expect("DONE strategy test", done, fmt(spec["test"][key]))
        expect("DONE strategy predicted_positive", done, str(spec["test"]["predicted_positive"]))
        cm = spec["test"]["confusion_matrix"]
        expect("DONE strategy FN/FP pair", done, f"{cm['fn']} / {cm['fp']}")
    require(not mismatches, f"doc numbers disagree with artifacts: {mismatches}")
    return {"checked_containers": ["DONE", "README"], "mismatches": mismatches}


def _expect_failure(label: str, fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        fn()
    except CheckFailure as exc:
        return {"control": label, "detected": True, "evidence": str(exc)[:300]}
    except Exception as exc:  # any crash is also a detection
        return {"control": label, "detected": True, "evidence": f"{type(exc).__name__}: {exc}"[:300]}
    return {"control": label, "detected": False, "evidence": "scenario passed on tampered input"}


def _driver_output(source: str) -> str:
    with tempfile.TemporaryDirectory(prefix="omp-neg-driver-") as tmp:
        driver = Path(tmp) / "control.py"
        driver.write_text(source.replace("<ROOT>", str(ROOT)), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(driver)], cwd=str(ROOT), capture_output=True, text=True, timeout=300)
        return proc.stdout + proc.stderr


def _driver_control(source: str, expect: str) -> dict[str, Any]:
    combined = _driver_output(source)
    return {
        "detected": expect in combined,
        "evidence": combined.strip().splitlines()[-1][:300] if combined.strip() else "no output",
    }


def negative_controls() -> list[dict[str, Any]]:
    """Tamper a copy of an input and prove the matching scenario goes red."""
    global OPT, ROOT
    base_opt, base_root = OPT, ROOT
    controls: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="omp-neg-") as tmp_dir:
            tmp = Path(tmp_dir)

            opt_a = tmp / "optA"
            shutil.copytree(base_opt, opt_a)
            fp = pd.read_csv(opt_a / "fp_audit.csv")
            fp.iloc[:-1].to_csv(opt_a / "fp_audit.csv", index=False)
            OPT = opt_a
            controls.append(_expect_failure("S01: one FP audit row removed", s01_audit_rows_match_predictions))

            opt_b = tmp / "optB"
            shutil.copytree(base_opt, opt_b)
            audit = pd.read_csv(opt_b / "fp_audit.csv", dtype=str, keep_default_na=False)
            audit.loc[0, "heuristic_tag"] = "design_review"
            audit.to_csv(opt_b / "fp_audit.csv", index=False)
            OPT = opt_b
            controls.append(_expect_failure("S03: heuristic tag rewritten", s03_audit_tags_defined_and_consistent))

            opt_c = tmp / "optC"
            shutil.copytree(base_opt, opt_c)
            thr = load_json(opt_c / "threshold_strategies.json")
            thr["strategies"]["max_f2"]["threshold"] += 0.001
            thr["strategies"]["max_f2"]["test"]["precision"] += 0.05
            (opt_c / "threshold_strategies.json").write_text(json.dumps(thr), encoding="utf-8")
            OPT = opt_c
            controls.append(_expect_failure("S06: reported test metric altered", s06_threshold_strategies_metrics))
            controls.append(_expect_failure("S07: strategy threshold shifted", s07_threshold_selection_on_val))

            opt_d = tmp / "optD"
            shutil.copytree(base_opt, opt_d)
            boot = load_json(opt_d / "bootstrap_ci.json")
            boot["metrics"]["recall"]["ci_low"] += 0.15
            (opt_d / "bootstrap_ci.json").write_text(json.dumps(boot), encoding="utf-8")
            OPT = opt_d
            controls.append(_expect_failure("S08: bootstrap CI inflated", s08_bootstrap_structure_and_independent_ci))

            opt_e = tmp / "optE"
            shutil.copytree(base_opt, opt_e)
            cv = load_json(opt_e / "group_cv.json")
            cv["per_fold"][0]["test"]["recall"] += 0.01
            (opt_e / "group_cv.json").write_text(json.dumps(cv), encoding="utf-8")
            OPT = opt_e
            controls.append(_expect_failure("S09: fold recall altered", s09_group_cv_protocol))

            OPT = base_opt
            root_f = tmp / "rootF"
            (root_f / "briefs").mkdir(parents=True)
            for name in ("OPTIMIZATION_PHASE1_DONE.md",):
                text = (base_root / "briefs" / name).read_text(encoding="utf-8")
                (root_f / "briefs" / name).write_text(text.replace("0.963", "0.960"), encoding="utf-8")
            (root_f / "README.md").write_text(
                (base_root / "README.md").read_text(encoding="utf-8").replace("0.963", "0.960"),
                encoding="utf-8",
            )
            ROOT = root_f
            controls.append(_expect_failure("S15: documented CI bound altered", s15_doc_numbers_match_artifacts))
    finally:
        OPT, ROOT = base_opt, base_root

    encoder_control = _driver_control(
        "import sys\n"
        "sys.path.insert(0, '<ROOT>')\n"
        "import encoder\n"
        "def _boom(*a, **k):\n"
        "    raise SystemExit('ENCODER_CALLED')\n"
        "encoder.document_features = _boom\n"
        "encoder.document_features('t', 'a', 'x')\n"
        "print('NOT_DETECTED')\n",
        "ENCODER_CALLED",
    )
    controls.append({"control": "S10: encoder entry point invoked directly", **encoder_control})

    network_out = _driver_output(
        "import socket, sys\n"
        "net = []\n"
        "def _audit(event, args):\n"
        "    if event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):\n"
        "        net.append(event)\n"
        "sys.addaudithook(_audit)\n"
        "try:\n"
        "    socket.getaddrinfo('127.0.0.1', 9)\n"
        "except OSError:\n"
        "    pass\n"
        "print('NETWORK_EVENTS=' + str(len(net)))\n"
    )
    match = re.search(r"NETWORK_EVENTS=(\d+)", network_out)
    controls.append(
        {
            "control": "S10: network audit hook sees a connect",
            "detected": bool(match) and int(match.group(1)) >= 1,
            "evidence": match.group(0) if match else network_out[-200:],
        }
    )
    return controls


SCENARIOS: list[tuple[str, Callable[[], Any]]] = [
    ("S01 audit rows == independent FN/FP selection", s01_audit_rows_match_predictions),
    ("S02 audit schema + blank manual columns", s02_audit_schema_and_manual_columns),
    ("S03 heuristic tags match definitions", s03_audit_tags_defined_and_consistent),
    ("S04 error_summary confusion/totals", s04_summary_confusion_and_totals),
    ("S05 documented empty-reason gap", s05_documented_reason_gap),
    ("S06 threshold strategies reproduce on test", s06_threshold_strategies_metrics),
    ("S07 thresholds re-derived on val", s07_threshold_selection_on_val),
    ("S08 bootstrap CI reproducible", s08_bootstrap_structure_and_independent_ci),
    ("S09 GroupKFold protocol and aggregates", s09_group_cv_protocol),
    ("S10 cached features only (encoder guard)", s10_cached_features_only),
    ("S11 reruns reproduce committed artifacts", s11_rerun_determinism),
    ("S12 protected artifacts unchanged", s12_protected_artifacts_unchanged),
    ("S13 docs + Phase 2 matrix", s13_docs_and_phase2_matrix),
    ("S15 doc numbers == artifacts", s15_doc_numbers_match_artifacts),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None, help="write the result JSON here")
    parser.add_argument("--no-rerun", action="store_true", help="skip S10/S11 subprocess runs")
    parser.add_argument("--only", default=None, help="comma-separated scenario id prefix list")
    parser.add_argument(
        "--negative-controls",
        action="store_true",
        help="tamper copies of the inputs and require the matching scenario to fail",
    )
    args = parser.parse_args()

    before = {str(p): sha256(p) for p in PROTECTED if p.exists()}
    results: list[dict[str, Any]] = []
    failures = 0
    for label, fn in SCENARIOS:
        sid = label.split()[0]
        if args.only and sid not in {s.strip() for s in args.only.split(",")}:
            continue
        if args.no_rerun and sid in {"S10", "S11"}:
            results.append({"id": sid, "name": label, "status": "SKIPPED", "detail": "--no-rerun"})
            print(f"SKIP {label} (--no-rerun)")
            continue
        started = time.time()
        try:
            detail = fn()
            status = "PASS"
        except Exception as exc:  # fail closed: any exception is a failure
            detail = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            status = "FAIL"
            failures += 1
        results.append(
            {"id": sid, "name": label, "status": status, "seconds": round(time.time() - started, 2), "detail": detail}
        )
        print(f"{status} {label}  ({time.time() - started:.1f}s)")
        if status == "FAIL":
            print(json.dumps(detail, indent=2, ensure_ascii=False)[:2000])

    after = {str(p): sha256(p) for p in PROTECTED if p.exists()}
    changed = [p for p in before if before[p] != after.get(p)]
    if changed:
        results.append(
            {"id": "S14", "name": "protected artifacts unchanged across the suite", "status": "FAIL",
             "detail": {"changed": changed}}
        )
        print(f"FAIL S14 protected artifacts changed: {changed}")
        failures += 1
    else:
        results.append(
            {"id": "S14", "name": "protected artifacts unchanged across the suite", "status": "PASS",
             "detail": {"checked": len(before)}}
        )
        print(f"PASS S14 protected artifacts unchanged across the suite ({len(before)} files)")

    payload = {
        "suite": str(Path(__file__).resolve()),
        "spec": "briefs/OPTIMIZATION_REVIEW_SPEC.md",
        "run_at": pd.Timestamp.now("UTC").isoformat(),
        "python": sys.version.split()[0],
        "failures": failures,
        "results": results,
    }
    if args.negative_controls:
        controls = negative_controls()
        payload["negative_controls"] = controls
        for control in controls:
            marker = "CTRL-OK  " if control["detected"] else "CTRL-MISS"
            print(f"{marker} {control['control']} :: {control['evidence']}")
            if not control["detected"]:
                failures += 1
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    print(f"\n{passed}/{len(results)} scenarios passed, {skipped} skipped, {failures} failures")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
