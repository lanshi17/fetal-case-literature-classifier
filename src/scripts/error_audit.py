#!/usr/bin/env python3
"""Export false negatives / false positives of the primary screening baseline.

Inputs (read-only)
------------------
artifacts/config.json            threshold + model path + feature metadata
artifacts/test_predictions.csv   primary-model test scores/decisions
artifacts/metrics.json           baseline confusion matrix (consistency check)
data/labeled_papers.csv          titles, design, reason, text metadata
data/final_labels.csv            optional richer reason/quote (merged by PMID)

Outputs
-------
artifacts/optimization/fn_audit.csv
artifacts/optimization/fp_audit.csv
artifacts/optimization/error_summary.json

Definitions match the deployed decision rule:
    predicted positive  <=> score >= threshold  <=> decision == "relevant"
    FN = label == 1 and score <  threshold
    FP = label == 0 and score >= threshold

The review grey band (threshold - 0.15) is reported but review rows still count
as "not relevant" for the audit, exactly as they do in the baseline confusion
matrix (metrics.json).

Usage
-----
    .venv/bin/python src/scripts/error_audit.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from train import load_dataset  # noqa: E402  (shared loader keeps definitions identical)

logger = logging.getLogger("error_audit")

DEFAULT_CONFIG = ROOT / "artifacts" / "config.json"
DEFAULT_PREDICTIONS = ROOT / "artifacts" / "test_predictions.csv"
DEFAULT_DATA = ROOT / "data" / "labeled_papers.csv"
DEFAULT_FINAL_LABELS = ROOT / "data" / "final_labels.csv"
DEFAULT_METRICS = ROOT / "artifacts" / "metrics.json"
DEFAULT_OUT = ROOT / "artifacts" / "optimization"

GREY_BAND = 0.15
TEXT_HEAD_CHARS = 500

# Definitions are embedded in error_summary.json so a reviewer can re-derive
# every tag without reading this file.
TAG_DEFINITIONS = {
    "empty_abstract": "abstract_available is false or abstract is blank",
    "pdf_text_source": "full text came from PDF fallback (text_source == 'pdf')",
    "weak_or_model_label": "weak == True or label_source == 'model'",
    "design_missing": "design field is empty",
    "design_review": "design contains 'review'",
    "design_case_report": "design contains 'case report' or 'case series'",
    "design_cohort_or_screening": "design contains cohort/screening/validation/registry",
    "design_method": "design is a methods paper",
    "design_other": "non-empty design outside the buckets above",
    "low_score_band": "error lies within 0.15 of the deployed threshold (review grey band)",
    "below_review_band": "FN score is below threshold-0.15 (not even review-eligible)",
    "mid_score_band": "FP score is >threshold+0.15 but <0.9 (mid-confidence FP)",
    "high_score_confident_error": "|score-0.5| >= 0.4, i.e. a high-confidence mistake",
}

AUDIT_COLUMNS = [
    "doc_id",
    "pmid",
    "title",
    "year",
    "label",
    "score",
    "threshold",
    "decision",
    "design",
    "reason",
    "quote",
    "label_source",
    "confidence",
    "weak",
    "text_source",
    "abstract_available",
    "text_chars",
    "heuristic_tag",
    "manual_tag",
    "manual_note",
    "text_head",
]


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _add(tags: list[str], tag: str) -> None:
    if tag not in tags:
        tags.append(tag)


def design_tags(design: str) -> list[str]:
    d = design.strip().lower()
    if not d:
        return ["design_missing"]
    if "review" in d:
        return ["design_review"]
    if "case report" in d or "case series" in d:
        return ["design_case_report"]
    if any(k in d for k in ("cohort", "screening", "validation", "registry")):
        return ["design_cohort_or_screening"]
    if d.startswith("method"):
        return ["design_method"]
    return ["design_other"]


def heuristic_tags(row: pd.Series, *, side: str, threshold: float) -> list[str]:
    """Multi-label heuristic attribution, deterministic and data-only."""
    tags: list[str] = []
    score = float(row["score"])

    if _as_bool(row.get("weak", "")) or _clean(row.get("label_source", "")).lower() == "model":
        _add(tags, "weak_or_model_label")
    if not _as_bool(row.get("abstract_available", "")) or not _clean(row.get("abstract", "")):
        _add(tags, "empty_abstract")
    if _clean(row.get("text_source", "")).lower() == "pdf":
        _add(tags, "pdf_text_source")
    for tag in design_tags(_clean(row.get("design", ""))):
        _add(tags, tag)

    if side == "FN":
        if score >= threshold - GREY_BAND:
            _add(tags, "low_score_band")
        else:
            _add(tags, "below_review_band")
    else:  # FP
        if score <= threshold + GREY_BAND:
            _add(tags, "low_score_band")
        elif score < 0.9:
            _add(tags, "mid_score_band")

    # High-confidence mistake in either direction (confidently wrong).
    if abs(score - 0.5) >= 0.4:
        _add(tags, "high_score_confident_error")

    return tags


def tag_counts(tags_series: Iterable[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in tags_series:
        for tag in str(value).split(";"):
            tag = tag.strip()
            if tag:
                counter[tag] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def load_inputs(
    config_path: Path,
    predictions_path: Path,
    data_path: Path,
    final_labels_path: Path | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preds = pd.read_csv(predictions_path, keep_default_na=False, dtype={"doc_id": str, "pmid": str})
    preds["doc_id"] = preds["doc_id"].astype(str)
    df = load_dataset(data_path)
    df["doc_id"] = df["doc_id"].astype(str)

    final_labels = None
    if final_labels_path is not None and final_labels_path.exists():
        final_labels = pd.read_csv(
            final_labels_path, encoding="utf-8-sig", keep_default_na=False, dtype=str
        )
        final_labels["pmid"] = final_labels["pmid"].astype(str)
        final_labels = final_labels.drop_duplicates("pmid", keep="first")
        logger.info("Loaded optional final_labels: %d rows", len(final_labels))
    return config, preds, df, final_labels


def build_audit_frame(
    preds: pd.DataFrame,
    df: pd.DataFrame,
    final_labels: pd.DataFrame | None,
    threshold: float,
    side: str,
) -> pd.DataFrame:
    if side == "FN":
        selected = preds[(preds["label"] == 1) & (preds["score"] < threshold)].copy()
    elif side == "FP":
        selected = preds[(preds["label"] == 0) & (preds["score"] >= threshold)].copy()
    else:  # pragma: no cover
        raise ValueError(side)

    merged = selected.merge(
        df[
            [
                "doc_id",
                "pmid",
                "title",
                "year",
                "abstract",
                "text",
                "design",
                "label_reason",
                "label_source",
                "confidence",
                "weak",
                "text_source",
                "abstract_available",
                "text_chars",
            ]
        ],
        on="doc_id",
        how="left",
        suffixes=("", "_data"),
    )
    if len(merged) != len(selected):
        raise RuntimeError(f"{side}: merge changed row count {len(selected)} -> {len(merged)}")

    if final_labels is not None:
        fl = final_labels[["pmid", "reason", "quote"]].rename(
            columns={"reason": "reason_final", "quote": "quote_final"}
        )
        merged = merged.merge(fl, on="pmid", how="left")
    else:
        merged["reason_final"] = ""
        merged["quote_final"] = ""

    for col in ("reason_final", "quote_final", "label_reason", "design", "text", "abstract"):
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("").astype(str)

    merged["reason"] = merged["label_reason"].where(
        merged["label_reason"].str.strip().ne(""), merged["reason_final"]
    )
    merged["quote"] = merged["quote_final"]
    merged["manual_tag"] = ""
    merged["manual_note"] = ""
    merged["threshold"] = float(threshold)
    merged["heuristic_tag"] = [
        ";".join(heuristic_tags(row, side=side, threshold=threshold))
        for _, row in merged.iterrows()
    ]
    merged["text_head"] = (
        merged["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, TEXT_HEAD_CHARS)
    )

    # Keep the schema stable even if a source column is missing.
    for col in AUDIT_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""
    out = merged[AUDIT_COLUMNS].copy()
    out["score"] = out["score"].astype(float)
    out["threshold"] = out["threshold"].astype(float)
    out["label"] = out["label"].astype(int)
    out = out.sort_values(["score", "doc_id"], ascending=[side == "FP", True]).reset_index(drop=True)
    return out


def confusion_from_predictions(preds: pd.DataFrame, threshold: float) -> dict[str, int]:
    y = preds["label"].astype(int).to_numpy()
    pred = (preds["score"].astype(float).to_numpy() >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def baseline_confusion(metrics_path: Path) -> dict[str, int] | None:
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cm = metrics.get("primary", {}).get("test", {}).get("confusion_matrix")
    return {k: int(v) for k, v in cm.items()} if cm else None


def make_findings(
    fn: pd.DataFrame, fp: pd.DataFrame, fn_tags: dict[str, int], fp_tags: dict[str, int]
) -> list[str]:
    findings: list[str] = []
    findings.append(f"Audited {len(fn)} FN and {len(fp)} FP from the default-subset test split.")
    if fn_tags:
        top = ", ".join(f"{k}={v}" for k, v in list(fn_tags.items())[:4])
        findings.append(f"FN dominant tags: {top}.")
    if fp_tags:
        top = ", ".join(f"{k}={v}" for k, v in list(fp_tags.items())[:4])
        findings.append(f"FP dominant tags: {top}.")

    n_fn_no_abs = int(fn["heuristic_tag"].str.contains("empty_abstract").sum())
    n_fp_no_abs = int(fp["heuristic_tag"].str.contains("empty_abstract").sum())
    findings.append(
        f"Missing abstracts: {n_fn_no_abs}/{len(fn)} FN, {n_fp_no_abs}/{len(fp)} FP."
    )
    n_fn_grey = int(fn["heuristic_tag"].str.contains("low_score_band").sum())
    n_fp_grey = int(fp["heuristic_tag"].str.contains("low_score_band").sum())
    findings.append(
        f"Grey-band (within {GREY_BAND} of threshold): {n_fn_grey}/{len(fn)} FN, "
        f"{n_fp_grey}/{len(fp)} FP."
    )
    n_fp_pdf = int(fp["heuristic_tag"].str.contains("pdf_text_source").sum())
    findings.append(f"PDF-sourced full text: {n_fp_pdf}/{len(fp)} FP.")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--final-labels", type=Path, default=DEFAULT_FINAL_LABELS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    config, preds, df, final_labels = load_inputs(
        args.config, args.predictions, args.data, args.final_labels
    )
    threshold = float(config["threshold"])
    logger.info("Threshold from config: %.6f", threshold)

    computed_cm = confusion_from_predictions(preds, threshold)
    expected_cm = baseline_confusion(args.metrics)
    consistent = expected_cm is not None and expected_cm == computed_cm
    if expected_cm is None:
        logger.warning("metrics.json has no baseline confusion matrix; skipping consistency check")
    elif not consistent:
        logger.warning("Computed confusion %s != baseline %s", computed_cm, expected_cm)

    fn = build_audit_frame(preds, df, final_labels, threshold, "FN")
    fp = build_audit_frame(preds, df, final_labels, threshold, "FP")
    logger.info("FN rows=%d, FP rows=%d", len(fn), len(fp))

    fn_path = args.out / "fn_audit.csv"
    fp_path = args.out / "fp_audit.csv"
    fn.to_csv(fn_path, index=False, encoding="utf-8")
    fp.to_csv(fp_path, index=False, encoding="utf-8")
    logger.info("Wrote %s and %s", fn_path, fp_path)

    fn_tags = tag_counts(fn["heuristic_tag"])
    fp_tags = tag_counts(fp["heuristic_tag"])
    all_tags = tag_counts(pd.concat([fn["heuristic_tag"], fp["heuristic_tag"]], ignore_index=True))

    checks = {
        "fn_rows_match_baseline": expected_cm is None or len(fn) == expected_cm["fn"],
        "fp_rows_match_baseline": expected_cm is None or len(fp) == expected_cm["fp"],
        "confusion_matrix_matches_baseline": consistent,
        "fn_scores_below_threshold": bool((fn["score"] < threshold).all()),
        "fp_scores_at_or_above_threshold": bool((fp["score"] >= threshold).all()),
        "all_rows_have_heuristic_tag": bool(
            (fn["heuristic_tag"].str.len() > 0).all() and (fp["heuristic_tag"].str.len() > 0).all()
        ),
        "manual_columns_empty": bool(
            (fn["manual_tag"] == "").all()
            and (fn["manual_note"] == "").all()
            and (fp["manual_tag"] == "").all()
            and (fp["manual_note"] == "").all()
        ),
    }
    checks["all_pass"] = all(checks.values())

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": str(args.config),
            "predictions": str(args.predictions),
            "data": str(args.data),
            "final_labels": str(args.final_labels) if final_labels is not None else None,
            "metrics": str(args.metrics),
        },
        "decision_rule": {
            "predicted_positive": "score >= threshold",
            "fn": "label == 1 and score < threshold",
            "fp": "label == 0 and score >= threshold",
            "grey_band": GREY_BAND,
            "threshold": threshold,
            "note": "review rows count as not relevant, matching metrics.json",
        },
        "totals": {
            "n_test": int(len(preds)),
            "positives": int((preds["label"] == 1).sum()),
            "negatives": int((preds["label"] == 0).sum()),
            "fn": int(len(fn)),
            "fp": int(len(fp)),
            "tp": int(computed_cm["tp"]),
            "tn": int(computed_cm["tn"]),
        },
        "computed_confusion_matrix": computed_cm,
        "baseline_confusion_matrix": expected_cm,
        "confusion_consistent": consistent,
        "checks": checks,
        "heuristic_tag_counts": {
            "all": all_tags,
            "fn": fn_tags,
            "fp": fp_tags,
        },
        "tag_definitions": TAG_DEFINITIONS,
        "design_breakdown": {
            "fn": fn["design"].replace("", "<missing>").value_counts().to_dict(),
            "fp": fp["design"].replace("", "<missing>").value_counts().to_dict(),
        },
        "score_stats": {
            "fn": {
                "min": float(fn["score"].min()) if len(fn) else None,
                "max": float(fn["score"].max()) if len(fn) else None,
                "mean": float(fn["score"].mean()) if len(fn) else None,
            },
            "fp": {
                "min": float(fp["score"].min()) if len(fp) else None,
                "max": float(fp["score"].max()) if len(fp) else None,
                "mean": float(fp["score"].mean()) if len(fp) else None,
            },
        },
        "manual_review": {
            "manual_tag_filled": int((fn["manual_tag"] != "").sum() + (fp["manual_tag"] != "").sum()),
            "manual_note_filled": int(
                (fn["manual_note"] != "").sum() + (fp["manual_note"] != "").sum()
            ),
            "note": "manual_tag/manual_note are intentionally blank for human review",
        },
        "findings": make_findings(fn, fp, fn_tags, fp_tags),
        "artifacts": {"fn_audit_csv": str(fn_path), "fp_audit_csv": str(fp_path)},
    }
    summary_path = args.out / "error_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", summary_path)

    print(json.dumps({
        "fn": len(fn),
        "fp": len(fp),
        "confusion": computed_cm,
        "consistent_with_metrics": consistent,
        "all_checks_pass": checks["all_pass"],
        "top_fn_tags": list(fn_tags.items())[:5],
        "top_fp_tags": list(fp_tags.items())[:5],
    }, indent=2))

    if not checks["all_pass"]:
        logger.error("Audit checks failed: %s", [k for k, v in checks.items() if not v])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
