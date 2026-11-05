#!/usr/bin/env python3
"""Fair retrain of casefilter head_v3 on the same PubMedBERT 1536-d features.

Why this exists
---------------
The shipped ``head_v3`` was finally fit on *all* ~965 Claude/curated papers, so
scoring our seed-42 test fold is contaminated (many test PMIDs were in that
final fit). This script retrains an identical-style LR head **only on the train
fold**, picks C by train CV (like retrain_v3.py), and evaluates on held-out
val/test. Shipped head is reported as an *unfair upper bound* only.

Recipe matched to pubmedbert/scripts/retrain_v3.py:
  - features: mean(all)||mean(first2) 1536-d, no StandardScaler
  - LogisticRegression(C, max_iter=5000) with source×label balanced sample weights
  - C ∈ {0.1, 1, 10, 100} by 5-fold StratifiedKFold AUC on train
  - thresholds: 0.5, balanced-acc on train-CV probs, and val Recall≥0.95

Also reports our sklearn ``class_weight='balanced'`` + StandardScaler baseline
on the same split for an apples-to-apples head comparison.

Usage
-----
    .venv/bin/python scripts/retrain_head_v3_fair.py
    .venv/bin/python scripts/retrain_head_v3_fair.py --promote
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoder import IdentityScaler, ShippedLogisticHead  # noqa: E402
from train import (  # noqa: E402
    binary_metrics,
    default_subset_mask,
    load_dataset,
    make_splits,
    select_threshold,
)

logger = logging.getLogger("retrain_v3_fair")


def source_label_weights(src: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Match casefilter retrain_v3.weights: balance (source, label) cells then classes."""
    cell = pd.Series(list(zip(src.tolist(), y.tolist())))
    w = np.array(1.0 / cell.map(cell.value_counts()), dtype=float)
    for lab in (0, 1):
        mask = y == lab
        if mask.any():
            w[mask] *= 0.5 / w[mask].sum()
    return w / w.mean()


def fit_v3(X: np.ndarray, y: np.ndarray, src: np.ndarray, C: float) -> LogisticRegression:
    clf = LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
    clf.fit(X, y, sample_weight=source_label_weights(src, y))
    return clf


def pick_C(X: np.ndarray, y: np.ndarray, src: np.ndarray, Cs: list[float]) -> tuple[float, float, np.ndarray]:
    best: tuple[float, float, np.ndarray] | None = None
    for C in Cs:
        oof = np.zeros(len(y), dtype=float)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, va in skf.split(X, y):
            clf = fit_v3(X[tr], y[tr], src[tr], C)
            oof[va] = clf.predict_proba(X[va])[:, 1]
        auc = float(roc_auc_score(y, oof))
        logger.info("train-CV C=%s AUC=%.4f", C, auc)
        if best is None or auc > best[0]:
            best = (auc, C, oof)
    assert best is not None
    return best[1], best[0], best[2]


def balanced_thr(y: np.ndarray, scores: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[np.argmax([balanced_accuracy_score(y, scores >= g) for g in grid])])


def score_bundle(name: str, scores: np.ndarray, y: np.ndarray, idx: np.ndarray, thr: float) -> dict[str, Any]:
    return {"name": name, "threshold": float(thr), **binary_metrics(y[idx], scores[idx], thr)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "labeled_papers.csv")
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--weights", type=Path, default=ROOT / "pubmedbert" / "weights")
    ap.add_argument("--gold", type=Path, default=ROOT / "data" / "gold_standard.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--Cs", type=str, default="0.1,1,10,100")
    ap.add_argument(
        "--promote",
        action="store_true",
        help="write fair head as artifacts/model.joblib (primary)",
    )
    args = ap.parse_args()
    out = args.artifacts / "head_v3_fair"
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((args.artifacts / "config.json").read_text())
    cache = Path(cfg["feature_cache"])
    df = load_dataset(args.data)
    data = np.load(cache, allow_pickle=False)
    if list(data["doc_ids"].astype(str)) != df["doc_id"].astype(str).tolist():
        raise SystemExit("feature cache / data order mismatch")

    X = np.concatenate([data["mean_all"], data["mean_head"]], axis=1).astype(np.float64)
    y = df["label"].to_numpy(dtype=int)
    src = df["text_source"].fillna("unknown").astype(str).to_numpy()
    subset = default_subset_mask(df).to_numpy()

    train_idx, val_idx, test_idx = make_splits(df, seed=args.seed)
    train_sub = train_idx[subset[train_idx]]
    val_sub = val_idx[subset[val_idx]]
    test_sub = test_idx[subset[test_idx]]

    Cs = [float(x) for x in args.Cs.split(",") if x.strip()]
    C, cv_auc, oof = pick_C(X[train_sub], y[train_sub], src[train_sub], Cs)
    logger.info("Selected C=%.3g (train-CV AUC=%.4f)", C, cv_auc)

    clf = fit_v3(X[train_sub], y[train_sub], src[train_sub], C)
    scores = clf.predict_proba(X)[:, 1]

    thr05 = 0.5
    thr_bal = balanced_thr(y[train_sub], oof)
    thr95 = select_threshold(y[val_sub], scores[val_sub], args.target_recall)

    # --- unfair shipped upper bound ---
    shipped_scores = None
    shipped_cfg = None
    if (args.weights / "config.json").exists():
        shipped_cfg = json.loads((args.weights / "config.json").read_text())
        z = np.load(args.weights / shipped_cfg["head"]["file"])
        coef = z["coef"].astype(np.float64).ravel()
        intercept = float(z["intercept"].ravel()[0])
        shipped_head = ShippedLogisticHead(coef, intercept)
        shipped_scores = shipped_head.predict_proba(X)[:, 1]
        thr_ship_05 = float(shipped_cfg.get("threshold", 0.5))
        thr_ship_95 = select_threshold(y[val_sub], shipped_scores[val_sub], args.target_recall)
    else:
        thr_ship_05 = thr_ship_95 = None

    # --- our StandardScaler + class_weight balanced (same split, C matched) ---
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X[train_sub])
    clf_sk = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=0)
    clf_sk.fit(Xs_tr, y[train_sub])
    scores_sk = clf_sk.predict_proba(scaler.transform(X))[:, 1]
    thr_sk95 = select_threshold(y[val_sub], scores_sk[val_sub], args.target_recall)

    rows = {
        "fair_v3_thr0.5": score_bundle("fair_v3", scores, y, test_sub, thr05),
        "fair_v3_thr_bal": score_bundle("fair_v3", scores, y, test_sub, thr_bal),
        "fair_v3_thr_r95": score_bundle("fair_v3", scores, y, test_sub, thr95),
        "sklearn_bal_thr_r95": score_bundle("sklearn_balanced", scores_sk, y, test_sub, thr_sk95),
    }
    val_rows = {
        "fair_v3_thr_r95": score_bundle("fair_v3", scores, y, val_sub, thr95),
        "sklearn_bal_thr_r95": score_bundle("sklearn_balanced", scores_sk, y, val_sub, thr_sk95),
    }
    if shipped_scores is not None:
        rows["shipped_v3_thr0.5_UNFAIR"] = score_bundle("shipped_UNFAIR", shipped_scores, y, test_sub, thr_ship_05)
        rows["shipped_v3_thr_r95_UNFAIR"] = score_bundle("shipped_UNFAIR", shipped_scores, y, test_sub, thr_ship_95)

    # gold-40 if present
    gold_rows: dict[str, Any] = {}
    if args.gold.exists():
        gold = pd.read_csv(args.gold, dtype={"pmid": str})
        gold = gold[gold["gold_label"].astype(str).isin(["0", "1"])].copy()
        gold["gold_label"] = gold["gold_label"].astype(int)
        gmap = gold.drop_duplicates("pmid", keep="last").set_index("pmid")["gold_label"]
        ids = df["doc_id"].astype(str).tolist()
        g_idx = np.array([i for i, d in enumerate(ids) if d in gmap.index], dtype=int)
        y_g = np.asarray([int(gmap.loc[ids[i]]) for i in g_idx], dtype=int)
        gold_rows = {
            "fair_v3_thr_r95": binary_metrics(y_g, scores[g_idx], thr95),
            "sklearn_bal_thr_r95": binary_metrics(y_g, scores_sk[g_idx], thr_sk95),
        }
        if shipped_scores is not None:
            gold_rows["shipped_v3_thr_r95_UNFAIR"] = binary_metrics(y_g, shipped_scores[g_idx], thr_ship_95)

    # persist fair head (IdentityScaler + sklearn LR coefs as ShippedLogisticHead OR dump clf)
    bundle = {
        "model": clf,  # sklearn LR, no scaler needed if we use IdentityScaler
        "scaler": IdentityScaler().fit(X[train_sub]),
        "threshold": float(thr95),
        "threshold_0.5": thr05,
        "threshold_balanced": thr_bal,
        "encoder_family": "pubmedbert",
        "feature_mode": "B",
        "include_aux": False,
        "feature_dim": 1536,
        "encoder": cfg.get("encoder"),
        "train_subset": "default train fold only (fair head_v3 retrain)",
        "primary_head": "head_v3_fair",
        "C": C,
        "train_cv_auc": cv_auc,
        "seed": args.seed,
        "target_recall": args.target_recall,
        "recipe": "casefilter retrain_v3 sample weights; no StandardScaler; train-only",
    }
    dump(bundle, out / "model_head_v3_fair.joblib")
    dump(bundle, args.artifacts / "model_head_v3_fair.joblib")

    # also export npz like original weights for inspection
    np.savez(
        out / "head_v3_fair.npz",
        coef=clf.coef_.astype(np.float64),
        intercept=clf.intercept_.astype(np.float64),
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "split": "GroupShuffleSplit seed=42, default subset (weak=False, label_source!=model)",
            "n_train": int(len(train_sub)),
            "n_val": int(len(val_sub)),
            "n_test": int(len(test_sub)),
            "C_selected": C,
            "train_cv_auc": cv_auc,
            "thr0.5": thr05,
            "thr_balanced": thr_bal,
            "thr_r95": thr95,
            "note": (
                "shipped_v3_*_UNFAIR rows use the exported head fit on all ~965 papers; "
                "fair_v3 was fit only on the train fold of this split."
            ),
        },
        "test": rows,
        "val": val_rows,
        "gold40": gold_rows,
        "shipped_config_caveat": (shipped_cfg or {}).get("caveat"),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.artifacts / "metrics_head_v3_fair.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # markdown summary
    def line(tag: str, m: dict[str, Any]) -> str:
        return (
            f"| {tag} | {m['threshold']:.3f} | {m['recall']:.3f} | {m['precision']:.3f} | "
            f"{m['f1']:.3f} | {m['ap']:.3f} | {m['roc_auc']:.3f} |"
        )

    md = [
        "# Fair head_v3 retrain",
        "",
        f"Train-only fit, C={C} (CV AUC={cv_auc:.3f}), n_train/val/test="
        f"{len(train_sub)}/{len(val_sub)}/{len(test_sub)}",
        "",
        "## Test set",
        "",
        "| system | thr | Recall | Precision | F1 | AP | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
        line("**fair head_v3 (R≥0.95)**", rows["fair_v3_thr_r95"]),
        line("fair head_v3 @0.5", rows["fair_v3_thr0.5"]),
        line("fair head_v3 @bal", rows["fair_v3_thr_bal"]),
        line("sklearn balanced+scaler (R≥0.95)", rows["sklearn_bal_thr_r95"]),
    ]
    if "shipped_v3_thr_r95_UNFAIR" in rows:
        md.append(line("shipped head_v3 R95 (**UNFAIR**)", rows["shipped_v3_thr_r95_UNFAIR"]))
        md.append(line("shipped head_v3 @0.5 (**UNFAIR**)", rows["shipped_v3_thr0.5_UNFAIR"]))
    if gold_rows:
        md += [
            "",
            "## Gold-40",
            "",
            "| system | Recall | Precision | F1 | AP |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, m in gold_rows.items():
            md.append(f"| {k} | {m['recall']:.3f} | {m['precision']:.3f} | {m['f1']:.3f} | {m['ap']:.3f} |")
    md += [
        "",
        "> Shipped rows are an upper bound only: that head was fit on nearly the full corpus.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")

    if args.promote:
        dump(bundle, args.artifacts / "model.joblib")
        primary = {
            **cfg,
            "model_path": str(args.artifacts / "model.joblib"),
            "primary_head": "head_v3_fair",
            "threshold": float(thr95),
            "review_lower_bound": max(0.0, float(thr95) - 0.15),
            "C": C,
            "train_subset": bundle["train_subset"],
            "fair_retrain": str(out / "metrics.json"),
        }
        (args.artifacts / "config.json").write_text(
            json.dumps(primary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Promoted fair head_v3 -> artifacts/model.joblib")

    print(json.dumps({"C": C, "thr_r95": thr95, "test": rows["fair_v3_thr_r95"], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
