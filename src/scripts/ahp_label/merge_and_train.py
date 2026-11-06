#!/usr/bin/env python3
"""Merge AHP hard labels into training-only data and retrain fair PubMedBERT+LR.

Protocol
--------
- AHP wave PMIDs with label in {0,1} enter **train only** (never test).
- Evaluation uses the original seed-42 default-subset test fold from
  labeled_papers.csv (and gold-40 if present).
- Features: PubMedBERT 1536-d; AHP docs are encoded on demand and cached.
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

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from encoder import IdentityScaler, document_features_batch, load_encoder  # noqa: E402
from train import (  # noqa: E402
    binary_metrics,
    default_subset_mask,
    load_dataset,
    make_splits,
    select_threshold,
)

logger = logging.getLogger("ahp_train")

DEFAULT_LABELS = ROOT / "data" / "ahp_wave1" / "labels_ahp.csv"
DEFAULT_CORPUS = ROOT / "data" / "ahp_wave1" / "corpus.jsonl"
DEFAULT_DATA = ROOT / "data" / "labeled_papers.csv"
DEFAULT_GOLD = ROOT / "data" / "gold_standard.csv"
DEFAULT_OUT = ROOT / "artifacts" / "ahp_wave1"


def load_corpus_map(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[str(rec["pmid"])] = rec
    return out


def encode_ahp(
    rows: pd.DataFrame,
    corpus: dict[str, dict],
    cache_path: Path,
    *,
    device: str | None,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    pmids = rows["pmid"].astype(str).tolist()
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=False)
        if list(data["doc_ids"].astype(str)) == pmids and data["X"].shape[0] == len(pmids):
            logger.info("AHP feature cache hit %s", cache_path)
            return data["X"].astype(np.float64)

    records = []
    for pmid in pmids:
        rec = corpus.get(pmid, {})
        records.append(
            {
                "title": rec.get("title") or "",
                "abstract": "",
                "text": rec.get("text") or "",
            }
        )
    enc = load_encoder(device=device, batch_size=batch_size)
    t0 = time.time()
    feats, _ = document_features_batch(
        records,
        enc,
        include_fulltext=True,
        include_aux=False,
        max_chunks=12,
        batch_size=batch_size,
        show_progress=True,
    )
    X = feats.astype(np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, doc_ids=np.asarray(pmids, dtype="U"), X=X)
    logger.info("Encoded %d AHP docs in %.1fs -> %s", len(pmids), time.time() - t0, cache_path)
    return X


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-reencode", action="store_true")
    ap.add_argument(
        "--recipe",
        choices=["v3_weights", "sklearn_balanced"],
        default="v3_weights",
        help="v3_weights: no scaler + source-balanced (simplified: class_weight balanced on pooled)",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.labels.exists():
        raise SystemExit(f"missing labels: {args.labels} (run aggregate.py first)")

    labels = pd.read_csv(args.labels, dtype=str)
    hard = labels[labels["label"].isin(["0", "1"])].copy()
    hard["pmid"] = hard["pmid"].astype(str)
    hard["y"] = hard["label"].astype(int)
    logger.info("AHP hard labels: %d (pos=%d)", len(hard), int(hard["y"].sum()))

    # base corpus features from existing PubMedBERT cache
    cfg = json.loads((args.artifacts / "config.json").read_text())
    cache = Path(cfg["feature_cache"])
    base_df = load_dataset(args.data)
    data = np.load(cache, allow_pickle=False)
    if list(data["doc_ids"].astype(str)) != base_df["doc_id"].astype(str).tolist():
        raise SystemExit("base feature cache mismatch")
    X_base = np.concatenate([data["mean_all"], data["mean_head"]], axis=1).astype(np.float64)
    y_base = base_df["label"].to_numpy(dtype=int)
    subset = default_subset_mask(base_df).to_numpy()
    train_idx, val_idx, test_idx = make_splits(base_df, seed=args.seed)
    train_sub = train_idx[subset[train_idx]]
    val_sub = val_idx[subset[val_idx]]
    test_sub = test_idx[subset[test_idx]]

    # encode AHP docs
    corpus = load_corpus_map(args.corpus) if args.corpus.exists() else {}
    missing = [p for p in hard["pmid"] if p not in corpus]
    if missing:
        logger.warning("AHP labels without corpus text: %d (dropped)", len(missing))
        hard = hard[hard["pmid"].isin(corpus)].reset_index(drop=True)
    X_ahp = encode_ahp(
        hard,
        corpus,
        args.out / f"ahp_features_n{len(hard)}.npz",
        device=args.device,
        batch_size=args.batch_size,
        force=args.force_reencode,
    )
    y_ahp = hard["y"].to_numpy(dtype=int)

    # Train = original train_sub + all AHP hard; Val/Test unchanged from base split
    X_train = np.vstack([X_base[train_sub], X_ahp])
    y_train = np.concatenate([y_base[train_sub], y_ahp])

    if args.recipe == "sklearn_balanced":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xva = scaler.transform(X_base[val_sub])
        Xte = scaler.transform(X_base[test_sub])
        clf = LogisticRegression(
            C=args.C, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=0
        )
        clf.fit(Xtr, y_train)
    else:
        # fair head_v3 style: IdentityScaler, balanced classes approx via class_weight
        scaler = IdentityScaler().fit(X_train)
        clf = LogisticRegression(
            C=args.C, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=0
        )
        clf.fit(X_train, y_train)
        Xva = X_base[val_sub]
        Xte = X_base[test_sub]

    val_scores = clf.predict_proba(Xva)[:, 1]
    test_scores = clf.predict_proba(Xte)[:, 1]
    thr = select_threshold(y_base[val_sub], val_scores, args.target_recall)
    val_m = binary_metrics(y_base[val_sub], val_scores, thr)
    test_m = binary_metrics(y_base[test_sub], test_scores, thr)

    gold_m = None
    if args.gold.exists():
        gold = pd.read_csv(args.gold, dtype={"pmid": str})
        gold = gold[gold["gold_label"].astype(str).isin(["0", "1"])].copy()
        gold["gold_label"] = gold["gold_label"].astype(int)
        gmap = gold.drop_duplicates("pmid", keep="last").set_index("pmid")["gold_label"]
        ids = base_df["doc_id"].astype(str).tolist()
        g_idx = [i for i, d in enumerate(ids) if d in gmap.index]
        y_g = np.asarray([int(gmap.loc[ids[i]]) for i in g_idx], dtype=int)
        if args.recipe == "sklearn_balanced":
            sc_g = clf.predict_proba(scaler.transform(X_base[g_idx]))[:, 1]
        else:
            sc_g = clf.predict_proba(X_base[g_idx])[:, 1]
        gold_m = binary_metrics(y_g, sc_g, thr)

    enc = cfg.get("encoder") or {}
    bundle = {
        "model": clf,
        "scaler": scaler,
        "threshold": float(thr),
        "encoder_family": "pubmedbert",
        "feature_mode": "B",
        "include_aux": False,
        "feature_dim": 1536,
        "encoder": enc,
        "primary_head": "ahp_wave1_fair",
        "train_subset": f"base_train_sub({len(train_sub)})+ahp_hard({len(hard)})",
        "C": args.C,
        "target_recall": args.target_recall,
        "seed": args.seed,
        "recipe": args.recipe,
    }
    model_path = args.out / "model_ahp.joblib"
    dump(bundle, model_path)
    # also optional promote path left to user; write sidecar metrics
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_ahp_hard": int(len(hard)),
        "n_train_total": int(len(y_train)),
        "n_val": int(len(val_sub)),
        "n_test": int(len(test_sub)),
        "threshold": thr,
        "val": val_m,
        "test": test_m,
        "gold40": gold_m,
        "model": str(model_path),
        "notes": [
            "AHP labels used only in training.",
            "Test/val indices are the original labeled_papers seed-42 default subset folds.",
        ],
    }
    (args.out / "metrics_train.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # compare vs current primary if present
    baseline = None
    primary = args.artifacts / "model.joblib"
    if primary.exists():
        from joblib import load

        base = load(primary)
        # score test with primary
        try:
            if hasattr(base["scaler"], "transform"):
                Xt = base["scaler"].transform(X_base[test_sub])
            else:
                Xt = X_base[test_sub]
            sc = base["model"].predict_proba(Xt)[:, 1]
            thr_b = float(base["threshold"])
            baseline = binary_metrics(y_base[test_sub], sc, thr_b)
        except Exception as exc:  # noqa: BLE001
            baseline = {"error": str(exc)}
    metrics["baseline_primary_test"] = baseline
    (args.out / "metrics_train.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    md = [
        "# AHP wave1 retrain",
        "",
        f"- AHP hard train add-on: **{len(hard)}** (pos={int(y_ahp.sum())})",
        f"- threshold (val R≥{args.target_recall}): **{thr:.4f}**",
        "",
        "## Test (original seed-42 default subset)",
        "",
        f"- AHP-augmented: R={test_m['recall']:.3f} P={test_m['precision']:.3f} "
        f"F1={test_m['f1']:.3f} AP={test_m['ap']:.3f}",
    ]
    if baseline and "recall" in baseline:
        md.append(
            f"- Previous primary: R={baseline['recall']:.3f} P={baseline['precision']:.3f} "
            f"F1={baseline['f1']:.3f} AP={baseline['ap']:.3f}"
        )
    if gold_m:
        md += [
            "",
            "## Gold-40",
            f"- R={gold_m['recall']:.3f} P={gold_m['precision']:.3f} F1={gold_m['f1']:.3f} AP={gold_m['ap']:.3f}",
        ]
    (args.out / "train_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(model_path), "test": test_m, "gold40": gold_m}, indent=2))


if __name__ == "__main__":
    main()
