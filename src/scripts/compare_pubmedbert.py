#!/usr/bin/env python3
"""Compare PubMedBERT (NeuML embeddings) vs SPECTER2 baseline on the same split.

Fair protocol (matches train.py defaults):
- data: data/labeled_papers.csv
- subset: weak=False and label_source != model
- GroupShuffleSplit seed=42 ≈ 64/16/20
- LogisticRegression(C=1, class_weight=balanced)
- threshold: max precision at val recall >= 0.95

PubMedBERT features (from pubmedbert/src/casefilter/embed.py):
  document vector = [mean_all_chunks || mean_first_2_chunks]  → 1536-d
  model: local NeuML/pubmedbert-base-embeddings snapshot

Also reports the frozen SPECTER2 numbers from artifacts/metrics.json and
evaluates both heads on the 40-paper gold_standard set when present.

Usage
-----
    .venv/bin/python src/scripts/compare_pubmedbert.py
    .venv/bin/python src/scripts/compare_pubmedbert.py --device cuda --batch-size 32
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
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
LEGACY_ROOT = ROOT / "pubmedbert"
if not LEGACY_ROOT.exists():
    sys.exit(
        "pubmedbert/ (legacy casefilter subproject) was removed; keeping only the DoRA\n"
        "final generation. Restore it with:\n"
        "    git checkout archive/pre-dora-legacy -- pubmedbert\n"
    )
sys.path.insert(0, str(ROOT / "pubmedbert" / "src"))

from train import (  # noqa: E402
    binary_metrics,
    default_subset_mask,
    feature_matrix,
    load_dataset,
    make_splits,
    select_threshold,
)

logger = logging.getLogger("compare_pmb")

DEFAULT_MODEL = Path(
    "/home/yangzs/.cache/huggingface/hub/models--NeuML--pubmedbert-base-embeddings"
    "/snapshots/b79526d6ef3645e0df4530322e266f24c829f5ef"
)
DEFAULT_DATA = ROOT / "data" / "labeled_papers.csv"
DEFAULT_GOLD = ROOT / "data" / "gold_standard.csv"
DEFAULT_OUT = ROOT / "artifacts" / "compare_pubmedbert"


def build_pubmedbert_features(
    df: pd.DataFrame,
    *,
    model_path: Path,
    cache_path: Path,
    device: str | None,
    batch_size: int,
    max_chunks: int,
    force: bool,
) -> np.ndarray:
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=False)
        ids = list(data["doc_ids"].astype(str))
        if ids == df["doc_id"].astype(str).tolist():
            logger.info("PubMedBERT feature cache hit: %s shape=%s", cache_path, data["X"].shape)
            return data["X"]
        logger.warning("Cache doc_ids mismatch; re-encoding")

    from casefilter.embed import Embedder, pool

    texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("") + "\n\n" + df["text"].fillna("")).tolist()
    logger.info("Encoding %d docs with PubMedBERT (%s)...", len(texts), model_path)
    t0 = time.time()
    emb = Embedder(str(model_path), device=device, batch_size=batch_size)
    chunks = emb.embed_docs(texts, max_chunks=max_chunks)
    X = pool(chunks, head=2).astype(np.float32)
    elapsed = time.time() - t0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        doc_ids=np.asarray(df["doc_id"].astype(str).tolist(), dtype="U"),
        X=X,
    )
    meta = {
        "model": str(model_path),
        "n_docs": len(df),
        "dim": int(X.shape[1]),
        "max_chunks": max_chunks,
        "pooling": "mean_all || mean_first_2",
        "encoding_seconds": round(elapsed, 1),
        "docs_per_second": round(len(df) / max(elapsed, 1e-9), 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    cache_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Cached PubMedBERT features -> %s (%.1fs)", cache_path, elapsed)
    return X


def train_eval(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
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
    Xtr = scaler.fit_transform(X[train_idx])
    Xva = scaler.transform(X[val_idx])
    Xte = scaler.transform(X[test_idx])
    clf.fit(Xtr, y[train_idx])
    val_scores = clf.predict_proba(Xva)[:, 1]
    test_scores = clf.predict_proba(Xte)[:, 1]
    thr = select_threshold(y[val_idx], val_scores, target_recall)
    return {
        "threshold": float(thr),
        "val": binary_metrics(y[val_idx], val_scores, thr),
        "test": binary_metrics(y[test_idx], test_scores, thr),
        "model": clf,
        "scaler": scaler,
        "test_scores": test_scores,
        "test_idx": test_idx,
    }


def eval_on_pmids(
    fit: dict[str, Any],
    X: np.ndarray,
    df: pd.DataFrame,
    pmids: set[str],
) -> dict[str, Any] | None:
    ids = df["doc_id"].astype(str).tolist()
    idx = np.array([i for i, d in enumerate(ids) if d in pmids], dtype=int)
    if idx.size == 0:
        return None
    scores = fit["model"].predict_proba(fit["scaler"].transform(X[idx]))[:, 1]
    y = df.iloc[idx]["label"].to_numpy(dtype=int)
    # If gold overrides exist in a separate file, caller should pass corrected y via df
    return binary_metrics(y, scores, fit["threshold"])


def load_specter2_baseline(artifacts: Path, df: pd.DataFrame, train_idx, val_idx, test_idx, y) -> dict[str, Any]:
    """Reload SPECTER2 cached features and retrain on the *same* indices for a true apples-to-apples split."""
    cfg = json.loads((artifacts / "config.json").read_text())
    cache = Path(cfg["feature_cache"])
    data = np.load(cache, allow_pickle=False)
    if list(data["doc_ids"].astype(str)) != df["doc_id"].astype(str).tolist():
        raise RuntimeError("SPECTER2 cache doc_ids mismatch")
    arrays = {k: data[k] for k in data.files}
    X = feature_matrix(arrays, "C")
    fit = train_eval(X, y, df, train_idx, val_idx, test_idx, target_recall=0.95, C=1.0)
    # also attach stored metrics for reference
    stored = json.loads((artifacts / "metrics.json").read_text())
    fit["stored_primary_test"] = stored["primary"]["test"]
    fit["feature_dim"] = int(X.shape[1])
    fit["X"] = X
    return fit


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-chunks", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--force-reencode", action="store_true")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"PubMedBERT model not found: {args.model}")

    df = load_dataset(args.data)
    mask = default_subset_mask(df).to_numpy()
    # Work on full df for feature alignment; split indices on default subset only
    train_idx, val_idx, test_idx = make_splits(df, seed=args.seed)
    # Restrict to default subset: drop weak/model from each split
    def keep(idx: np.ndarray) -> np.ndarray:
        return idx[mask[idx]]

    train_idx, val_idx, test_idx = keep(train_idx), keep(val_idx), keep(test_idx)
    for name, idx in ("train", train_idx), ("val", val_idx), ("test", test_idx):
        if len(np.unique(df.iloc[idx]["label"])) < 2:
            raise SystemExit(f"{name} split lost a class after default-subset filter")

    y = df["label"].to_numpy(dtype=np.int64)
    logger.info(
        "Split (default subset) train/val/test=%d/%d/%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    pmb_cache = args.out / f"pubmedbert_features_n{len(df)}_mc{args.max_chunks}.npz"
    X_pmb = build_pubmedbert_features(
        df,
        model_path=args.model,
        cache_path=pmb_cache,
        device=args.device,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
        force=args.force_reencode,
    )

    pmb = train_eval(
        X_pmb, y, df, train_idx, val_idx, test_idx,
        target_recall=args.target_recall, C=args.C,
    )
    dump(
        {
            "model": pmb["model"],
            "scaler": pmb["scaler"],
            "threshold": pmb["threshold"],
            "feature_mode": "pubmedbert_mean_all||mean_first2",
            "feature_dim": int(X_pmb.shape[1]),
            "encoder": {"model": str(args.model), "max_chunks": args.max_chunks},
        },
        args.out / "model_pubmedbert.joblib",
    )

    specter = load_specter2_baseline(ROOT / "artifacts", df, train_idx, val_idx, test_idx, y)

    # Gold evaluation with gold labels applied
    gold_metrics = {}
    if args.gold.exists():
        gold = pd.read_csv(args.gold, dtype={"pmid": str})
        gold = gold[gold["gold_label"].astype(str).isin(["0", "1"])]
        gmap = {str(r.pmid): int(r.gold_label) for r in gold.itertuples()}
        df_g = df.copy()
        for i, doc_id in enumerate(df_g["doc_id"].astype(str)):
            if doc_id in gmap:
                df_g.at[df_g.index[i], "label"] = gmap[doc_id]
        gold_pmids = set(gmap)
        gold_metrics["pubmedbert"] = eval_on_pmids(pmb, X_pmb, df_g, gold_pmids)
        gold_metrics["specter2"] = eval_on_pmids(specter, specter["X"], df_g, gold_pmids)

    def slim(block: dict[str, Any]) -> dict[str, Any]:
        keys = ["n", "positives", "negatives", "threshold", "recall", "precision", "f1", "f2", "ap", "roc_auc", "confusion_matrix"]
        return {k: block[k] for k in keys if k in block}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "subset": "weak=False & label_source!=model",
            "split": "GroupShuffleSplit seed=42 then filter to subset",
            "sizes": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "classifier": "StandardScaler + LogisticRegression(C=1, class_weight=balanced)",
            "threshold": "max precision s.t. val recall >= 0.95",
        },
        "pubmedbert": {
            "model": str(args.model),
            "feature_dim": int(X_pmb.shape[1]),
            "pooling": "mean_all_chunks || mean_first_2_chunks",
            "max_chunks": args.max_chunks,
            "threshold": pmb["threshold"],
            "val": slim(pmb["val"]),
            "test": slim(pmb["test"]),
        },
        "specter2": {
            "feature_mode": "C",
            "feature_dim": specter["feature_dim"],
            "threshold": specter["threshold"],
            "val": slim(specter["val"]),
            "test": slim(specter["test"]),
            "stored_primary_test_note": "retrained on identical indices; stored metrics may differ slightly from original run due to subset-filter-after-split",
            "stored_primary_test": slim(specter["stored_primary_test"]),
        },
        "gold_eval_n40": gold_metrics,
        "delta_test": {
            "recall_pmb_minus_specter2": pmb["test"]["recall"] - specter["test"]["recall"],
            "precision_pmb_minus_specter2": pmb["test"]["precision"] - specter["test"]["precision"],
            "ap_pmb_minus_specter2": pmb["test"]["ap"] - specter["test"]["ap"],
            "f1_pmb_minus_specter2": pmb["test"]["f1"] - specter["test"]["f1"],
        },
    }

    (args.out / "compare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown table
    lines = [
        "# PubMedBERT vs SPECTER2",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Split sizes (default subset): train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}",
        "",
        "## Test set",
        "",
        "| Model | Dim | Thr | Recall | Precision | F1 | F2 | AP | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| PubMedBERT | {X_pmb.shape[1]} | {pmb['threshold']:.4f} | "
            f"{pmb['test']['recall']:.3f} | {pmb['test']['precision']:.3f} | "
            f"{pmb['test']['f1']:.3f} | {pmb['test']['f2']:.3f} | "
            f"{pmb['test']['ap']:.3f} | {pmb['test']['roc_auc']:.3f} |"
        ),
        (
            f"| SPECTER2-C | {specter['feature_dim']} | {specter['threshold']:.4f} | "
            f"{specter['test']['recall']:.3f} | {specter['test']['precision']:.3f} | "
            f"{specter['test']['f1']:.3f} | {specter['test']['f2']:.3f} | "
            f"{specter['test']['ap']:.3f} | {specter['test']['roc_auc']:.3f} |"
        ),
        "",
        "## Validation (threshold selection)",
        "",
        f"- PubMedBERT val R={pmb['val']['recall']:.3f} P={pmb['val']['precision']:.3f}",
        f"- SPECTER2-C val R={specter['val']['recall']:.3f} P={specter['val']['precision']:.3f}",
        "",
    ]
    if gold_metrics:
        lines += ["## Gold set (n=40, corrected labels)", ""]
        for name, block in gold_metrics.items():
            if not block:
                continue
            lines.append(
                f"- **{name}**: R={block['recall']:.3f} P={block['precision']:.3f} "
                f"F1={block['f1']:.3f} AP={block['ap']:.3f} "
                f"cm={block['confusion_matrix']}"
            )
        lines.append("")
    (args.out / "compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print((args.out / "compare.md").read_text())
    print(f"Wrote {args.out / 'compare.json'}")


if __name__ == "__main__":
    main()
