#!/usr/bin/env python3
"""Evaluate shipped PubMedBERT head (weights.rar) vs SPECTER2 on this project's labels.

Uses pubmedbert/weights/{config.json,head_v3.npz} as-is (no head retrain).
Embeddings follow the exported config: NeuML/pubmedbert-base-embeddings,
max_chunks=12, pool = mean(all) || mean(first 2) → 1536-d.

Reports on:
1) the same default-subset GroupShuffleSplit test fold as SPECTER2 (seed=42)
2) the 40-paper human gold_standard.csv (corrected labels)

Also tunes a recall>=0.95 threshold on the SPECTER2 val fold for a protocol-matched
operating point (the shipped head defaults to 0.5).

Usage
-----
    .venv/bin/python scripts/compare_pubmedbert_weights.py --device cuda
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pubmedbert" / "src"))

from train import (  # noqa: E402
    binary_metrics,
    default_subset_mask,
    feature_matrix,
    load_dataset,
    make_splits,
    select_threshold,
)

logger = logging.getLogger("compare_pmb_weights")

DEFAULT_WEIGHTS = ROOT / "pubmedbert" / "weights"
DEFAULT_MODEL = Path(
    "/home/yangzs/.cache/huggingface/hub/models--NeuML--pubmedbert-base-embeddings"
    "/snapshots/b79526d6ef3645e0df4530322e266f24c829f5ef"
)
DEFAULT_DATA = ROOT / "data" / "labeled_papers.csv"
DEFAULT_GOLD = ROOT / "data" / "gold_standard.csv"
DEFAULT_OUT = ROOT / "artifacts" / "compare_pubmedbert"


def load_head(weights_dir: Path) -> tuple[dict[str, Any], np.ndarray, float]:
    cfg = json.loads((weights_dir / "config.json").read_text(encoding="utf-8"))
    z = np.load(weights_dir / cfg["head"]["file"])
    coef = z["coef"].astype(np.float64).ravel()
    intercept = float(z["intercept"].ravel()[0])
    return cfg, coef, intercept


def embed_corpus(
    df: pd.DataFrame,
    *,
    model_path: Path,
    cache_path: Path,
    max_chunks: int,
    device: str | None,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=False)
        if list(data["doc_ids"].astype(str)) == df["doc_id"].astype(str).tolist():
            logger.info("cache hit %s %s", cache_path, data["X"].shape)
            return data["X"]
        logger.warning("cache mismatch; re-encoding")

    from casefilter.embed import Embedder, pool

    texts = (
        df["title"].fillna("") + ". " + df["abstract"].fillna("") + "\n\n" + df["text"].fillna("")
    ).tolist()
    logger.info("Encoding %d docs with %s (max_chunks=%d)...", len(texts), model_path, max_chunks)
    t0 = time.time()
    emb = Embedder(str(model_path), device=device, batch_size=batch_size)
    X = pool(emb.embed_docs(texts, max_chunks=max_chunks), head=2).astype(np.float64)
    elapsed = time.time() - t0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, doc_ids=np.asarray(df["doc_id"].astype(str).tolist(), dtype="U"), X=X)
    cache_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": str(model_path),
                "max_chunks": max_chunks,
                "dim": int(X.shape[1]),
                "encoding_seconds": round(elapsed, 1),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("cached features in %.1fs -> %s", elapsed, cache_path)
    return X


def score_head(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    logits = X @ coef + intercept
    return 1.0 / (1.0 + np.exp(-logits))


def slim(m: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "n",
        "positives",
        "negatives",
        "threshold",
        "recall",
        "precision",
        "f1",
        "f2",
        "ap",
        "roc_auc",
        "confusion_matrix",
    ]
    return {k: m[k] for k in keys if k in m}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force-reencode", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg, coef, intercept = load_head(args.weights)
    max_chunks = int(cfg.get("max_chunks", 12))
    shipped_thr = float(cfg.get("threshold", 0.5))
    logger.info(
        "Loaded head %s dim=%d thr=%.3f C=%s",
        cfg.get("name"),
        coef.size,
        shipped_thr,
        cfg.get("head", {}).get("C"),
    )

    df = load_dataset(args.data)
    y = df["label"].to_numpy(dtype=np.int64)
    subset = default_subset_mask(df).to_numpy()
    train_idx, val_idx, test_idx = make_splits(df, seed=args.seed)
    train_sub, val_sub, test_sub = (
        train_idx[subset[train_idx]],
        val_idx[subset[val_idx]],
        test_idx[subset[test_idx]],
    )

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / f"pubmedbert_features_n{len(df)}_mc{max_chunks}.npz"
    X = embed_corpus(
        df,
        model_path=args.model,
        cache_path=cache,
        max_chunks=max_chunks,
        device=args.device,
        batch_size=args.batch_size,
        force=args.force_reencode,
    )
    if X.shape[1] != coef.size:
        raise SystemExit(f"feature dim {X.shape[1]} != coef dim {coef.size}")

    probs = score_head(X, coef, intercept)

    # Ship threshold 0.5
    pmb_test_05 = binary_metrics(y[test_sub], probs[test_sub], shipped_thr)
    pmb_val_05 = binary_metrics(y[val_sub], probs[val_sub], shipped_thr)

    # Protocol-matched threshold on val (our labels)
    thr95 = select_threshold(y[val_sub], probs[val_sub], 0.95)
    pmb_test_95 = binary_metrics(y[test_sub], probs[test_sub], thr95)
    pmb_val_95 = binary_metrics(y[val_sub], probs[val_sub], thr95)

    # SPECTER2 on identical indices (reload features + retrain head for fair split;
    # also load stored metrics)
    art = ROOT / "artifacts"
    s_cfg = json.loads((art / "config.json").read_text())
    s_data = np.load(s_cfg["feature_cache"], allow_pickle=False)
    if list(s_data["doc_ids"].astype(str)) != df["doc_id"].astype(str).tolist():
        raise SystemExit("SPECTER2 cache order mismatch")
    arrays = {k: s_data[k] for k in s_data.files}
    Xs = feature_matrix(arrays, "C")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, solver="lbfgs", random_state=0)
    clf.fit(scaler.fit_transform(Xs[train_sub]), y[train_sub])
    s_val = clf.predict_proba(scaler.transform(Xs[val_sub]))[:, 1]
    s_te = clf.predict_proba(scaler.transform(Xs[test_sub]))[:, 1]
    s_thr = select_threshold(y[val_sub], s_val, 0.95)
    specter_test = binary_metrics(y[test_sub], s_te, s_thr)
    specter_val = binary_metrics(y[val_sub], s_val, s_thr)

    # Gold set with corrected labels
    gold_block: dict[str, Any] = {}
    if args.gold.exists():
        gold = pd.read_csv(args.gold, dtype={"pmid": str})
        gold = gold[gold["gold_label"].astype(str).isin(["0", "1"])]
        gmap = {str(r.pmid): int(r.gold_label) for r in gold.itertuples()}
        ids = df["doc_id"].astype(str).tolist()
        g_idx = np.array([i for i, d in enumerate(ids) if d in gmap], dtype=int)
        y_g = np.array([gmap[ids[i]] for i in g_idx], dtype=int)
        gold_block = {
            "n": int(len(g_idx)),
            "pubmedbert_thr0.5": slim(binary_metrics(y_g, probs[g_idx], shipped_thr)),
            "pubmedbert_thr_valR95": slim(binary_metrics(y_g, probs[g_idx], thr95)),
            "specter2_thr_valR95": slim(binary_metrics(y_g, clf.predict_proba(scaler.transform(Xs[g_idx]))[:, 1], s_thr)),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pubmedbert_weights": {
            "dir": str(args.weights),
            "config": cfg,
            "note": "Head trained offline for UltraPhenoCards intake (Claude+curated); evaluated zero-shot on this repo's labels.",
        },
        "split": {
            "seed": args.seed,
            "train": int(len(train_sub)),
            "val": int(len(val_sub)),
            "test": int(len(test_sub)),
        },
        "test": {
            "pubmedbert_thr0.5": slim({**pmb_test_05, "threshold": shipped_thr}),
            "pubmedbert_thr_valR95": slim({**pmb_test_95, "threshold": float(thr95)}),
            "specter2_thr_valR95": slim({**specter_test, "threshold": float(s_thr)}),
        },
        "val": {
            "pubmedbert_thr0.5": slim({**pmb_val_05, "threshold": shipped_thr}),
            "pubmedbert_thr_valR95": slim({**pmb_val_95, "threshold": float(thr95)}),
            "specter2_thr_valR95": slim({**specter_val, "threshold": float(s_thr)}),
        },
        "gold": gold_block,
    }
    (args.out / "compare_weights.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# PubMedBERT 现成权重 vs SPECTER2",
        "",
        f"权重：`pubmedbert/weights`（{cfg.get('name')}，dim={coef.size}，默认阈值 {shipped_thr}）",
        f"划分：default subset train/val/test = {len(train_sub)}/{len(val_sub)}/{len(test_sub)}（与 SPECTER2 同 seed）",
        "",
        "## 测试集（本仓库标签）",
        "",
        "| 系统 | 阈值 | Recall | Precision | F1 | AP | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| PubMedBERT 现成 head | {shipped_thr:.3f} | {pmb_test_05['recall']:.3f} | {pmb_test_05['precision']:.3f} | {pmb_test_05['f1']:.3f} | {pmb_test_05['ap']:.3f} | {pmb_test_05['roc_auc']:.3f} |",
        f"| PubMedBERT 现成 head（val 上调到 R≥0.95） | {thr95:.3f} | {pmb_test_95['recall']:.3f} | {pmb_test_95['precision']:.3f} | {pmb_test_95['f1']:.3f} | {pmb_test_95['ap']:.3f} | {pmb_test_95['roc_auc']:.3f} |",
        f"| SPECTER2-C + LR（本仓库训练） | {s_thr:.3f} | {specter_test['recall']:.3f} | {specter_test['precision']:.3f} | {specter_test['f1']:.3f} | {specter_test['ap']:.3f} | {specter_test['roc_auc']:.3f} |",
        "",
        "> 注意：PubMedBERT head 的训练目标/标签来源与本仓库不完全相同；`thr=0.5` 是原任务默认点，`val R≥0.95` 是为对齐本仓库筛查协议。",
        "",
    ]
    if gold_block:
        md += [
            "## 金标准 40 篇（人工纠正标签）",
            "",
            "| 系统 | Recall | Precision | F1 | AP |",
            "|---|---:|---:|---:|---:|",
            f"| PubMedBERT thr=0.5 | {gold_block['pubmedbert_thr0.5']['recall']:.3f} | {gold_block['pubmedbert_thr0.5']['precision']:.3f} | {gold_block['pubmedbert_thr0.5']['f1']:.3f} | {gold_block['pubmedbert_thr0.5']['ap']:.3f} |",
            f"| PubMedBERT thr=R95 | {gold_block['pubmedbert_thr_valR95']['recall']:.3f} | {gold_block['pubmedbert_thr_valR95']['precision']:.3f} | {gold_block['pubmedbert_thr_valR95']['f1']:.3f} | {gold_block['pubmedbert_thr_valR95']['ap']:.3f} |",
            f"| SPECTER2-C | {gold_block['specter2_thr_valR95']['recall']:.3f} | {gold_block['specter2_thr_valR95']['precision']:.3f} | {gold_block['specter2_thr_valR95']['f1']:.3f} | {gold_block['specter2_thr_valR95']['ap']:.3f} |",
            "",
        ]
    (args.out / "compare_weights.md").write_text("\n".join(md), encoding="utf-8")
    print((args.out / "compare_weights.md").read_text())
    print("Wrote", args.out / "compare_weights.json")


if __name__ == "__main__":
    main()
