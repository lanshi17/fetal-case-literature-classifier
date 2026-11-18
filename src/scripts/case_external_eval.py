#!/usr/bin/env python3
"""External-validity splits for the case-only DoRA classifier.

Two protocols, both reusing the identical training recipe
(`finetune_case_dora.train_model`):

temporal   train on papers published <= --cutoff-year, test on the rest.
journal    5-fold GroupKFold on journal (all papers of a journal in one fold).

Outputs (``artifacts/case_dora_external``):
    temporal/fold_0.json, temporal/oof.csv
    journal_cv/fold_{i}.json, journal_cv/oof.csv
    SUMMARY.md, summary.json

Usage
-----
    .venv/bin/python src/scripts/case_external_eval.py --protocol temporal
    .venv/bin/python src/scripts/case_external_eval.py --protocol journal --folds 0,1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from finetune_case_dora import (  # noqa: E402
    DEFAULT_DATA,
    evaluate_scores,
    load_examples,
    threshold_max_f1,
    threshold_target_precision,
    train_model,
)
from finetune_chunk_fair import predict_proba, resolve_model_path  # noqa: E402

DEFAULT_OUT = ROOT / "artifacts" / "case_dora_external"
DEFAULT_RESULTS = ROOT / "artifacts" / "screen_full" / "results.csv"


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        dora_r=args.dora_r,
        dora_alpha=args.dora_alpha,
        target_precision=args.target_precision,
        val_frac=args.val_frac,
        input_mode="full",
    )


def inner_val(train_idx: np.ndarray, y: np.ndarray, fold: int, frac: float) -> tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=42 + fold)
    rel_tr, rel_va = next(sss.split(np.zeros(len(train_idx)), y[train_idx]))
    return train_idx[rel_tr], train_idx[rel_va]


def run_split(
    name: str,
    fold: int,
    examples: list,
    y: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    tokenizer: Any,
    model_path: str,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    meta: pd.DataFrame,
) -> dict[str, Any]:
    fold_path = out_dir / name / f"fold_{fold}.json"
    if fold_path.exists():
        print(f"{name} fold {fold}: exists, skipping", flush=True)
        return json.loads(fold_path.read_text(encoding="utf-8"))

    tr_rel, va_idx = inner_val(tr_idx, y, fold, args.val_frac)
    tr = [examples[int(i)] for i in tr_rel]
    va = [examples[int(i)] for i in va_idx]
    te = [examples[int(i)] for i in te_idx]
    print(
        f"{name} fold {fold}: train={len(tr)} val={len(va)} test={len(te)} "
        f"(pos {int(y[tr_rel].sum())}/{int(y[te_idx].sum())})",
        flush=True,
    )

    t0 = time.time()
    model, info = train_model(tr, va, tokenizer, model_path, args, device, seed=42 + fold)
    yv = y[va_idx]
    yt = y[te_idx]
    va_scores = predict_proba(model, va, tokenizer, max_length=args.max_length, batch_size=args.eval_batch_size, device=device)
    te_scores = predict_proba(model, te, tokenizer, max_length=args.max_length, batch_size=args.eval_batch_size, device=device)
    thr_f1 = threshold_max_f1(yv, va_scores)
    thr_p90 = threshold_target_precision(yv, va_scores, args.target_precision)

    result = {
        "protocol": name,
        "fold": fold,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "pos_test": int(yt.sum()),
        "best_epoch": info["best_epoch"],
        "threshold_f1": thr_f1,
        "threshold_p90": thr_p90,
        "test_ap": float(average_precision_score(yt, te_scores)) if len(np.unique(yt)) > 1 else float("nan"),
        "test_roc_auc": float(roc_auc_score(yt, te_scores)) if len(np.unique(yt)) > 1 else float("nan"),
        "test_at_f1_thr": evaluate_scores(yt, te_scores, thr_f1),
        "test_at_shared_thr": evaluate_scores(yt, te_scores, 0.39624494),
        "elapsed_sec": round(time.time() - t0, 1),
        "history": info["history"],
    }
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    fold_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    oof = pd.DataFrame(
        {
            "pmid": [e.pmid for e in te],
            "y": yt,
            "score": te_scores,
            "fold": fold,
            "protocol": name,
            "year": meta.iloc[te_idx]["year"].to_numpy(),
            "journal": meta.iloc[te_idx]["journal"].to_numpy(),
        }
    )
    oof_path = out_dir / name / "oof.csv"
    oof.to_csv(oof_path, mode="a", header=not oof_path.exists(), index=False)
    print(
        f"{name} fold {fold}: AP={result['test_ap']:.4f} AUC={result['test_roc_auc']:.4f} "
        f"F1={result['test_at_f1_thr']['f1']:.4f} ({result['elapsed_sec']/60:.1f} min)",
        flush=True,
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", choices=["temporal", "journal"], required=True)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cutoff-year", type=int, default=2019, help="temporal: train <= cutoff, test > cutoff")
    ap.add_argument("--folds", type=str, default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=480)
    ap.add_argument("--dora-r", type=int, default=16)
    ap.add_argument("--dora-alpha", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--target-precision", type=float, default=0.90)
    args = ap.parse_args()
    args.out_dir = args.out_dir.expanduser().resolve()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    examples, sub, stats = load_examples(tokenizer, args.data, "full")
    y = np.asarray([e.label for e in examples], dtype=int)
    print(f"loaded {len(examples)} rows: {stats}", flush=True)

    res = pd.read_csv(args.results, dtype={"pmid": str}, low_memory=False)[["pmid", "journal"]].drop_duplicates("pmid")
    meta = sub[["pmid", "year"]].copy()
    meta["pmid"] = meta["pmid"].astype(str)
    meta = meta.merge(res, on="pmid", how="left")
    missing = int(meta["journal"].isna().sum())
    if missing:
        print(f"warning: {missing} rows without journal; bucketed as (missing)", flush=True)
        meta["journal"] = meta["journal"].fillna("(missing)")
    meta["year"] = pd.to_numeric(meta["year"], errors="coerce").fillna(0)

    train_args = build_args(args)
    idx = np.arange(len(examples))
    if args.protocol == "temporal":
        tr_idx = idx[meta["year"].to_numpy() <= args.cutoff_year]
        te_idx = idx[meta["year"].to_numpy() > args.cutoff_year]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            raise SystemExit("empty temporal split")
        run_split("temporal", 0, examples, y, tr_idx, te_idx, tokenizer, model_path, train_args, device, args.out_dir, meta)
    else:
        groups = meta["journal"].to_numpy()
        gkf = GroupKFold(n_splits=args.n_folds)
        splits = list(gkf.split(idx.reshape(-1, 1), y, groups))
        wanted = [int(x) for x in args.folds.split(",")] if args.folds else list(range(len(splits)))
        for fold in wanted:
            tr_idx, te_idx = splits[fold]
            run_split("journal_cv", fold, examples, y, tr_idx, te_idx, tokenizer, model_path, train_args, device, args.out_dir, meta)


if __name__ == "__main__":
    main()
