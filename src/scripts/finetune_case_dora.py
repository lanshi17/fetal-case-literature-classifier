#!/usr/bin/env python3
"""Fine-tune a case-only (case report / case series) DoRA classifier.

Target
------
Binary design label derived from ``data/labeled_papers.csv``:
    positive: design matches ``^case (report|series)``
    negative: design matches cohort|screening|review|method|validation|registry
Rows with missing / "other" design are excluded.

Architecture matches the relevance model (fair comparison):
    NeuML/pubmedbert-base-embeddings + DoRA(r=16, a=32, query/value)
    chunk 480 tokens, max 12 chunks, keep first 2 -> 1536-d pooled feature
    linear head on 1536-d

Evaluation
----------
5-fold stratified CV. Each fold holds out 12% of its training part for early
stopping and threshold selection; the held-out fold is untouched until scoring.
Out-of-fold scores are saved so thresholds can be derived globally.

Modes
-----
    # one fold (resumable; skips if fold json exists)
    .venv/bin/python src/scripts/finetune_case_dora.py --folds 0
    # several folds
    .venv/bin/python src/scripts/finetune_case_dora.py --folds 0,1
    # train final model on all labeled rows after CV
    .venv/bin/python src/scripts/finetune_case_dora.py --final

Outputs (``artifacts/case_dora``):
    cv/fold_{i}.json        per-fold metrics
    cv/oof.csv              out-of-fold pmid / y / score
    checkpoint/model.pt     final model state (+ tokenizer files)
    config.json             encoder, adapter, threshold, CV summary
    report.md               human-readable results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from finetune_chunk_fair import (  # noqa: E402
    ChunkDataset,
    ChunkPoolClassifier,
    Example,
    class_weights,
    collate_examples,
    doc_to_chunks,
    predict_proba,
    resolve_model_path,
    wrap_dora,
)
from train import binary_metrics  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "labeled_papers.csv"
DEFAULT_OUT = ROOT / "artifacts" / "case_dora"
CASE_RE = r"^case\s+(report|series)\b"
NEG_PATTERN = r"cohort|screening|review|method|validation|registry"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_examples(
    tokenizer: Any, data_path: Path, input_mode: str = "full"
) -> tuple[list[Example], pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(data_path, low_memory=False)
    design = df["design"].fillna("").astype(str).str.strip().str.lower()
    pos = design.str.match(CASE_RE)
    neg = design.str.contains(NEG_PATTERN, regex=True) & ~pos
    mask = pos | neg
    sub = df[mask].reset_index(drop=True)
    y = pos[mask].astype(int).to_numpy()
    examples: list[Example] = []
    for (_, row), label in zip(sub.iterrows(), y):
        title = str(row.get("title") or "")
        abstract = str(row.get("abstract") or "")
        text = str(row.get("text") or "")
        if input_mode == "title":
            abstract, text = "", ""
        elif input_mode == "title_abstract":
            text = ""
        chunks = doc_to_chunks(title, abstract, text, tokenizer)
        examples.append(Example(str(row["pmid"]), chunks, int(label), "design"))
    stats = {
        "pos": int(y.sum()),
        "neg": int((1 - y).sum()),
        "excluded_missing_or_other": int((~mask).sum()),
        "input_mode": input_mode,
    }
    return examples, sub, stats


# ---------------------------------------------------------------------------
# Thresholds / evaluation
# ---------------------------------------------------------------------------
def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    uniq = np.unique(scores)
    if uniq.size == 0:
        return np.array([0.5])
    return np.unique(np.concatenate(([0.0], (uniq[:-1] + uniq[1:]) / 2.0, [1.0])))


def threshold_max_f1(y: np.ndarray, scores: np.ndarray) -> float:
    best_t, best = 0.5, (-1.0, -1.0, -1.0)
    for t in _threshold_candidates(scores):
        pred = scores >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if (f1, p, r) > best:
            best, best_t = (f1, p, r), float(t)
    return best_t


def threshold_target_precision(y: np.ndarray, scores: np.ndarray, target: float = 0.90) -> float:
    best_t, best_recall = None, -1.0
    for t in _threshold_candidates(scores):
        pred = scores >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        if p + 1e-12 >= target and r > best_recall:
            best_t, best_recall = float(t), r
    return best_t if best_t is not None else threshold_max_f1(y, scores)


def evaluate_scores(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    m = binary_metrics(y, scores, threshold)
    if len(np.unique(y)) > 1:
        m["ap"] = float(average_precision_score(y, scores))
        m["roc_auc"] = float(roc_auc_score(y, scores))
    return m


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    train: list[Example],
    val: list[Example] | None,
    tokenizer: Any,
    model_path: str,
    args: argparse.Namespace,
    device: torch.device,
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ChunkPoolClassifier(model_path)
    model = wrap_dora(model, args.dora_r, args.dora_alpha)
    model.to(device)

    weights = class_weights(train, device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        ChunkDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_examples(b, tokenizer, args.max_length),
    )
    steps_per_epoch = max(1, math.ceil(len(loader) / max(args.grad_accum, 1)))
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(total_steps * args.warmup_ratio), total_steps
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_ap, best_epoch, best_state = -1.0, -1, None
    patience = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            labels = batch.pop("labels").to(device)
            n = batch.pop("batch_size")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    owners=batch["owners"],
                    batch_size=n,
                )["logits"]
                loss = F.cross_entropy(logits, labels, weight=weights) / max(args.grad_accum, 1)
            scaler.scale(loss).backward()
            running += float(loss.detach()) * max(args.grad_accum, 1) * labels.size(0)
            n_seen += int(labels.size(0))
            if step % args.grad_accum == 0 or step == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

        entry: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running / max(n_seen, 1),
        }
        if val:
            scores = predict_proba(
                model,
                val,
                tokenizer,
                max_length=args.max_length,
                batch_size=args.eval_batch_size,
                device=device,
            )
            yv = np.asarray([e.label for e in val])
            ap = float(average_precision_score(yv, scores))
            f1 = float(evaluate_scores(yv, scores, threshold_max_f1(yv, scores))["f1"])
            entry.update({"val_ap": ap, "val_f1_best": f1})
            if ap > best_ap + 1e-6:
                best_ap, best_epoch = ap, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
        history.append(entry)
        print(
            f"  epoch {epoch}/{args.epochs} loss={entry['train_loss']:.4f}"
            + (f" val_ap={entry['val_ap']:.4f} val_f1={entry['val_f1_best']:.4f}" if val else ""),
            flush=True,
        )
        if val and patience >= args.patience:
            print(f"  early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_ap": best_ap, "history": history}


# ---------------------------------------------------------------------------
# CV / final
# ---------------------------------------------------------------------------
def run_fold(
    fold: int,
    examples: list[Example],
    y: np.ndarray,
    tokenizer: Any,
    model_path: str,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    fold_path = out_dir / "cv" / f"fold_{fold}.json"
    if fold_path.exists():
        print(f"fold {fold}: already done, skipping", flush=True)
        return json.loads(fold_path.read_text())

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    tr_idx, te_idx = splits[fold]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed + fold)
    tr_rel, va_rel = next(sss.split(np.zeros(len(tr_idx)), y[tr_idx]))
    tr = [examples[int(tr_idx[i])] for i in tr_rel]
    va = [examples[int(tr_idx[i])] for i in va_rel]
    te = [examples[int(i)] for i in te_idx]
    print(
        f"fold {fold}: train={len(tr)} val={len(va)} test={len(te)} "
        f"(pos {int(y[tr_idx].sum())}/{int(y[te_idx].sum())})",
        flush=True,
    )

    t0 = time.time()
    model, info = train_model(
        tr, va, tokenizer, model_path, args, device, seed=args.seed + fold
    )
    yv = np.asarray([e.label for e in va])
    yt = np.asarray([e.label for e in te])
    va_scores = predict_proba(
        model, va, tokenizer, max_length=args.max_length,
        batch_size=args.eval_batch_size, device=device,
    )
    te_scores = predict_proba(
        model, te, tokenizer, max_length=args.max_length,
        batch_size=args.eval_batch_size, device=device,
    )
    thr_f1 = threshold_max_f1(yv, va_scores)
    thr_p90 = threshold_target_precision(yv, va_scores, args.target_precision)

    result = {
        "fold": fold,
        "input_mode": args.input_mode,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "pos_train": int(sum(e.label for e in tr)),
        "pos_val": int(yv.sum()),
        "pos_test": int(yt.sum()),
        "best_epoch": info["best_epoch"],
        "best_val_ap": info["best_val_ap"],
        "threshold_f1": thr_f1,
        "threshold_p90": thr_p90,
        "val_ap": float(average_precision_score(yv, va_scores)),
        "test_at_f1_thr": evaluate_scores(yt, te_scores, thr_f1),
        "test_at_p90_thr": evaluate_scores(yt, te_scores, thr_p90),
        "test_ap": float(average_precision_score(yt, te_scores)),
        "test_roc_auc": float(roc_auc_score(yt, te_scores)),
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
        }
    )
    oof_path = out_dir / "cv" / "oof.csv"
    oof.to_csv(oof_path, mode="a", header=not oof_path.exists(), index=False)
    print(
        f"fold {fold}: test AP={result['test_ap']:.4f} AUC={result['test_roc_auc']:.4f} "
        f"F1={result['test_at_f1_thr']['f1']:.4f} "
        f"P={result['test_at_f1_thr']['precision']:.4f} R={result['test_at_f1_thr']['recall']:.4f}"
        f" ({result['elapsed_sec']/60:.1f} min)",
        flush=True,
    )
    return result


def train_final(
    examples: list[Example],
    tokenizer: Any,
    model_path: str,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    fold_files = sorted((out_dir / "cv").glob("fold_*.json"))
    if not fold_files:
        raise SystemExit("run CV folds before --final")
    folds = [json.loads(p.read_text()) for p in fold_files]
    epochs = int(round(np.median([f["best_epoch"] for f in folds])))
    epochs = max(1, min(args.epochs, epochs))
    print(f"final: training on {len(examples)} rows for {epochs} epochs", flush=True)
    t0 = time.time()
    model, info = train_model(
        examples, None, tokenizer, model_path, args, device, seed=args.seed
    )
    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "model_path": model_path, "method": "dora_case"},
        ckpt / "model.pt",
    )
    tokenizer.save_pretrained(ckpt)

    oof = pd.read_csv(out_dir / "cv" / "oof.csv")
    thr_f1 = threshold_max_f1(oof["y"].to_numpy(), oof["score"].to_numpy())
    thr_p90 = threshold_target_precision(
        oof["y"].to_numpy(), oof["score"].to_numpy(), args.target_precision
    )
    metrics = {
        "oof_n": int(len(oof)),
        "oof_ap": float(average_precision_score(oof["y"], oof["score"])),
        "oof_roc_auc": float(roc_auc_score(oof["y"], oof["score"])),
        "oof_at_f1_thr": evaluate_scores(
            oof["y"].to_numpy(), oof["score"].to_numpy(), thr_f1
        ),
        "oof_at_p90_thr": evaluate_scores(
            oof["y"].to_numpy(), oof["score"].to_numpy(), thr_p90
        ),
        "fold_test_ap_mean": float(np.mean([f["test_ap"] for f in folds])),
        "fold_test_ap_std": float(np.std([f["test_ap"] for f in folds])),
        "fold_test_f1_mean": float(
            np.mean([f["test_at_f1_thr"]["f1"] for f in folds])
        ),
        "fold_test_f1_std": float(np.std([f["test_at_f1_thr"]["f1"] for f in folds])),
        "fold_test_precision_mean": float(
            np.mean([f["test_at_f1_thr"]["precision"] for f in folds])
        ),
        "fold_test_recall_mean": float(
            np.mean([f["test_at_f1_thr"]["recall"] for f in folds])
        ),
    }
    config = {
        "name": "case_dora_chunk",
        "description": "case report / case series vs cohort/screening/review/method, DoRA chunk-pool",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "encoder": {
            "model_id": "NeuML/pubmedbert-base-embeddings",
            "model_path": model_path,
            "chunk_tokens": args.max_length,
            "max_chunks": 12,
            "keep_first": 2,
            "feature": "L2(mean_all) || L2(mean_first_2) -> 1536-d",
        },
        "adapter": {
            "type": "DoRA",
            "r": args.dora_r,
            "alpha": args.dora_alpha,
            "target_modules": ["query", "value"],
        },
        "threshold": thr_f1,
        "threshold_p90": thr_p90,
        "target_precision": args.target_precision,
        "final_epochs": epochs,
        "cv_metrics": metrics,
        "paths": {"checkpoint_dir": str(ckpt.relative_to(ROOT))},
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Case-only DoRA (case report / case series)",
        "",
        f"- train: {len(examples)} rows (case {sum(e.label for e in examples)}, "
        f"non-case {sum(1 for e in examples if not e.label)})",
        f"- final epochs: {epochs}",
        f"- OOF AP **{metrics['oof_ap']:.3f}**, AUC **{metrics['oof_roc_auc']:.3f}**",
        f"- 5-fold test AP {metrics['fold_test_ap_mean']:.3f} ± {metrics['fold_test_ap_std']:.3f}",
        f"- 5-fold test F1 {metrics['fold_test_f1_mean']:.3f} ± {metrics['fold_test_f1_std']:.3f}",
        f"- threshold (F1) {thr_f1:.4f}: "
        f"P={metrics['oof_at_f1_thr']['precision']:.3f} "
        f"R={metrics['oof_at_f1_thr']['recall']:.3f}",
        f"- threshold (P≥{args.target_precision:.2f}) {thr_p90:.4f}: "
        f"P={metrics['oof_at_p90_thr']['precision']:.3f} "
        f"R={metrics['oof_at_p90_thr']['recall']:.3f}",
        f"- elapsed {round(time.time()-t0)/60:.1f} min",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(config["cv_metrics"], ensure_ascii=False, indent=2), flush=True)

    outdated = {k: v for k, v in info.items() if k != "history"}
    (out_dir / "final_metrics.json").write_text(
        json.dumps(outdated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--folds", type=str, default=None, help="comma-separated fold ids, e.g. 0,1")
    ap.add_argument("--final", action="store_true", help="train final model on all rows")
    ap.add_argument("--n-folds", type=int, default=5)
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-mode", choices=["full", "title_abstract", "title"], default="full")
    ap.add_argument("--limit", type=int, default=None, help="subsample for smoke tests")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if args.input_mode != "full" and args.out_dir == DEFAULT_OUT:
        args.out_dir = DEFAULT_OUT.parent / f"case_dora_abl_{args.input_mode}"
    examples, sub, stats = load_examples(tokenizer, args.data, args.input_mode)
    if args.limit:
        rng = np.random.RandomState(args.seed)
        keep = rng.choice(len(examples), size=min(args.limit, len(examples)), replace=False)
        examples = [examples[int(i)] for i in keep]
        stats = {"pos": int(sum(e.label for e in examples)),
                 "neg": int(sum(1 for e in examples if not e.label)),
                 "excluded_missing_or_other": stats["excluded_missing_or_other"]}
    y = np.asarray([e.label for e in examples], dtype=int)
    print(f"loaded {len(examples)} rows: {stats}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.folds:
        for fold in [int(x) for x in args.folds.split(",") if x.strip() != ""]:
            run_fold(fold, examples, y, tokenizer, model_path, args, device, args.out_dir)
    if args.final:
        train_final(examples, tokenizer, model_path, args, device, args.out_dir)


if __name__ == "__main__":
    main()
