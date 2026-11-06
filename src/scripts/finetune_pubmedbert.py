#!/usr/bin/env python3
"""Fine-tune NeuML PubMedBERT for variant case screening.

Methods
-------
- full   : standard full-parameter AdamW
- dora   : DoRA (PEFT, attention q/v)
- galore : full-parameter updates with GaLoreAdamW (low-rank gradient projection)

Protocol (aligned with AHP / fair head)
--------------------------------------
- Base labels: data/labeled_papers.csv, seed-42 GroupShuffleSplit
- Val/Test: default subset only (weak=False, label_source != model)
- Train: default train subset + AHP hard labels (train-only; never in test)
- Threshold: val Recall ≥ target (default 0.95), max precision
- Report: original test fold + gold-40

Usage
-----
    .venv/bin/python src/scripts/finetune_pubmedbert.py --method full
    .venv/bin/python src/scripts/finetune_pubmedbert.py --method dora
    .venv/bin/python src/scripts/finetune_pubmedbert.py --method galore
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from encoder import DEFAULT_MODEL, compose_document_text  # noqa: E402
from train import (  # noqa: E402
    binary_metrics,
    default_subset_mask,
    load_dataset,
    make_splits,
    select_threshold,
)

logger = logging.getLogger("finetune")

DEFAULT_DATA = ROOT / "data" / "labeled_papers.csv"
DEFAULT_GOLD = ROOT / "data" / "gold_standard.csv"
DEFAULT_AHP_LABELS = ROOT / "data" / "ahp_wave1" / "labels_ahp.csv"
DEFAULT_AHP_CORPUS = ROOT / "data" / "ahp_wave1" / "corpus.jsonl"


def resolve_model_path(path: str | None) -> str:
    cand = Path(path or DEFAULT_MODEL).expanduser()
    if cand.is_dir() and (cand / "config.json").exists():
        return str(cand)
    return "NeuML/pubmedbert-base-embeddings"


@dataclass
class Example:
    pmid: str
    text: str
    label: int
    source: str  # base | ahp


class TextClsDataset(Dataset):
    def __init__(self, rows: list[Example], tokenizer, max_length: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.rows[idx]
        enc = self.tok(
            ex.text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex.label, dtype=torch.long)
        return item


def load_corpus_map(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[str(rec["pmid"])] = rec
    return out


def build_examples(
    *,
    data: Path,
    ahp_labels: Path,
    ahp_corpus: Path,
    seed: int,
    include_ahp: bool,
) -> tuple[list[Example], list[Example], list[Example], list[Example], dict[str, Any]]:
    """Return train/val/test/gold examples + split meta."""
    base_df = load_dataset(data)
    subset = default_subset_mask(base_df).to_numpy()
    train_idx, val_idx, test_idx = make_splits(base_df, seed=seed)
    train_sub = train_idx[subset[train_idx]]
    val_sub = val_idx[subset[val_idx]]
    test_sub = test_idx[subset[test_idx]]

    def from_base(idxs: np.ndarray, source: str = "base") -> list[Example]:
        rows: list[Example] = []
        for i in idxs:
            r = base_df.iloc[int(i)]
            text = compose_document_text(r["title"], r["abstract"], r["text"])
            rows.append(
                Example(
                    pmid=str(r["pmid"]),
                    text=text,
                    label=int(r["label"]),
                    source=source,
                )
            )
        return rows

    train = from_base(train_sub)
    val = from_base(val_sub)
    test = from_base(test_sub)

    ahp_n = 0
    if include_ahp and ahp_labels.exists():
        labels = pd.read_csv(ahp_labels, dtype=str)
        hard = labels[labels["label"].isin(["0", "1"])].copy()
        corpus = load_corpus_map(ahp_corpus) if ahp_corpus.exists() else {}
        for _, r in hard.iterrows():
            pmid = str(r["pmid"])
            rec = corpus.get(pmid)
            if not rec:
                continue
            text = compose_document_text(rec.get("title") or "", "", rec.get("text") or "")
            if len(text.strip()) < 50:
                continue
            train.append(Example(pmid=pmid, text=text, label=int(r["label"]), source="ahp"))
            ahp_n += 1

    gold_rows: list[Example] = []
    gold_path = DEFAULT_GOLD
    if gold_path.exists():
        gold = pd.read_csv(gold_path, dtype={"pmid": str})
        gold = gold[gold["gold_label"].astype(str).isin(["0", "1"])].copy()
        gmap = gold.drop_duplicates("pmid", keep="last").set_index("pmid")["gold_label"].astype(int)
        id_to_i = {str(p): i for i, p in enumerate(base_df["pmid"].astype(str))}
        for pmid, y in gmap.items():
            if pmid not in id_to_i:
                continue
            r = base_df.iloc[id_to_i[pmid]]
            gold_rows.append(
                Example(
                    pmid=pmid,
                    text=compose_document_text(r["title"], r["abstract"], r["text"]),
                    label=int(y),
                    source="gold",
                )
            )

    meta = {
        "n_train_base": int(len(train_sub)),
        "n_train_ahp": int(ahp_n),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_gold": int(len(gold_rows)),
        "seed": seed,
        "pos_train": int(sum(e.label for e in train)),
        "neg_train": int(len(train) - sum(e.label for e in train)),
    }
    return train, val, test, gold_rows, meta


def class_weights(train: list[Example], device: torch.device) -> torch.Tensor:
    n1 = sum(e.label for e in train)
    n0 = len(train) - n1
    # inverse frequency, normalized to mean 1
    w0 = len(train) / (2 * max(n0, 1))
    w1 = len(train) / (2 * max(n1, 1))
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def build_model(method: str, model_path: str, dora_r: int, dora_alpha: int):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    info: dict[str, Any] = {"method": method}
    if method == "dora":
        from peft import LoraConfig, TaskType, get_peft_model

        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=dora_r,
            lora_alpha=dora_alpha,
            lora_dropout=0.05,
            use_dora=True,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info.update({"dora_r": dora_r, "dora_alpha": dora_alpha, "trainable": trainable, "total": total})
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        info.update({"trainable": trainable, "total": total})
    return model, info


def build_optimizer(
    method: str,
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    galore_rank: int,
    galore_gap: int,
    galore_scale: float,
):
    if method == "galore":
        from galore_torch import GaLoreAdamW

        galore_params = []
        other_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # 2-D weights in encoder (attention/FFN); skip embeddings & classifier head
            if (
                p.ndim == 2
                and "classifier" not in name
                and "embeddings" not in name
                and "pooler" not in name
            ):
                galore_params.append(p)
            else:
                other_params.append(p)
        logger.info("GaLore matrices=%d other_tensors=%d rank=%d", len(galore_params), len(other_params), galore_rank)
        return GaLoreAdamW(
            [
                {"params": other_params},
                {
                    "params": galore_params,
                    "rank": galore_rank,
                    "update_proj_gap": galore_gap,
                    "scale": galore_scale,
                    "proj_type": "std",
                },
            ],
            lr=lr,
            weight_decay=weight_decay,
        )

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


@torch.no_grad()
def predict_proba(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        p = F.softmax(logits, dim=-1)[:, 1].detach().float().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float64)


def evaluate_split(
    model,
    rows: list[Example],
    tokenizer,
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
    threshold: float | None,
    target_recall: float,
) -> tuple[dict[str, Any], float]:
    if not rows:
        return {"n": 0}, 0.5
    ds = TextClsDataset(rows, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores = predict_proba(model, loader, device)
    y = np.asarray([e.label for e in rows], dtype=int)
    if threshold is None:
        thr = select_threshold(y, scores, target_recall)
    else:
        thr = float(threshold)
    return binary_metrics(y, scores, thr), thr


def train_one(
    method: str,
    *,
    model_path: str,
    out_dir: Path,
    train: list[Example],
    val: list[Example],
    test: list[Example],
    gold: list[Example],
    meta: dict[str, Any],
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    max_length: int,
    target_recall: float,
    seed: int,
    dora_r: int,
    dora_alpha: int,
    galore_rank: int,
    galore_gap: int,
    galore_scale: float,
    eval_batch_size: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model, model_info = build_model(method, model_path, dora_r, dora_alpha)
    model.to(device)
    if method == "full":
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    train_ds = TextClsDataset(train, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    weights = class_weights(train, device)

    optimizer = build_optimizer(
        method, model, lr, weight_decay, galore_rank, galore_gap, galore_scale
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(grad_accum, 1)))
    total_steps = steps_per_epoch * epochs
    warmup = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_val_ap = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    t0 = time.time()

    logger.info(
        "Start method=%s device=%s train=%d val=%d epochs=%d bs=%d accum=%d lr=%s",
        method,
        device,
        len(train),
        len(val),
        epochs,
        batch_size,
        grad_accum,
        lr,
    )

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(**batch).logits
                loss = F.cross_entropy(logits, labels, weight=weights)
                loss = loss / max(grad_accum, 1)
            scaler.scale(loss).backward()
            running += float(loss.detach()) * max(grad_accum, 1) * labels.size(0)
            n_seen += int(labels.size(0))

            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        val_m, _ = evaluate_split(
            model,
            val,
            tokenizer,
            max_length=max_length,
            batch_size=eval_batch_size,
            device=device,
            threshold=0.5,
            target_recall=target_recall,
        )
        train_loss = running / max(n_seen, 1)
        ap = float(val_m.get("ap") or -1.0)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_ap@0.5": ap, "val": val_m})
        logger.info(
            "epoch %d/%d loss=%.4f val_ap@0.5=%.4f val_f1@0.5=%.3f",
            epoch,
            epochs,
            train_loss,
            ap,
            float(val_m.get("f1") or 0.0),
        )
        if ap > best_val_ap + 1e-6:
            best_val_ap = ap
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best epoch=%d val_ap@0.5=%.4f", best_epoch, best_val_ap)

    val_m, thr = evaluate_split(
        model,
        val,
        tokenizer,
        max_length=max_length,
        batch_size=eval_batch_size,
        device=device,
        threshold=None,
        target_recall=target_recall,
    )
    test_m, _ = evaluate_split(
        model,
        test,
        tokenizer,
        max_length=max_length,
        batch_size=eval_batch_size,
        device=device,
        threshold=thr,
        target_recall=target_recall,
    )
    gold_m = None
    if gold:
        gold_m, _ = evaluate_split(
            model,
            gold,
            tokenizer,
            max_length=max_length,
            batch_size=eval_batch_size,
            device=device,
            threshold=thr,
            target_recall=target_recall,
        )

    # save
    save_dir = out_dir / "checkpoint"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    elapsed = time.time() - t0
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "model_path": model_path,
        "model_info": model_info,
        "hyper": {
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "max_length": max_length,
            "target_recall": target_recall,
            "seed": seed,
            "galore_rank": galore_rank if method == "galore" else None,
            "galore_gap": galore_gap if method == "galore" else None,
            "galore_scale": galore_scale if method == "galore" else None,
            "dora_r": dora_r if method == "dora" else None,
            "dora_alpha": dora_alpha if method == "dora" else None,
        },
        "data": meta,
        "best_epoch": best_epoch,
        "best_val_ap_at_0_5": best_val_ap,
        "threshold": thr,
        "val": val_m,
        "test": test_m,
        "gold40": gold_m,
        "history": history,
        "elapsed_sec": elapsed,
        "checkpoint": str(save_dir),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = [
        f"# PubMedBERT fine-tune — `{method}`",
        "",
        f"- train={meta['n_train']} (base {meta['n_train_base']} + ahp {meta['n_train_ahp']})",
        f"- best_epoch={best_epoch} val_AP@0.5={best_val_ap:.4f}",
        f"- threshold (val R≥{target_recall}): **{thr:.4f}**",
        f"- elapsed: {elapsed / 60:.1f} min",
        "",
        "## Test",
        f"- R={test_m['recall']:.3f} P={test_m['precision']:.3f} F1={test_m['f1']:.3f} AP=**{test_m['ap']:.3f}**",
        "",
    ]
    if gold_m:
        md += [
            "## Gold-40",
            f"- R={gold_m['recall']:.3f} P={gold_m['precision']:.3f} F1={gold_m['f1']:.3f} AP=**{gold_m['ap']:.3f}**",
            "",
        ]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"method": method, "test": test_m, "gold40": gold_m, "out": str(out_dir)}, indent=2))
    return result


def default_lr(method: str) -> float:
    return {"full": 2e-5, "galore": 2e-5, "dora": 2e-4}[method]


def default_batch(method: str) -> int:
    return {"full": 4, "galore": 4, "dora": 8}[method]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["full", "dora", "galore"], required=True)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--ahp-labels", type=Path, default=DEFAULT_AHP_LABELS)
    ap.add_argument("--ahp-corpus", type=Path, default=DEFAULT_AHP_CORPUS)
    ap.add_argument("--out-root", type=Path, default=ROOT / "artifacts" / "finetune")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ahp", action="store_true")
    ap.add_argument("--dora-r", type=int, default=16)
    ap.add_argument("--dora-alpha", type=int, default=32)
    ap.add_argument("--galore-rank", type=int, default=128)
    ap.add_argument("--galore-gap", type=int, default=50)
    ap.add_argument("--galore-scale", type=float, default=1.0)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    args = ap.parse_args()

    model_path = resolve_model_path(args.model)
    lr = args.lr if args.lr is not None else default_lr(args.method)
    batch_size = args.batch_size if args.batch_size is not None else default_batch(args.method)
    out_dir = args.out_root / args.method

    train, val, test, gold, meta = build_examples(
        data=args.data,
        ahp_labels=args.ahp_labels,
        ahp_corpus=args.ahp_corpus,
        seed=args.seed,
        include_ahp=not args.no_ahp,
    )
    logger.info("Data meta: %s", meta)

    train_one(
        args.method,
        model_path=model_path,
        out_dir=out_dir,
        train=train,
        val=val,
        test=test,
        gold=gold,
        meta=meta,
        epochs=args.epochs,
        batch_size=batch_size,
        grad_accum=args.grad_accum,
        lr=lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        target_recall=args.target_recall,
        seed=args.seed,
        dora_r=args.dora_r,
        dora_alpha=args.dora_alpha,
        galore_rank=args.galore_rank,
        galore_gap=args.galore_gap,
        galore_scale=args.galore_scale,
        eval_batch_size=args.eval_batch_size,
    )


if __name__ == "__main__":
    main()
