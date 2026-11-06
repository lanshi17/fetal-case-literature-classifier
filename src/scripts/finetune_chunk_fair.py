#!/usr/bin/env python3
"""Fair fine-tune: same multi-chunk pooling as frozen PubMedBERT+LR.

Encoding protocol (matches encoder.py / fair head)
-------------------------------------------------
- compose_document_text(title, abstract, text)
- chunk_text(..., chunk_tokens=480, max_chunks=12, keep_first=2)
- per-chunk mean-pool of last hidden states (attention-masked)
- document feature = L2(mean_all) ‖ L2(mean_first_2) → 1536-d
- linear classifier on 1536-d

Methods: full | dora | galore
Eval: seed-42 default-subset val/test + gold-40; val R≥0.95 threshold.
AHP hard labels train-only.

Usage
-----
    .venv/bin/python src/scripts/finetune_chunk_fair.py --method full
    .venv/bin/python src/scripts/finetune_chunk_fair.py --method dora
    .venv/bin/python src/scripts/finetune_chunk_fair.py --method galore
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from encoder import (  # noqa: E402
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_KEEP_FIRST,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_MODEL,
    EMBED_DIM,
    chunk_text,
    compose_document_text,
)
from train import (  # noqa: E402
    binary_metrics,
    default_subset_mask,
    load_dataset,
    make_splits,
    select_threshold,
)

logger = logging.getLogger("finetune_chunk")

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
    chunks: list[str]
    label: int
    source: str


def load_corpus_map(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[str(rec["pmid"])] = rec
    return out


def doc_to_chunks(title: str, abstract: str, text: str, tokenizer) -> list[str]:
    doc = compose_document_text(title, abstract, text)
    return chunk_text(
        doc,
        tokenizer,
        chunk_tokens=DEFAULT_CHUNK_TOKENS,
        max_chunks=DEFAULT_MAX_CHUNKS,
        keep_first=DEFAULT_KEEP_FIRST,
    )


def build_examples(
    *,
    data: Path,
    ahp_labels: Path,
    ahp_corpus: Path,
    tokenizer,
    seed: int,
    include_ahp: bool,
) -> tuple[list[Example], list[Example], list[Example], list[Example], dict[str, Any]]:
    base_df = load_dataset(data)
    subset = default_subset_mask(base_df).to_numpy()
    train_idx, val_idx, test_idx = make_splits(base_df, seed=seed)
    train_sub = train_idx[subset[train_idx]]
    val_sub = val_idx[subset[val_idx]]
    test_sub = test_idx[subset[test_idx]]

    def from_base(idxs: np.ndarray) -> list[Example]:
        rows: list[Example] = []
        for i in idxs:
            r = base_df.iloc[int(i)]
            chunks = doc_to_chunks(r["title"], r["abstract"], r["text"], tokenizer)
            rows.append(Example(str(r["pmid"]), chunks, int(r["label"]), "base"))
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
            chunks = doc_to_chunks(rec.get("title") or "", "", rec.get("text") or "", tokenizer)
            if not chunks or (len(chunks) == 1 and chunks[0].strip() in {"", " "}):
                continue
            train.append(Example(pmid, chunks, int(r["label"]), "ahp"))
            ahp_n += 1

    gold_rows: list[Example] = []
    if DEFAULT_GOLD.exists():
        gold = pd.read_csv(DEFAULT_GOLD, dtype={"pmid": str})
        gold = gold[gold["gold_label"].astype(str).isin(["0", "1"])].copy()
        gmap = gold.drop_duplicates("pmid", keep="last").set_index("pmid")["gold_label"].astype(int)
        id_to_i = {str(p): i for i, p in enumerate(base_df["pmid"].astype(str))}
        for pmid, y in gmap.items():
            if pmid not in id_to_i:
                continue
            r = base_df.iloc[id_to_i[pmid]]
            gold_rows.append(
                Example(
                    pmid,
                    doc_to_chunks(r["title"], r["abstract"], r["text"], tokenizer),
                    int(y),
                    "gold",
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
        "chunk_tokens": DEFAULT_CHUNK_TOKENS,
        "max_chunks": DEFAULT_MAX_CHUNKS,
        "keep_first": DEFAULT_KEEP_FIRST,
        "feature_dim": 2 * EMBED_DIM,
        "pos_train": int(sum(e.label for e in train)),
        "neg_train": int(len(train) - sum(e.label for e in train)),
        "mean_chunks_train": float(np.mean([len(e.chunks) for e in train])),
    }
    return train, val, test, gold_rows, meta


class ChunkDataset(Dataset):
    def __init__(self, rows: list[Example]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Example:
        return self.rows[idx]


def collate_examples(batch: list[Example], tokenizer, max_length: int) -> dict[str, Any]:
    """Flatten chunks across docs; return owner index for pooling."""
    all_chunks: list[str] = []
    owners: list[int] = []
    labels: list[int] = []
    n_chunks: list[int] = []
    for i, ex in enumerate(batch):
        ch = ex.chunks or [" "]
        all_chunks.extend(ch)
        owners.extend([i] * len(ch))
        n_chunks.append(len(ch))
        labels.append(ex.label)
    enc = tokenizer(
        all_chunks,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "owners": torch.tensor(owners, dtype=torch.long),
        "n_chunks": torch.tensor(n_chunks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "batch_size": len(batch),
    }


class ChunkPoolClassifier(nn.Module):
    """BERT → per-chunk mean pool → L2(mean_all)‖L2(mean_head) → logits."""

    def __init__(self, model_path: str, keep_first: int = DEFAULT_KEEP_FIRST):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        hidden = int(self.encoder.config.hidden_size)
        self.keep_first = keep_first
        self.classifier = nn.Linear(2 * hidden, 2)
        self.hidden = hidden

    def _chunk_vecs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # (N, L, H)
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom  # (N, H)

    def pool_doc(self, vecs: torch.Tensor) -> torch.Tensor:
        # vecs: (C, H)
        mean_all = F.normalize(vecs.mean(dim=0), dim=0, eps=1e-9)
        k = max(1, min(self.keep_first, vecs.size(0)))
        mean_head = F.normalize(vecs[:k].mean(dim=0), dim=0, eps=1e-9)
        return torch.cat([mean_all, mean_head], dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        owners: torch.Tensor,
        batch_size: int,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        chunk_vecs = self._chunk_vecs(input_ids, attention_mask)  # (sum_C, H)
        feats = []
        for i in range(batch_size):
            vecs = chunk_vecs[owners == i]
            if vecs.numel() == 0:
                vecs = chunk_vecs.new_zeros((1, self.hidden))
            feats.append(self.pool_doc(vecs))
        feat = torch.stack(feats, dim=0)  # (B, 2H)
        logits = self.classifier(feat)
        out: dict[str, torch.Tensor] = {"logits": logits, "features": feat}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out


def wrap_dora(model: ChunkPoolClassifier, r: int, alpha: int) -> ChunkPoolClassifier:
    from peft import LoraConfig, TaskType, get_peft_model

    # PEFT on the inner encoder; classifier stays fully trainable
    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        use_dora=True,
        target_modules=["query", "value"],
    )
    model.encoder = get_peft_model(model.encoder, cfg)
    for p in model.classifier.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("DoRA trainable=%d / %d (%.2f%%)", trainable, total, 100 * trainable / total)
    return model


def build_optimizer(
    method: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
    galore_rank: int,
    galore_gap: int,
    galore_scale: float,
):
    if method == "galore":
        from galore_torch import GaLoreAdamW

        galore_params, other = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                p.ndim == 2
                and "classifier" not in name
                and "embeddings" not in name
                and "pooler" not in name
            ):
                galore_params.append(p)
            else:
                other.append(p)
        logger.info("GaLore n_mat=%d n_other=%d rank=%d", len(galore_params), len(other), galore_rank)
        return GaLoreAdamW(
            [
                {"params": other},
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
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def class_weights(train: list[Example], device: torch.device) -> torch.Tensor:
    n1 = sum(e.label for e in train)
    n0 = len(train) - n1
    w0 = len(train) / (2 * max(n0, 1))
    w1 = len(train) / (2 * max(n1, 1))
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


@torch.no_grad()
def predict_proba(
    model: ChunkPoolClassifier,
    rows: list[Example],
    tokenizer,
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    ds = ChunkDataset(rows)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_examples(b, tokenizer, max_length),
    )
    probs: list[np.ndarray] = []
    for batch in loader:
        labels = batch.pop("labels")
        n = batch.pop("batch_size")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            owners=batch["owners"],
            batch_size=n,
        )["logits"]
        p = F.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
        probs.append(p)
        del labels
    return np.concatenate(probs) if probs else np.zeros(0)


def evaluate(
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
    scores = predict_proba(
        model, rows, tokenizer, max_length=max_length, batch_size=batch_size, device=device
    )
    y = np.asarray([e.label for e in rows], dtype=int)
    thr = select_threshold(y, scores, target_recall) if threshold is None else float(threshold)
    return binary_metrics(y, scores, thr), thr


def train_method(args) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = resolve_model_path(args.model)
    out_dir = args.out_root / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train, val, test, gold, meta = build_examples(
        data=args.data,
        ahp_labels=args.ahp_labels,
        ahp_corpus=args.ahp_corpus,
        tokenizer=tokenizer,
        seed=args.seed,
        include_ahp=not args.no_ahp,
    )
    logger.info("meta=%s", meta)

    model = ChunkPoolClassifier(model_path, keep_first=DEFAULT_KEEP_FIRST)
    model_info: dict[str, Any] = {"method": args.method, "encode": "chunk_pool_1536"}
    if args.method == "dora":
        model = wrap_dora(model, args.dora_r, args.dora_alpha)
        model_info.update({"dora_r": args.dora_r, "dora_alpha": args.dora_alpha})
    if args.method == "full":
        model.encoder.gradient_checkpointing_enable()
        model.encoder.config.use_cache = False

    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model_info.update({"trainable": trainable, "total": total})
    logger.info("trainable=%d / %d", trainable, total)

    lr = args.lr if args.lr is not None else {"full": 2e-5, "galore": 2e-5, "dora": 2e-4}[args.method]
    bs = args.batch_size if args.batch_size is not None else {"full": 1, "galore": 1, "dora": 2}[args.method]

    weights = class_weights(train, device)
    opt = build_optimizer(
        args.method, model, lr, args.weight_decay, args.galore_rank, args.galore_gap, args.galore_scale
    )
    train_loader = DataLoader(
        ChunkDataset(train),
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda b: collate_examples(b, tokenizer, args.max_length),
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(args.grad_accum, 1)))
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * args.warmup_ratio), total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_ap = -1.0
    best_epoch = -1
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
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
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

        val_m, _ = evaluate(
            model,
            val,
            tokenizer,
            max_length=args.max_length,
            batch_size=max(1, args.eval_batch_size),
            device=device,
            threshold=0.5,
            target_recall=args.target_recall,
        )
        ap = float(val_m.get("ap") or -1.0)
        loss_ep = running / max(n_seen, 1)
        history.append({"epoch": epoch, "train_loss": loss_ep, "val_ap@0.5": ap, "val": val_m})
        logger.info("epoch %d/%d loss=%.4f val_ap@0.5=%.4f", epoch, args.epochs, loss_ep, ap)
        if ap > best_ap + 1e-6:
            best_ap = ap
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("restored best epoch=%d ap=%.4f", best_epoch, best_ap)

    val_m, thr = evaluate(
        model,
        val,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
        threshold=None,
        target_recall=args.target_recall,
    )
    test_m, _ = evaluate(
        model,
        test,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
        threshold=thr,
        target_recall=args.target_recall,
    )
    gold_m = None
    if gold:
        gold_m, _ = evaluate(
            model,
            gold,
            tokenizer,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            device=device,
            threshold=thr,
            target_recall=args.target_recall,
        )

    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    # save encoder + classifier state
    torch.save(
        {"model_state": model.state_dict(), "method": args.method, "model_path": model_path},
        ckpt / "model.pt",
    )
    tokenizer.save_pretrained(ckpt)

    elapsed = time.time() - t0
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "encode_mode": "chunk_pool_1536_fair",
        "model_path": model_path,
        "model_info": model_info,
        "hyper": {
            "epochs": args.epochs,
            "batch_size": bs,
            "grad_accum": args.grad_accum,
            "lr": lr,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "target_recall": args.target_recall,
            "seed": args.seed,
        },
        "data": meta,
        "best_epoch": best_epoch,
        "best_val_ap_at_0_5": best_ap,
        "threshold": thr,
        "val": val_m,
        "test": test_m,
        "gold40": gold_m,
        "history": history,
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt),
        "note": "Same chunking/pooling as frozen PubMedBERT+LR (fair vs prior 512-trunc FT).",
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = [
        f"# Fair chunk-pool fine-tune — `{args.method}`",
        "",
        f"- encode: chunk_tokens={DEFAULT_CHUNK_TOKENS}, max_chunks={DEFAULT_MAX_CHUNKS}, feat=1536",
        f"- train={meta['n_train']} (base {meta['n_train_base']}+ahp {meta['n_train_ahp']}), mean_chunks={meta['mean_chunks_train']:.1f}",
        f"- best_epoch={best_epoch} val_AP@0.5={best_ap:.4f} thr={thr:.4f}",
        f"- elapsed={elapsed/60:.1f} min",
        "",
        f"## Test: R={test_m['recall']:.3f} P={test_m['precision']:.3f} F1={test_m['f1']:.3f} AP=**{test_m['ap']:.3f}**",
    ]
    if gold_m:
        md.append(
            f"## Gold-40: R={gold_m['recall']:.3f} P={gold_m['precision']:.3f} F1={gold_m['f1']:.3f} AP=**{gold_m['ap']:.3f}**"
        )
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"method": args.method, "test": test_m, "gold40": gold_m, "out": str(out_dir)}, indent=2))
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["full", "dora", "galore"], required=True)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--ahp-labels", type=Path, default=DEFAULT_AHP_LABELS)
    ap.add_argument("--ahp-corpus", type=Path, default=DEFAULT_AHP_CORPUS)
    ap.add_argument("--out-root", type=Path, default=ROOT / "artifacts" / "finetune_fair")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=DEFAULT_CHUNK_TOKENS)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ahp", action="store_true")
    ap.add_argument("--dora-r", type=int, default=16)
    ap.add_argument("--dora-alpha", type=int, default=32)
    ap.add_argument("--galore-rank", type=int, default=128)
    ap.add_argument("--galore-gap", type=int, default=50)
    ap.add_argument("--galore-scale", type=float, default=1.0)
    ap.add_argument("--eval-batch-size", type=int, default=2)
    args = ap.parse_args()
    train_method(args)


if __name__ == "__main__":
    main()
