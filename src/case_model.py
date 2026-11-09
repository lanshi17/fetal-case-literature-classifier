#!/usr/bin/env python3
"""Load the case-only DoRA model and score (title, text) documents."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

DEFAULT_CONFIG = ROOT / "artifacts" / "case_dora" / "config.json"


class CaseDoraPredictor:
    """Chunk-pool DoRA classifier: case report / case series vs other designs."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG, *, device: str | None = None) -> None:
        from transformers import AutoTokenizer
        from finetune_chunk_fair import ChunkPoolClassifier, resolve_model_path, wrap_dora

        self.config_path = Path(config_path)
        cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.cfg = cfg
        ckpt = ROOT / cfg["paths"]["checkpoint_dir"]
        if not (ckpt / "model.pt").exists():
            raise FileNotFoundError(f"case checkpoint not found: {ckpt / 'model.pt'}")

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model_path = resolve_model_path(cfg["encoder"].get("model_path") or cfg["encoder"]["model_id"])
        self.threshold = float(cfg["threshold"])
        self.threshold_p90 = float(cfg.get("threshold_p90", cfg["threshold"]))
        self.chunk_tokens = int(cfg["encoder"].get("chunk_tokens", 480))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model = ChunkPoolClassifier(model_path)
        model = wrap_dora(model, int(cfg["adapter"]["r"]), int(cfg["adapter"]["alpha"]))
        state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
        model.to(device).eval()
        self.model = model

    def decide(self, score: float, *, p90: bool = False) -> str:
        thr = self.threshold_p90 if p90 else self.threshold
        return "case" if score >= thr else "non-case"

    def score_docs(self, docs: list[dict[str, Any]], *, batch_docs: int = 16) -> list[float]:
        """Score docs with keys ``title`` and ``text``; returns P(case)."""
        from finetune_chunk_fair import doc_to_chunks

        out: list[float] = []
        for start in range(0, len(docs), batch_docs):
            batch = docs[start : start + batch_docs]
            chunks = [
                doc_to_chunks(str(d.get("title") or ""), "", str(d.get("text") or ""), self.tokenizer)
                for d in batch
            ]
            flat = [c for cs in chunks for c in cs]
            enc = self.tokenizer(
                flat,
                truncation=True,
                max_length=self.chunk_tokens,
                padding=True,
                return_tensors="pt",
            )
            owners = torch.repeat_interleave(
                torch.arange(len(batch), dtype=torch.long),
                torch.tensor([len(cs) for cs in chunks], dtype=torch.long),
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            owners = owners.to(self.device)
            with torch.inference_mode():
                logits = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    owners=owners,
                    batch_size=len(batch),
                )["logits"]
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            out.extend(float(p) for p in probs)
        return out
