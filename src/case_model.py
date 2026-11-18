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

# Below this many tokens the design signal is unreliable: a fragment can sit just
# above the case threshold without reporting any study design. Short inputs must
# clear ``SHORT_THRESHOLD``; the [case threshold, SHORT_THRESHOLD) band is
# reported as ``insufficient`` and routed to human review instead of asserted.
DEFAULT_MIN_TOKENS = 128
DEFAULT_SHORT_THRESHOLD = 0.75


def guarded_decision(
    score: float,
    n_tokens: int | None,
    *,
    threshold: float,
    threshold_p90: float,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    short_threshold: float = DEFAULT_SHORT_THRESHOLD,
    p90: bool = False,
) -> str:
    """Map P(case) to ``case`` / ``non-case`` / ``insufficient``.

    Short documents (< ``min_tokens``) need ``short_threshold`` for a positive
    call; scores between the normal case threshold and that bar are ambiguous,
    so they abstain rather than becoming false positives.
    """
    thr = threshold_p90 if p90 else threshold
    if n_tokens is not None and n_tokens < min_tokens:
        if score >= short_threshold:
            return "case"
        if score < thr:
            return "non-case"
        return "insufficient"
    return "case" if score >= thr else "non-case"


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
        evidence = cfg.get("evidence", {})
        self.min_tokens = int(evidence.get("min_tokens", DEFAULT_MIN_TOKENS))
        self.short_threshold = float(evidence.get("short_threshold", DEFAULT_SHORT_THRESHOLD))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model = ChunkPoolClassifier(model_path)
        model = wrap_dora(model, int(cfg["adapter"]["r"]), int(cfg["adapter"]["alpha"]))
        state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
        model.to(device).eval()
        self.model = model

    def decide(self, score: float, *, p90: bool = False, n_tokens: int | None = None) -> str:
        return guarded_decision(
            score,
            n_tokens,
            threshold=self.threshold,
            threshold_p90=self.threshold_p90,
            min_tokens=self.min_tokens,
            short_threshold=self.short_threshold,
            p90=p90,
        )

    def count_tokens(self, title: str = "", text: str = "") -> int:
        """Token count of the composed document, before chunking."""
        from finetune_chunk_fair import compose_document_text

        doc = compose_document_text(str(title or ""), "", str(text or ""))
        return len(self.tokenizer(doc, add_special_tokens=False, truncation=False)["input_ids"])

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
