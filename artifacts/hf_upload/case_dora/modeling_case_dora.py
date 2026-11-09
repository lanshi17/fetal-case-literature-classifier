"""Self-contained inference code for the prenatal case-only DoRA model.

Usage
-----
    from modeling_case_dora import CaseDoraPredictor

    model = CaseDoraPredictor.from_pretrained("your-username/prenatal-case-dora")
    out = model.score(title="...", text="full article text ...")
    # {"case_score": 0.93, "case_decision": "case", "threshold": 0.396...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

DEFAULT_BASE = "NeuML/pubmedbert-base-embeddings"
CHUNK_TOKENS = 480
MAX_CHUNKS = 12
KEEP_FIRST = 2
EMBED_DIM = 768


def compose_document_text(title: str, abstract: str, text: str) -> str:
    head = ". ".join(x for x in [str(title or "").strip(), str(abstract or "").strip()] if x)
    body = str(text or "").strip()
    if head and body:
        return f"{head}\n\n{body}"
    return head or body or " "


def chunk_text(
    text: str,
    tokenizer: Any,
    *,
    chunk_tokens: int = CHUNK_TOKENS,
    max_chunks: int = MAX_CHUNKS,
    keep_first: int = KEEP_FIRST,
) -> list[str]:
    if not text or not str(text).strip():
        return [" "]
    ids = tokenizer(str(text), add_special_tokens=False, truncation=False)["input_ids"]
    if not ids:
        return [" "]
    chunks = [ids[i : i + chunk_tokens] for i in range(0, len(ids), chunk_tokens)] or [[]]
    if max_chunks and len(chunks) > max_chunks:
        keep = max(0, min(int(keep_first), max_chunks))
        if max_chunks <= keep:
            chunks = chunks[:max_chunks]
        else:
            rest_idx = np.linspace(keep, len(chunks) - 1, max_chunks - keep).round().astype(int)
            rest_idx = sorted({int(i) for i in rest_idx if int(i) >= keep})
            selected = list(range(keep)) + rest_idx
            seen: set[int] = set()
            ordered: list[int] = []
            for i in selected:
                if i not in seen:
                    seen.add(i)
                    ordered.append(i)
            chunks = [chunks[i] for i in ordered[:max_chunks]]
    return [tokenizer.decode(c) if c else " " for c in chunks]


class ChunkPoolClassifier(nn.Module):
    """BERT -> per-chunk mean pool -> L2(mean_all) || L2(mean_head) -> logits."""

    def __init__(self, model_path: str, keep_first: int = KEEP_FIRST) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        hidden = int(self.encoder.config.hidden_size)
        self.keep_first = keep_first
        self.classifier = nn.Linear(2 * hidden, 2)
        self.hidden = hidden

    def _chunk_vecs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def pool_doc(self, vecs: torch.Tensor) -> torch.Tensor:
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
        chunk_vecs = self._chunk_vecs(input_ids, attention_mask)
        feats = []
        for i in range(batch_size):
            vecs = chunk_vecs[owners == i]
            if vecs.numel() == 0:
                vecs = chunk_vecs.new_zeros((1, self.hidden))
            feats.append(self.pool_doc(vecs))
        feat = torch.stack(feats, dim=0)
        logits = self.classifier(feat)
        out: dict[str, torch.Tensor] = {"logits": logits, "features": feat}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out


def wrap_dora(model: ChunkPoolClassifier, r: int, alpha: int) -> ChunkPoolClassifier:
    from peft import LoraConfig, TaskType, get_peft_model

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
    return model


class CaseDoraPredictor:
    def __init__(
        self,
        model: ChunkPoolClassifier,
        tokenizer: Any,
        config: dict[str, Any],
        device: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.threshold = float(config["threshold"])
        self.threshold_p90 = float(config.get("threshold_p90", self.threshold))
        self.chunk_tokens = int(config.get("encoder", {}).get("chunk_tokens", CHUNK_TOKENS))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @classmethod
    def from_pretrained(cls, repo_id: str, *, device: str | None = None, revision: str | None = None) -> "CaseDoraPredictor":
        config_path = hf_hub_download(repo_id, "config.json", revision=revision)
        weights_path = hf_hub_download(repo_id, "model.safetensors", revision=revision)
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        base = config.get("encoder", {}).get("base_model") or DEFAULT_BASE
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = ChunkPoolClassifier(base)
        model = wrap_dora(
            model,
            int(config.get("adapter", {}).get("r", 16)),
            int(config.get("adapter", {}).get("alpha", 32)),
        )
        state = load_file(weights_path)
        model.load_state_dict(state, strict=True)
        return cls(model, tokenizer, config, device=device)

    def score(self, title: str = "", text: str = "", *, abstract: str = "") -> dict[str, Any]:
        doc = compose_document_text(title, abstract, text)
        chunks = chunk_text(doc, self.tokenizer, chunk_tokens=self.chunk_tokens)
        enc = self.tokenizer(
            chunks,
            truncation=True,
            max_length=self.chunk_tokens,
            padding=True,
            return_tensors="pt",
        )
        owners = torch.zeros(enc["input_ids"].size(0), dtype=torch.long)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        owners = owners.to(self.device)
        with torch.inference_mode():
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                owners=owners,
                batch_size=1,
            )["logits"]
            prob = float(F.softmax(logits, dim=-1)[0, 1].detach().cpu())
        return {
            "case_score": round(prob, 6),
            "case_decision": "case" if prob >= self.threshold else "non-case",
            "case_decision_p90": "case" if prob >= self.threshold_p90 else "non-case",
            "threshold": self.threshold,
            "threshold_p90": self.threshold_p90,
            "n_chunks": len(chunks),
        }
