#!/usr/bin/env python3
"""PubMedBERT encoder utilities (NeuML/pubmedbert-base-embeddings).

Feature layout (per document), matching casefilter-head-v3:
    [ mean(all chunks) L2-normed (768) | mean(first 2 chunks) L2-normed (768) ]
    optionally + [ case-term density, variant/test-term density ]  (2)

Frozen encoder only — no fine-tuning of PubMedBERT weights.

Usage
-----
    python src/encoder.py --smoke
    python src/encoder.py --text "A case report of a novel variant ..."
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

_CACHE_ROOT = ROOT / ".cache"
PROJECT_HF_HOME = _CACHE_ROOT / "huggingface"
PROJECT_HF_HUB = PROJECT_HF_HOME / "hub"
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DEFAULT_MODEL = os.environ.get(
    "PUBMEDBERT_MODEL_PATH",
    os.environ.get(
        "PUBMEDBERT_MODEL",
        "/home/yangzs/.cache/huggingface/hub/models--NeuML--pubmedbert-base-embeddings/"
        "snapshots/b79526d6ef3645e0df4530322e266f24c829f5ef",
    ),
)
DEFAULT_MODEL_ID = "NeuML/pubmedbert-base-embeddings"

DEFAULT_CHUNK_TOKENS = 480
DEFAULT_MAX_CHUNKS = 12
DEFAULT_KEEP_FIRST = 2
EMBED_DIM = 768
POOL_DIM = 2 * EMBED_DIM  # 1536
AUX_DIM = 2
FEATURE_DIM_A = EMBED_DIM  # first-2 mean only
FEATURE_DIM_B = POOL_DIM  # mean-all || first-2
FEATURE_DIM_C = POOL_DIM + AUX_DIM


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _offline_mode() -> bool:
    return (
        _truthy_env("HF_HUB_OFFLINE")
        or _truthy_env("TRANSFORMERS_OFFLINE")
        or _truthy_env("PUBMEDBERT_LOCAL_FILES_ONLY")
    )


def _writable_cache(path: str | Path) -> bool:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return os.access(probe, os.W_OK)


def _configure_hf_cache() -> None:
    default_home = Path.home() / ".cache" / "huggingface"
    requested_home = os.environ.get("HF_HOME")
    home = Path(requested_home).expanduser() if requested_home else default_home
    if not _writable_cache(home) and _writable_cache(PROJECT_HF_HOME):
        home = PROJECT_HF_HOME
    os.environ.setdefault("HF_HOME", str(home))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))


_configure_hf_cache()


CASE_TERMS = (
    "case report",
    "case series",
    "case study",
    "proband",
    "patient",
    "patients",
    "fetus",
    "fetuses",
    "foetus",
    "prenatal",
    "antenatal",
    "family",
    "pedigree",
    "phenotype",
    "clinical",
    "pregnancy",
    "infant",
    "neonate",
    "counselling",
    "counseling",
)
VARIANT_TEST_TERMS = (
    "variant",
    "variants",
    "mutation",
    "mutations",
    "pathogenic",
    "likely pathogenic",
    "vus",
    "cnv",
    "copy number",
    "karyotype",
    "karyotyping",
    "exome",
    "genome",
    "sequencing",
    "microarray",
    "pcr",
    "qpcr",
    "qfpcr",
    "panel",
    "deletion",
    "duplication",
    "trisomy",
    "aneuploidy",
    "aneuploid",
    "genotype",
    "screening",
    "amniocentesis",
    "cvs",
    "nipt",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class PubMedBertEncoder:
    """Frozen NeuML PubMedBERT sentence embedding model."""

    model: Any
    device: str
    model_path: str
    max_length: int = 512
    normalize: bool = True
    batch_size: int = 32
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_dim(self) -> int:
        return int(getattr(self.model, "get_sentence_embedding_dimension", lambda: EMBED_DIM)())

    @property
    def tokenizer(self) -> Any:
        return self.model.tokenizer

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        normalize: bool | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        batch_size = int(batch_size or self.batch_size)
        normalize = self.normalize if normalize is None else normalize
        cleaned = [t if isinstance(t, str) and t.strip() else " " for t in texts]
        vecs = self.model.encode(
            cleaned,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)


def _resolve_model_path(model_path: str | Path | None) -> str:
    """Prefer a local snapshot; fall back to the HF model id."""
    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path).expanduser())
    candidates.append(Path(DEFAULT_MODEL).expanduser())
    for cand in candidates:
        if cand.is_dir() and (cand / "config.json").exists():
            return str(cand)
    return DEFAULT_MODEL_ID


def load_encoder(
    model_path: str | Path | None = None,
    *,
    device: str | None = None,
    max_length: int = 512,
    normalize: bool = True,
    batch_size: int = 32,
    local_files_only: bool | None = None,
    # Back-compat kwargs from the SPECTER2 era (ignored).
    adapter_name: str | None = None,
    allow_base_fallback: bool = False,
    base_path: str | Path | None = None,
) -> PubMedBertEncoder:
    """Load frozen NeuML/pubmedbert-base-embeddings."""
    del adapter_name, allow_base_fallback  # unused; kept for call-site compatibility
    import torch
    from sentence_transformers import SentenceTransformer

    if base_path and not model_path:
        model_path = base_path
    resolved = _resolve_model_path(model_path)
    offline = _offline_mode() if local_files_only is None else bool(local_files_only)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device = "cpu"

    logger.info("Loading PubMedBERT from %s on %s (offline=%s)", resolved, device, offline)
    kwargs: dict[str, Any] = {}
    if offline and Path(resolved).is_dir():
        # Local snapshot: SentenceTransformer loads from path without network.
        pass
    model = SentenceTransformer(resolved, device=device, **kwargs)
    model.max_seq_length = int(max_length)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return PubMedBertEncoder(
        model=model,
        device=str(device),
        model_path=resolved,
        max_length=int(max_length),
        normalize=normalize,
        batch_size=batch_size,
        metadata={
            "encoder": "pubmedbert",
            "model_id": DEFAULT_MODEL_ID,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "pooling": "concat[mean(all), mean(first 2)] L2-normed -> 1536",
        },
    )


# ---------------------------------------------------------------------------
# Chunking (token windows, keep first 2 + even sample)
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    tokenizer: Any,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    keep_first: int = DEFAULT_KEEP_FIRST,
) -> list[str]:
    """Token-window chunks. Always keeps the first ``keep_first`` (title/abstract), samples the rest."""
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
            # Preserve order, drop duplicates, cap length.
            seen: set[int] = set()
            ordered: list[int] = []
            for i in selected:
                if i not in seen:
                    seen.add(i)
                    ordered.append(i)
            chunks = [chunks[i] for i in ordered[:max_chunks]]
    return [tokenizer.decode(c) if c else " " for c in chunks]


def compose_document_text(title: str, abstract: str, text: str) -> str:
    head = ". ".join(x for x in [str(title or "").strip(), str(abstract or "").strip()] if x)
    body = str(text or "").strip()
    if head and body:
        return f"{head}\n\n{body}"
    return head or body or " "


# ---------------------------------------------------------------------------
# Auxiliary lexical features
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def _term_density(text: str, terms: Sequence[str]) -> float:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    lowered = " " + " ".join(words) + " "
    hits = sum(lowered.count(term) for term in terms)
    return 1000.0 * hits / max(len(words), 1)


def aux_features(title: str, abstract: str, text: str, *, max_chars: int = 20000) -> np.ndarray:
    sample = " ".join([title or "", abstract or "", (text or "")[:max_chars]])
    return np.asarray(
        [_term_density(sample, CASE_TERMS), _term_density(sample, VARIANT_TEST_TERMS)],
        dtype=np.float32,
    )


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-9
    return (v / n).astype(np.float32)


def pool_chunk_vectors(chunk_vecs: np.ndarray, *, keep_first: int = DEFAULT_KEEP_FIRST) -> np.ndarray:
    """Return 1536-d vector: mean(all) || mean(first k), each L2-normalised."""
    if chunk_vecs.size == 0:
        z = np.zeros(EMBED_DIM, dtype=np.float32)
        return np.concatenate([z, z])
    mean_all = _l2_normalize(chunk_vecs.mean(axis=0))
    head = chunk_vecs[: max(1, keep_first)].mean(axis=0)
    mean_head = _l2_normalize(head)
    return np.concatenate([mean_all, mean_head]).astype(np.float32)


# ---------------------------------------------------------------------------
# Document features
# ---------------------------------------------------------------------------
def document_features(
    title: str,
    abstract: str,
    text: str,
    encoder: PubMedBertEncoder,
    *,
    include_fulltext: bool = True,
    include_aux: bool = False,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    keep_first: int = DEFAULT_KEEP_FIRST,
    batch_size: int | None = None,
    # legacy kwargs ignored
    max_chars: int | None = None,
    overlap_chars: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    del max_chars, overlap_chars
    feats, meta = document_features_batch(
        [{"title": title, "abstract": abstract, "text": text}],
        encoder,
        include_fulltext=include_fulltext,
        include_aux=include_aux,
        chunk_tokens=chunk_tokens,
        max_chunks=max_chunks,
        keep_first=keep_first,
        batch_size=batch_size,
    )
    return feats[0], meta[0]


def document_features_batch(
    records: Iterable[dict[str, Any]],
    encoder: PubMedBertEncoder,
    *,
    include_fulltext: bool = True,
    include_aux: bool = False,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    keep_first: int = DEFAULT_KEEP_FIRST,
    batch_size: int | None = None,
    show_progress: bool = True,
    # legacy kwargs ignored
    max_chars: int | None = None,
    overlap_chars: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Encode many documents. Returns (n, 1536[+2]) and per-doc metadata.

    Columns: [mean_all 768 | mean_first_k 768 | optional aux 2].
    """
    del max_chars, overlap_chars
    records = list(records)
    tok = encoder.tokenizer
    all_chunks: list[str] = []
    owners: list[int] = []
    chunk_counts: list[int] = []
    metas: list[dict[str, Any]] = []

    for i, r in enumerate(records):
        title = str(r.get("title") or "")
        abstract = str(r.get("abstract") or "")
        text = str(r.get("text") or "") if include_fulltext else ""
        has_text = bool(str(r.get("text") or "").strip())
        doc = compose_document_text(title, abstract, text)
        chunks = chunk_text(
            doc,
            tok,
            chunk_tokens=chunk_tokens,
            max_chunks=max_chunks,
            keep_first=keep_first,
        )
        all_chunks.extend(chunks)
        owners.extend([i] * len(chunks))
        chunk_counts.append(len(chunks))
        metas.append(
            {
                "full_text_available": bool(has_text),
                "n_chunks": len(chunks),
                "text_chars": len(str(r.get("text") or "")),
            }
        )

    if all_chunks:
        chunk_vecs = encoder.encode_texts(
            all_chunks,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    else:
        chunk_vecs = np.zeros((0, encoder.embedding_dim), dtype=np.float32)

    n = len(records)
    mean_all = np.zeros((n, EMBED_DIM), dtype=np.float32)
    mean_head = np.zeros((n, EMBED_DIM), dtype=np.float32)
    offset = 0
    for i in range(n):
        c = chunk_counts[i]
        if c:
            vecs = chunk_vecs[offset : offset + c]
            pooled = pool_chunk_vectors(vecs, keep_first=keep_first)
            mean_all[i] = pooled[:EMBED_DIM]
            mean_head[i] = pooled[EMBED_DIM:]
            offset += c

    parts = [mean_all, mean_head]
    if include_aux:
        aux = np.stack(
            [
                aux_features(
                    str(r.get("title") or ""),
                    str(r.get("abstract") or ""),
                    str(r.get("text") or ""),
                )
                for r in records
            ]
        )
        parts.append(aux)
    features = np.concatenate(parts, axis=1).astype(np.float32)
    return features, metas


def feature_dim(include_fulltext: bool = True, include_aux: bool = False) -> int:
    # include_fulltext kept for API compatibility; pooling always uses composed doc.
    del include_fulltext
    return FEATURE_DIM_C if include_aux else FEATURE_DIM_B


# ---------------------------------------------------------------------------
# Shipped linear head (casefilter-head-v3) helpers for joblib bundles
# ---------------------------------------------------------------------------
class IdentityScaler:
    """No-op scaler: shipped head expects raw L2-pooled features."""

    def fit(self, X: np.ndarray, y: Any = None) -> "IdentityScaler":
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)


class ShippedLogisticHead:
    """Logistic head with fixed coef/intercept (no sklearn dependency at score time)."""

    def __init__(self, coef: np.ndarray, intercept: float):
        self.coef_ = np.asarray(coef, dtype=np.float64).reshape(1, -1)
        self.intercept_ = np.asarray([float(intercept)], dtype=np.float64)
        self.classes_ = np.asarray([0, 1])
        self.n_features_in_ = int(self.coef_.shape[1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        logits = X @ self.coef_.ravel() + float(self.intercept_[0])
        p1 = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - p1, p1])


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="load model and encode a few texts")
    parser.add_argument("--text", type=str, default=None, help="encode one piece of text")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    enc = load_encoder(args.model, device=args.device)
    print(f"device={enc.device} dim={enc.embedding_dim} model={enc.model_path}")
    print(f"metadata={enc.metadata}")

    if args.text:
        vec = enc.encode_texts([args.text])[0]
        print(f"text vector shape={vec.shape} norm={float(np.linalg.norm(vec)):.4f}")
        return

    if args.smoke:
        sample = (
            "A prenatal case report of a fetus with a de novo pathogenic variant "
            "in the COL1A1 gene detected by exome sequencing. The parents were "
            "tested and the variant was confirmed by Sanger sequencing."
        )
        chunks = chunk_text(sample, enc.tokenizer)
        print(f"chunks={len(chunks)} lengths={[len(c) for c in chunks]}")
        vec = enc.encode_texts([sample])[0]
        feats, meta = document_features(
            "Prenatal case report",
            "A novel pathogenic variant detected by exome sequencing.",
            sample,
            enc,
            include_aux=True,
        )
        print(f"embedding_shape={vec.shape} embedding_norm={np.linalg.norm(vec):.4f}")
        print(f"feature_shape={feats.shape} meta={meta}")
        print(f"finite={np.isfinite(feats).all()}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
