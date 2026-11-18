#!/usr/bin/env python3
"""Section-aware encoding and attention pooling (frozen PubMedBERT).

Compares three document representations on the 864 design-labelled papers with
identical folds:

flat       L2(mean all chunks) || L2(mean first 2)  -> 1536-d (current protocol)
section    per-section means (head/methods/results/discussion), L2, concat -> 3072-d
attention  learned attention over chunk vectors -> 768-d -> linear head

Flat and section use logistic regression; attention trains a small torch head.
Thresholds are selected on an inner validation split, then applied to the test
fold. This isolates representation/pooling from DoRA fine-tuning.

Outputs: artifacts/case_structured_pooling/{metrics.json,report.md}
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from encoder import chunk_text  # noqa: E402
from finetune_chunk_fair import resolve_model_path  # noqa: E402

DATA = ROOT / "data" / "labeled_papers.csv"
OUT = ROOT / "artifacts" / "case_structured_pooling"
CASE_RE = r"^case\s+(report|series)\b"
NEG_PATTERN = r"cohort|screening|review|method|validation|registry"
MAX_CHUNKS = 12
CHUNK_TOKENS = 480
SECTIONS = ("head", "methods", "results", "discussion")
HEAD_RE = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?(abstract|introduction|background|methods?|materials and methods|"
    r"patients and methods|subjects and methods|results?|discussion|conclusions?)\s*:?\s*$"
)


def split_sections(title: str, abstract: str, text: str) -> dict[str, str]:
    head = ". ".join(x for x in [str(title or "").strip(), str(abstract or "").strip()] if x)
    body = str(text or "")
    marks = [(m.start(), m.group(1).lower()) for m in HEAD_RE.finditer(body)]
    out = {k: "" for k in SECTIONS}
    out["head"] = head
    if not marks:
        return out
    for i, (start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(body)
        if name.startswith("method") or name in ("materials and methods", "patients and methods", "subjects and methods"):
            key = "methods"
        elif name.startswith("result"):
            key = "results"
        elif name.startswith("discussion") or name.startswith("conclusion"):
            key = "discussion"
        else:
            continue
        if not out[key]:
            out[key] = body[start:end]
    return out


def encode_sections(rows: pd.DataFrame, tok: Any, st: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode sections in one batched pass, then pool per section and per document."""
    section_chunks: list[list[list[str]]] = []
    for r in rows.itertuples(index=False):
        sections = split_sections(r.title, r.abstract, r.text)
        per_doc = []
        for key in SECTIONS:
            txt = sections[key]
            per_doc.append(chunk_text(txt, tok, chunk_tokens=CHUNK_TOKENS, max_chunks=2, keep_first=2) if txt.strip() else [])
        section_chunks.append(per_doc)

    flat = [c for doc in section_chunks for sec in doc for c in sec]
    print(f"  encoding {len(flat)} section chunks", flush=True)
    vecs = st.encode(flat, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sec_vecs = np.zeros((len(rows), len(SECTIONS), 768), dtype=np.float32)
    flat_all = np.zeros((len(rows), 768), dtype=np.float32)
    flat_head = np.zeros((len(rows), 768), dtype=np.float32)
    cursor = 0
    for i, doc in enumerate(section_chunks):
        doc_vecs = []
        for j, chunks in enumerate(doc):
            if not chunks:
                continue
            v = vecs[cursor:cursor + len(chunks)]
            cursor += len(chunks)
            sec_vecs[i, j] = v.mean(axis=0)
            doc_vecs.append(v)
        if doc_vecs:
            merged = np.concatenate(doc_vecs, axis=0)
            flat_all[i] = merged.mean(axis=0)
            flat_head[i] = merged[:2].mean(axis=0)
    for arr in (sec_vecs, flat_all, flat_head):
        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        norm[norm == 0] = 1.0
        arr /= norm
    flat_feat = np.concatenate([flat_all, flat_head], axis=1)
    section = sec_vecs.reshape(len(rows), -1)
    return flat_feat, section, sec_vecs


def encode_documents(rows: pd.DataFrame, tok: Any, st: Any) -> list[np.ndarray]:
    chunks = [
        chunk_text(f"{r.title}. {r.abstract}\n\n{r.text}", tok, chunk_tokens=CHUNK_TOKENS, max_chunks=MAX_CHUNKS, keep_first=2)
        for r in rows.itertuples(index=False)
    ]
    flat = [c for cs in chunks for c in cs]
    print(f"  encoding {len(flat)} document chunks", flush=True)
    vecs = st.encode(flat, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    out, cursor = [], 0
    for cs in chunks:
        out.append(vecs[cursor:cursor + len(cs)].astype(np.float32))
        cursor += len(cs)
    return out


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    ts = np.unique(np.concatenate(([0.0], (np.unique(s)[:-1] + np.unique(s)[1:]) / 2, [1.0])))
    best = (-1.0, 0.5)
    for t in ts:
        p = s >= t
        tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def score(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, float]:
    p = s >= thr
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return {"ap": float(average_precision_score(y, s)), "auc": float(roc_auc_score(y, s)),
            "precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-9), "threshold": float(thr)}


class AttentionPool(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.att = nn.Linear(dim, 1)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = self.att(x).squeeze(-1).masked_fill(~mask, -1e9).softmax(dim=-1)
        v = (x * a.unsqueeze(-1)).sum(dim=1)
        return self.head(v).squeeze(-1)


def train_attention(chunk_lists: list[np.ndarray], y: np.ndarray, idx: np.ndarray, *, epochs: int = 40, device: str = "cuda") -> AttentionPool:
    model = AttentionPool().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    pos_weight = torch.tensor([(len(idx) - y[idx].sum()) / max(y[idx].sum(), 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(idx)
        for start in range(0, len(order), 16):
            batch_idx = order[start:start + 16]
            seqs = [torch.tensor(chunk_lists[i], dtype=torch.float32) for i in batch_idx]
            maxlen = max(len(s) for s in seqs)
            x = torch.zeros(len(seqs), maxlen, 768, device=device)
            mask = torch.zeros(len(seqs), maxlen, dtype=torch.bool, device=device)
            for k, s in enumerate(seqs):
                x[k, :len(s)] = s.to(device); mask[k, :len(s)] = True
            logits = model(x, mask)
            loss = lossf(logits, torch.tensor(y[batch_idx], dtype=torch.float32, device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def predict_attention(model: AttentionPool, chunk_lists: list[np.ndarray], idx: np.ndarray, device: str = "cuda") -> np.ndarray:
    out = np.empty(len(idx))
    with torch.inference_mode():
        for start in range(0, len(idx), 32):
            batch = idx[start:start + 32]
            seqs = [torch.tensor(chunk_lists[i], dtype=torch.float32) for i in batch]
            maxlen = max(len(s) for s in seqs)
            x = torch.zeros(len(seqs), maxlen, 768, device=device)
            mask = torch.zeros(len(seqs), maxlen, dtype=torch.bool, device=device)
            for k, s in enumerate(seqs):
                x[k, :len(s)] = s.to(device); mask[k, :len(s)] = True
            out[start:start + len(batch)] = model(x, mask).sigmoid().cpu().numpy()
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, low_memory=False, dtype={"pmid": str})
    design = df["design"].fillna("").astype(str).str.strip().str.lower()
    pos = design.str.match(CASE_RE)
    neg = design.str.contains(NEG_PATTERN, regex=True) & ~pos
    rows = df[pos | neg].reset_index(drop=True)
    y = pos[pos | neg].astype(int).to_numpy()
    print(f"rows {len(rows)} pos {int(y.sum())}", flush=True)

    model_path = resolve_model_path(None)
    tok = AutoTokenizer.from_pretrained(model_path)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    flat, section, sec_vecs = encode_sections(rows, tok, st)
    np.savez_compressed(OUT / "features.npz", flat=flat, section=section, sec_vecs=sec_vecs, y=y,
                        pmid=rows["pmid"].astype(str).to_numpy())
    # chunks in a single list per doc for the attention head
    chunk_lists = encode_documents(rows, tok, st)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {"flat": [], "section": [], "attention": []}
    for fold, (tr, te) in enumerate(skf.split(flat, y)):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=42 + fold)
        rel_tr, rel_va = next(sss.split(tr.reshape(-1, 1), y[tr]))
        tr_i, va_i = tr[rel_tr], tr[rel_va]
        for name, X in (("flat", flat), ("section", section)):
            clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
            clf.fit(X[tr_i], y[tr_i])
            va = clf.predict_proba(X[va_i])[:, 1]
            te_s = clf.predict_proba(X[te])[:, 1]
            thr = best_threshold(y[va_i], va)
            results[name].append(score(y[te], te_s, thr))
        model = train_attention(chunk_lists, y, tr_i, device=device)
        va = predict_attention(model, chunk_lists, va_i, device)
        te_s = predict_attention(model, chunk_lists, te, device)
        thr = best_threshold(y[va_i], va)
        results["attention"].append(score(y[te], te_s, thr))
        print(f"fold {fold} done", flush=True)

    summary = {}
    for name, rows_ in results.items():
        summary[name] = {
            "ap_mean": float(np.mean([r["ap"] for r in rows_])),
            "ap_std": float(np.std([r["ap"] for r in rows_])),
            "f1_mean": float(np.mean([r["f1"] for r in rows_])),
            "f1_std": float(np.std([r["f1"] for r in rows_])),
            "per_fold": rows_,
        }
    (OUT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Section-aware pooling (frozen encoder)", "",
             "| representation | AP | F1 |", "|---|---:|---:|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['ap_mean']:.3f} ± {s['ap_std']:.3f} | {s['f1_mean']:.3f} ± {s['f1_std']:.3f} |")
    lines += ["", "5-fold, same splits; thresholds selected on each inner validation split.",
              "The current DoRA model is end-to-end fine-tuned, so this table compares pooling strategies on frozen features, not against DoRA."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
