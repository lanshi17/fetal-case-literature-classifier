"""Standalone inference with the exported weights (local embedding + logistic head, no Claude)."""
import csv
import json
import re
from pathlib import Path

import numpy as np

from .embed import Embedder, pool
from .fulltext import best_text, STUB_CHARS


def load_head(weights_dir="weights"):
    d = Path(weights_dir)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    z = np.load(d / cfg["head"]["file"])
    return cfg, z["coef"], float(z["intercept"][0])


def collect(folder):
    """Group files in a folder by stem; returns {stem: {'xml':..,'txt':..,'pdf':..}} (skips macOS '._' files)."""
    papers = {}
    for p in sorted(Path(folder).rglob("*")):
        ext = p.suffix.lower().lstrip(".")
        if ext in ("xml", "txt", "pdf") and not p.name.startswith("._"):
            papers.setdefault(str(p.with_suffix("")), {})[ext] = p
    return papers


def predict_folder(folder, weights_dir="weights", model=None, device="cpu", out_csv=None):
    cfg, coef, b = load_head(weights_dir)
    papers = collect(folder)
    keys, texts, srcs = [], [], []
    for k, files in papers.items():
        t, src = best_text(files)
        keys.append(k); texts.append(t); srcs.append(src)
    emb = Embedder(model or cfg["embedding_model"], device=device)
    X = pool(emb.embed_docs(texts, max_chunks=cfg["max_chunks"]))
    prob = 1.0 / (1.0 + np.exp(-(X @ coef + b)))
    rows = []
    for k, t, s, p in zip(keys, texts, srcs, prob):
        m = re.search(r"(\d{5,9})", Path(k).name)
        rows.append(dict(file=Path(k).name, pmid=m.group(1) if m else "", source=s, n_chars=len(t),
                         stub=len(t) < STUB_CHARS, prob=round(float(p), 4), label=int(p >= cfg["threshold"])))
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    return rows
