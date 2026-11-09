#!/usr/bin/env python3
"""Baselines and cheap ablations for the case-only design classifier.

All methods use the same 5-fold splits as ``finetune_case_dora.py``
(StratifiedKFold seed 42; inner 12% validation split for threshold selection).

Methods
-------
- majority (always non-case)
- TF-IDF (word 1-2gram) + logistic regression: title, title+abstract
- frozen PubMedBERT chunk-pool features + logistic regression:
    title, title+abstract, full text (mean+head), full text (mean only),
    full text truncated to one 512-token chunk
- DoRA full text (existing CV JSON) and DoRA title+abstract (optional CV JSON)

Usage
-----
    .venv/bin/python src/scripts/case_baselines.py
    .venv/bin/python src/scripts/case_baselines.py --limit 120   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from encoder import document_features_batch, load_encoder  # noqa: E402
from finetune_case_dora import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_OUT,
    evaluate_scores,
    load_examples,
    threshold_max_f1,
)
from finetune_chunk_fair import resolve_model_path  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

FOLDS = 5
VAL_FRAC = 0.12
SEED = 42


def split_indices(y: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    tr_idx, te_idx = list(skf.split(np.zeros(len(y)), y))[fold]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED + fold)
    tr_rel, va_rel = next(sss.split(np.zeros(len(tr_idx)), y[tr_idx]))
    return tr_idx[tr_rel], tr_idx[va_rel], te_idx


def fit_lr(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray) -> dict[str, Any]:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
    )
    pipe.fit(X[tr], y[tr])
    va_s = pipe.predict_proba(X[va])[:, 1]
    te_s = pipe.predict_proba(X[te])[:, 1]
    thr = threshold_max_f1(y[va], va_s)
    m = evaluate_scores(y[te], te_s, thr)
    return {"scores": te_s, "metrics": m, "threshold": thr, "te": te}


def fit_tfidf(texts: list[str], y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray) -> dict[str, Any]:
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True),
        LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"),
    )
    tr_t = [texts[i] for i in tr]
    va_t = [texts[i] for i in va]
    te_t = [texts[i] for i in te]
    pipe.fit(tr_t, y[tr])
    va_s = pipe.predict_proba(va_t)[:, 1]
    te_s = pipe.predict_proba(te_t)[:, 1]
    thr = threshold_max_f1(y[va], va_s)
    return {"scores": te_s, "metrics": evaluate_scores(y[te], te_s, thr), "threshold": thr, "te": te}


def summarise(name: str, y: np.ndarray, fold_results: list[dict[str, Any]], *, baseline: bool = False) -> dict[str, Any]:
    if baseline:
        return {
            "method": name,
            "accuracy": float(accuracy_score(y, np.zeros_like(y))),
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    oof_scores = np.zeros(len(y))
    for r in fold_results:
        oof_scores[r["te"]] = r["scores"]
    oof_thr = threshold_max_f1(y, oof_scores)
    oof = evaluate_scores(y, oof_scores, oof_thr)
    def agg(key: str) -> tuple[float, float]:
        vals = [r["metrics"][key] for r in fold_results]
        return float(np.mean(vals)), float(np.std(vals))
    out: dict[str, Any] = {"method": name, "oof": oof, "oof_threshold": oof_thr}
    for key in ("ap", "roc_auc", "precision", "recall", "f1"):
        mean, std = agg(key)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    model_path = resolve_model_path(None)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    examples, sub, stats = load_examples(tokenizer, args.data, "full")
    if args.limit:
        rng = np.random.RandomState(SEED)
        keep = rng.choice(len(examples), size=min(args.limit, len(examples)), replace=False)
        examples = [examples[int(i)] for i in keep]
        sub = sub.iloc[keep].reset_index(drop=True)
    y = np.asarray([e.label for e in examples], dtype=int)
    print(f"rows={len(y)} pos={int(y.sum())} stats={stats}", flush=True)

    titles = sub["title"].fillna("").astype(str).tolist()
    abstracts = sub["abstract"].fillna("").astype(str).tolist()
    texts = sub["text"].fillna("").astype(str).tolist()
    title_abs = [f"{t} \n {a}" for t, a in zip(titles, abstracts)]

    encoder = load_encoder(model_path, device=device, max_length=512, normalize=True)
    records = {
        "title": [{"title": t, "abstract": "", "text": ""} for t in titles],
        "title_abstract": [{"title": t, "abstract": a, "text": ""} for t, a in zip(titles, abstracts)],
        "full": [{"title": t, "abstract": a, "text": x} for t, a, x in zip(titles, abstracts, texts)],
    }
    feats: dict[str, np.ndarray] = {}
    for name, recs in records.items():
        f, _ = document_features_batch(
            recs,
            encoder,
            include_fulltext=(name == "full"),
            chunk_tokens=480,
            max_chunks=12,
            keep_first=2,
            batch_size=16,
        )
        feats[name] = np.asarray(f, dtype=np.float32)
        print(f"features {name}: {feats[name].shape}", flush=True)
    f, _ = document_features_batch(
        records["full"],
        encoder,
        include_fulltext=True,
        chunk_tokens=512,
        max_chunks=1,
        keep_first=1,
        batch_size=16,
    )
    feats["single512"] = np.asarray(f, dtype=np.float32)
    print(f"features single512: {feats['single512'].shape}", flush=True)

    results: dict[str, Any] = {}
    results["majority"] = summarise("majority", y, [], baseline=True)

    fold_metrics: dict[str, list[dict[str, Any]]] = {k: [] for k in ("tfidf_title", "tfidf_title_abstract")}
    for fold in range(FOLDS):
        tr, va, te = split_indices(y, fold)
        for name, corpus in (("tfidf_title", titles), ("tfidf_title_abstract", title_abs)):
            r = fit_tfidf(corpus, y, tr, va, te)
            fold_metrics[name].append(r)
        print(f"fold {fold}: tfidf done", flush=True)
    for name in ("tfidf_title", "tfidf_title_abstract"):
        results[name] = summarise(name, y, fold_metrics[name])

    for name, X in (
        ("frozen_title", feats["title"]),
        ("frozen_title_abstract", feats["title_abstract"]),
        ("frozen_full", feats["full"]),
        ("frozen_full_mean", feats["full"][:, :768]),
        ("frozen_single512", feats["single512"][:, :768]),
    ):
        folds: list[dict[str, Any]] = []
        for fold in range(FOLDS):
            tr, va, te = split_indices(y, fold)
            folds.append(fit_lr(X, y, tr, va, te))
        results[name] = summarise(name, y, folds)
        print(f"{name}: AP={results[name]['ap_mean']:.4f} F1={results[name]['f1_mean']:.4f}", flush=True)

    # Existing DoRA CV runs
    for name, path in (
        ("dora_full", args.out_dir / "cv"),
        ("dora_title_abstract", args.out_dir.parent / "case_dora_abl_title_abstract" / "cv"),
    ):
        files = sorted(path.glob("fold_*.json")) if path.exists() else []
        if len(files) != FOLDS:
            print(f"skip {name}: need {FOLDS} folds in {path}", flush=True)
            continue
        folds = [json.loads(p.read_text()) for p in files]
        d = {
            "method": name,
            "fold_test_ap_mean": float(np.mean([f["test_ap"] for f in folds])),
            "fold_test_ap_std": float(np.std([f["test_ap"] for f in folds])),
            "fold_test_f1_mean": float(np.mean([f["test_at_f1_thr"]["f1"] for f in folds])),
            "fold_test_f1_std": float(np.std([f["test_at_f1_thr"]["f1"] for f in folds])),
            "fold_precision_mean": float(np.mean([f["test_at_f1_thr"]["precision"] for f in folds])),
            "fold_recall_mean": float(np.mean([f["test_at_f1_thr"]["recall"] for f in folds])),
            "fold_auc_mean": float(np.mean([f["test_roc_auc"] for f in folds])),
        }
        results[name] = d

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "baselines.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Case-only baselines and ablations",
        "",
        "Identical 5-fold splits as the DoRA model; thresholds selected on each fold's inner 12% validation split.",
        "",
        "| Method | AP | AUC | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = [
        "majority", "tfidf_title", "tfidf_title_abstract",
        "frozen_title", "frozen_title_abstract", "frozen_full_mean",
        "frozen_single512", "frozen_full",
        "dora_title_abstract", "dora_full",
    ]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        if name == "majority":
            lines.append(f"| {name} | — | — | 0.000 | 0.000 | 0.000 |")
            continue
        if name.startswith("dora"):
            lines.append(
                f"| {name} | {r['fold_test_ap_mean']:.3f} ± {r['fold_test_ap_std']:.3f} | "
                f"{r['fold_auc_mean']:.3f} | {r['fold_precision_mean']:.3f} | "
                f"{r['fold_recall_mean']:.3f} | {r['fold_test_f1_mean']:.3f} ± {r['fold_test_f1_std']:.3f} |"
            )
        else:
            lines.append(
                f"| {name} | {r['ap_mean']:.3f} ± {r['ap_std']:.3f} | "
                f"{r['roc_auc_mean']:.3f} | {r['precision_mean']:.3f} | "
                f"{r['recall_mean']:.3f} | {r['f1_mean']:.3f} ± {r['f1_std']:.3f} |"
            )
    lines += ["", "OOF metrics for the same methods are in `baselines.json`.", ""]
    (args.out_dir / "BASELINES.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
