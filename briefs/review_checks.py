#!/usr/bin/env python3
"""Reproducible scenario checker for the dsh SPECTER2 deliverable (review only).

Every scenario comes from `briefs/REVIEW_SPEC.md`. The checker fails closed:
an unexpected exception in a scenario is a FAIL, and the process exits nonzero
if any scenario fails. It writes JSON + human-readable lines.

Usage:
    .venv/bin/python briefs/review_checks.py [--rerun-dir /tmp/omp-review-train]

`--rerun-dir` must contain the output of a cached `train.py` re-run
(metrics.json/config.json) used by S16.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "labeled_papers.csv"
LABELS = ROOT / "data" / "final_labels.csv"
PACKAGE = ROOT / "data" / "fulltext_package"
ARTIFACTS = ROOT / "artifacts"

RESULTS: list[dict[str, Any]] = []


def record(sid: str, status: str, detail: str) -> None:
    RESULTS.append({"id": sid, "status": status, "detail": detail})
    print(f"{status:4s} {sid}: {detail}", flush=True)


def scenario(sid: str) -> Callable[[Callable[[], str]], Callable[[], None]]:
    def deco(fn: Callable[[], str]) -> Callable[[], None]:
        def wrapper(*args: Any, **kwargs: Any) -> None:
            try:
                detail = fn(*args, **kwargs)
                record(sid, "PASS", detail)
            except SkipCheck as exc:
                record(sid, "SKIP", str(exc))
            except Exception as exc:  # noqa: BLE001 - fail closed
                record(sid, "FAIL", f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
        wrapper.__name__ = fn.__name__
        return wrapper

    return deco


class SkipCheck(Exception):
    pass


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_frames():
    labels = pd.read_csv(LABELS, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    out = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    return labels, out


def as_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


# ---------------------------------------------------------------------------
# S01
# ---------------------------------------------------------------------------
@scenario("S01")
def s01_join() -> str:
    labels, out = load_frames()
    need(len(out) == 974, f"expected 974 output rows, got {len(out)}")
    need(out["doc_id"].is_unique, "duplicate doc_id")
    need((out["doc_id"] == out["group_id"]).all(), "doc_id != group_id")
    need((out["doc_id"] == out["pmid"]).all(), "doc_id != pmid")
    need(set(labels["pmid"]) == set(out["pmid"]), "pmid sets differ")
    m = labels.set_index("pmid")
    o = out.set_index("pmid")
    need(int((m.loc[o.index, "label"].astype(int) != o["label"].astype(int)).sum()) == 0, "label mismatch")
    need(int((m.loc[o.index, "reason"].fillna("") != o["label_reason"].fillna("")).sum()) == 0, "reason mismatch")
    need(
        int(
            (
                m.loc[o.index, "weak"].map(as_bool).astype(bool).to_numpy()
                != o["weak"].map(as_bool).astype(bool).to_numpy()
            ).sum()
        )
        == 0,
        "weak mismatch",
    )
    manifest = pd.read_csv(PACKAGE / "manifest_fulltext.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    need(set(manifest["PMID"].astype(str)) >= set(out["pmid"]), "some PMID missing from manifest")
    return f"974/974 rows, unique doc_id, doc_id==group_id==pmid, labels/reason/weak identical, manifest covers all PMIDs"


# ---------------------------------------------------------------------------
# S02
# ---------------------------------------------------------------------------
@scenario("S02")
def s02_filter() -> str:
    labels, out = load_frames()
    weak = labels["weak"].map(as_bool)
    model = labels["label_source"].str.strip().str.lower().eq("model")
    default_n = int((~weak & ~model).sum())
    config = json.loads((ARTIFACTS / "config.json").read_text())
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    need(default_n == 883, f"independent default subset = {default_n}, expected 883")
    need(metrics["data"]["n_default_subset"] == default_n, "metrics n_default_subset mismatch")
    need("weak=False" in config["train_subset"] and "model" in config["train_subset"], "config train_subset not declared")
    need(metrics["full_label_control"] is not None, "full-label control missing")
    need(metrics["full_label_control"]["train_subset"] == "all", "full-label control not on all rows")
    return f"default subset {default_n} (weak={int(weak.sum())}, model={int(model.sum())}); full-label control present"


# ---------------------------------------------------------------------------
# S03
# ---------------------------------------------------------------------------
@scenario("S03")
def s03_fulltext_semantics(enc) -> str:
    _, out = load_frames()
    text_len = out["text"].str.len()
    avail = out["full_text_available"].map(as_bool)
    need(bool((avail == (text_len > 0)).all()), "full_text_available disagrees with len(text)")
    need(set(out["text_source"]) <= {"txt", "pdf", "none"}, f"unexpected text_source {set(out['text_source'])}")
    need(int(((out["text_source"] == "none") & avail).sum()) == 0, "text_source=none but full_text_available=True")
    from encoder import document_features

    feats, meta = document_features("Prenatal variant report", "A de novo variant.", "", enc, include_aux=True)
    need(meta["full_text_available"] is False, "empty text reported as full text")
    need(meta["n_chunks"] == 0, "empty text produced chunks")
    need(bool(np.allclose(feats[768:2304], 0.0)), "body blocks not zero without full text")
    need(bool(np.any(feats[:768] != 0.0)), "title+abstract block unexpectedly zero")
    return f"974 rows consistent; empty-text probe -> full_text_available=False, n_chunks=0, mean/max zero, TA non-zero"


# ---------------------------------------------------------------------------
# S04
# ---------------------------------------------------------------------------
@scenario("S04")
def s04_abstracts() -> str:
    import build_dataset as bd

    labels, out = load_frames()
    stats = json.loads((ROOT / "data" / "build_stats.json").read_text())
    nonempty = int((out["abstract"].str.len() > 0).sum())
    need(nonempty == stats["with_abstract"] == 764, f"abstract count {nonempty} != 764")
    manifest = pd.read_csv(PACKAGE / "manifest_fulltext.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    man = {str(r["PMID"]): r for _, r in manifest.iterrows()}
    checked = 0
    for _, row in out.head(40).iterrows():
        m = man.get(row["pmid"])
        if m is None:
            continue
        xml = bd.resolve_member(PACKAGE, m.get("XML_file", ""), m.get("XML", ""), "xml", row["pmid"])
        extracted = bd.extract_jats_abstract(xml) if xml else ""
        if extracted:
            need(extracted == row["abstract"], f"abstract mismatch for PMID {row['pmid']}")
            checked += 1
    empties = out[out["abstract"].str.len() == 0]
    xml_empties = 0
    for _, row in empties.iterrows():
        m = man.get(row["pmid"])
        if m is not None and bd.resolve_member(PACKAGE, m.get("XML_file", ""), m.get("XML", ""), "xml", row["pmid"]):
            xml_empties += 1
    need(int((out["text_source"] == "pdf").sum()) == 206, "PDF-sourced rows != 206")
    return (
        f"764 abstracts; {checked} sampled XML abstracts re-extract identically; "
        f"{len(empties)} empty ({int((out['text_source'] == 'pdf').sum())} PDF-only + {xml_empties} XML without abstract)"
    )


# ---------------------------------------------------------------------------
# S05
# ---------------------------------------------------------------------------
@scenario("S05")
def s05_encoder(enc) -> str:
    need(enc.metadata["adapter_loaded"] is True, f"adapter not loaded: {enc.metadata['adapter_error']}")
    need("CLF" in str(enc.metadata.get("active_adapter_setup", "")), f"no active CLF adapter: {enc.metadata}")
    need(enc.embedding_dim == 768, f"embedding dim {enc.embedding_dim}")
    trainable = [n for n, p in enc.model.named_parameters() if p.requires_grad]
    need(not trainable, f"{len(trainable)} trainable params (e.g. {trainable[:3]})")
    need(enc.model.training is False, "model not in eval mode")

    from encoder import document_features, feature_dim

    need(feature_dim(False, False) == 768, "A dim")
    need(feature_dim(True, False) == 2304, "B dim")
    need(feature_dim(True, True) == 2306, "C dim")

    text = "A prenatal case report of a fetus with a de novo pathogenic COL1A1 variant detected by exome sequencing."
    with_adapter = enc.encode_texts([text])[0]
    stack = str(enc.model.active_adapters)
    name = stack.removeprefix("Stack[").removesuffix("]")
    enc.model.set_active_adapters(None)
    try:
        base_only = enc.encode_texts([text])[0]
    finally:
        enc.model.set_active_adapters(name)
    cos = float(np.dot(with_adapter, base_only))
    need(cos < 0.999, f"base-only and adapted embeddings nearly identical (cos={cos:.4f}) -> adapter not in forward path")
    need(str(enc.model.active_adapters) == stack, "adapter stack not restored")
    detail = f"cos(adapted, base-only)={cos:.4f} (adapter={name})"

    feats, _ = document_features("t", "a", "body text", enc, include_aux=True)
    need(feats.shape == (2306,), f"feature shape {feats.shape}")
    import joblib

    bundle = joblib.load(ARTIFACTS / "model.joblib")
    need(int(bundle["model"].n_features_in_) == 2306, "model expects different dim")
    cfg = json.loads((ARTIFACTS / "config.json").read_text())
    need(cfg["feature_dim"] == 2306 and cfg["feature_mode"] == "C", "config dim/mode mismatch")
    return f"adapter CLF active, 0 trainable params, dims 768/2304/2306, {detail}, model n_features_in_=2306"


# ---------------------------------------------------------------------------
# S06
# ---------------------------------------------------------------------------
@scenario("S06")
def s06_chunking() -> str:
    from encoder import chunk_text

    long_text = "HEADMARK " + ("Fetal ultrasound findings and genetic testing results were reviewed. " * 60)
    long_text += "\n\n" + ("Karyotype and microarray results were concordant. " * 60) + " TAILMARK"
    chunks = chunk_text(long_text, max_chars=3000, overlap_chars=400, max_chunks=24)
    need(0 < len(chunks) <= 24, f"{len(chunks)} chunks")
    need(max(len(c) for c in chunks) <= 3000, "chunk exceeds max_chars")
    need("HEADMARK" in chunks[0], "head marker missing from first chunk")
    need("TAILMARK" in chunks[-1], "tail marker missing from last chunk")

    short = " ".join(f"Sentence number {i} about prenatal exome sequencing." for i in range(80))
    chunks3 = chunk_text(short, max_chars=1000, overlap_chars=200, max_chunks=24)
    need(len(chunks3) >= 2, "expected multiple chunks for overlap probe")
    overlaps = []
    for a, b in zip(chunks3, chunks3[1:]):
        wa, wb = a.split(), b.split()
        n = 0
        for k in range(1, min(len(wa), len(wb)) + 1):
            if wa[-k:] == wb[:k]:
                n = k
        overlaps.append(n)
    need(max(overlaps) >= 5, f"no meaningful chunk overlap: {overlaps}")

    _, out = load_frames()
    global_max = 0
    total_chunks = 0
    for text in out["text"]:
        cs = chunk_text(text, max_chars=3000, overlap_chars=400, max_chunks=24)
        if cs:
            global_max = max(global_max, max(len(c) for c in cs))
            total_chunks += len(cs)
    need(global_max <= 3000, f"global max chunk len {global_max}")
    return f"24-chunk cap, max len {global_max}/3000 over 974 docs ({total_chunks} chunks), head+tail kept, overlap {max(overlaps)} words"


# ---------------------------------------------------------------------------
# S07
# ---------------------------------------------------------------------------
@scenario("S07")
def s07_split() -> str:
    import train as tr

    df = tr.load_dataset(DATA)
    train_idx, val_idx, test_idx = tr.make_splits(df, seed=42)
    groups = df["group_id"].to_numpy()
    sets = [set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])]
    need(not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]), "group leakage across splits")
    y = df["label"].to_numpy()
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        need(len(np.unique(y[idx])) == 2, f"{name} split single-class")
    subset = tr.default_subset_mask(df).to_numpy()
    sizes = {n: int(subset[idx].sum()) for n, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))}
    need(sizes == {"train": 566, "val": 140, "test": 177}, f"default sizes {sizes}")
    all_sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    need(all_sizes == {"train": 623, "val": 156, "test": 195}, f"all sizes {all_sizes}")
    return f"disjoint PMID groups, both classes; all {all_sizes}, default {sizes}"


# ---------------------------------------------------------------------------
# S08
# ---------------------------------------------------------------------------
@scenario("S08")
def s08_threshold_metrics() -> str:
    from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score, precision_score, recall_score
    import joblib
    import train as tr

    summary = json.loads((ARTIFACTS / "train_summary.json").read_text())
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    pred = pd.read_csv(ARTIFACTS / "test_predictions.csv")
    need(len(pred) == 177, f"test_predictions rows {len(pred)}")
    t = float(summary["threshold"])
    y = pred["label"].to_numpy(int)
    s = pred["score"].to_numpy(float)
    p = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    block = summary["test"]
    recomputed = {
        "recall": recall_score(y, p),
        "precision": precision_score(y, p),
        "f1": fbeta_score(y, p, beta=1),
        "f2": fbeta_score(y, p, beta=2),
    }
    for k, v in recomputed.items():
        need(abs(v - block[k]) < 1e-9, f"{k}: recomputed {v} vs reported {block[k]}")
    need(abs(average_precision_score(y, s) - block["ap"]) < 1e-9, "AP mismatch")
    need(
        (int(tn), int(fp), int(fn), int(tp))
        == (block["confusion_matrix"]["tn"], block["confusion_matrix"]["fp"], block["confusion_matrix"]["fn"], block["confusion_matrix"]["tp"]),
        "confusion matrix mismatch",
    )
    val_recall = metrics["primary"]["val"]["recall"]
    need(val_recall >= 0.95 - 1e-9, f"validation recall {val_recall} < 0.95")

    # Reproduce the threshold-selection argmax from cached features.
    df = tr.load_dataset(DATA)
    arrays = dict(np.load(next(ARTIFACTS.glob("features/*.npz")), allow_pickle=False))
    X = tr.feature_matrix(arrays, "C")
    train_idx, val_idx, test_idx = tr.make_splits(df, seed=42)
    subset = tr.default_subset_mask(df).to_numpy()
    tr_i, va_i, te_i = train_idx[subset[train_idx]], val_idx[subset[val_idx]], test_idx[subset[test_idx]]
    bundle = joblib.load(ARTIFACTS / "model.joblib")
    val_scores = bundle["model"].predict_proba(bundle["scaler"].transform(X[va_i]))[:, 1]
    yv = df["label"].to_numpy()[va_i]
    t_re = tr.select_threshold(yv, val_scores, 0.95)
    need(abs(t_re - t) < 1e-9, f"reproduced threshold {t_re} != delivered {t}")
    cand = np.unique(np.concatenate(([0.0], (np.unique(val_scores)[:-1] + np.unique(val_scores)[1:]) / 2.0, [1.0])))
    best = max(
        (precision_score(yv, val_scores >= c, zero_division=0) for c in cand if recall_score(yv, val_scores >= c, zero_division=0) >= 0.95),
        default=-1.0,
    )
    need(abs(best - precision_score(yv, val_scores >= t, zero_division=0)) < 1e-9, "threshold is not max-precision at target recall")
    return (
        f"val recall {val_recall:.4f}>=0.95; threshold {t:.6f} reproduced; "
        f"test recall {recomputed['recall']:.4f} (below 0.95, reported), AP/confusion reproduce"
    )


# ---------------------------------------------------------------------------
# S09 / S10 (need predictor; encoder reused)
# ---------------------------------------------------------------------------
@scenario("S09")
def s09_bands(enc) -> str:
    from predict import Predictor

    predictor = Predictor(ARTIFACTS / "model.joblib", load_encoder_now=False)
    predictor.encoder = enc
    t = predictor.threshold
    need(predictor.decide(t) == "relevant", "t not relevant")
    need(predictor.decide(t - 0.15) == "review", "t-0.15 not review")
    need(predictor.decide(t - 0.15 + 1e-9) == "review", "inside band not review")
    need(predictor.decide(t - 0.15 - 1e-6) == "irrelevant", "below band not irrelevant")
    res = predictor.predict_record({"title": "Prenatal exome sequencing case report", "abstract": "A de novo variant.", "text": "Case report text about prenatal diagnosis."})
    need(res["decision"] in {"relevant", "review", "irrelevant"}, "bad decision")
    need(np.isfinite(res["relevance_score"]), "score not finite")
    for k in ("decision", "relevance_score", "threshold", "review_threshold", "full_text_available"):
        need(k in res, f"missing output key {k}")
    return f"boundaries exact (t={t:.6f}, t-0.15={t-0.15:.6f}); end-to-end decision={res['decision']} score={res['relevance_score']}"


@scenario("S10")
def s10_degraded(enc) -> str:
    from predict import Predictor

    predictor = Predictor(ARTIFACTS / "model.joblib", load_encoder_now=False)
    predictor.encoder = enc
    title_only = predictor.predict_record({"title": "Prenatal genetic variant screening by exome sequencing"})
    need(title_only["full_text_available"] is False, "title-only claimed full text")
    need(title_only["pdf_used"] is False, "title-only claimed pdf_used")
    need(np.isfinite(title_only["relevance_score"]), "title-only score not finite")
    pdfs = sorted(p for p in (PACKAGE / "pdfs").glob("*.pdf") if not p.name.startswith("._"))
    need(pdfs, "no PDFs in package")
    pdf_res = predictor.predict_pdf(pdfs[0])
    need(pdf_res["pdf_used"] is True, "pdf_used False for PDF input")
    need(pdf_res["full_text_available"] is True, "PDF produced no text")
    missing = predictor.predict_record({"title": "x", "pdf_path": "/nonexistent/file.pdf"})
    need(np.isfinite(missing["relevance_score"]), "missing PDF path crashed")
    return f"title-only -> full_text_available=False score={title_only['relevance_score']}; PDF -> pdf_used=True score={pdf_res['relevance_score']}"


# ---------------------------------------------------------------------------
# S12
# ---------------------------------------------------------------------------
@scenario("S12")
def s12_gitignore() -> str:
    def ignored(path: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0

    npz = next((ARTIFACTS / "features").glob("*.npz"))
    must = [
        "data/fulltext_package/",
        str(npz.relative_to(ROOT)),
        "artifacts/model.joblib",
        ".venv/",
        ".cache/",
    ]
    bad = [p for p in must if not ignored(p)]
    need(not bad, f"not ignored: {bad}")
    need(not ignored("data/final_labels.csv"), "final_labels.csv unexpectedly ignored")
    labeled_ignored = ignored("data/labeled_papers.csv")
    return f"ignored: {must}; final_labels trackable; labeled_papers.csv ignored={labeled_ignored} (27 MB, intended deliverable)"


# ---------------------------------------------------------------------------
# S13
# ---------------------------------------------------------------------------
@scenario("S13")
def s13_no_finetune(enc) -> str:
    needles = [".backward(", "optimizer", "AdamW", "model.train(", "Trainer("]
    hits = {}
    for name in ("train.py", "encoder.py", "predict.py", "api.py", "build_dataset.py"):
        src = (ROOT / name).read_text()
        found = [n for n in needles if n in src]
        if found:
            hits[name] = found
    need(not hits, f"fine-tuning constructs found: {hits}")
    frozen = [n for n, p in enc.model.named_parameters() if p.requires_grad]
    need(not frozen, f"{len(frozen)} trainable params")
    base = Path(json.loads((ARTIFACTS / "config.json").read_text())["encoder"]["base_path"])
    need(base.exists() and (base / "pytorch_model.bin").exists() or any(base.glob("*.safetensors")), "base snapshot weights missing")
    return "no optimizer/backward/train() in pipeline; all encoder params requires_grad=False; only LogisticRegression fitted"


# ---------------------------------------------------------------------------
# S14
# ---------------------------------------------------------------------------
@scenario("S14")
def s14_artifacts() -> str:
    import joblib

    need((ARTIFACTS / "model.joblib").exists(), "model.joblib missing")
    bundle = joblib.load(ARTIFACTS / "model.joblib")
    for k in ("model", "scaler", "threshold", "feature_mode", "encoder"):
        need(k in bundle, f"bundle missing {k}")
    cfg = json.loads((ARTIFACTS / "config.json").read_text())
    for k in ("threshold", "feature_dim", "encoder", "split", "train_subset", "target_recall"):
        need(k in cfg, f"config missing {k}")
    met = json.loads((ARTIFACTS / "metrics.json").read_text())
    need(set(met["ablations"]) >= {"A", "B", "C"}, f"ablations {list(met['ablations'])}")
    for mode, dim in (("A", 768), ("B", 2304), ("C", 2306)):
        need(met["ablations"][mode]["feature_dim"] == dim, f"ablation {mode} dim")
    need(met["full_label_control"] is not None, "full-label control missing")
    need("test_by_full_text" in met["primary"], "stratified report missing")
    test = met["primary"]["test"]
    for k in ("recall", "precision", "f1", "f2", "ap", "roc_auc", "confusion_matrix"):
        need(k in test, f"test block missing {k}")
    need(len(pd.read_csv(ARTIFACTS / "test_predictions.csv")) == 177, "test_predictions rows")
    return "model/config/metrics/test_predictions present; ablations A=768 B=2304 C=2306; stratification + full control present"


# ---------------------------------------------------------------------------
# S15
# ---------------------------------------------------------------------------
@scenario("S15")
def s15_readme() -> str:
    text = (ROOT / "README.md").read_text()
    required = [
        "requirements.txt",
        "build_dataset.py",
        "train.py",
        "predict.py",
        "uvicorn api:app",
        "--workers 1",
        "allenai/specter2_classification",
        "specter2_base",
        "fulltext_package",
        "/health",
        "/predict",
    ]
    missing = [r for r in required if r not in text]
    need(not missing, f"README missing: {missing}")
    return f"README covers install/build/train/predict/API; all {len(required)} required strings present"


# ---------------------------------------------------------------------------
# S16
# ---------------------------------------------------------------------------
@scenario("S16")
def s16_reproducible(rerun_dir: Path | None) -> str:
    npz_path = next((ARTIFACTS / "features").glob("*.npz"))
    arrays = np.load(npz_path, allow_pickle=False)
    _, out = load_frames()
    need(list(arrays["doc_ids"].astype(str)) == out["doc_id"].tolist(), "cache order != dataset order")
    need(bool((arrays["labels"] == out["label"].astype(int).to_numpy()).all()), "cached labels mismatch")
    need(
        bool((arrays["full_text_available"] == out["full_text_available"].map(as_bool).to_numpy()).all()),
        "cached full_text_available mismatch",
    )
    need(arrays["ta"].shape == (974, 768) and arrays["body_mean"].shape == (974, 768) and arrays["aux"].shape == (974, 2), "cache shapes")
    if rerun_dir is None:
        raise SkipCheck("no --rerun-dir given; cached-feature integrity passed but rerun comparison skipped")
    metrics = json.loads(Path(rerun_dir, "metrics.json").read_text())
    delivered = json.loads((ARTIFACTS / "metrics.json").read_text())
    for k in ("recall", "precision", "f1", "f2", "ap", "confusion_matrix"):
        a, b = metrics["primary"]["test"][k], delivered["primary"]["test"][k]
        if isinstance(a, dict):
            need(a == b, f"{k} differs")
        else:
            need(abs(a - b) <= 1e-9, f"{k} differs: {a} vs {b}")
    need(abs(metrics["primary"]["threshold"] - delivered["primary"]["threshold"]) <= 1e-12, "threshold differs")
    need(metrics["split"] == delivered["split"], "split differs")
    return "cache loads with allow_pickle=False, order/labels match; cached rerun reproduces threshold, split and test metrics to 1e-9"


# ---------------------------------------------------------------------------
# S17
# ---------------------------------------------------------------------------
@scenario("S17")
def s17_no_network() -> str:
    src = (ROOT / "build_dataset.py").read_text()
    needles = ["requests.", "urllib", "http://", "https://", "eutils", "Entrez", "esearch", "efetch", "Bio."]
    hits = [n for n in needles if n in src]
    need(not hits, f"network/full-text retrieval references in build_dataset.py: {hits}")
    return "build_dataset.py has no HTTP/Entrez references; text comes only from data/fulltext_package"


# ---------------------------------------------------------------------------
# S18
# ---------------------------------------------------------------------------
@scenario("S18")
def s18_deps() -> str:
    import importlib

    versions = {}
    for mod in ("torch", "transformers", "adapters", "sklearn", "pandas", "numpy", "joblib", "pymupdf", "fastapi", "uvicorn", "multipart"):
        m = importlib.import_module(mod)
        versions[mod] = getattr(m, "__version__", "?")
    req = (ROOT / "requirements.txt").read_text()
    missing = [p for p in ("torch", "transformers", "adapters", "scikit-learn", "pandas", "numpy", "joblib", "pymupdf", "fastapi", "uvicorn", "python-multipart") if p not in req]
    need(not missing, f"requirements.txt missing {missing}")
    return "imports ok: " + ", ".join(f"{k}={v}" for k, v in versions.items())


# ---------------------------------------------------------------------------
# S19
# ---------------------------------------------------------------------------
@scenario("S19")
def s19_default_offline_load() -> str:
    env = {k: v for k, v in os.environ.items() if k != "HF_HOME"}
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, "encoder.py", "--smoke"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        tail = [ln for ln in (proc.stderr or "").strip().splitlines() if ln.strip()][-2:]
        raise AssertionError("default-env offline load failed: " + " | ".join(tail))
    return "HF_HOME unset + HF_HUB_OFFLINE=1: base and adapter resolved from repo caches"


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=Path("/tmp/omp-review-checks.json"))
    args = parser.parse_args()

    enc = None
    from encoder import load_encoder

    enc = load_encoder(device="cpu")

    s01_join()
    s02_filter()
    s03_fulltext_semantics(enc)
    s04_abstracts()
    s05_encoder(enc)
    s06_chunking()
    s07_split()
    s08_threshold_metrics()
    s09_bands(enc)
    s10_degraded(enc)
    s12_gitignore()
    s13_no_finetune(enc)
    s14_artifacts()
    s15_readme()
    s16_reproducible(args.rerun_dir)
    s17_no_network()
    s18_deps()
    s19_default_offline_load()

    args.json_out.write_text(json.dumps(RESULTS, indent=2))
    fails = [r for r in RESULTS if r["status"] == "FAIL"]
    print(f"\n{len(RESULTS)} scenarios: {sum(r['status'] == 'PASS' for r in RESULTS)} PASS, "
          f"{sum(r['status'] == 'SKIP' for r in RESULTS)} SKIP, {len(fails)} FAIL")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
