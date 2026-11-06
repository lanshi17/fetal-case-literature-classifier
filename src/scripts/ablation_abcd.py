#!/usr/bin/env python3
"""Mandatory ablations A/B/C/D for PubMedBERT screening.

A — title + abstract only (speed / accuracy baseline)
B — title + abstract + fulltext (fulltext recall gain)
C — fulltext + explicit case/variant lexical features (FP reduction?)
D — hard-negative stress test (prediction / cohort / review / database)

Protocol: seed-42 GroupShuffleSplit; default subset (weak=False, ≠model);
val R≥0.95 threshold; report test + gold-49. Encoder = frozen PubMedBERT
chunk-pool 1536-d (+2 aux for C). Optional: DoRA inference A vs B.

Usage
-----
    .venv/bin/python src/scripts/ablation_abcd.py --device cuda
    .venv/bin/python src/scripts/ablation_abcd.py --skip-dora   # LR-only, faster
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from encoder import (  # noqa: E402
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_KEEP_FIRST,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_MODEL,
    EMBED_DIM,
    IdentityScaler,
    aux_features,
    document_features_batch,
    load_encoder,
)
from train import (  # noqa: E402
    binary_metrics,
    default_subset_mask,
    load_dataset,
    make_splits,
    select_threshold,
)

logger = logging.getLogger("ablation_abcd")

OUT = ROOT / "artifacts" / "ablation_abcd"
GOLD = ROOT / "data" / "gold_standard.csv"


# ---------------------------------------------------------------------------
# Hard-negative tagging (experiment D)
# ---------------------------------------------------------------------------
HARD_DESIGN = re.compile(
    r"(review|cohort|screening|validation|method|registry|observational|gwas|"
    r"database|retrospective|prospective)",
    re.I,
)
HARD_TITLE = re.compile(
    r"\b(review|meta[- ]analysis|guideline|database|registry|polygenic|gwas|"
    r"genome[- ]wide|prediction\s+score|pathogenicity\s+prediction|"
    r"in\s+silico|machine\s+learning|deep\s+learning|noninvasive\s+prenatal|"
    r"nipt|cell[- ]free\s+dna|screening\s+for|cohort\s+study|"
    r"large[- ]scale|population[- ]based)\b",
    re.I,
)
HARD_GOLD_REASONS = {
    "gwas_large_sequencing",
    "review_or_editorial",
    "prediction_only",
    "method_algorithm_db",
}


def tag_hard_negative(row: pd.Series) -> tuple[bool, str]:
    design = str(row.get("design") or "")
    title = str(row.get("title") or "")
    reason = str(row.get("label_reason") or "")
    tags = []
    if HARD_DESIGN.search(design):
        tags.append(f"design:{design.strip()[:40] or 'match'}")
    if HARD_TITLE.search(title):
        tags.append("title_kw")
    blob = f"{title} {reason}".lower()
    if any(k in blob for k in ("database", "in silico", "polygenic", "prediction only")):
        tags.append("reason_kw")
    return (bool(tags), ";".join(tags))


def encode_corpus(
    df: pd.DataFrame,
    *,
    include_fulltext: bool,
    cache_path: Path,
    device: str | None,
    batch_size: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (X_1536, aux_2, encode_seconds)."""
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=False)
        if list(data["doc_ids"].astype(str)) == df["doc_id"].astype(str).tolist():
            logger.info("cache hit %s", cache_path)
            return data["X"].astype(np.float64), data["aux"].astype(np.float64), float(data["encode_sec"][0])

    enc = load_encoder(DEFAULT_MODEL, device=device, batch_size=batch_size)
    records = []
    for r in df.itertuples(index=False):
        records.append(
            {
                "title": r.title,
                "abstract": r.abstract,
                "text": r.text if include_fulltext else "",
            }
        )
    t0 = time.time()
    feats, _ = document_features_batch(
        records,
        enc,
        include_fulltext=include_fulltext,
        include_aux=True,
        chunk_tokens=DEFAULT_CHUNK_TOKENS,
        max_chunks=DEFAULT_MAX_CHUNKS,
        keep_first=DEFAULT_KEEP_FIRST,
        batch_size=batch_size,
        show_progress=True,
    )
    elapsed = time.time() - t0
    X = feats[:, : 2 * EMBED_DIM].astype(np.float64)
    aux = feats[:, 2 * EMBED_DIM :].astype(np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        doc_ids=np.asarray(df["doc_id"].astype(str).tolist(), dtype="U"),
        X=X,
        aux=aux,
        encode_sec=np.asarray([elapsed], dtype=np.float64),
        include_fulltext=np.asarray([include_fulltext]),
    )
    logger.info("encoded %d docs in %.1fs -> %s", len(df), elapsed, cache_path)
    return X, aux, elapsed


def train_lr(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    C: float,
    target_recall: float,
) -> dict[str, Any]:
    scaler = IdentityScaler().fit(X[train_idx])
    clf = LogisticRegression(
        C=C, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=0
    )
    clf.fit(X[train_idx], y[train_idx])
    val_s = clf.predict_proba(X[val_idx])[:, 1]
    test_s = clf.predict_proba(X[test_idx])[:, 1]
    thr = select_threshold(y[val_idx], val_s, target_recall)
    return {
        "threshold": thr,
        "val": binary_metrics(y[val_idx], val_s, thr),
        "test": binary_metrics(y[test_idx], test_s, thr),
        "test_scores": test_s,
        "model": clf,
        "scaler": scaler,
    }


def gold_metrics(clf, scaler, X: np.ndarray, base: pd.DataFrame, thr: float) -> dict[str, Any] | None:
    if not GOLD.exists():
        return None
    gold = pd.read_csv(GOLD, dtype={"pmid": str})
    gold = gold[gold["gold_label"].astype(str).isin(["0", "1"])].drop_duplicates("pmid", keep="last")
    gold["gold_label"] = gold["gold_label"].astype(int)
    id_to_i = {str(p): i for i, p in enumerate(base["pmid"].astype(str))}
    idx = [id_to_i[p] for p in gold["pmid"] if p in id_to_i]
    if not idx:
        return None
    gmap = gold.set_index("pmid")["gold_label"]
    y = np.asarray([int(gmap.loc[str(base.iloc[i]["pmid"])]) for i in idx], dtype=int)
    Xt = np.asarray(scaler.transform(X[idx]), dtype=np.float64)
    sc = clf.predict_proba(Xt)[:, 1]
    return binary_metrics(y, sc, thr)


def eval_hard_negatives(
    df: pd.DataFrame,
    scores: np.ndarray,
    thr: float,
    idxs: np.ndarray,
    *,
    name: str,
) -> dict[str, Any]:
    """On hard negatives within idxs (typically test), report exclusion ability."""
    sub = df.iloc[idxs].reset_index(drop=True)
    sc = scores
    hard_mask = []
    tags = []
    for _, r in sub.iterrows():
        is_h, tag = tag_hard_negative(r)
        # only true negatives that look hard, OR labeled 0 with hard tag
        hard_mask.append(bool(is_h) and int(r["label"]) == 0)
        tags.append(tag)
    hard_mask = np.asarray(hard_mask, dtype=bool)
    y = sub["label"].to_numpy(dtype=int)
    pred = (sc >= thr).astype(int)
    # among hard negatives (y=0 & hard): specificity = 1 - FP_rate
    hn = hard_mask
    n = int(hn.sum())
    if n == 0:
        return {"name": name, "n_hard_neg": 0}
    fp = int(((pred == 1) & hn).sum())
    tn = int(((pred == 0) & hn).sum())
    return {
        "name": name,
        "n_hard_neg": n,
        "tn": tn,
        "fp": fp,
        "specificity": float(tn / n),
        "fp_rate": float(fp / n),
        "mean_score": float(sc[hn].mean()),
        "examples_fp": sub.loc[hn & (pred == 1), ["pmid", "title", "design"]].head(10).to_dict("records"),
    }


def run_dora_ab(
    base: pd.DataFrame,
    test_idx: np.ndarray,
    gold_df: pd.DataFrame | None,
    *,
    device: str,
    thr: float,
) -> dict[str, Any]:
    """DoRA inference: A=title+abstract chunks vs B=fulltext chunks."""
    sys.path.insert(0, str(ROOT / "src" / "scripts"))
    import torch
    from transformers import AutoTokenizer
    from finetune_chunk_fair import (
        ChunkPoolClassifier,
        Example,
        doc_to_chunks,
        predict_proba,
        resolve_model_path,
        wrap_dora,
    )

    dora_dir = ROOT / "artifacts" / "finetune_fair" / "dora"
    meta = json.loads((dora_dir / "metrics.json").read_text())
    thr_path = dora_dir / "threshold.json"
    if thr_path.exists():
        thr = float(json.loads(thr_path.read_text())["threshold"])
    model_path = resolve_model_path(meta.get("model_path"))
    tok = AutoTokenizer.from_pretrained(model_path)
    model = ChunkPoolClassifier(model_path)
    model = wrap_dora(model, int(meta["model_info"]["dora_r"]), int(meta["model_info"]["dora_alpha"]))
    state = torch.load(dora_dir / "checkpoint" / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()

    def score(rows: list[Example]) -> np.ndarray:
        return predict_proba(
            model, rows, tok, max_length=DEFAULT_CHUNK_TOKENS, batch_size=2, device=torch.device(device)
        )

    def build(idxs, fulltext: bool) -> list[Example]:
        out = []
        for i in idxs:
            r = base.iloc[int(i)]
            text = r["text"] if fulltext else ""
            chunks = doc_to_chunks(r["title"], r["abstract"], text, tok)
            out.append(Example(str(r["pmid"]), chunks, int(r["label"]), "t"))
        return out

    t0 = time.time()
    ex_a = build(test_idx, False)
    sc_a = score(ex_a)
    time_a = time.time() - t0
    t0 = time.time()
    ex_b = build(test_idx, True)
    sc_b = score(ex_b)
    time_b = time.time() - t0
    y = np.asarray([e.label for e in ex_a], dtype=int)
    out: dict[str, Any] = {
        "threshold": thr,
        "A": {**binary_metrics(y, sc_a, thr), "infer_sec": time_a, "sec_per_doc": time_a / max(len(ex_a), 1)},
        "B": {**binary_metrics(y, sc_b, thr), "infer_sec": time_b, "sec_per_doc": time_b / max(len(ex_b), 1)},
        "delta_recall_B_minus_A": float(
            binary_metrics(y, sc_b, thr)["recall"] - binary_metrics(y, sc_a, thr)["recall"]
        ),
        "delta_ap_B_minus_A": float(binary_metrics(y, sc_b, thr)["ap"] - binary_metrics(y, sc_a, thr)["ap"]),
    }
    # hard neg on test with B scores
    out["D_hard_neg_B"] = eval_hard_negatives(base, sc_b, thr, test_idx, name="dora_B")
    out["D_hard_neg_A"] = eval_hard_negatives(base, sc_a, thr, test_idx, name="dora_A")

    if gold_df is not None and len(gold_df):
        id_to_i = {str(p): i for i, p in enumerate(base["pmid"].astype(str))}
        g_idx = np.asarray([id_to_i[p] for p in gold_df["pmid"] if p in id_to_i], dtype=int)

        def gold_ex(fulltext: bool) -> list[Example]:
            rows = []
            gmap = gold_df.set_index("pmid")["gold_label"]
            for i in g_idx:
                r = base.iloc[int(i)]
                text = r["text"] if fulltext else ""
                rows.append(
                    Example(
                        str(r["pmid"]),
                        doc_to_chunks(r["title"], r["abstract"], text, tok),
                        int(gmap.loc[str(r["pmid"])]),
                        "g",
                    )
                )
            return rows

        ga, gb = gold_ex(False), gold_ex(True)
        y_g = np.asarray([e.label for e in ga], dtype=int)
        out["gold_A"] = binary_metrics(y_g, score(ga), thr)
        out["gold_B"] = binary_metrics(y_g, score(gb), thr)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-reencode", action="store_true")
    ap.add_argument("--skip-dora", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset(ROOT / "data" / "labeled_papers.csv")
    subset = default_subset_mask(df).to_numpy()
    train_idx, val_idx, test_idx = make_splits(df, seed=args.seed)
    train_sub = train_idx[subset[train_idx]]
    val_sub = val_idx[subset[val_idx]]
    test_sub = test_idx[subset[test_idx]]
    y = df["label"].to_numpy(dtype=int)

    gold_df = None
    if GOLD.exists():
        gold_df = pd.read_csv(GOLD, dtype={"pmid": str})
        gold_df = gold_df[gold_df["gold_label"].astype(str).isin(["0", "1"])].drop_duplicates("pmid", keep="last")
        gold_df["gold_label"] = gold_df["gold_label"].astype(int)

    # ---- encode A / B ----
    X_a, aux_a, sec_a = encode_corpus(
        df,
        include_fulltext=False,
        cache_path=OUT / "features_A_title_abstract.npz",
        device=args.device,
        batch_size=args.batch_size,
        force=args.force_reencode,
    )
    X_b, aux_b, sec_b = encode_corpus(
        df,
        include_fulltext=True,
        cache_path=OUT / "features_B_fulltext.npz",
        device=args.device,
        batch_size=args.batch_size,
        force=args.force_reencode,
    )
    # aux for C: prefer fulltext-aware aux (same as encoder aux_features on title+abs+text)
    aux_c = np.stack(
        [aux_features(str(r.title), str(r.abstract), str(r.text)) for r in df.itertuples(index=False)]
    ).astype(np.float64)
    X_c = np.concatenate([X_b, aux_c], axis=1)

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "seed": args.seed,
            "split": "GroupShuffleSplit 64/16/20, default subset val/test/train",
            "n_train": int(len(train_sub)),
            "n_val": int(len(val_sub)),
            "n_test": int(len(test_sub)),
            "C": args.C,
            "target_recall": args.target_recall,
            "encoder": "frozen PubMedBERT chunk-pool (480×≤12 → 1536)",
        },
        "encode_timing": {
            "A_title_abstract_sec": sec_a,
            "B_fulltext_sec": sec_b,
            "A_sec_per_doc": sec_a / max(len(df), 1),
            "B_sec_per_doc": sec_b / max(len(df), 1),
            "speedup_A_vs_B": (sec_b / sec_a) if sec_a > 0 else None,
        },
        "experiments": {},
    }

    # ---- A / B / C train ----
    for key, X, desc in (
        ("A", X_a, "title + abstract only"),
        ("B", X_b, "title + abstract + fulltext"),
        ("C", X_c, "fulltext + explicit case/variant lexical features (1536+2)"),
    ):
        logger.info("Training experiment %s: %s", key, desc)
        fit = train_lr(X, y, train_sub, val_sub, test_sub, C=args.C, target_recall=args.target_recall)
        g = gold_metrics(fit["model"], fit["scaler"], X, df, fit["threshold"])
        hard = eval_hard_negatives(df, fit["test_scores"], fit["threshold"], test_sub, name=key)
        results["experiments"][key] = {
            "description": desc,
            "feature_dim": int(X.shape[1]),
            "threshold": fit["threshold"],
            "val": fit["val"],
            "test": fit["test"],
            "gold": g,
            "hard_negatives_test": hard,
        }
        dump(
            {"model": fit["model"], "scaler": fit["scaler"], "threshold": fit["threshold"], "mode": key},
            OUT / f"model_{key}.joblib",
        )
        logger.info(
            "%s test AP=%.3f R=%.3f P=%.3f | hard-neg spec=%.3f (n=%d)",
            key,
            fit["test"]["ap"],
            fit["test"]["recall"],
            fit["test"]["precision"],
            hard.get("specificity") or -1,
            hard.get("n_hard_neg") or 0,
        )

    # deltas
    ea, eb, ec = results["experiments"]["A"], results["experiments"]["B"], results["experiments"]["C"]
    results["deltas"] = {
        "B_minus_A_test_recall": eb["test"]["recall"] - ea["test"]["recall"],
        "B_minus_A_test_ap": eb["test"]["ap"] - ea["test"]["ap"],
        "C_minus_B_test_precision": ec["test"]["precision"] - eb["test"]["precision"],
        "C_minus_B_hard_neg_specificity": (ec["hard_negatives_test"].get("specificity") or 0)
        - (eb["hard_negatives_test"].get("specificity") or 0),
    }

    # ---- D catalog ----
    hard_rows = []
    for i, r in df.iterrows():
        is_h, tag = tag_hard_negative(r)
        if is_h and int(r["label"]) == 0:
            hard_rows.append(
                {
                    "pmid": r["pmid"],
                    "title": r["title"],
                    "design": r["design"],
                    "label": r["label"],
                    "tag": tag,
                    "in_test": bool(i in set(test_sub.tolist())),
                }
            )
    hard_df = pd.DataFrame(hard_rows)
    hard_df.to_csv(OUT / "hard_negatives_catalog.csv", index=False)
    results["D"] = {
        "description": "hard negatives: cohort/review/screening/method designs + title keywords",
        "n_corpus_hard_neg": int(len(hard_df)),
        "n_test_hard_neg": int(hard_df["in_test"].sum()) if len(hard_df) else 0,
        "per_model": {
            k: results["experiments"][k]["hard_negatives_test"] for k in ("A", "B", "C")
        },
    }

    # gold hard reasons
    if gold_df is not None:
        g_hard = gold_df[gold_df["gold_reason_code"].isin(HARD_GOLD_REASONS) & (gold_df["gold_label"] == 0)]
        results["D"]["gold_hard_neg_n"] = int(len(g_hard))
        results["D"]["gold_hard_reason_counts"] = g_hard["gold_reason_code"].value_counts().to_dict()

    # ---- DoRA optional ----
    if not args.skip_dora and (ROOT / "artifacts/finetune_fair/dora/checkpoint/model.pt").exists():
        logger.info("DoRA inference ablation A vs B ...")
        device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        thr = 0.762
        results["dora_inference"] = run_dora_ab(df, test_sub, gold_df, device=device, thr=thr)

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # markdown report
    md = [
        "# Ablation A/B/C/D",
        "",
        f"Generated: {results['generated_at']}",
        "",
        "## Encode speed (frozen PubMedBERT, n=974)",
        f"- A (title+abstract): **{sec_a:.1f}s** ({sec_a/len(df):.3f}s/doc)",
        f"- B (fulltext): **{sec_b:.1f}s** ({sec_b/len(df):.3f}s/doc)",
        f"- Speedup A vs B: **{(sec_b/sec_a):.2f}×**" if sec_a else "",
        "",
        "## Experiments A/B/C (frozen + LR, val R≥0.95)",
        "",
        "| exp | input | test AP | test R | test P | test F1 | gold AP | hard-neg Spec |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ("A", "B", "C"):
        e = results["experiments"][k]
        g = e.get("gold") or {}
        h = e.get("hard_negatives_test") or {}
        md.append(
            f"| {k} | {e['description'][:42]} | {e['test']['ap']:.3f} | {e['test']['recall']:.3f} | "
            f"{e['test']['precision']:.3f} | {e['test']['f1']:.3f} | "
            f"{(g.get('ap') if g else float('nan')):.3f} | {(h.get('specificity') if h else float('nan')):.3f} |"
        )
    d = results["deltas"]
    md += [
        "",
        "### Key deltas",
        f"- Fulltext recall gain (B−A): **{d['B_minus_A_test_recall']:+.3f}** (AP {d['B_minus_A_test_ap']:+.3f})",
        f"- Lexical features precision (C−B): **{d['C_minus_B_test_precision']:+.3f}**",
        f"- Hard-neg specificity (C−B): **{d['C_minus_B_hard_neg_specificity']:+.3f}**",
        "",
        "## Experiment D — hard negatives",
        f"- Catalog size (label=0 & hard tag): **{results['D']['n_corpus_hard_neg']}** "
        f"(in test: {results['D']['n_test_hard_neg']})",
        f"- See `hard_negatives_catalog.csv`",
        "",
    ]
    if "dora_inference" in results:
        di = results["dora_inference"]
        md += [
            "## DoRA inference A vs B (recal thr)",
            f"- thr={di['threshold']:.4f}",
            f"- A test: R={di['A']['recall']:.3f} P={di['A']['precision']:.3f} AP={di['A']['ap']:.3f} "
            f"({di['A']['sec_per_doc']:.3f}s/doc)",
            f"- B test: R={di['B']['recall']:.3f} P={di['B']['precision']:.3f} AP={di['B']['ap']:.3f} "
            f"({di['B']['sec_per_doc']:.3f}s/doc)",
            f"- Δ recall B−A = **{di['delta_recall_B_minus_A']:+.3f}**, Δ AP = **{di['delta_ap_B_minus_A']:+.3f}**",
            f"- Hard-neg Spec A={di['D_hard_neg_A'].get('specificity')} B={di['D_hard_neg_B'].get('specificity')}",
            "",
        ]
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
