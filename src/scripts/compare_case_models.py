#!/usr/bin/env python3
"""Compare two case-only DoRA runs trained on the same folds (encoder swap).

Reads ``cv/fold_*.json``, ``cv/oof.csv`` and ``config.json`` from two run
directories, verifies that both scored the same PMIDs with the same fold
assignment, and reports per-fold and out-of-fold differences with a paired
bootstrap CI for AP and an exact McNemar test at a shared threshold.

Usage
-----
    .venv/bin/python src/scripts/compare_case_models.py \
        --base artifacts/case_dora --new artifacts/case_dora_biomedbert \
        --out artifacts/case_dora_biomedbert/COMPARE.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SHARED_THRESHOLD = 0.39624494


def load_run(run_dir: Path) -> dict[str, Any]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    folds = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run_dir / "cv").glob("fold_*.json"))]
    oof = pd.read_csv(run_dir / "cv" / "oof.csv", dtype={"pmid": str})
    oof = oof.drop_duplicates(subset=["pmid"], keep="last").sort_values("pmid").reset_index(drop=True)
    return {"dir": run_dir, "cfg": cfg, "folds": folds, "oof": oof}


def prf(y: np.ndarray, score: np.ndarray, thr: float) -> dict[str, float]:
    pred = score >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "tp": tp, "fp": fp, "fn": fn}


def boot_delta_ap(y: np.ndarray, a: np.ndarray, b: np.ndarray, *, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    idx = np.arange(len(y))
    for i in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        yy = y[take]
        if yy.min() == yy.max():
            out[i] = np.nan
            continue
        out[i] = average_precision_score(yy, b[take]) - average_precision_score(yy, a[take])
    out = out[~np.isnan(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, int | float]:
    ok_a, ok_b = pred_a == y, pred_b == y
    b_only = int((ok_b & ~ok_a).sum())
    a_only = int((ok_a & ~ok_b).sum())
    p = binomtest(b_only, b_only + a_only, 0.5).pvalue if (b_only + a_only) else 1.0
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p_value": float(p)}


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "artifacts" / "case_dora")
    ap.add_argument("--new", type=Path, default=ROOT / "artifacts" / "case_dora_biomedbert")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label-base", default="NeuML")
    ap.add_argument("--label-new", default="BiomedBERT")
    args = ap.parse_args()

    base, new = load_run(args.base), load_run(args.new)
    ob, nw = base["oof"], new["oof"]
    if list(ob["pmid"]) != list(nw["pmid"]):
        raise SystemExit("OOF PMIDs differ; runs are not comparable on the same folds")
    if not np.array_equal(ob["y"].to_numpy(), nw["y"].to_numpy()) or not np.array_equal(ob["fold"], nw["fold"]):
        raise SystemExit("OOF labels/folds differ; runs are not comparable")

    y = ob["y"].to_numpy()
    sb, sn = ob["score"].to_numpy(), nw["score"].to_numpy()
    thr_b = float(base["cfg"]["threshold"])
    thr_n = float(new["cfg"]["threshold"])

    lines = [
        "# Encoder swap: NeuML vs BiomedBERT",
        "",
        f"- data: {len(y)} design-labelled papers ({int(y.sum())} case / {int((1 - y).sum())} non-case), identical 5-fold split",
        f"- base run: `{args.base}` ({base['cfg']['encoder'].get('model_id')})",
        f"- new run: `{args.new}` ({new['cfg']['encoder'].get('model_id')})",
        "",
        "## Per-fold test metrics",
        "",
        f"| Fold | {args.label_base} AP | {args.label_new} AP | ΔAP | {args.label_base} F1 | {args.label_new} F1 | best epoch (base/new) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    deltas = []
    for fb, fn in zip(base["folds"], new["folds"]):
        db = fn["test_ap"] - fb["test_ap"]
        deltas.append(db)
        lines.append(
            f"| {fb['fold']} | {fmt(fb['test_ap'])} | {fmt(fn['test_ap'])} | {db:+.3f} | "
            f"{fmt(fb['test_at_f1_thr']['f1'])} | {fmt(fn['test_at_f1_thr']['f1'])} | "
            f"{fb['best_epoch']}/{fn['best_epoch']} |"
        )

    def mean_std(folds: list[dict], key: str) -> tuple[float, float]:
        v = np.array([f[key] for f in folds], dtype=float)
        return float(v.mean()), float(v.std())

    for name, key in (("test AP", "test_ap"), ("test AUC", "test_roc_auc")):
        mb, sb_ = mean_std(base["folds"], key)
        mn, sn_ = mean_std(new["folds"], key)
        lines.append(f"| **mean {name}** | **{fmt(mb)} ± {fmt(sb_)}** | **{fmt(mn)} ± {fmt(sn_)}** | **{mn - mb:+.3f}** | | | |")

    lines += ["", "## Out-of-fold", "", "| Metric | " + args.label_base + " | " + args.label_new + " | Δ |", "|---|---:|---:|---:|"]
    thr_b_fold = {f["fold"]: f["threshold_f1"] for f in base["folds"]}
    thr_n_fold = {f["fold"]: f["threshold_f1"] for f in new["folds"]}
    pred_b = np.array([s >= thr_b_fold[f] for s, f in zip(sb, ob["fold"])])
    pred_n = np.array([s >= thr_n_fold[f] for s, f in zip(sn, nw["fold"])])

    def prf_pred(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}

    oof_rows = [
        ("AP", average_precision_score(y, sb), average_precision_score(y, sn)),
        ("ROC AUC", roc_auc_score(y, sb), roc_auc_score(y, sn)),
    ]
    for name, a, b in oof_rows:
        lines.append(f"| {name} | {fmt(a, 4)} | {fmt(b, 4)} | {b - a:+.4f} |")

    pb, pn = prf_pred(y, pred_b), prf_pred(y, pred_n)
    lines.append(
        f"| F1 at per-fold val thresholds (primary) | P {fmt(pb['precision'])} / R {fmt(pb['recall'])} / F1 {fmt(pb['f1'])} "
        f"| P {fmt(pn['precision'])} / R {fmt(pn['recall'])} / F1 {fmt(pn['f1'])} | F1 {pn['f1'] - pb['f1']:+.4f} |"
    )
    for name, thr in ((f"F1 at own OOF threshold ({args.label_base} {thr_b:.4f} / {args.label_new} {thr_n:.4f})", None), ("F1 at shared threshold 0.3962", SHARED_THRESHOLD)):
        pb = prf(y, sb, thr_b if thr is None else thr)
        pn = prf(y, sn, thr_n if thr is None else thr)
        lines.append(
            f"| {name} | P {fmt(pb['precision'])} / R {fmt(pb['recall'])} / F1 {fmt(pb['f1'])} "
            f"| P {fmt(pn['precision'])} / R {fmt(pn['recall'])} / F1 {fmt(pn['f1'])} | F1 {pn['f1'] - pb['f1']:+.4f} |"
        )

    lo, hi = boot_delta_ap(y, sb, sn)
    mc = mcnemar(y, pred_b, pred_n)
    mc_shared = mcnemar(y, sb >= SHARED_THRESHOLD, sn >= SHARED_THRESHOLD)
    wins = int((np.array([f["test_ap"] for f in new["folds"]]) > np.array([f["test_ap"] for f in base["folds"]])).sum())
    delta_ap = average_precision_score(y, sn) - average_precision_score(y, sb)
    lines += [
        "",
        f"Paired bootstrap ΔAP ({args.label_new} − {args.label_base}): **{delta_ap:+.4f}** "
        f"(95% CI {lo:+.4f} … {hi:+.4f}, 2000 resamples). Per-fold test AP: {args.label_new} wins {wins}/5.",
        f"McNemar at per-fold val thresholds: only-{args.label_base} correct {mc['a_only_correct']}, only-{args.label_new} correct {mc['b_only_correct']} (p = {mc['p_value']:.3g}).",
        f"McNemar at shared 0.3962: only-{args.label_base} correct {mc_shared['a_only_correct']}, only-{args.label_new} correct {mc_shared['b_only_correct']} (p = {mc_shared['p_value']:.3g}).",
        "",
        "## Conclusion",
        "",
        ("No statistically significant AP difference (CI includes 0). " if lo <= 0 <= hi else "AP difference is outside the bootstrap CI. ")
        + "Choose the encoder on operational grounds (license, packaging, reproducibility), not on this delta.",
        "",
        "## Caveats",
        "",
        "- Same folds, seed, hyperparameters and chunking; only the encoder checkpoint differs.",
        "- Thresholds are selected per fold on the inner validation split, then applied to the held-out fold.",
        "- BiomedBERT weights are the original `pytorch_model.bin` (BertForMaskedLM head discarded by `AutoModel`); NeuML is a SentenceTransformer fine-tune of the same architecture.",
    ]

    out = args.out or (args.new / "COMPARE.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "n": int(len(y)),
        "base": {"threshold": thr_b, "oof_ap": average_precision_score(y, sb), "oof_auc": roc_auc_score(y, sb)},
        "new": {"threshold": thr_n, "oof_ap": average_precision_score(y, sn), "oof_auc": roc_auc_score(y, sn)},
        "delta_ap": average_precision_score(y, sn) - average_precision_score(y, sb),
        "delta_ap_ci": [lo, hi],
        "mcnemar": mc,
        "fold_delta_ap": deltas,
    }
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
