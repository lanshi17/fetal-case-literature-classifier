#!/usr/bin/env python3
"""Build AHP criterion + expert weights; calibrate experts on gold seed when available."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ahp_core import ahp_weights, soft_score_from_result  # noqa: E402
from criteria import CRITERION_NAMES, DEFAULT_CRITERION_PAIRWISE  # noqa: E402
from judge_models import EXPERTS, load_cache  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "ahp_wave1" / "ahp_config.json"
DEFAULT_CACHE = ROOT / "data" / "ahp_wave1" / "cache"
DEFAULT_GOLD = ROOT / "data" / "gold_standard.csv"


def saaty_from_scores(scores: dict[str, float], names: list[str]) -> list[list[float]]:
    """Map performance scores in [0,1] to a Saaty pairwise matrix."""
    eps = 1e-6
    s = {k: max(eps, float(scores.get(k, eps))) for k in names}
    # ratio clipped to Saaty 1..9
    mat = []
    for a in names:
        row = []
        for b in names:
            r = s[a] / s[b]
            # map ratio to 1/9..9
            if r >= 1:
                val = float(min(9, max(1, round(r * 3))))  # gentle scale
            else:
                inv = 1.0 / r
                val = 1.0 / float(min(9, max(1, round(inv * 3))))
            row.append(val)
        mat.append(row)
    # force diagonal 1
    for i in range(len(names)):
        mat[i][i] = 1.0
    return mat


def expert_seed_metrics(
    gold: pd.DataFrame,
    cache_dir: Path,
    experts: list[str],
) -> dict[str, dict[str, float]]:
    g = gold.copy()
    g["pmid"] = g["pmid"].astype(str)
    g = g[g["gold_label"].astype(str).isin(["0", "1"])]
    g["gold_label"] = g["gold_label"].astype(int)
    gmap = g.drop_duplicates("pmid", keep="last").set_index("pmid")["gold_label"]

    out: dict[str, dict[str, float]] = {}
    for name in experts:
        cache = load_cache(cache_dir / f"{name}.jsonl")
        y_true, y_pred, confs = [], [], []
        for pmid, ylab in gmap.items():
            if pmid not in cache:
                continue
            res = cache[pmid].get("result") or {}
            if res.get("error") or "qualifies" not in res:
                continue
            y_true.append(int(ylab))
            y_pred.append(1 if res.get("qualifies") else 0)
            confs.append(float(res.get("confidence") or 0.5))
        if len(y_true) < 5 or len(set(y_true)) < 2:
            out[name] = {"n": float(len(y_true)), "accuracy": 0.5, "macro_f1": 0.5, "score": 0.5}
            continue
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        acc = float((yt == yp).mean())
        f1s = []
        for lab in (0, 1):
            tp = int(((yp == lab) & (yt == lab)).sum())
            fp = int(((yp == lab) & (yt != lab)).sum())
            fn = int(((yp != lab) & (yt == lab)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        macro_f1 = float(np.mean(f1s))
        # composite for Saaty
        score = 0.5 * acc + 0.5 * macro_f1
        out[name] = {
            "n": float(len(y_true)),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "score": score,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--experts", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument(
        "--uniform-experts",
        action="store_true",
        help="force equal expert weights (ignore gold calibration)",
    )
    args = ap.parse_args()

    # criterion weights
    w_c, lam_c, cr_c = ahp_weights(DEFAULT_CRITERION_PAIRWISE)
    if cr_c > 0.1:
        print(f"WARNING: criterion CR={cr_c:.3f} > 0.1")

    available = []
    for name in args.experts:
        path = args.cache_dir / f"{name}.jsonl"
        if path.exists() and path.stat().st_size > 0:
            available.append(name)
    if not available:
        available = list(args.experts)

    expert_metrics: dict[str, Any] = {}
    if args.gold.exists() and not args.uniform_experts:
        expert_metrics = expert_seed_metrics(pd.read_csv(args.gold, dtype=str), args.cache_dir, available)
        scores = {k: float(v.get("score", 0.5)) for k, v in expert_metrics.items()}
        # if no overlapping judgments, fall back to uniform
        if all(v.get("n", 0) < 5 for v in expert_metrics.values()):
            pairwise_e = [[1.0 if i == j else 1.0 for j in range(len(available))] for i in range(len(available))]
            note = "uniform_experts_insufficient_gold_overlap"
        else:
            pairwise_e = saaty_from_scores(scores, available)
            note = "experts_calibrated_on_gold"
    else:
        pairwise_e = [[1.0 if i == j else 1.0 for j in range(len(available))] for i in range(len(available))]
        note = "uniform_experts"

    w_e, lam_e, cr_e = ahp_weights(pairwise_e)
    # zero-out missing caches and renormalize if some experts absent
    present_mask = np.array([(args.cache_dir / f"{n}.jsonl").exists() for n in available], dtype=bool)
    if present_mask.any() and not present_mask.all():
        w_e = w_e * present_mask.astype(float)
        if w_e.sum() > 0:
            w_e = w_e / w_e.sum()

    models = {
        n: {
            "default_model": EXPERTS[n]["default_model"],
            "env_model": EXPERTS[n]["env_model"],
        }
        for n in available
        if n in EXPERTS
    }

    cfg = {
        "note": note,
        "criterion_names": list(CRITERION_NAMES),
        "criterion_pairwise": DEFAULT_CRITERION_PAIRWISE,
        "criterion_weights": {n: float(w) for n, w in zip(CRITERION_NAMES, w_c)},
        "criterion_lambda_max": lam_c,
        "criterion_CR": cr_c,
        "experts": available,
        "expert_models": models,
        "expert_pairwise": pairwise_e,
        "expert_weights": {n: float(w) for n, w in zip(available, w_e)},
        "expert_lambda_max": lam_e,
        "expert_CR": cr_e,
        "expert_seed_metrics": expert_metrics,
        "hard_label": {"hi": 0.55, "lo": 0.45},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "criterion_CR": cr_c, "expert_CR": cr_e, "expert_weights": cfg["expert_weights"], "note": note}, indent=2))


if __name__ == "__main__":
    main()
