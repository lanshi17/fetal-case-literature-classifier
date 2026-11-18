#!/usr/bin/env python3
"""Calibration and abstention analysis for the case-only classifier.

Inputs are out-of-fold scores (pmid, y, score, fold). Reports:
    AP / AUC / Brier / ECE, reliability bins;
    cross-fitted temperature scaling and isotonic regression;
    an abstention band just below the case threshold.

Usage
-----
    .venv/bin/python src/scripts/calibrate_case_model.py
    .venv/bin/python src/scripts/calibrate_case_model.py \
        --oof artifacts/case_dora/cv/oof.csv \
        --external artifacts/case_dora_external/temporal/oof.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OOF = ROOT / "artifacts" / "case_dora" / "cv" / "oof.csv"
DEFAULT_OUT = ROOT / "artifacts" / "case_dora_calibration"


def ece_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "pred": float(p[m].mean()), "obs": float(y[m].mean())})
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece), rows


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    z = logit(p)

    def nll(log_t: np.ndarray) -> float:
        t = np.exp(log_t[0])
        q = 1 / (1 + np.exp(-z / t))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    return float(np.exp(minimize(nll, x0=np.array([0.0]), method="Nelder-Mead").x[0]))


def temperature_cv(y: np.ndarray, p: np.ndarray, fold: np.ndarray) -> tuple[np.ndarray, list[float]]:
    out = np.empty_like(p)
    temps = []
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        t = fit_temperature(y[tr], p[tr])
        temps.append(t)
        z = logit(p[te]) / t
        out[te] = 1 / (1 + np.exp(-z))
    return out, temps


def isotonic_cv(y: np.ndarray, p: np.ndarray, fold: np.ndarray) -> np.ndarray:
    out = np.empty_like(p)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
        out[te] = iso.predict(p[te])
    return out


def abstention(y: np.ndarray, p: np.ndarray, thr: float, margin: float) -> dict[str, float]:
    case = p >= thr
    abstain = (p >= thr - margin) & (p < thr)
    noncase = p < thr - margin
    kept = case | noncase
    tp = int((case & (y == 1)).sum())
    fp = int((case & (y == 0)).sum())
    fn = int((~case & (y == 1)).sum())
    return {
        "margin": margin,
        "coverage": float(kept.mean()),
        "abstain_rate": float(abstain.mean()),
        "abstain_positives": int((abstain & (y == 1)).sum()),
        "abstain_negatives": int((abstain & (y == 0)).sum()),
        "case_precision": tp / max(tp + fp, 1),
        "case_recall": tp / max(tp + fn, 1),
    }


def analyse(name: str, d: pd.DataFrame, thr: float) -> dict[str, Any]:
    y = d["y"].to_numpy()
    p = d["score"].to_numpy()
    fold = d["fold"].to_numpy() if "fold" in d else np.zeros(len(d), dtype=int)
    if len(np.unique(fold)) < 2:
        from sklearn.model_selection import KFold
        fold = np.zeros(len(y), dtype=int)
        for k, (_, te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=0).split(y)):
            fold[te] = k
    ece, rows = ece_bins(y, p)
    tp2, tlist = temperature_cv(y, p, fold)
    iso = isotonic_cv(y, p, fold)
    return {
        "name": name,
        "n": int(len(d)),
        "ap": float(average_precision_score(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece,
        "reliability": rows,
        "temperature": {"temps": tlist, "brier": float(brier_score_loss(y, tp2)), "ece": ece_bins(y, tp2)[0]},
        "isotonic": {"brier": float(brier_score_loss(y, iso)), "ece": ece_bins(y, iso)[0]},
        "abstention": [abstention(y, p, thr, m) for m in (0.0, 0.05, 0.10, 0.15, 0.20)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    ap.add_argument("--external", type=Path, nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=0.39624494)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = [("random_cv", pd.read_csv(args.oof, dtype={"pmid": str}).drop_duplicates("pmid"))]
    for path in args.external:
        p = Path(path)
        runs.append((p.parent.name, pd.read_csv(p, dtype={"pmid": str}).drop_duplicates("pmid")))

    results = [analyse(name, d, args.threshold) for name, d in runs]
    (args.out / "calibration.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Case-model calibration", "", f"threshold = {args.threshold}", ""]
    for r in results:
        lines += [
            f"## {r['name']} (n={r['n']})",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| AP | {r['ap']:.4f} |",
            f"| AUC | {r['auc']:.4f} |",
            f"| Brier | {r['brier']:.4f} |",
            f"| ECE (15 bins) | {r['ece']:.4f} |",
            f"| Brier / ECE after temperature scaling | {r['temperature']['brier']:.4f} / {r['temperature']['ece']:.4f} |",
            f"| Brier / ECE after isotonic | {r['isotonic']['brier']:.4f} / {r['isotonic']['ece']:.4f} |",
            "",
            "| predicted | observed | n |",
            "|---:|---:|---:|",
        ]
        lines += [f"| {b['pred']:.3f} | {b['obs']:.3f} | {b['n']} |" for b in r["reliability"]]
        lines += ["", "| abstain margin | coverage | abstain rate | pos abstained | neg abstained | case P | case R |", "|---:|---:|---:|---:|---:|---:|---:|"]
        lines += [
            f"| {a['margin']:.2f} | {a['coverage']:.3f} | {a['abstain_rate']:.3f} | {a['abstain_positives']} | "
            f"{a['abstain_negatives']} | {a['case_precision']:.3f} | {a['case_recall']:.3f} |"
            for a in r["abstention"]
        ]
        lines.append("")
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
