#!/usr/bin/env python3
"""Summarise external-validity runs (temporal / journal_cv).

Reads fold JSONs and OOF scores written by ``case_external_eval.py`` and writes
``artifacts/case_dora_external/SUMMARY.md`` + ``summary.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "case_dora_external"
PROTOCOLS = ("temporal", "journal_cv")


def summarise(folds: list[dict], oof: pd.DataFrame) -> dict:
    ap = np.array([f["test_ap"] for f in folds], dtype=float)
    auc = np.array([f["test_roc_auc"] for f in folds], dtype=float)
    f1 = np.array([f["test_at_f1_thr"]["f1"] for f in folds], dtype=float)
    pr = np.array([f["test_at_f1_thr"]["precision"] for f in folds], dtype=float)
    rc = np.array([f["test_at_f1_thr"]["recall"] for f in folds], dtype=float)
    return {
        "n_folds": len(folds),
        "n_test": int(len(oof)),
        "ap_mean": float(ap.mean()), "ap_std": float(ap.std()),
        "auc_mean": float(auc.mean()), "auc_std": float(auc.std()),
        "f1_mean": float(f1.mean()), "f1_std": float(f1.std()),
        "precision_mean": float(pr.mean()), "recall_mean": float(rc.mean()),
        "oof_ap": float(average_precision_score(oof["y"], oof["score"])),
        "oof_auc": float(roc_auc_score(oof["y"], oof["score"])),
        "thresholds": [float(f["threshold_f1"]) for f in folds],
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.dir
    summary = {}
    lines = ["# External validity", ""]
    for protocol in PROTOCOLS:
        d = out / protocol
        folds = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("fold_*.json"))]
        if not folds:
            continue
        oof = pd.read_csv(d / "oof.csv", dtype={"pmid": str})
        s = summarise(folds, oof)
        summary[protocol] = s
        label = "temporal (train <=2019, test >2019)" if protocol == "temporal" else "journal GroupKFold (5 folds)"
        lines += [
            f"## {label}",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| folds | {s['n_folds']} |",
            f"| test docs | {s['n_test']} |",
            f"| AP (mean ± std) | {s['ap_mean']:.3f} ± {s['ap_std']:.3f} |",
            f"| AUC (mean ± std) | {s['auc_mean']:.3f} ± {s['auc_std']:.3f} |",
            f"| F1 (mean ± std) | {s['f1_mean']:.3f} ± {s['f1_std']:.3f} |",
            f"| precision / recall (mean) | {s['precision_mean']:.3f} / {s['recall_mean']:.3f} |",
            f"| pooled OOF AP / AUC | {s['oof_ap']:.3f} / {s['oof_auc']:.3f} |",
            f"| selected thresholds | {', '.join(f'{t:.3f}' for t in s['thresholds'])} |",
            "",
        ]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
