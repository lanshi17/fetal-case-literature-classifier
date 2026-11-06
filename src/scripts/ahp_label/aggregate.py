#!/usr/bin/env python3
"""Aggregate multi-model caches with AHP weights into labels_ahp.csv + report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ahp_core import hard_label, soft_score_from_result  # noqa: E402
from criteria import CRITERION_NAMES  # noqa: E402
from judge_models import load_cache  # noqa: E402

DEFAULT_CFG = ROOT / "data" / "ahp_wave1" / "ahp_config.json"
DEFAULT_CACHE = ROOT / "data" / "ahp_wave1" / "cache"
DEFAULT_CORPUS = ROOT / "data" / "ahp_wave1" / "corpus.jsonl"
DEFAULT_OUT = ROOT / "data" / "ahp_wave1" / "labels_ahp.csv"
DEFAULT_REPORT = ROOT / "artifacts" / "ahp_wave1" / "report.md"


def majority_vote(votes: list[int | None]) -> str:
    vals = [v for v in votes if v is not None]
    if not vals:
        return "uncertain"
    c = Counter(vals)
    top, n = c.most_common(1)[0]
    if n < (len(vals) + 1) // 2:
        return "uncertain"
    # tie
    if len([k for k, v in c.items() if v == n]) > 1:
        return "uncertain"
    return str(top)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    experts = list(cfg["experts"])
    w_c = np.asarray([cfg["criterion_weights"][n] for n in CRITERION_NAMES], dtype=float)
    w_e = {k: float(v) for k, v in cfg["expert_weights"].items()}
    hi = float(cfg.get("hard_label", {}).get("hi", 0.55))
    lo = float(cfg.get("hard_label", {}).get("lo", 0.45))

    caches = {name: load_cache(args.cache_dir / f"{name}.jsonl") for name in experts}
    pmids: list[str] = []
    if args.corpus.exists():
        with args.corpus.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pmids.append(str(json.loads(line)["pmid"]))
    else:
        # union of caches
        s: set[str] = set()
        for c in caches.values():
            s |= set(c.keys())
        pmids = sorted(s)

    rows = []
    agree_pairs = Counter()
    pair_n = Counter()
    for pmid in pmids:
        p_m = {}
        votes = []
        reasons = {}
        ok_experts = []
        for name in experts:
            entry = caches.get(name, {}).get(pmid)
            if not entry:
                p_m[name] = None
                votes.append(None)
                continue
            res = entry.get("result") or {}
            if res.get("error") or "qualifies" not in res:
                p_m[name] = None
                votes.append(None)
                reasons[name] = res.get("error") or "bad_result"
                continue
            p = soft_score_from_result(res, CRITERION_NAMES, w_c)
            p_m[name] = p
            votes.append(1 if res.get("qualifies") else 0)
            reasons[name] = str(res.get("reason") or "")[:120]
            ok_experts.append(name)

        # pairwise agreement among available hard qualifies
        for i, a in enumerate(experts):
            for b in experts[i + 1 :]:
                va = votes[experts.index(a)]
                vb = votes[experts.index(b)]
                if va is None or vb is None:
                    continue
                pair_n[(a, b)] += 1
                if va == vb:
                    agree_pairs[(a, b)] += 1

        # renormalize expert weights over available
        avail = [n for n in experts if p_m.get(n) is not None]
        if not avail:
            P = float("nan")
            lab = "uncertain"
        else:
            wsum = sum(w_e.get(n, 0.0) for n in avail)
            if wsum <= 0:
                ww = {n: 1.0 / len(avail) for n in avail}
            else:
                ww = {n: w_e.get(n, 0.0) / wsum for n in avail}
            P = float(sum(ww[n] * float(p_m[n]) for n in avail))
            lab = hard_label(P, hi=hi, lo=lo)

        rows.append(
            {
                "pmid": pmid,
                "P": P,
                "label": lab,
                "majority": majority_vote(votes),
                "n_experts": len(avail),
                **{f"p_{n}": p_m.get(n) for n in experts},
                **{f"vote_{n}": votes[experts.index(n)] for n in experts},
                "reasons": json.dumps(reasons, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    vc = df["label"].value_counts(dropna=False).to_dict()
    lines = [
        "# AHP wave1 aggregation report",
        "",
        f"Experts: {experts}",
        f"Expert weights: {cfg.get('expert_weights')}",
        f"Criterion weights: {cfg.get('criterion_weights')}",
        f"criterion_CR={cfg.get('criterion_CR')} expert_CR={cfg.get('expert_CR')} note={cfg.get('note')}",
        "",
        f"N={len(df)} label counts: {vc}",
        "",
        "## Pairwise hard-vote agreement",
        "",
    ]
    for (a, b), n in sorted(pair_n.items()):
        rate = agree_pairs[(a, b)] / n if n else 0
        lines.append(f"- {a} vs {b}: {agree_pairs[(a, b)]}/{n} = {rate:.3f}")
    lines += [
        "",
        f"Labels: `{args.out}`",
        f"Config: `{args.config}`",
        "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n": len(df), "label_counts": vc, "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
