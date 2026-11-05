#!/usr/bin/env python3
"""Stratified sample of N PMIDs from AutomatedPubMed downloads, excluding labeled set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = Path("/data/yangzs/Projects/05_Tools/04_AutomatedPubMed/downloads/index.csv")
DEFAULT_LABELED = ROOT / "data" / "labeled_papers.csv"
DEFAULT_OUT = ROOT / "data" / "ahp_wave1" / "sample_3000.csv"


def journal_bucket(name: str) -> str:
    s = (name or "").strip().lower()
    if not s:
        return "unknown"
    # first letter bucket keeps coarse diversity without exploding strata
    return s[0] if s[0].isalpha() else "other"


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample ~n rows with decade × journal_bucket proportional allocation."""
    rng = np.random.default_rng(seed)
    work = df.copy()
    work["year_bin"] = pd.to_numeric(work["Year"], errors="coerce").fillna(0).astype(int)
    work["decade"] = (work["year_bin"] // 10) * 10
    work["jbucket"] = work["Journal"].map(journal_bucket)
    work["stratum"] = work["decade"].astype(str) + "|" + work["jbucket"]

    groups = {k: g for k, g in work.groupby("stratum", sort=False)}
    keys = list(groups.keys())
    counts = np.array([len(groups[k]) for k in keys], dtype=float)
    props = counts / counts.sum()
    raw = props * n
    targets = np.floor(raw).astype(int)
    rem = int(n - targets.sum())
    if rem > 0:
        for i in np.argsort(-(raw - targets))[:rem]:
            targets[i] += 1
    for i, k in enumerate(keys):
        targets[i] = min(int(targets[i]), len(groups[k]))
    short = n - int(targets.sum())
    if short > 0:
        order = np.argsort(-counts)
        for i in order:
            room = len(groups[keys[i]]) - int(targets[i])
            if room <= 0:
                continue
            take = min(room, short)
            targets[i] += take
            short -= take
            if short <= 0:
                break

    parts: list[pd.DataFrame] = []
    for i, k in enumerate(keys):
        need = int(targets[i])
        if need <= 0:
            continue
        g = groups[k]
        if need >= len(g):
            parts.append(g)
        else:
            pick = rng.choice(g.index.to_numpy(), size=need, replace=False)
            parts.append(work.loc[pick])
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(out) > n:
        out = out.iloc[:n].reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_INDEX.parent / "pdf")
    args = ap.parse_args()

    idx = pd.read_csv(args.index, dtype=str)
    for col in ("PMID", "File"):
        if col not in idx.columns:
            raise SystemExit(f"index missing column {col}")
    labeled = pd.read_csv(args.labeled, dtype=str)
    exclude = set(labeled["pmid"].astype(str)) | set(labeled.get("doc_id", labeled["pmid"]).astype(str))

    idx["PMID"] = idx["PMID"].astype(str)
    pool = idx[~idx["PMID"].isin(exclude)].copy()
    # require file exists
    pdf_dir = args.pdf_dir
    pool["pdf_path"] = pool["File"].map(lambda f: str(pdf_dir / f))
    exists = pool["File"].map(lambda f: (pdf_dir / f).is_file())
    pool = pool[exists].reset_index(drop=True)
    if len(pool) < args.n:
        raise SystemExit(f"only {len(pool)} eligible PDFs after exclude/exists; need {args.n}")

    sample = stratified_sample(pool, args.n, args.seed)
    sample["wave"] = "ahp_wave1"
    sample["seed"] = args.seed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(
        f"Wrote {len(sample)} rows -> {args.out}\n"
        f"excluded_labeled={len(exclude)} pool={len(pool)} "
        f"years={sample['Year'].value_counts().head(5).to_dict()}"
    )


if __name__ == "__main__":
    main()
