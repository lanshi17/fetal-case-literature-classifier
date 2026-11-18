#!/usr/bin/env python3
"""Label audit for the case report / case series boundary.

Flags papers whose ``design`` label conflicts with explicit text cues and writes
a human-review queue. This is audit support, not re-annotation.

Flags
-----
case_series_large      design=case series but a reported series size >= 50
cohort_with_case_cue   design=cohort but title/abstract says case report/series
case_cue_unlabeled     design missing/other but title says case report/series
mixed_design_language  title/abstract mentions both cohort and case series

Outputs (``artifacts/label_audit``): report.md, metrics.json, review_queue.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "labeled_papers.csv"
OOF = ROOT / "artifacts" / "case_dora" / "cv" / "oof.csv"
OUT = ROOT / "artifacts" / "label_audit"

CASE_CUE = re.compile(r"(?i)\b(?:case report|case series)\b")
COHORT_CUE = re.compile(r"(?i)\bcohort\b")
SIZE_RE = re.compile(r"(?i)\b(\d{1,4})\s*(?:patients|fetuses|fetuses|cases|individuals|probands)\b")


def max_series_size(text: str) -> int:
    nums = [int(n) for n in SIZE_RE.findall(str(text or "")) if int(n) < 5000]
    return max(nums) if nums else 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, low_memory=False, dtype={"pmid": str})
    df["design_l"] = df["design"].fillna("").astype(str).str.strip().str.lower()
    oof = pd.read_csv(OOF, dtype={"pmid": str}).drop_duplicates("pmid")[["pmid", "score"]]
    df = df.merge(oof, on="pmid", how="left")

    head = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.strip()
    body = (head + " " + df["text"].fillna("").str.slice(0, 4000))
    df["case_cue"] = head.str.contains(CASE_CUE)
    df["cohort_cue"] = head.str.contains(COHORT_CUE)
    df["series_size"] = body.map(max_series_size)

    is_cs = df["design_l"].str.match(r"^case\s+series\b")
    is_cr = df["design_l"].str.match(r"^case\s+report\b")
    is_cohort = df["design_l"].str.contains(r"cohort", regex=True, na=False)
    is_other = ~(is_cr | is_cs | is_cohort | df["design_l"].str.contains(
        r"screening|review|method|validation|registry", regex=True, na=False))

    flags = {
        "case_series_large": is_cs & (df["series_size"] >= 50),
        "cohort_with_case_cue": is_cohort & df["case_cue"],
        "case_cue_unlabeled": is_other & df["case_cue"],
        "mixed_design_language": df["case_cue"] & df["cohort_cue"],
    }
    for name, mask in flags.items():
        df[name] = mask

    queue = df[df[list(flags)].any(axis=1)].copy()
    queue["review_flags"] = queue[list(flags)].apply(lambda r: ",".join(k for k in flags if r[k]), axis=1)
    queue = queue.sort_values(["review_flags", "score"], ascending=[True, False])
    cols = ["pmid", "title", "year", "design", "label", "score", "series_size", "case_cue", "cohort_cue", "review_flags"]
    queue[cols].to_csv(OUT / "review_queue.csv", index=False, encoding="utf-8")

    med = df.loc[is_cs & (df["series_size"] > 0), "series_size"].median()
    med = 0.0 if pd.isna(med) else float(med)
    metrics = {
        "n": int(len(df)),
        "n_case_report": int(is_cr.sum()),
        "n_case_series": int(is_cs.sum()),
        "n_cohort": int(is_cohort.sum()),
        "flags": {k: int(v.sum()) for k, v in flags.items()},
        "queue_size": int(len(queue)),
        "cohort_with_case_cue_rate": float((is_cohort & df["case_cue"]).sum() / max(is_cohort.sum(), 1)),
        "case_series_ge50": int((is_cs & (df["series_size"] >= 50)).sum()),
        "case_series_median_size": med,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Label audit: case report / case series boundary",
        "",
        f"- n = {metrics['n']}: case report {metrics['n_case_report']}, case series {metrics['n_case_series']}, cohort {metrics['n_cohort']}",
        f"- cohort papers whose title/abstract says `case report/series`: **{metrics['flags']['cohort_with_case_cue']}** "
        f"({metrics['cohort_with_case_cue_rate']:.1%} of cohort)",
        f"- case series reporting a size >= 50: **{metrics['flags']['case_series_large']}** (median reported size {metrics['case_series_median_size']:.0f})",
        f"- title says case report/series but design is missing/other: **{metrics['flags']['case_cue_unlabeled']}**",
        f"- title/abstract mixes cohort and case-series language: **{metrics['flags']['mixed_design_language']}**",
        f"- review queue: **{metrics['queue_size']}** rows -> `review_queue.csv`",
        "",
        "These flags are structural, not ground truth. The queue is the input for a human audit;",
        "labels must not be changed automatically.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
