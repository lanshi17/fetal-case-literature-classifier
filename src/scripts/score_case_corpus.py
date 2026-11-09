#!/usr/bin/env python3
"""Score screening results with the trained case-only DoRA model.

Default: score only retained rows (``decision != irrelevant``), because the
final case set is retained ∩ case.  Results stream to JSONL and resume by PMID.

Outputs (``--out-dir``, default ``artifacts/screen_full``):
    case_scores.jsonl         pmid / case_score / case_decision per doc
    results_case_dora.csv     screening rows joined with case scores
    case_dora_retained.csv    retained AND case (final case-only set)
    case_dora_relevant.csv    relevant AND case
    case_dora_review.csv      review AND case
    CASE_DORA.md              summary

Usage
-----
    .venv/bin/python src/scripts/score_case_corpus.py --limit 200
    .venv/bin/python src/scripts/score_case_corpus.py --start 0 --limit 5000
    .venv/bin/python src/scripts/score_case_corpus.py --build-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

DEFAULT_PDF_DIR = Path("/data/yangzs/Projects/05_Tools/04_AutomatedPubMed/downloads/pdf")
DEFAULT_RESULTS = ROOT / "artifacts" / "screen_full" / "results.csv"
DEFAULT_OUT = DEFAULT_RESULTS.parent

# Explicit title cues override the design model: a paper that calls itself a
# case report / case series is kept even if the model learned a cohort-like
# pattern from noisy design labels (e.g. large retrospective case series).
EXPLICIT_CASE_RE = re.compile(
    r"(?i)\b(?:case report|case series|"
    r"report (?:of|on) [1-9][0-9]? cases|"
    r"(?:a )?series of [1-9][0-9]? (?:cases|patients|fetuses))\b"
)


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[str(rec["pmid"])] = rec
    return done


def score_pending(args: argparse.Namespace) -> None:
    from case_model import CaseDoraPredictor
    from screen_corpus import ExtractorPool

    pred = CaseDoraPredictor(args.config, device=args.device)
    df = pd.read_csv(args.results, dtype={"pmid": str})
    if args.only_retained:
        df = df[df["decision"] != "irrelevant"].copy()
    df = df.iloc[args.start : (args.start + args.limit if args.limit else None)].reset_index(drop=True)
    done = load_done(args.out_dir / "case_scores.jsonl") if not args.no_resume else {}
    df = df[~df["pmid"].isin(done)].reset_index(drop=True)
    print(f"scoring {len(df)} docs (already done {len(done)})", flush=True)

    out = args.out_dir / "case_scores.jsonl"
    pool = ExtractorPool(args.extract_workers, args.extract_timeout)
    try:
        for start in range(0, len(df), args.batch_docs):
            batch = df.iloc[start : start + args.batch_docs].to_dict("records")
            docs = [
                {
                    "pmid": r["pmid"],
                    "title": r["title"],
                    "text": pool.extract(args.pdf_dir / r["pdf_file"])["text"],
                }
                for r in batch
            ]
            scores = pred.score_docs(docs, batch_docs=args.batch_docs)
            with out.open("a", encoding="utf-8") as f:
                for d, s in zip(docs, scores):
                    explicit = bool(EXPLICIT_CASE_RE.search(str(d.get("title") or "")))
                    model_decision = pred.decide(float(s))
                    rec = {
                        "pmid": d["pmid"],
                        "case_score": round(float(s), 6),
                        "case_decision": "case" if explicit or model_decision == "case" else "non-case",
                        "case_decision_model": model_decision,
                        "explicit_case_title": explicit,
                        "case_decision_p90": pred.decide(float(s), p90=True),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            n = start + len(batch)
            if n % 200 < args.batch_docs or n == len(df):
                print(f"progress {n}/{len(df)}", flush=True)
    finally:
        pool.close()


def build_outputs(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.results, dtype={"pmid": str})
    done = load_done(args.out_dir / "case_scores.jsonl")
    if not done:
        raise SystemExit("no case_scores.jsonl rows")
    scores = pd.DataFrame(done.values())
    if "explicit_case_title" not in scores.columns:
        scores["explicit_case_title"] = False
    if "case_decision_model" not in scores.columns:
        scores["case_decision_model"] = scores["case_decision"]
    scores = scores[
        ["pmid", "case_score", "case_decision", "case_decision_model", "explicit_case_title", "case_decision_p90"]
    ]
    merged = df.merge(scores, on="pmid", how="left")
    explicit = merged["title"].fillna("").str.contains(EXPLICIT_CASE_RE)
    merged["explicit_case_title"] = merged["explicit_case_title"].fillna(False) | explicit
    merged["case_decision"] = np.where(
        merged["explicit_case_title"], "case", merged["case_decision"]
    )
    merged.to_csv(args.out_dir / "results_case_dora.csv", index=False)

    case = merged[merged["case_decision"] == "case"]
    retained = case[case["decision"] != "irrelevant"].sort_values("relevance_score", ascending=False)
    retained.to_csv(args.out_dir / "case_dora_retained.csv", index=False)
    for decision, name in (("relevant", "case_dora_relevant.csv"), ("review", "case_dora_review.csv")):
        case[case["decision"] == decision].sort_values(
            "relevance_score", ascending=False
        ).to_csv(args.out_dir / name, index=False)

    by_decision_total = merged["decision"].value_counts().to_dict()
    by_decision_case = case["decision"].value_counts().to_dict()
    summary = {
        "total": int(len(merged)),
        "scored": int(merged["case_score"].notna().sum()),
        "by_decision_total": {k: int(by_decision_total.get(k, 0)) for k in ("relevant", "review", "irrelevant")},
        "by_decision_case": {k: int(by_decision_case.get(k, 0)) for k in ("relevant", "review", "irrelevant")},
        "retained_total": int((merged["decision"] != "irrelevant").sum()),
        "retained_case": int(len(retained)),
        "retained_case_model": int(((merged["decision"] != "irrelevant") & (merged["case_decision_model"] == "case")).sum()),
        "explicit_title_overrides": int(((merged["decision"] != "irrelevant") & merged["explicit_case_title"] & (merged["case_decision_model"] == "non-case")).sum()),
        "outputs": {
            "case_dora_retained": str(args.out_dir / "case_dora_retained.csv"),
            "case_dora_relevant": str(args.out_dir / "case_dora_relevant.csv"),
            "case_dora_review": str(args.out_dir / "case_dora_review.csv"),
        },
    }
    (args.out_dir / "case_dora_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Case-only DoRA 全量推理",
        "",
        "| 相关性决策 | 总数 | 其中 case | 占比 |",
        "|---|---:|---:|---:|",
    ]
    for d in ("relevant", "review", "irrelevant"):
        t = summary["by_decision_total"][d]
        c = summary["by_decision_case"][d]
        lines.append(f"| {d} | {t:,} | {c:,} | {c/max(t,1):.2%} |")
    lines += [
        f"| **保留（relevant+review）** | **{summary['retained_total']:,}** | "
        f"**{summary['retained_case']:,}** | **{summary['retained_case']/max(summary['retained_total'],1):.2%}** |",
        "",
        f"其中模型判 case {summary['retained_case_model']:,} 篇；标题显式写 case report / case series "
        f"的追加 {summary['explicit_title_overrides']:,} 篇（含 report of N<100 cases / series of N<100 表述）。",
        "",
        "最终文件：`case_dora_retained.csv`（相关性保留 ∩ case）。",
        "",
    ]
    (args.out_dir / "CASE_DORA.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    ap.add_argument("--config", type=Path, default=ROOT / "artifacts" / "case_dora" / "config.json")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-docs", type=int, default=16)
    ap.add_argument("--extract-workers", type=int, default=1)
    ap.add_argument("--extract-timeout", type=float, default=120.0)
    ap.add_argument("--only-retained", action="store_true", default=True)
    ap.add_argument("--all-decisions", dest="only_retained", action="store_false")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.build_only:
        score_pending(args)
    build_outputs(args)


if __name__ == "__main__":
    main()
