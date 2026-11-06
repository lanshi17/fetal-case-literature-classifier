#!/usr/bin/env python3
"""Export DoRA (fair chunk) predictions + FN/FP queues for human review.

Runs on CPU by default so GaLore can keep the GPU.
Writes under artifacts/finetune_fair/dora/:
  gold_predictions.csv, test_predictions.csv
  gold_fn.csv, gold_fp.csv, test_fn.csv, test_fp.csv
  review_queue.csv  (gold errors first, then test FN/FP not yet in gold)
  REVIEW.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from finetune_chunk_fair import (  # noqa: E402
    ChunkPoolClassifier,
    build_examples,
    predict_proba,
    resolve_model_path,
    wrap_dora,
)
from transformers import AutoTokenizer

from train import select_threshold  # noqa: E402

OUT = ROOT / "artifacts" / "finetune_fair" / "dora"


def load_dora(ckpt_dir: Path, device: str):
    meta = json.loads((OUT / "metrics.json").read_text())
    model_path = resolve_model_path(meta.get("model_path"))
    dora_r = int(meta["hyper"].get("dora_r") or meta["model_info"].get("dora_r") or 16)
    dora_alpha = int(meta["hyper"].get("dora_alpha") or meta["model_info"].get("dora_alpha") or 32)
    tok = AutoTokenizer.from_pretrained(ckpt_dir if (ckpt_dir / "tokenizer_config.json").exists() else model_path)
    model = ChunkPoolClassifier(model_path)
    model = wrap_dora(model, dora_r, dora_alpha)
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device)
    model.eval()
    thr = float(meta["threshold"])
    return model, tok, thr, meta


def score_rows(model, tok, rows, device, max_length, batch_size, thr, band=0.15):
    scores = predict_proba(
        model, rows, tok, max_length=max_length, batch_size=batch_size, device=torch.device(device)
    )
    out = []
    for ex, s in zip(rows, scores):
        if s >= thr:
            dec = "relevant"
        elif s >= thr - band:
            dec = "review"
        else:
            dec = "irrelevant"
        out.append(
            {
                "pmid": ex.pmid,
                "doc_id": ex.pmid,
                "label": ex.label,
                "score": float(s),
                "threshold": thr,
                "decision": dec,
                "pred": int(s >= thr),
                "source": ex.source,
                "n_chunks": len(ex.chunks),
            }
        )
    return pd.DataFrame(out)


def enrich(df: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    p = papers.drop_duplicates("pmid").set_index("pmid")
    cols = [c for c in ("title", "year", "design", "label_source", "label_reason", "abstract", "text") if c in p.columns]
    for c in cols:
        df[c] = df["pmid"].map(lambda x, col=c: p.loc[x, col] if x in p.index else "")
    df["text_head"] = df.get("text", pd.Series([""] * len(df))).fillna("").astype(str).str.slice(0, 800)
    if "abstract" in df.columns:
        df["abstract_head"] = df["abstract"].fillna("").astype(str).str.slice(0, 500)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=480)
    args = ap.parse_args()

    ckpt = OUT / "checkpoint"
    if not (ckpt / "model.pt").exists():
        raise SystemExit(f"missing DoRA checkpoint: {ckpt}/model.pt")

    print(f"loading DoRA on {args.device} ...", flush=True)
    model, tok, thr, meta = load_dora(ckpt, args.device)

    train, val, test, gold, data_meta = build_examples(
        data=ROOT / "data" / "labeled_papers.csv",
        ahp_labels=ROOT / "data" / "ahp_wave1" / "labels_ahp.csv",
        ahp_corpus=ROOT / "data" / "ahp_wave1" / "corpus.jsonl",
        tokenizer=tok,
        seed=42,
        include_ahp=False,
    )
    del train, val  # unused for export
    papers = pd.read_csv(ROOT / "data" / "labeled_papers.csv", dtype=str)

    print(f"scoring gold n={len(gold)} thr={thr:.4f}", flush=True)
    gold_df = enrich(score_rows(model, tok, gold, args.device, args.max_length, args.batch_size, thr), papers)
    print(f"scoring test n={len(test)}", flush=True)
    test_df = enrich(score_rows(model, tok, test, args.device, args.max_length, args.batch_size, thr), papers)

    gold_fn = gold_df[(gold_df["label"] == 1) & (gold_df["pred"] == 0)].copy()
    gold_fp = gold_df[(gold_df["label"] == 0) & (gold_df["pred"] == 1)].copy()
    test_fn = test_df[(test_df["label"] == 1) & (test_df["pred"] == 0)].copy()
    test_fp = test_df[(test_df["label"] == 0) & (test_df["pred"] == 1)].copy()

    gold_done = set(pd.read_csv(ROOT / "data" / "gold_standard.csv", dtype=str)["pmid"]) if (
        ROOT / "data" / "gold_standard.csv"
    ).exists() else set()
    # review queue: gold mistakes first, then test errors not in gold
    q_parts = []
    for tag, part in (("gold_fn", gold_fn), ("gold_fp", gold_fp)):
        p = part.copy()
        p["queue_tag"] = tag
        q_parts.append(p)
    for tag, part in (("test_fn", test_fn), ("test_fp", test_fp)):
        p = part[~part["pmid"].isin(gold_done)].copy()
        p["queue_tag"] = tag
        q_parts.append(p)
    queue = pd.concat(q_parts, ignore_index=True) if q_parts else pd.DataFrame()

    OUT.mkdir(parents=True, exist_ok=True)
    gold_df.to_csv(OUT / "gold_predictions.csv", index=False)
    test_df.to_csv(OUT / "test_predictions.csv", index=False)
    gold_fn.to_csv(OUT / "gold_fn.csv", index=False)
    gold_fp.to_csv(OUT / "gold_fp.csv", index=False)
    test_fn.to_csv(OUT / "test_fn.csv", index=False)
    test_fp.to_csv(OUT / "test_fp.csv", index=False)
    # combined FN/FP for gold_review --queue fn_fp (via --fn/--fp once patched)
    pd.concat([gold_fn, test_fn], ignore_index=True).to_csv(OUT / "fn_audit.csv", index=False)
    pd.concat([gold_fp, test_fp], ignore_index=True).to_csv(OUT / "fp_audit.csv", index=False)
    queue.to_csv(OUT / "review_queue.csv", index=False)

    # also write predictions in artifacts format for gold_review
    pred_out = test_df[["pmid", "doc_id", "score", "threshold", "decision"]].copy()
    pred_out.to_csv(OUT / "predictions_for_review.csv", index=False)

    md = [
        "# DoRA review export",
        "",
        f"- threshold={thr:.4f} (val R≥0.95)",
        f"- gold: n={len(gold_df)} FN={len(gold_fn)} FP={len(gold_fp)}",
        f"- test: n={len(test_df)} FN={len(test_fn)} FP={len(test_fp)}",
        f"- review_queue rows={len(queue)} (gold errors first; test errors excluding existing gold)",
        "",
        "## Gold FN",
    ]
    for _, r in gold_fn.iterrows():
        md.append(f"- {r['pmid']} score={r['score']:.3f} | {str(r.get('title') or '')[:80]}")
    md.append("")
    md.append("## Gold FP")
    for _, r in gold_fp.iterrows():
        md.append(f"- {r['pmid']} score={r['score']:.3f} | {str(r.get('title') or '')[:80]}")
    md += [
        "",
        "## How to review",
        "",
        "```bash",
        ".venv/bin/python src/scripts/gold_review.py \\",
        f"  --predictions {OUT / 'predictions_for_review.csv'} \\",
        f"  --fn {OUT / 'fn_audit.csv'} --fp {OUT / 'fp_audit.csv'} \\",
        "  --queue fn_fp --port 8765",
        "```",
        "",
        "Or open `review_queue.csv` / `REVIEW.md` offline.",
    ]
    (OUT / "REVIEW.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "thr": thr,
        "gold_fn": int(len(gold_fn)),
        "gold_fp": int(len(gold_fp)),
        "test_fn": int(len(test_fn)),
        "test_fp": int(len(test_fp)),
        "queue": int(len(queue)),
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
