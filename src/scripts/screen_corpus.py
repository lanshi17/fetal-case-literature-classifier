#!/usr/bin/env python3
"""Screen the full AutomatedPubMed PDF corpus with the frozen best solution.

Pipeline
--------
1. read the download index (one row per PDF)
2. extract full text from each PDF (isolated subprocess; crashes/hangs are contained)
3. chunk + batch-score with the DoRA model (``artifacts/best``)
4. apply the hard-negative rule layer
5. stream one JSONL row per PMID and checkpoint (resume-safe)

Outputs (``--out-dir``, default ``artifacts/screen_full``):
    results.jsonl    one row per screened document (append-only, resumable)
    results.csv      same rows, sorted relevant -> review -> irrelevant, score desc
    relevant.csv / review.csv / irrelevant.csv
    summary.json     counts, rates, rule hits, score quantiles
    REPORT.md        human-readable summary

Usage
-----
    .venv/bin/python src/scripts/screen_corpus.py --limit 50
    .venv/bin/python src/scripts/screen_corpus.py            # full 23k corpus
    .venv/bin/python src/scripts/screen_corpus.py --resume   # continue an interrupted run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

DEFAULT_INDEX = Path(
    "/data/yangzs/Projects/05_Tools/04_AutomatedPubMed/downloads/index.csv"
)
DEFAULT_PDF_DIR = Path(
    "/data/yangzs/Projects/05_Tools/04_AutomatedPubMed/downloads/pdf"
)
DEFAULT_OUT_DIR = ROOT / "artifacts" / "screen_full"

log = logging.getLogger("screen_corpus")


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def extract_pdf(path: Path) -> dict[str, Any]:
    """Return text plus extraction metadata for one PDF (never raises)."""
    meta: dict[str, Any] = {
        "text": "",
        "n_pages": 0,
        "ocr_needed": False,
        "extract_error": "",
    }
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover
        import fitz

    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # noqa: BLE001
        meta["extract_error"] = f"{type(exc).__name__}: {exc}"
        return meta
    try:
        meta["n_pages"] = doc.page_count
        parts = [(page.get_text("text") or "") for page in doc]
        text = "\n\n".join(parts).replace("\x00", " ").strip()
    finally:
        doc.close()
    meta["text"] = text
    meta["ocr_needed"] = bool(meta["n_pages"] and len(text) < 200)
    return meta


def _error_meta(message: str) -> dict[str, Any]:
    return {"text": "", "n_pages": 0, "ocr_needed": False, "extract_error": message}


class ExtractorClient:
    """One long-lived extraction subprocess (MuPDF can segfault on bad PDFs).

    A crash or hang only loses the current document: the client is restarted
    and the document is recorded with ``extract_error`` instead of killing the
    whole screening run.
    """

    def __init__(self, python: str, timeout: float) -> None:
        self.python = python
        self.timeout = timeout
        self.proc: subprocess.Popen[str] | None = None
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [self.python, "-u", str(Path(__file__).resolve()), "--extract-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def restart(self) -> None:
        self.close()
        self._start()

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                pipe.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
        self.proc = None

    def extract(self, path: Path) -> dict[str, Any]:
        assert self.proc is not None
        assert self.proc.stdin is not None and self.proc.stdout is not None
        try:
            self.proc.stdin.write(json.dumps({"path": str(path)}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.restart()
            return _error_meta("extractor_crash")

        box: list[str] = []

        def _read() -> None:
            try:
                box.append(self.proc.stdout.readline())  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                box.append("")

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(self.timeout)
        if reader.is_alive():
            self.restart()
            return _error_meta("extractor_timeout")
        line = box[0] if box else ""
        if not line:
            self.restart()
            return _error_meta("extractor_crash")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            self.restart()
            return _error_meta("extractor_protocol_error")


class ExtractorPool:
    def __init__(self, n: int, timeout: float) -> None:
        self._clients: queue.Queue[ExtractorClient] = queue.Queue()
        for _ in range(max(1, n)):
            self._clients.put(ExtractorClient(sys.executable, timeout))

    def extract(self, path: Path) -> dict[str, Any]:
        client = self._clients.get()
        try:
            return client.extract(path)
        finally:
            self._clients.put(client)

    def close(self) -> None:
        while not self._clients.empty():
            self._clients.get_nowait().close()


def run_extract_server() -> None:
    """stdin/stdout JSON-line server used by :class:`ExtractorClient`."""
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover
        import fitz
    for name in ("mupdf_display_errors", "mupdf_display_warnings"):
        fn = getattr(fitz.TOOLS, name, None)
        if fn:
            try:
                fn(False)
            except Exception:  # noqa: BLE001
                pass
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            meta = extract_pdf(Path(json.loads(line)["path"]))
        except Exception as exc:  # noqa: BLE001
            meta = _error_meta(f"server:{type(exc).__name__}: {exc}")
        out.write(json.dumps(meta, ensure_ascii=False) + "\n")
        out.flush()


# ---------------------------------------------------------------------------
# Batched scoring
# ---------------------------------------------------------------------------
def score_batch(
    predictor: Any,
    docs: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
) -> list[float]:
    """Score a batch of docs exactly like ``predict_record`` but batched on GPU."""
    import torch
    from finetune_chunk_fair import doc_to_chunks

    chunks = [
        doc_to_chunks(d["title"], "", d["text"], tokenizer) for d in docs
    ]
    flat = [c for doc_chunks in chunks for c in doc_chunks]
    enc = tokenizer(
        flat,
        truncation=True,
        max_length=int(predictor._chunk_tokens),
        padding=True,
        return_tensors="pt",
    )
    owners = torch.repeat_interleave(
        torch.arange(len(docs), dtype=torch.long),
        torch.tensor([len(c) for c in chunks], dtype=torch.long),
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    owners = owners.to(device)
    with torch.inference_mode():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            owners=owners,
            batch_size=len(docs),
        )["logits"]
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
    return [float(p) for p in probs]


def screen_docs(
    predictor: Any,
    docs: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    use_rules: bool,
) -> list[dict[str, Any]]:
    from rule_layer import apply_rules

    probs = score_batch(
        predictor, docs, tokenizer=tokenizer, model=model, device=device
    )
    rows: list[dict[str, Any]] = []
    for doc, score in zip(docs, probs):
        rule = apply_rules(
            doc["title"],
            "",
            doc["text"],
            score=score,
            threshold=predictor.threshold,
            design="",
            enabled=use_rules,
        )
        rows.append(
            {
                **{k: doc[k] for k in DOC_META_FIELDS},
                "decision": rule.decision,
                "model_decision": rule.model_decision,
                "relevance_score": round(score, 6),
                "threshold": predictor.threshold,
                "review_threshold": round(predictor.review_threshold, 6),
                "full_text_available": bool(doc["text"].strip()),
                "text_chars": len(doc["text"]),
                "n_pages": doc.get("n_pages", 0),
                "ocr_needed": bool(doc.get("ocr_needed", False)),
                "extract_error": doc.get("extract_error", ""),
                "rules_fired": [h.code for h in rule.rules_fired],
                "rules_overridden": rule.overridden,
                "rules_protected_by_case": rule.protected_by_case,
            }
        )
    return rows


DOC_META_FIELDS = [
    "pmid",
    "title",
    "year",
    "journal",
    "doi",
    "pdf_file",
]


# ---------------------------------------------------------------------------
# Corpus I/O
# ---------------------------------------------------------------------------
def load_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.rename(
        columns={
            "PMID": "pmid",
            "Title": "title",
            "Journal": "journal",
            "Year": "year",
            "DOI": "doi",
            "File": "pdf_file",
        }
    )
    missing = [c for c in DOC_META_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"index missing columns: {missing}")
    df = df[DOC_META_FIELDS].copy()
    df["pmid"] = df["pmid"].astype(str).str.strip()
    df = df[df["pmid"] != ""]
    df = df.drop_duplicates("pmid", keep="first").reset_index(drop=True)
    return df


def load_screened(out_dir: Path) -> dict[str, dict[str, Any]]:
    path = out_dir / "results.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[str(rec["pmid"])] = rec
    return done


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, append: bool) -> int:
    n = 0
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    return n


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_tables(
    records: list[dict[str, Any]], out_dir: Path, *, corpus_rows: int | None = None
) -> dict[str, Any]:
    df = pd.DataFrame(records)
    rank = {"relevant": 0, "review": 1, "irrelevant": 2}
    df["_rank"] = df["decision"].map(rank).fillna(3)
    df = df.sort_values(["_rank", "relevance_score"], ascending=[True, False])
    df = df.drop(columns=["_rank"])
    cols = [c for c in df.columns if c != "rules_fired"] + ["rules_fired"]

    df.to_csv(out_dir / "results.csv", index=False, columns=cols)
    for decision in ("relevant", "review", "irrelevant"):
        sub = df[df["decision"] == decision]
        sub.to_csv(out_dir / f"{decision}.csv", index=False, columns=cols)

    decisions = Counter(df["decision"])
    fired = Counter(code for codes in df["rules_fired"] for code in codes)
    scores = df["relevance_score"].astype(float)
    years = Counter(df.loc[df["decision"] == "relevant", "year"])
    summary = {
        "corpus_rows": int(corpus_rows) if corpus_rows else int(len(df)),
        "screened": int(len(df)),
        "unique_pmids": int(df["pmid"].nunique()),
        "decisions": {k: int(decisions.get(k, 0)) for k in ("relevant", "review", "irrelevant")},
        "decision_rates": {
            k: round(decisions.get(k, 0) / max(len(df), 1), 4)
            for k in ("relevant", "review", "irrelevant")
        },
        "retained_non_irrelevant": int(decisions.get("relevant", 0) + decisions.get("review", 0)),
        "fulltext": {
            "with_text": int(df["full_text_available"].sum()),
            "without_text": int((~df["full_text_available"]).sum()),
            "ocr_needed": int(df["ocr_needed"].sum()),
            "extract_errors": int((df["extract_error"].astype(str) != "").sum()),
        },
        "rules": {
            "overridden": int(df["rules_overridden"].sum()),
            "protected_by_case": int(df["rules_protected_by_case"].sum()),
            "fired": dict(fired.most_common()),
        },
        "scores": {
            "min": round(float(scores.min()), 4),
            "p05": round(float(scores.quantile(0.05)), 4),
            "p25": round(float(scores.quantile(0.25)), 4),
            "median": round(float(scores.median()), 4),
            "p75": round(float(scores.quantile(0.75)), 4),
            "p95": round(float(scores.quantile(0.95)), 4),
            "max": round(float(scores.max()), 4),
        },
        "relevant_by_year_top": dict(years.most_common(15)),
        "outputs": {
            "results_jsonl": str(out_dir / "results.jsonl"),
            "results_csv": str(out_dir / "results.csv"),
            "relevant_csv": str(out_dir / "relevant.csv"),
            "review_csv": str(out_dir / "review.csv"),
            "irrelevant_csv": str(out_dir / "irrelevant.csv"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "REPORT.md").write_text(
        render_report(summary, df), encoding="utf-8"
    )
    return summary


def render_report(summary: dict[str, Any], df: pd.DataFrame) -> str:
    d = summary["decisions"]
    total = summary["screened"]
    lines = [
        "# 全量文献筛选报告",
        "",
        f"- 语料：AutomatedPubMed PDF 全量（{summary.get('corpus_rows', total):,} 篇，按 PMID 去重）",
        f"- 已筛选：**{total:,}** 篇（唯一 PMID {summary['unique_pmids']:,}）",
        f"- 模型：DoRA + 规则层（阈值 {df['threshold'].iloc[0]:.4f}，review 阈值 {df['review_threshold'].iloc[0]:.4f}）",
        "",
        "## 三档分布",
        "",
        "| 决策 | 篇数 | 占比 |",
        "|---|---:|---:|",
        f"| relevant | {d['relevant']:,} | {summary['decision_rates']['relevant']:.2%} |",
        f"| review | {d['review']:,} | {summary['decision_rates']['review']:.2%} |",
        f"| irrelevant | {d['irrelevant']:,} | {summary['decision_rates']['irrelevant']:.2%} |",
        f"| **保留（relevant+review）** | **{summary['retained_non_irrelevant']:,}** | "
        f"**{summary['retained_non_irrelevant'] / max(total, 1):.2%}** |",
        "",
        "## 全文与规则",
        "",
        f"- 成功抽取正文：{summary['fulltext']['with_text']:,} 篇；"
        f"正文缺失/过短：{summary['fulltext']['without_text']:,} 篇"
        f"（OCR 疑似需要 {summary['fulltext']['ocr_needed']:,}，抽取报错 {summary['fulltext']['extract_errors']:,}）",
        f"- 规则层改判：{summary['rules']['overridden']:,} 篇；"
        f"病例线索保护：{summary['rules']['protected_by_case']:,} 篇",
    ]
    if summary["rules"]["fired"]:
        lines += ["", "| 规则 | 命中篇数 |", "|---|---:|"]
        for code, n in summary["rules"]["fired"].items():
            lines.append(f"| `{code}` | {n:,} |")

    lines += [
        "",
        "## 分数分布",
        "",
        "| min | p05 | p25 | median | p75 | p95 | max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            f"{summary['scores'][k]:.4f}"
            for k in ("min", "p05", "p25", "median", "p75", "p95", "max")
        )
        + " |",
        "",
        "## relevant 年份分布（前 15）",
        "",
        "| 年份 | 篇数 |",
        "|---|---:|",
    ]
    for year, n in summary["relevant_by_year_top"].items():
        lines.append(f"| {year or '未知'} | {n:,} |")

    lines += ["", "## 高置信 relevant 示例（前 10）", ""]
    top = df[df["decision"] == "relevant"].head(10)
    for _, r in top.iterrows():
        lines.append(
            f"- `{r['pmid']}` ({r['year'] or '?'}) score={r['relevance_score']:.4f} — {r['title']}"
        )
    lines += [
        "",
        "## 文件",
        "",
        "- `results.jsonl` / `results.csv`：全部筛选结果（含分数、规则命中、抽取状态）",
        "- `relevant.csv` / `review.csv` / `irrelevant.csv`：三档子集",
        "- `summary.json`：机器可读汇总",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-docs", type=int, default=16)
    ap.add_argument(
        "--extract-workers",
        type=int,
        default=1,
        help="number of isolated extraction subprocesses",
    )
    ap.add_argument(
        "--extract-timeout", type=float, default=120.0, help="seconds per PDF before restart"
    )
    ap.add_argument("--start", type=int, default=0, help="index offset (for sharded runs)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows for a smoke test")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    if "--extract-server" in sys.argv[1:]:
        run_extract_server()
        return
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(args.index)
    full_rows = len(index)
    stop = None if args.limit is None else args.start + args.limit
    index = index.iloc[args.start : stop].reset_index(drop=True)
    total = len(index)
    log.info("index rows=%d out=%s", total, args.out_dir)

    done = load_screened(args.out_dir) if args.resume else {}
    if not args.resume:
        (args.out_dir / "results.jsonl").unlink(missing_ok=True)
    pending = index[~index["pmid"].isin(done)].reset_index(drop=True)
    log.info("already screened=%d pending=%d", len(done), len(pending))

    if pending.empty and args.report_only:
        pass
    elif not pending.empty:
        from predict import load_predictor

        predictor = load_predictor(
            best=True, device=args.device, use_rules=not args.no_rules
        )
        tokenizer, model = predictor.tokenizer, predictor.model
        device = predictor.device
        log.info(
            "model ready device=%s thr=%.4f batch_docs=%d workers=%d",
            device,
            predictor.threshold,
            args.batch_docs,
            args.extract_workers,
        )

        results_path = args.out_dir / "results.jsonl"
        rows = pending[["pmid", "title", "year", "journal", "doi", "pdf_file"]].to_dict(
            "records"
        )
        n_done = 0
        t0 = time.time()
        pool = ExtractorPool(args.extract_workers, args.extract_timeout)
        try:
            for start in range(0, len(rows), args.batch_docs):
                batch = rows[start : start + args.batch_docs]
                docs = [
                    {**r, **pool.extract(args.pdf_dir / r["pdf_file"])} for r in batch
                ]
                scored = screen_docs(
                    predictor,
                    docs,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    use_rules=not args.no_rules,
                )
                write_jsonl(results_path, scored, append=True)
                n_done += len(scored)
                if n_done % 200 < args.batch_docs or n_done == len(rows):
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-9)
                    eta = (len(rows) - n_done) / max(rate, 1e-9)
                    log.info(
                        "progress %d/%d (%.1f%%) %.2f docs/s elapsed=%.0fs eta=%.0fs",
                        n_done,
                        len(rows),
                        100.0 * n_done / len(rows),
                        rate,
                        elapsed,
                        eta,
                    )
        finally:
            pool.close()
        log.info("scored %d docs in %.0fs", n_done, time.time() - t0)

    records = list(load_screened(args.out_dir).values())
    if not records:
        raise SystemExit("no screened records found")
    summary = build_tables(records, args.out_dir, corpus_rows=full_rows)
    log.info(
        "summary: %s",
        json.dumps(
            {
                k: summary[k]
                for k in ("screened", "decisions", "retained_non_irrelevant")
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
