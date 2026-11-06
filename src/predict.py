#!/usr/bin/env python3
"""Score papers with the frozen best solution (DoRA + rules) or legacy LR.

Best solution (default with --best / MODEL_BACKEND=best)
-------------------------------------------------------
    Fair chunk-pool DoRA + gold-recal threshold + rule_layer
    Config: artifacts/best/config.json

Decision rule
-------------
    score >= t                 -> relevant
    t - 0.15 <= score < t      -> review
    score <  t - 0.15          -> irrelevant
    then rule_layer may force irrelevant / demote to review

Usage
-----
    python src/predict.py --best --sample 1
    python src/predict.py --pdf paper.pdf --title "Prenatal case report"
    python src/predict.py --model artifacts/model.joblib   # legacy LR
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

from rule_layer import REVIEW_MARGIN, apply_rules

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "model.joblib"
DEFAULT_BEST_CONFIG = ROOT / "artifacts" / "best" / "config.json"
DEFAULT_DATA = ROOT / "data" / "labeled_papers.csv"

logger = logging.getLogger("predict")


def extract_pdf_text(source: str | Path | bytes) -> tuple[str, dict[str, Any]]:
    """Extract text + basic metadata from a PDF path or raw bytes."""
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover
        import fitz

    if isinstance(source, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(source), filetype="pdf")
    else:
        doc = fitz.open(str(source))
    try:
        parts = [page.get_text("text") for page in doc]
        meta = dict(doc.metadata or {})
    finally:
        doc.close()
    text = "\n\n".join(parts).replace("\x00", " ").strip()
    return text, meta


class Predictor:
    """Loads the frozen PubMedBERT encoder + trained linear head once."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        *,
        device: str | None = None,
        load_encoder_now: bool = True,
        use_rules: bool = True,
    ) -> None:
        from encoder import load_encoder

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model bundle not found: {self.model_path}. Run `python src/train.py` first."
            )
        self.bundle = load(self.model_path)
        self.clf = self.bundle["model"]
        self.scaler = self.bundle["scaler"]
        self.threshold = float(self.bundle["threshold"])
        self.review_threshold = self.threshold - REVIEW_MARGIN
        self.feature_mode = self.bundle.get("feature_mode", "B")
        self.use_rules = bool(use_rules)
        enc_cfg = self.bundle.get("encoder", {})
        self.enc_cfg = enc_cfg
        self.encoder = None
        if load_encoder_now:
            self.encoder = load_encoder(
                enc_cfg.get("model_path") or enc_cfg.get("base_path"),
                device=device,
                max_length=int(enc_cfg.get("max_length", 512)),
                normalize=bool(enc_cfg.get("normalize", True)),
            )

    def _features(self, title: str, abstract: str, text: str) -> tuple[np.ndarray, bool]:
        if self.encoder is None:
            raise RuntimeError("encoder not loaded")
        from encoder import EMBED_DIM, document_features

        feats, meta = document_features(
            title,
            abstract,
            text,
            self.encoder,
            include_fulltext=True,
            include_aux=bool(self.bundle.get("include_aux", False)),
            chunk_tokens=int(self.enc_cfg.get("chunk_tokens", 480)),
            max_chunks=int(self.enc_cfg.get("max_chunks", 12)),
            keep_first=int(self.enc_cfg.get("keep_first", 2)),
            batch_size=8,
        )
        if self.feature_mode == "A":
            feats = feats[EMBED_DIM : 2 * EMBED_DIM]
        elif self.feature_mode == "B":
            feats = feats[: 2 * EMBED_DIM]
        else:
            feats = feats[: 2 * EMBED_DIM + 2]
        return feats.reshape(1, -1), bool(meta["full_text_available"])

    def decide(self, score: float) -> str:
        if score >= self.threshold:
            return "relevant"
        if score >= self.review_threshold:
            return "review"
        return "irrelevant"

    def predict_record(self, record: dict[str, Any]) -> dict[str, Any]:
        title = str(record.get("title") or "").strip()
        abstract = str(record.get("abstract") or "").strip()
        text = str(record.get("text") or "").strip()

        pdf_value = record.get("pdf_path") or record.get("pdf") or record.get("PDF_file")
        pdf_used = bool(record.get("pdf_used", False))
        if pdf_value and not text:
            path = Path(str(pdf_value))
            if not path.is_absolute():
                path = ROOT / path
            if path.exists():
                text, pdf_meta = extract_pdf_text(path)
                pdf_used = True
                if not title:
                    title = str(pdf_meta.get("title") or path.stem)
            else:
                logger.warning("PDF not found: %s", path)

        feats, has_fulltext = self._features(title, abstract, text)
        scaled = self.scaler.transform(feats)
        score = float(self.clf.predict_proba(scaled)[0, 1])
        rule = apply_rules(
            title,
            abstract,
            text,
            score=score,
            threshold=self.threshold,
            design=str(record.get("design") or ""),
            enabled=self.use_rules,
        )
        return {
            "doc_id": record.get("doc_id") or record.get("pmid"),
            "title": title,
            "decision": rule.decision,
            "model_decision": rule.model_decision,
            "relevance_score": round(score, 6),
            "threshold": self.threshold,
            "review_threshold": round(self.review_threshold, 6),
            "full_text_available": has_fulltext,
            "pdf_used": pdf_used,
            "feature_mode": self.feature_mode,
            "encoder_family": self.bundle.get("encoder_family", "pubmedbert"),
            "rules_enabled": self.use_rules,
            "rules_fired": [h.code for h in rule.rules_fired],
            "rules_overridden": rule.overridden,
            "rules_protected_by_case": rule.protected_by_case,
        }

    def predict_pdf(self, pdf_path: str | Path, title: str = "", abstract: str = "") -> dict[str, Any]:
        text, meta = extract_pdf_text(pdf_path)
        return self.predict_record(
            {
                "title": title or str(meta.get("title") or Path(pdf_path).stem),
                "abstract": abstract,
                "text": text,
                "pdf_path": None,
                "pdf_used": True,
                "doc_id": Path(pdf_path).stem,
            }
        )

    def predict_input_file(self, path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("records", payload) if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                records = [records]
        elif suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, keep_default_na=False, dtype=str)
            records = df.to_dict("records")
        else:
            raise ValueError(f"unsupported input type: {path.suffix} (use .json/.csv/.tsv)")
        if limit:
            records = records[:limit]
        return [self.predict_record(r) for r in records]


class BestDoraPredictor:
    """Canonical best solution: chunk-pool DoRA + recal threshold + rules."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_BEST_CONFIG,
        *,
        device: str | None = None,
        use_rules: bool | None = None,
    ) -> None:
        import torch
        from transformers import AutoTokenizer

        sys.path.insert(0, str(ROOT / "src" / "scripts"))
        from finetune_chunk_fair import (  # noqa: WPS433
            ChunkPoolClassifier,
            Example,
            doc_to_chunks,
            resolve_model_path,
            wrap_dora,
        )
        from encoder import DEFAULT_CHUNK_TOKENS, DEFAULT_MODEL as ENC_MODEL

        self.config_path = Path(config_path)
        cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.cfg = cfg
        ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"]
        thr_path = ROOT / cfg["paths"]["threshold_json"]
        self.threshold = float(json.loads(thr_path.read_text())["threshold"]) if thr_path.exists() else float(cfg["threshold"])
        self.review_threshold = self.threshold - float(cfg.get("review_margin", REVIEW_MARGIN))
        self.use_rules = cfg.get("use_rules", True) if use_rules is None else bool(use_rules)
        self.feature_mode = "dora_chunk_1536"
        self.backend = "dora_chunk"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        train_meta = json.loads((ROOT / cfg["paths"]["train_metrics"]).read_text())
        model_path = resolve_model_path(train_meta.get("model_path") or ENC_MODEL)
        dora_r = int(cfg["adapter"]["r"])
        dora_alpha = int(cfg["adapter"]["alpha"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_dir if (ckpt_dir / "tokenizer_config.json").exists() else model_path
        )
        model = ChunkPoolClassifier(model_path)
        model = wrap_dora(model, dora_r, dora_alpha)
        state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
        model.to(device).eval()
        self.model = model
        self._Example = Example
        self._doc_to_chunks = doc_to_chunks
        self._chunk_tokens = int(cfg["encoder"].get("chunk_tokens", DEFAULT_CHUNK_TOKENS))
        logger.info(
            "Best DoRA ready thr=%.4f rules=%s device=%s ckpt=%s",
            self.threshold,
            self.use_rules,
            device,
            ckpt_dir,
        )

    def decide(self, score: float) -> str:
        if score >= self.threshold:
            return "relevant"
        if score >= self.review_threshold:
            return "review"
        return "irrelevant"

    def _score(self, title: str, abstract: str, text: str) -> tuple[float, bool]:
        import torch
        import torch.nn.functional as F

        chunks = self._doc_to_chunks(title, abstract, text, self.tokenizer)
        enc = self.tokenizer(
            chunks,
            truncation=True,
            max_length=self._chunk_tokens,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        owners = torch.zeros(enc["input_ids"].size(0), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                owners=owners,
                batch_size=1,
            )["logits"]
            prob = float(F.softmax(logits, dim=-1)[0, 1].detach().cpu())
        return prob, bool(str(text or "").strip())

    def predict_record(self, record: dict[str, Any]) -> dict[str, Any]:
        title = str(record.get("title") or "").strip()
        abstract = str(record.get("abstract") or "").strip()
        text = str(record.get("text") or "").strip()
        pdf_value = record.get("pdf_path") or record.get("pdf") or record.get("PDF_file")
        pdf_used = bool(record.get("pdf_used", False))
        if pdf_value and not text:
            path = Path(str(pdf_value))
            if not path.is_absolute():
                path = ROOT / path
            if path.exists():
                text, pdf_meta = extract_pdf_text(path)
                pdf_used = True
                if not title:
                    title = str(pdf_meta.get("title") or path.stem)

        score, has_fulltext = self._score(title, abstract, text)
        rule = apply_rules(
            title,
            abstract,
            text,
            score=score,
            threshold=self.threshold,
            design=str(record.get("design") or ""),
            enabled=self.use_rules,
        )
        return {
            "doc_id": record.get("doc_id") or record.get("pmid"),
            "title": title,
            "decision": rule.decision,
            "model_decision": rule.model_decision,
            "relevance_score": round(score, 6),
            "threshold": self.threshold,
            "review_threshold": round(self.review_threshold, 6),
            "full_text_available": has_fulltext,
            "pdf_used": pdf_used,
            "feature_mode": self.feature_mode,
            "encoder_family": "pubmedbert_dora",
            "backend": self.backend,
            "rules_enabled": self.use_rules,
            "rules_fired": [h.code for h in rule.rules_fired],
            "rules_overridden": rule.overridden,
            "rules_protected_by_case": rule.protected_by_case,
        }

    def predict_pdf(self, pdf_path: str | Path, title: str = "", abstract: str = "") -> dict[str, Any]:
        text, meta = extract_pdf_text(pdf_path)
        return self.predict_record(
            {
                "title": title or str(meta.get("title") or Path(pdf_path).stem),
                "abstract": abstract,
                "text": text,
                "pdf_used": True,
                "doc_id": Path(pdf_path).stem,
            }
        )

    def predict_input_file(self, path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("records", payload) if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                records = [records]
        elif suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, keep_default_na=False, dtype=str)
            records = df.to_dict("records")
        else:
            raise ValueError(f"unsupported input type: {path.suffix} (use .json/.csv/.tsv)")
        if limit:
            records = records[:limit]
        return [self.predict_record(r) for r in records]


def load_predictor(
    *,
    best: bool = False,
    model_path: str | Path | None = None,
    device: str | None = None,
    use_rules: bool = True,
) -> Predictor | BestDoraPredictor:
    """Factory: best DoRA (default when --best / MODEL_BACKEND=best) or legacy LR."""
    env_backend = os.environ.get("MODEL_BACKEND", "").strip().lower()
    if best or env_backend in {"best", "dora", "dora_chunk"}:
        if not DEFAULT_BEST_CONFIG.exists():
            raise FileNotFoundError(f"missing best config: {DEFAULT_BEST_CONFIG}")
        return BestDoraPredictor(DEFAULT_BEST_CONFIG, device=device, use_rules=use_rules)
    path = Path(model_path or DEFAULT_MODEL)
    return Predictor(path, device=device, use_rules=use_rules)


def _print_result(result: dict[str, Any], as_json_lines: bool = False) -> None:
    if as_json_lines:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None, help="legacy joblib LR path")
    parser.add_argument("--best", action="store_true", default=None, help="use artifacts/best DoRA+rules (default)")
    parser.add_argument("--legacy-lr", action="store_true", help="force frozen PubMedBERT+LR joblib")
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--abstract", type=str, default="")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-rules", action="store_true", help="disable hard-negative rule layer")
    parser.add_argument("--no-json-lines", action="store_true")
    args = parser.parse_args()

    use_best = not args.legacy_lr and args.model is None
    if args.best is True:
        use_best = True
    predictor = load_predictor(
        best=use_best,
        model_path=args.model or DEFAULT_MODEL,
        device=args.device,
        use_rules=not args.no_rules,
    )
    results: list[dict[str, Any]] = []

    if args.pdf:
        results.append(predictor.predict_pdf(args.pdf, args.title, args.abstract))
    elif args.json:
        results.append(predictor.predict_record(json.loads(args.json)))
    elif args.input:
        results.extend(predictor.predict_input_file(args.input, args.limit))
    elif args.sample:
        df = pd.read_csv(DEFAULT_DATA, keep_default_na=False, dtype=str)
        records = df.head(args.sample).to_dict("records")
        results.extend(predictor.predict_record(r) for r in records)
    elif args.title or args.abstract or args.text:
        results.append(
            predictor.predict_record({"title": args.title, "abstract": args.abstract, "text": args.text})
        )
    else:
        parser.error("provide --pdf, --input, --json, --sample, or metadata/text")

    payload = results[0] if len(results) == 1 else results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(results)} prediction(s) -> {args.output}")
    else:
        _print_result(payload, as_json_lines=len(results) > 1 and not args.no_json_lines)


if __name__ == "__main__":
    main()
