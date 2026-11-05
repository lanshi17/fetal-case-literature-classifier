#!/usr/bin/env python3
"""FastAPI service for the PubMedBERT genetic-variant screening baseline.

The model and frozen encoder are loaded once at startup. Run a single worker:

    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

Endpoints
---------
GET  /health
POST /predict     multipart/form-data (title, abstract, year, pmid, pdf) or
                  application/json (same fields; pdf_base64 optional)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from predict import DEFAULT_BEST_CONFIG, DEFAULT_MODEL, load_predictor, extract_pdf_text

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_PDF_BYTES = int(os.environ.get("MAX_PDF_BYTES", 30 * 1024 * 1024))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    app.state.load_error = None
    model_path = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL)))
    device = os.environ.get("DEVICE") or None
    try:
        logger.info("Loading predictor from %s ...", model_path)
        use_rules = os.environ.get("USE_RULES", "1").strip().lower() not in {"0", "false", "no", "off"}
        backend = os.environ.get("MODEL_BACKEND", "best").strip().lower()
        use_best = backend in {"best", "dora", "dora_chunk"} and DEFAULT_BEST_CONFIG.exists()
        if use_best:
            app.state.predictor = load_predictor(best=True, device=device, use_rules=use_rules)
            enc_dev = getattr(app.state.predictor, "device", "?")
        else:
            app.state.predictor = load_predictor(
                best=False, model_path=model_path, device=device, use_rules=use_rules
            )
            enc_dev = getattr(getattr(app.state.predictor, "encoder", None), "device", "?")
        logger.info(
            "Predictor ready: backend=%s threshold=%.4f feature_mode=%s device=%s rules=%s",
            "best" if use_best else "legacy_lr",
            app.state.predictor.threshold,
            app.state.predictor.feature_mode,
            enc_dev,
            use_rules,
        )
    except Exception as exc:  # noqa: BLE001 - health endpoint reports the failure
        logger.exception("Failed to load predictor")
        app.state.load_error = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(
    title="PubMedBERT genetic-variant screening baseline",
    version="0.2.0",
    description="Relevance screening for prenatal genetic-variant case reports.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "pubmedbert-variant-screening", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return {
            "status": "unavailable",
            "model_loaded": False,
            "error": getattr(request.app.state, "load_error", "model not loaded"),
        }
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": str(predictor.model_path),
        "threshold": predictor.threshold,
        "review_threshold": predictor.review_threshold,
        "feature_mode": predictor.feature_mode,
        "feature_dim": int(predictor.scaler.n_features_in_),
        "device": predictor.encoder.device,
        "encoder_family": predictor.bundle.get("encoder_family", "pubmedbert"),
        "model_path_encoder": predictor.enc_cfg.get("model_path")
        or predictor.enc_cfg.get("base_path"),
        "max_chunks": predictor.enc_cfg.get("max_chunks"),
    }


def _payload_from_form(form: Any) -> tuple[dict[str, Any], bytes | None]:
    record: dict[str, Any] = {}
    for key in ("title", "abstract", "text", "year", "pmid", "doc_id"):
        value = form.get(key)
        if value is not None and not hasattr(value, "filename"):
            record[key] = str(value)
    upload = form.get("pdf") or form.get("file")
    pdf_bytes = None
    if upload is not None and hasattr(upload, "read"):
        if int(getattr(upload, "size", 0) or 0) > MAX_PDF_BYTES:
            raise HTTPException(status_code=413, detail="PDF too large")
        pdf_bytes = upload.file.read()
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise HTTPException(status_code=413, detail="PDF too large")
    return record, pdf_bytes


async def _payload_from_json(request: Request) -> tuple[dict[str, Any], bytes | None]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    record = {
        k: payload[k]
        for k in ("title", "abstract", "text", "year", "pmid", "doc_id")
        if k in payload and payload[k] is not None
    }
    pdf_bytes = None
    if payload.get("pdf_base64"):
        try:
            pdf_bytes = base64.b64decode(payload["pdf_base64"], validate=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="invalid pdf_base64") from exc
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise HTTPException(status_code=413, detail="PDF too large")
    return record, pdf_bytes


@app.post("/predict")
async def predict(request: Request):
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("application/json"):
        record, pdf_bytes = await _payload_from_json(request)
    else:
        form = await request.form()
        record, pdf_bytes = _payload_from_form(form)

    if pdf_bytes:
        try:
            text, meta = await run_in_threadpool(extract_pdf_text, pdf_bytes)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"could not parse PDF: {exc}") from exc
        record["text"] = text
        record["pdf_used"] = True
        if not record.get("title"):
            record["title"] = str(meta.get("title") or "")

    if not any(str(record.get(k) or "").strip() for k in ("title", "abstract", "text")):
        raise HTTPException(
            status_code=422,
            detail="provide title/abstract/text, or upload a PDF with at least one character of text",
        )

    try:
        result = await run_in_threadpool(predictor.predict_record, record)
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), workers=1)
