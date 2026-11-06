#!/usr/bin/env python3
"""MCP entry for the fetal-case literature classifier (DoRA + rules).

Exposes screening tools over Model Context Protocol (stdio by default).

Cursor / Claude Desktop example
-------------------------------
    {
      "mcpServers": {
        "fetal-case-screener": {
          "command": "/abs/path/to/.venv/bin/python",
          "args": ["/abs/path/to/src/mcp_server.py"],
          "env": {"MODEL_BACKEND": "best", "USE_RULES": "1"}
        }
      }
    }

Or from this repo:

    .venv/bin/python src/mcp_server.py
    .venv/bin/python src/mcp_server.py --transport streamable-http --port 8765
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator

from predict import DEFAULT_BEST_CONFIG, DEFAULT_MODEL, extract_pdf_text, load_predictor

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("mcp_server")

mcp = MCPServer(
    "fetal_case_screener_mcp",
    instructions=(
        "Screen prenatal genetic-variant case reports as relevant / review / irrelevant. "
        "Prefer screen_paper with title+abstract+full text when available; use screen_pdf for local PDFs. "
        "Call get_screener_info for model/threshold/rule status. Research use only — not clinical advice."
    ),
)

_predictor: Any | None = None
_load_error: str | None = None
_lock = threading.Lock()


def _env_use_rules() -> bool:
    return os.environ.get("USE_RULES", "1").strip().lower() not in {"0", "false", "no", "off"}


def _env_use_best() -> bool:
    backend = os.environ.get("MODEL_BACKEND", "best").strip().lower()
    return backend in {"best", "dora", "dora_chunk"} and DEFAULT_BEST_CONFIG.exists()


def get_predictor() -> Any:
    """Lazy-load the shared predictor (heavy GPU/CPU model)."""
    global _predictor, _load_error
    if _predictor is not None:
        return _predictor
    with _lock:
        if _predictor is not None:
            return _predictor
        if _load_error is not None:
            raise RuntimeError(_load_error)
        try:
            device = os.environ.get("DEVICE") or None
            use_rules = _env_use_rules()
            if _env_use_best():
                _predictor = load_predictor(best=True, device=device, use_rules=use_rules)
            else:
                model_path = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL)))
                _predictor = load_predictor(
                    best=False, model_path=model_path, device=device, use_rules=use_rules
                )
            logger.info(
                "MCP predictor ready backend=%s thr=%.4f rules=%s",
                getattr(_predictor, "backend", "legacy_lr"),
                float(_predictor.threshold),
                use_rules,
            )
            return _predictor
        except Exception as exc:  # noqa: BLE001
            _load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to load predictor for MCP")
            raise RuntimeError(_load_error) from exc


def _dumps(payload: dict[str, Any] | list[Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


class ScreenPaperInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(default="", description="Paper title")
    abstract: str = Field(default="", description="Abstract text")
    text: str = Field(
        default="",
        description="Full text or body excerpt (strongly preferred when available)",
    )
    pmid: str | None = Field(default=None, description="Optional PubMed ID / doc_id")
    design: str = Field(
        default="",
        description="Optional study-design hint passed to the rule layer (e.g. review, NIPT)",
    )
    use_rules: bool | None = Field(
        default=None,
        description="Override rule layer for this call; default follows USE_RULES env",
    )

    @field_validator("title", "abstract", "text", "design", mode="before")
    @classmethod
    def _none_to_empty(cls, v: Any) -> str:
        return "" if v is None else str(v)


class ScreenPdfInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    pdf_path: str = Field(..., description="Absolute or repo-relative path to a PDF", min_length=1)
    title: str = Field(default="", description="Optional title override")
    abstract: str = Field(default="", description="Optional abstract")
    use_rules: bool | None = Field(
        default=None,
        description="Override rule layer for this call; default follows USE_RULES env",
    )


class ScreenBatchItem(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(default="")
    abstract: str = Field(default="")
    text: str = Field(default="")
    pmid: str | None = Field(default=None)
    design: str = Field(default="")


class ScreenBatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    papers: list[ScreenBatchItem] = Field(
        ...,
        description="Papers to screen (max 20)",
        min_length=1,
        max_length=20,
    )
    use_rules: bool | None = Field(default=None)


def _apply_rules_override(predictor: Any, use_rules: bool | None) -> bool:
    previous = bool(getattr(predictor, "use_rules", True))
    if use_rules is not None:
        predictor.use_rules = bool(use_rules)
    return previous


@mcp.tool(
    name="get_screener_info",
    description=(
        "Return classifier backend, threshold, rule-layer flag, and load status. "
        "Call first when unsure whether the model is ready."
    ),
    annotations=ToolAnnotations(
        title="Get screener info",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def get_screener_info() -> str:
    """Report model readiness and screening configuration."""
    info: dict[str, Any] = {
        "service": "fetal_case_screener_mcp",
        "repo_root": str(ROOT),
        "model_backend_env": os.environ.get("MODEL_BACKEND", "best"),
        "use_rules_env": _env_use_rules(),
        "best_config_exists": DEFAULT_BEST_CONFIG.exists(),
        "legacy_model_exists": DEFAULT_MODEL.exists(),
    }
    try:
        predictor = get_predictor()
        info.update(
            {
                "status": "ok",
                "model_loaded": True,
                "backend": getattr(predictor, "backend", "legacy_lr"),
                "threshold": float(predictor.threshold),
                "review_threshold": float(predictor.review_threshold),
                "feature_mode": getattr(predictor, "feature_mode", None),
                "rules_enabled": bool(getattr(predictor, "use_rules", True)),
                "device": getattr(predictor, "device", None)
                or getattr(getattr(predictor, "encoder", None), "device", None),
            }
        )
    except Exception as exc:  # noqa: BLE001
        info.update({"status": "unavailable", "model_loaded": False, "error": str(exc)})
    return _dumps(info)


@mcp.tool(
    name="screen_paper",
    description=(
        "Screen one paper for prenatal genetic-variant case relevance. "
        "Provide title/abstract and full text when possible. "
        "Returns decision in {relevant, review, irrelevant}, score, and fired rules."
    ),
    annotations=ToolAnnotations(
        title="Screen paper",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def screen_paper(params: ScreenPaperInput) -> str:
    """Classify a single paper from title / abstract / text."""
    if not any(s.strip() for s in (params.title, params.abstract, params.text)):
        return _dumps(
            {
                "error": "provide at least one of title, abstract, or text",
                "hint": "Full text improves recall; title+abstract alone is a fast baseline.",
            }
        )
    try:
        predictor = get_predictor()
    except Exception as exc:  # noqa: BLE001
        return _dumps({"error": f"model not loaded: {exc}"})

    previous = _apply_rules_override(predictor, params.use_rules)
    try:
        result = predictor.predict_record(
            {
                "title": params.title,
                "abstract": params.abstract,
                "text": params.text,
                "pmid": params.pmid,
                "doc_id": params.pmid,
                "design": params.design,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("screen_paper failed")
        return _dumps({"error": f"prediction failed: {type(exc).__name__}: {exc}"})
    finally:
        predictor.use_rules = previous
    return _dumps(result)


@mcp.tool(
    name="screen_pdf",
    description=(
        "Screen a local PDF path. Extracts text with PyMuPDF, then runs the same "
        "DoRA+rules pipeline as screen_paper."
    ),
    annotations=ToolAnnotations(
        title="Screen PDF",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def screen_pdf(params: ScreenPdfInput) -> str:
    """Classify a paper from a PDF file on disk."""
    path = Path(params.pdf_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return _dumps({"error": f"PDF not found: {path}"})
    if path.suffix.lower() != ".pdf":
        return _dumps({"error": f"expected a .pdf file, got: {path.suffix}"})

    try:
        predictor = get_predictor()
    except Exception as exc:  # noqa: BLE001
        return _dumps({"error": f"model not loaded: {exc}"})

    previous = _apply_rules_override(predictor, params.use_rules)
    try:
        text, meta = extract_pdf_text(path)
        if not text.strip() and not params.title.strip() and not params.abstract.strip():
            return _dumps({"error": "PDF has no extractable text and no title/abstract provided"})
        result = predictor.predict_record(
            {
                "title": params.title or str(meta.get("title") or path.stem),
                "abstract": params.abstract,
                "text": text,
                "pdf_used": True,
                "doc_id": path.stem,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("screen_pdf failed")
        return _dumps({"error": f"prediction failed: {type(exc).__name__}: {exc}"})
    finally:
        predictor.use_rules = previous
    return _dumps(result)


@mcp.tool(
    name="screen_batch",
    description=(
        "Screen up to 20 papers in one call. Same fields as screen_paper per item. "
        "Use for small queues; prefer screen_paper for interactive single-item work."
    ),
    annotations=ToolAnnotations(
        title="Screen batch",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
def screen_batch(params: ScreenBatchInput) -> str:
    """Classify a small batch of papers."""
    try:
        predictor = get_predictor()
    except Exception as exc:  # noqa: BLE001
        return _dumps({"error": f"model not loaded: {exc}"})

    previous = _apply_rules_override(predictor, params.use_rules)
    results: list[dict[str, Any]] = []
    try:
        for i, paper in enumerate(params.papers):
            if not any(s.strip() for s in (paper.title, paper.abstract, paper.text)):
                results.append({"index": i, "error": "empty title/abstract/text"})
                continue
            try:
                out = predictor.predict_record(
                    {
                        "title": paper.title,
                        "abstract": paper.abstract,
                        "text": paper.text,
                        "pmid": paper.pmid,
                        "doc_id": paper.pmid or str(i),
                        "design": paper.design,
                    }
                )
                out["index"] = i
                results.append(out)
            except Exception as exc:  # noqa: BLE001
                results.append({"index": i, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        predictor.use_rules = previous
    return _dumps({"count": len(results), "results": results})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
    )
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8765")))
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the predictor before accepting MCP traffic",
    )
    args = parser.parse_args()

    if args.preload:
        get_predictor()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
