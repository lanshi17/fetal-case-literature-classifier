#!/usr/bin/env python3
"""Regression check for the api.py /health dual-backend fix.

Bug: at 5dd99e2 /health read only legacy-Predictor attributes
(model_path/scaler/encoder/bundle/enc_cfg), so the default best backend
(BestDoraPredictor, none of those) raised AttributeError -> HTTP 500.
Verified live 2026-11-06: pre-fix handler raises, post-fix returns both shapes.

Uses stdlib SimpleNamespace as the app.state boundary only — health()'s
attribute-resolution logic runs unmocked. No test runner in this repo; run as
a script, nonzero exit on any failure:

    .venv/bin/python src/scripts/check_api_health.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from api import health  # noqa: E402


def _req(predictor: Any) -> Any:
    state = types.SimpleNamespace(predictor=predictor)
    app = types.SimpleNamespace(state=state)
    return types.SimpleNamespace(app=app)


BEST = types.SimpleNamespace(
    threshold=0.7619361281394958,
    review_threshold=0.6119361281394958,
    feature_mode="dora_chunk_1536",
    backend="dora_chunk",
    device="cuda",
    use_rules=True,
    cfg={"encoder": {"model_id": "NeuML/pubmedbert-base-embeddings", "max_chunks": 12}},
)

LEGACY = types.SimpleNamespace(
    model_path=ROOT / "artifacts" / "model.joblib",
    threshold=0.192,
    review_threshold=0.042,
    feature_mode="B",
    encoder=types.SimpleNamespace(device="cuda"),
    bundle={"encoder_family": "pubmedbert"},
    enc_cfg={"model_path": "/snap/pubmedbert", "max_chunks": 12},
    scaler=types.SimpleNamespace(n_features_in_=1536),
)

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        failures.append(f"{name}: {detail}")


best_out = health(_req(BEST))
check("best.status", best_out.get("status") == "ok", str(best_out))
check("best.backend", best_out.get("backend") == "dora_chunk", str(best_out.get("backend")))
check("best.family", best_out.get("encoder_family") == "pubmedbert_dora", str(best_out.get("encoder_family")))
check("best.device", best_out.get("device") == "cuda", str(best_out.get("device")))
check("best.no_legacy_fields", "feature_dim" not in best_out and "model_path" not in best_out, str(sorted(best_out)))
check("best.encoder_model", best_out.get("model_path_encoder") == "NeuML/pubmedbert-base-embeddings",
      str(best_out.get("model_path_encoder")))

legacy_out = health(_req(LEGACY))
check("legacy.status", legacy_out.get("status") == "ok", str(legacy_out))
check("legacy.backend", legacy_out.get("backend") == "legacy_lr", str(legacy_out.get("backend")))
check("legacy.dim", legacy_out.get("feature_dim") == 1536, str(legacy_out.get("feature_dim")))
check("legacy.model_path", str(legacy_out.get("model_path")).endswith("model.joblib"),
      str(legacy_out.get("model_path")))
check("legacy.encoder_model", legacy_out.get("model_path_encoder") == "/snap/pubmedbert",
      str(legacy_out.get("model_path_encoder")))

unloaded = health(_req(None))
check("none.status", unloaded.get("status") == "unavailable", str(unloaded))

if failures:
    print("FAIL")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"PASS: /health ok for best+legacy+unloaded ({len(best_out) + len(legacy_out)} keys asserted)")
