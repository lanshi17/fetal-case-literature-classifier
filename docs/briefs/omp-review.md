# omp review — SPECTER2 screening baseline (dsh deliverable)

Verdict: **PARTIAL PASS — 1 blocking defect, 17/18 executable scenarios PASS.**
Business logic matches the plan (join, filtering, encoder, split, threshold,
three-band decision, API, git hygiene, no fine-tuning). The defect is operational:
the delivered offline load path cannot find the classification adapter.

Evidence: `briefs/EVIDENCE.md` (spec: `briefs/REVIEW_SPEC.md`; checker:
`briefs/review_checks.py`). Spec approval: not obtained (autonomous review).
No business code was modified by the review.

## P0 — blocking (reproducibility)

### 1. Adapter cache is unreachable in a fresh shell → `predict.py` / `api.py` cannot start offline
- **Symptom** (`briefs/EVIDENCE.md` S19):
  ```bash
  env -u HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python encoder.py --smoke
  # RuntimeError: Could not load classification adapter 'allenai/specter2_classification'
  # OSError: Unable to load adapter ... from any source
  ```
- **Cause**: the adapter exists only under
  `$PWD/.cache/huggingface/hub/models--allenai--specter2_classification/snapshots/d843816b…`.
  `encoder.py` takes `HF_HOME`/`$HOME/.cache/huggingface` whenever writable and
  `_resolve_adapter()` (encoder.py:196-218) falls back to the network instead of the
  repo-local cache. README §2 line 64 documents only the `$HOME`-read-only case.
- **Fix (pick one, prefer 1+2)**:
  1. In `_resolve_adapter`, before any network path, probe
     `ROOT/.cache/huggingface/hub/models--<org>--<name>/snapshots/*/adapter_config.json`
     and return that directory; pass `local_files_only` through (env
     `SPECTER2_LOCAL_FILES_ONLY=1` or a `load_encoder(..., local_files_only=True)`
     default for predict/api).
  2. Set `HF_HOME` to `ROOT/.cache/huggingface` when the repo cache contains the
     adapter, instead of leaving it to `$HOME` discovery.
  3. Document explicitly: `export HF_HOME="$PWD/.cache/huggingface"` and add
     `SPECTER2_ADAPTER=/abs/path/snapshot` to README and to the `dsh-done` command list.
- **Acceptance**:
  ```bash
  env -u HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python encoder.py --smoke
  # exit 0, adapter_loaded=True, feature_shape=(2306,)
  env -u HF_HOME HF_HUB_OFFLINE=1 MODEL_PATH=$PWD/artifacts/model.joblib \
    .venv/bin/uvicorn api:app --port 8124 --workers 1   # /health -> status ok
  ```
  Then the S19 line in EVIDENCE can be re-run for PASS.

## P1 — should fix before calling this a baseline

### 2. No automated tests at all
The repo has zero tests (`no tests/`, no pytest config), so coverage/mutation layers
are SKIP and every regression relies on the reviewer's harness. Add a small pytest
suite with no GPU:
- `build_dataset`: PMID join is lossless; `full_text_available == bool(text)`;
  abstract empty when XML has no `<abstract>`; label/reason/weak preserved.
- `encoder.chunk_text`: `len(c) <= max_chars`, `len(chunks) <= max_chunks`, head/tail
  markers preserved, overlap > 0 (the S06 probes can be lifted verbatim).
- `train.select_threshold`: returns a threshold with recall ≥ target, max precision;
  handles all-positive / zero-positive inputs.
- `predict.Predictor.decide`: the four S09 boundary cases.
- `api`: `TestClient` `/health` and `POST /predict` (monkeypatch `Predictor` so no
  encoder is loaded), plus 422 on `{}` and 400 on malformed JSON.
Acceptance: `python -m pytest -q` passes locally, and `pytest --cov` exits nonzero
under a changed-line threshold.

### 3. Dependencies are floor-pinned only
`requirements.txt` has 13 `>=` specifiers and no upper bounds/hashes (only
`transformers<5`). The delivered state was tested with exact versions recorded in
`dsh-done`/EVIDENCE. Add `constraints.txt` (or a lock file) with the tested versions
and run `pip-audit` once; record the result.

### 4. `train.py:main` is 290 lines
Split into `prepare_split()`, `run_ablations()`, `run_full_control()`,
`persist_artifacts()`. Behaviour must stay byte-identical: the cached re-run already
reproduces threshold/metrics to 1e-9, so use that as the regression gate.

## P2 — quality / hygiene

5. **`/health` returns HTTP 200 when the model failed to load** (`api.py:81-93`:
   `status="unavailable"` in a 200 body). Orchestrators that watch status codes will
   consider a broken service healthy. Return `JSONResponse(..., status_code=503)`.
6. **`data/labeled_papers.csv` is 27 MB and not ignored.** Decide explicitly:
   commit as-is, move to Git LFS, or ignore + require `build_dataset.py`. Do not let it
   land in git by accident; the repo currently has no commits, so this is the moment.
7. **`config.json` / bundle embed absolute paths** (`/data/yangzs/...`). Prefer
   relative paths or `SPECTER2_BASE_PATH` overrides; `load_encoder` already falls back
   to `allenai/specter2_base`, but the delivered config is machine-specific.
8. **`predict.py:_features` hardcodes slices 768/2304/2306** while the bundle carries
   `feature_dim`/`feature_mode`/`include_aux`. Derive the slice from the bundle and
   raise a clear error on mismatch (a mode/`include_aux` inconsistency would currently
   fail inside sklearn with a dimension error).
9. **`review_threshold` rounding is inconsistent**: `predict.py` returns
   `round(self.review_threshold, 6)` next to an unrounded `threshold`. Keep both at full
   precision (or both rounded) so clients can reproduce the boundary exactly.
10. **Report drift**: `dsh-done` says GPU unavailable/`nvidia-smi` fails; at review time
    CUDA is available (RTX 4060). The artifacts were encoded on CPU
    (`features/*.json: device=cpu`, 2366.8 s), so this is environment drift — correct
    the wording so the report is not read as a permanent constraint.
11. **`abstract_only` stratum is empty** (974/974 have full text). The stratification
    code is correct but uninformative. Optional: evaluate a deterministic
    "drop-text" variant (reuse mode A features) to quantify degradation; state it as an
    evaluation-only ablation, not a second production path.
12. **Manifest/package mismatch is handled but should stay visible**: manifest says
    TXT/XML for 929, package contains 768 (`txt/`, `xml/`), 206 PDFs; build falls back
    to PDF and records `text_source`. Keep this note in README; it is already accurate.
13. **`build_dataset.resolve_member` glob fallback** (`f"{pmid}_*"`) returns the first
    match. Safe for this package (one file per PMID), but if the archive ever ships
    multiple renditions the choice is silent. Sort by name and log ambiguity if >1.

## Leakage assessment (requested)

- Feature extraction runs a frozen encoder over `title/abstract/text` only
  (`encoder.document_features_batch`); no label, `reason`, `confidence`, or
  `label_source` column enters the feature vector or the cache fingerprint.
- `StandardScaler` is fitted on the training indices only
  (`train.py:389-391`); val/test are transformed with train statistics.
- The threshold is chosen on validation scores only (`train.py:400`); the test split
  is scored once per feature mode and only reported.
- `group_id == PMID` and PMIDs are unique, so `GroupShuffleSplit` cannot leak a
  document across splits (S07 checks disjointness explicitly). Feature cache order
  is asserted equal to the CSV order (S16).
- Cited risk, not a defect: the single 177-paper test set is used to report A/B/C
  ablations, so comparing modes is a mild test-set selection. There is no nested CV
  or confidence interval. For a baseline this is acceptable if the numbers are read
  as point estimates (EVIDENCE §5 says so).

## What passed (no action)

- PMID join 974/974; `doc_id == group_id == pmid`; labels/reason/weak identical.
- `full_text_available` is exactly "non-empty extracted text"; empty body yields zero
  mean/max and `full_text_available=false` (no fake text).
- Encoder: classification adapter `[CLF]` is active and materially changes embeddings
  (cos 0.768 vs base); features 768 / 2304 / **2306** with the documented +2 aux;
  chunks ≤3000 chars, ≤24, head/middle/tail covered.
- Training default subset excludes `weak` (91) and `label_source=model` (71) → 883;
  full-label control reported separately; `GroupShuffleSplit` with disjoint PMIDs,
  both classes, 566/140/177.
- Threshold selected on validation at recall 0.9633 ≥ 0.95 (max precision), test
  recall 0.9231 reported honestly; AP/F1/F2/confusion reproduce to 1e-9.
- Three-band decision exact at `t`, `t-0.15`; API `/health`, `/predict` JSON and
  multipart PDF work; `{}` → 422, malformed JSON → 400.
- `.gitignore` covers the 338 MB package, feature cache, models, venv, project cache.
- No SPECTER2 fine-tuning: 0 trainable params, no optimizer/backward, only the linear head.
- Cached retrain reproduces the delivered artifacts in 4.7 s (`Cache hit`).

## Re-dispatch acceptance (run after fixes)

```bash
cd /data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening
# 1) offline start, no HF_HOME, must pass
env -u HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python encoder.py --smoke
# 2) full acceptance: 18/18 checker PASS, exit 0 (S11 live API is exercised separately)
env -u HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PWD/.venv/bin/python" briefs/review_checks.py --rerun-dir /tmp/omp-review-train
# 3) tests (after P1 #2)
.venv/bin/python -m pytest -q
```
