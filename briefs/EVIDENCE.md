# EVIDENCE — SPECTER2 genetic-variant screening baseline (review of dsh deliverable)

Spec: `/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening/briefs/REVIEW_SPEC.md`
Spec approval: **not obtained (autonomous review run)** — the spec was authored after
the implementation, so it is a post-hoc acceptance spec. RED/GREEN gating did not
apply to the implementation; confidence is correspondingly lower.
Tier: **2/3 hybrid** (data pipeline + model metrics + public HTTP API).
Reviewer: omp + skill `old-coder`.
Date: 2026-11-02.

## 0. Source state under review

Git repository has **no commits** (`master` unborn); everything is untracked, so
the state is identified by content hashes instead of a commit SHA.

```
9dccf446ce5844124446a18bbd40baebca4d47df39d8204850db5d911af38d87  build_dataset.py
b1ac8c61ce0932264ff13af8c5afd798a018ed364842452c1ff310abc6214b6f  encoder.py
e90851f3a052d60c5a66a85291b8815e80db816d9e2f24ae1bafe9c67004dc2f  train.py
081b879370cf11f4eac14cf86a14b53139e46ae308c8e256f2eb158da8034d9c  predict.py
8968878d3d1c6b144a9155c185b8e6bc06533823c74384188509e15cd4d5eacd  api.py
e8b7b01cb4135b0cbe441ae3e69d540944d4c1c52ace2a3fe60c3150565c6d17  requirements.txt
8c85b7168e3f65159206928ba9de329ad2b103a7d47cd05d100d06963f10873d  README.md
b2f2774750181ed40a43cab51a7c21ae5fd161f863b5f0c520efdc6db7704390  .gitignore
4e612a1b2731d90426f8f8629442002868a30079f6a3aee9b3fb04b1d571bdad  data/final_labels.csv
```

Hashes re-computed after the review run: **identical** (no business file touched).
`git status --porcelain` shows only the pre-existing untracked top-level entries;
review added `briefs/REVIEW_SPEC.md`, `briefs/EVIDENCE.md`, `briefs/omp-review.md`,
`briefs/review_checks.py`.

Environment (`.venv`): Python 3.12.13; torch 2.15.0+cu132; transformers 4.57.6;
adapters 1.3.0; scikit-learn 1.9.1; pandas 3.1.0; numpy 2.5.4; joblib 1.6.0;
PyMuPDF 2.0; fastapi 0.162.0; uvicorn 0.54.0; python-multipart 0.0.33.

Device note: at review time `torch.cuda.is_available() == True`
(RTX 4060 visible). The delivered `artifacts/features/*.json` records
`"device": "cpu", "cuda_available": false` and `encoding_seconds: 2366.8`.
The report's "GPU unavailable" claim was true at encoding time and is stale in the
current container; it does not indicate a code defect.

## 1. Entry point and final result

```bash
cd /data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening
HF_HOME=$PWD/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python briefs/review_checks.py \
    --rerun-dir /tmp/omp-review-train \
    --json-out /tmp/omp-review-checks.json
```

Final fresh run after the last edit to the checker:

```
18 scenarios: 17 PASS, 0 SKIP, 1 FAIL     (process exit code 1)
```

| # | Scenario | Result |
|---|---|---|
| S01 | PMID join complete/lossless | PASS |
| S02 | default subset excludes weak + model | PASS |
| S03 | `full_text_available` semantics | PASS |
| S04 | abstracts from JATS XML / explicit absence | PASS |
| S05 | classification adapter + 768/2304/2306 dims | PASS |
| S06 | chunk ≤3000 chars, ≤24 chunks, head/tail, overlap | PASS |
| S07 | group split 64/16/20, both classes | PASS |
| S08 | recall-targeted threshold, metrics consistent | PASS |
| S09 | three-band boundaries | PASS |
| S10 | no-full-text degradation, PDF input | PASS |
| S11 | FastAPI `/health` + `/predict` (live) | PASS (separate run, §2.11) |
| S12 | git hygiene for large payloads | PASS |
| S13 | no SPECTER2 fine-tuning | PASS |
| S14 | artifact schema | PASS |
| S15 | README workflow | PASS |
| S16 | reproducibility from feature cache | PASS |
| S17 | no network full-text retrieval | PASS |
| S18 | dependencies import | PASS |
| S19 | offline load from repo caches (**added during review**) | **FAIL** |

## 2. Scenario evidence

### S01 — PMID join
Command: checker S01.
Result: `974/974 rows, unique doc_id, doc_id==group_id==pmid, labels/reason/weak
identical, manifest covers all PMIDs`. Independently: 974 labels rows, 974 unique
PMIDs, 974 manifest rows, 0 duplicate `doc_id`, 0 label/reason/weak mismatches,
`data/labeled_papers.csv` = 974 rows × 21 columns.

### S02 — default subset filter
`weak=True` = 91 rows; `label_source=model` = 71 rows; union excluded = 91, so the
default subset is 883/974. `artifacts/metrics.json:data.n_default_subset = 883`.
`config.json:train_subset = "default (weak=False, label_source!=model)"`.
Full-label control exists with `train_subset = "all"` (test n=195).

### S03 — full-text semantics
All 974 rows satisfy `full_text_available == (len(text) > 0)`; `text_source` ∈
{`txt` 768, `pdf` 206}; no row has `text_source=none` with `full_text_available=True`.
Synthetic empty-body probe via `document_features`:
`full_text_available=False`, `n_chunks=0`, body mean/max blocks all zero,
title+abstract block non-zero.

### S04 — abstracts
Non-empty abstracts = 764 = `build_stats.json.with_abstract`.
24 sampled rows re-extracted from their JATS XML produce byte-identical abstracts.
Empty abstracts = 210 = 206 PDF-only papers + 4 XML files without `<abstract>`.

### S05 — encoder / features
`adapter_loaded=True`, `adapter_error=""`, `active_adapter_setup="Stack[[CLF]]"`,
adapter name `[CLF]`, all model parameters `requires_grad=False`, `model.training=False`.
Adapter-in-forward-path proof: embeddings with the adapter vs
`set_active_adapters(None)` on the same text have **cosine 0.7681** (adapter stack
restored afterwards); i.e. the adapter materially changes the representation.
`feature_dim` = 768 (A) / 2304 (B) / 2306 (C); `document_features(..., include_aux=True)`
returns shape `(2306,)`; `model.joblib` classifier `n_features_in_ = 2306`;
`config.json` declares `feature_mode="C"`, `feature_dim=2306`.
Feature layout: `ta 768 + body_mean 768 + body_max 768 + 2 lexical aux = 2306`
(the 2304 base matches the plan; +2 aux is the optional, documented extra).
No fine-tuning: 0 trainable parameters, no optimizer/backward anywhere (§S13).

### S06 — chunking
Synthetic long document: 24 chunks max, max chunk length 3000, first chunk contains
the head marker, last chunk contains the tail marker, adjacent chunks share a
28-word overlap (probe run with `max_chars=1000`). Across all 974 real documents:
11,275 chunks, global maximum chunk length **3000** (equal to `max_chars`, none over).

### S07 — split
`train.make_splits(seed=42)`: all rows 623/156/195, default subset **566/140/177**.
PMID groups are pairwise disjoint across train/val/test; every split contains both
classes. Default positives: 439/109/130.

### S08 — threshold and metrics
Delivered/reproduced threshold **0.1525784656405449**. Validation (n=140):
recall **0.9633**, precision **0.8898**, confusion TN18/FP13/FN4/TP105 → satisfies
target recall 0.95. Recomputing the threshold from cached features reproduces
0.152578… exactly, and no candidate threshold with validation recall ≥0.95 has
higher precision (max-precision-at-target-recall verified).
Test (n=177): recall 0.9231, precision 0.8000, F1 0.8571, F2 0.8955, AP 0.9315,
ROC-AUC 0.8280, TN17/FP30/FN10/TP120 — recomputed from
`artifacts/test_predictions.csv`, all within 1e-9 of `train_summary.json`.
Test recall is below 0.95 because the threshold was tuned on validation; this is
reported, not hidden.

### S09 — three-band decisions
`decide(t)="relevant"`, `decide(t-0.15)="review"`, `decide(t-0.15+1e-9)="review"`,
`decide(t-0.15-1e-6)="irrelevant"` with t=0.152578 (review_threshold=0.002578).
End-to-end record → `decision`, finite `relevance_score`, `threshold`,
`review_threshold`, `full_text_available` all present.
Test-set distribution: relevant 150 / review 19 / irrelevant 8; review band holds
9 positives and 10 negatives (a meaningful manual-review band).

### S10 — degradation / PDF
Title-only record → `full_text_available=False`, `pdf_used=False`, score 0.0,
valid decision. Real PDF → `pdf_used=True`, `full_text_available=True`, finite
score. Nonexistent PDF path → warning, no crash, finite score.

### S11 — live API (separate run)
```bash
HF_HOME=$PWD/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
DEVICE=cpu MODEL_PATH=$PWD/artifacts/model.joblib \
  .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8123 --workers 1
```
- `GET /health` → HTTP 200
  `{"status":"ok","model_loaded":true,"threshold":0.1525784656405449,
  "review_threshold":0.002578465640544897,"feature_mode":"C","feature_dim":2306,
  "device":"cpu","adapter":"allenai/specter2_classification", ...}`.
- `POST /predict` JSON (title/abstract/text) → HTTP 200
  `{"decision":"relevant","relevance_score":0.99999,"full_text_available":true,
  "pdf_used":false,"feature_mode":"C",...}`.
- `POST /predict` multipart with `pdf=@11218737_...pdf` → HTTP 200
  `{"decision":"irrelevant","relevance_score":4e-06,"full_text_available":true,
  "pdf_used":true,...}` (real model output for that PDF, not a stub).
- `POST /predict` JSON `{}` → HTTP **422**
  `{"detail":"provide title/abstract/text, or upload a PDF ..."}`.
- `POST /predict` malformed JSON → HTTP **400**
  `{"detail":"invalid JSON: ..."}`.
Service started single-worker as the plan requires; startup log confirms adapter
`[CLF]` loaded and `threshold=0.1526 feature_mode=C device=cpu`.

### S12 — git hygiene
`git check-ignore` matches `data/fulltext_package/` (338 MB),
`artifacts/features/*.npz` (8.9 MB), `artifacts/model.joblib`, `.venv/`, `.cache/`.
`data/final_labels.csv` remains trackable.
`data/labeled_papers.csv` (27 MB) is **not** ignored and is intended as a
deliverable; flagged in `omp-review.md` as a repository-size decision.

### S13 — no fine-tuning
No `optimizer`, `AdamW`, `.backward(`, `model.train(`, `Trainer(` in
`train.py|encoder.py|predict.py|api.py|build_dataset.py`; all encoder parameters
`requires_grad=False`; only `StandardScaler` + `LogisticRegression` are fitted.
Base snapshot loads from the read-only HF cache; no code path writes to it.

### S14 — artifacts
`model.joblib` keys: model, scaler, threshold, feature_mode, encoder.
`config.json` keys: threshold, feature_dim, encoder, split, train_subset, target_recall.
`metrics.json`: primary + ablations A (768) / B (2304) / C (2306) + full-label
control (n=195) + `test_by_full_text` (`abstract_only: {"n": 0, "note": "empty stratum"}`).
`test_predictions.csv` = 177 rows with score/decision.

### S15 — README
Present: `requirements.txt`, `build_dataset.py`, `train.py`, `predict.py`,
`uvicorn api:app`, `--workers 1`, `allenai/specter2_classification`,
`specter2_base`, `fulltext_package`, `/health`, `/predict`.
README documents the local base snapshot path, `max_chars=3000`, `max_chunks=24`,
the 768/768/768 (+2) feature layout, and the no-fine-tuning/research-only limits.

### S16 — reproducibility
Feature cache loads with `allow_pickle=False`; shapes (974×768, 974×768, 974×768,
974×2); `doc_ids` equal the CSV order; cached `labels` and `full_text_available`
match the CSV. Cached training re-run:

```bash
.venv/bin/python train.py --artifacts /tmp/omp-review-train \
  --cache-dir "$PWD/artifacts/features"
# real 4.678s; "Cache hit"; no re-encoding
```
Reproduces threshold 0.1525784656405449, split 623/156/195 (default 566/140/177),
and every test metric to ≤1e-9, including AP 0.9314845362025383 and confusion
TN17/FP30/FN10/TP120.

### S17 — no network full-text retrieval
`build_dataset.py` contains no `requests.`, `urllib`, `http(s)://`, `eutils`,
`Entrez`, `esearch`, `efetch`, `Bio.` references; text comes only from
`data/fulltext_package/{txt,pdfs,xml}`. (The encoder may use the HF cache; it did
so with `HF_HUB_OFFLINE=1`.)

### S18 — dependencies
All imports resolve: torch 2.15.0+cu132, transformers 4.57.6, adapters 1.3.0,
sklearn 1.9.1, pandas 3.1.0, numpy 2.5.4, joblib 1.6.0, pymupdf 2.0,
fastapi 0.162.0, uvicorn 0.54.0, multipart 0.0.33. `requirements.txt` lists all
runtime packages.

### S19 — FAIL: delivered offline load cannot resolve the adapter
Expected (spec S19, dsh-done §3 claim): with `HF_HOME` unset and
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, `encoder.py --smoke` loads base +
adapter from repo caches.

```bash
env -u HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python encoder.py --smoke
# RuntimeError: Could not load classification adapter 'allenai/specter2_classification':
#   OSError: Unable to load adapter ... from any source.
```

Root cause: the adapter exists only in `$PWD/.cache/huggingface/hub/models--allenai--specter2_classification`.
`encoder.py` chooses `HF_HOME`/`$HOME/.cache/huggingface` whenever it is writable
(it is, in this container) and `_resolve_adapter()` falls back to the network, never
to the repo-local cache. README line 64 only covers the `$HOME` read-only case.
Workaround used for S03/S05/S09/S10/S11:
`HF_HOME=$PWD/.cache/huggingface` (then the same smoke run prints
`device=... dim=768 adapter=allenai/specter2_classification`,
`active_adapter_setup='Stack[[CLF]]'`, `feature_shape=(2306,)`, `finite=True`).
Impact: `predict.py` and `api.py` will fail to start offline on a fresh shell even
though the repo contains the adapter cache. This is a reproducibility defect, not a
model-quality defect.

## 3. Gauntlet layers

| Layer | Command | Result |
|---|---|---|
| Full test suite | (none exists) | **SKIP** — no tests, no `tests/`, no pytest config in the repo |
| Static types | (none configured) | **SKIP** — no mypy/pyright config; no CI |
| Lint + format | (none configured) | **SKIP** — no ruff/flake8 config |
| Changed-line coverage | (no tests) | **SKIP** — undefined without a suite |
| Mutation testing | (no tests) | **SKIP** — no mutants could be killed; not meaningful without tests |
| Property-based tests | (none) | **SKIP** — no hypothesis tests delivered; boundary probes live in S06/S09 |
| Complexity budget | AST function-length scan | **FLAG** — `train.py:main` 290 lines, `build_dataset.py:build` 139, `train.py:build_feature_cache` 102 (maintainability only) |
| Real execution | encoder smoke, predict sample/PDF/JSON, live uvicorn, cached retrain | **PASS** — see S03/S05/S10/S11/S16 |
| Supply chain & secrets | grep for key patterns; requirements review | **PARTIAL** — no secret patterns; 13 `>=` specifiers, no hashes/pins, `pip-audit` not installed/used |
| Suite health | — | **N/A** — no suite |

Checker soundness (home-grown gate): fails closed on exceptions (S19 proves a real
failure reaches FAIL, process exit 1); every `need()` raises `AssertionError`.
Negative control: a copy of the rerun metrics with `test.recall` decreased by 0.01
(`/tmp/omp-review-train-bad`) produced
`FAIL S16: recall differs: 0.9130769230769231 vs 0.9230769230769231`
and the run exited **1** (verified without a pipeline).
The checker itself was corrected for three harness bugs found during review (Series
truth-value test, AppleDouble PDF selection, numpy `bool()` on an array) and rerun;
those were checker defects, not implementation defects.

## 4. Reported numbers vs independently reproduced numbers

| Quantity | Reported (`dsh-done`) | Reproduced |
|---|---|---|
| Dataset rows / PMID overlap | 974 / 974 | 974 / 974 |
| Full text available | 974 | 974 (768 txt + 206 pdf) |
| Abstracts | 764 | 764 |
| Default subset | 883 (91 weak) | 883 (91 weak, 71 model) |
| Split (default) | 566/140/177 | 566/140/177 |
| Threshold | 0.152578 | 0.152578 (recomputed from cache) |
| Test recall / precision / F1 / F2 / AP | 0.9231 / 0.8000 / 0.8571 / 0.8955 / 0.9315 | all match to 1e-9 |
| Encoder dimension | 2306 (mode C) | 2306; B=2304, A=768 |
| Adapter | `allenai/specter2_classification` | active `[CLF]`, cos(base)=0.768 |
| Chunks | 11,275; max 3000 | 11,275; max 3000 |

## 5. Known limits / confidence statement

- This is a **review of a delivered implementation**, not a TDD session: no
  implementation test suite exists, so none of the usual RED→GREEN proof applies.
  The 18 scenarios above are the reviewer's executable acceptance checks; they cover
  the plan's clauses and the failure modes listed in the spec, not every possible bug.
- S19 is a real FAIL; the delivered offline claim is false as stated.
  Everything else was executed on this machine, with the workaround documented.
- Metric values are a single 64/16/20 group split (177 test papers); no cross-validation
  or confidence intervals. Recall target 0.95 is met on validation (0.9633), not on test
  (0.9231); the report states this.
- The `abstract_only` stratum is empty (all 974 papers have text), so "model without
  full text" performance is unmeasured; only the title+abstract ablation (A) approximates it.
- `data/labeled_papers.csv` (27 MB) is unignored by design; repository size/git-LFS is a
  project decision, not a correctness issue.
- Device/report discrepancy: the report says no GPU; CUDA is available in the current
  container. The artifacts were produced on CPU (recorded in cache metadata).
