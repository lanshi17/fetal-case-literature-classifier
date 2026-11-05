# REVIEW_SPEC — SPECTER2 screening baseline (dsh deliverable)

Status: **spec approval not obtained (autonomous review run)**.
Reviewer: omp + skill `old-coder`. Spec authored *after* implementation (post-hoc
acceptance spec derived from the plan and the implementation brief); RED/GREEN
gating was therefore not applied to the implementation. Confidence claims in
`briefs/EVIDENCE.md` are correspondingly lower and every scenario below is
mapped to an executed check, a SKIP with reason, or FAIL.

Authoritative documents:

- Plan: `/home/yangzs/.cursor/plans/SPECTER2 screening pipeline-e4a3448e.plan.md`
- Implementation brief: `briefs/dsh-implement.md`
- Implementer report: `briefs/dsh-done.md`
- Working directory: `/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening`

Source state under review: git repository with **no commits** (`master` unborn);
all 11 top-level entries are untracked. Source identification therefore uses a
sha256 of the source files recorded in EVIDENCE, not a commit SHA.

Tier: **Tier 2/3 hybrid** — data pipeline + model metrics + public HTTP API.
The failure model that drives the scenarios:

| Failure mode | Scenario |
|---|---|
| PMID join silently drops or duplicates papers | S01 |
| `full_text_available` lies (claims text that isn't there) | S03 |
| Full text / abstract mixed up or fetched from network | S04, S17 |
| Encoder silently skips the classification adapter (base-only) | S05 |
| Feature dimension or chunking drifts from the trained model | S05, S06 |
| `weak`/`model` labels leak into the default training subset | S02, S07 |
| Group split leaks a PMID across train/val/test | S07 |
| Threshold not recall-targeted, or metrics not reproducible | S08, S16 |
| Three-band decision off-by-boundary | S09 |
| API missing/unhealthy or PDF path broken | S11 |
| Large data/model files committed to git | S12 |
| SPECTER2 base accidentally fine-tuned | S13 |
| Reported metrics not backed by artifact files | S14, S16 |

## Setup plan (authorized by this spec)

- No new dependencies. Use the existing project `.venv`.
- No business-code modifications during review. Only review artifacts are written:
  `briefs/REVIEW_SPEC.md`, `briefs/EVIDENCE.md`, `briefs/omp-review.md`,
  `briefs/review_checks.py` (reproducible checker), and scratch output under
  `/tmp/omp-review-*`.
- `train.py` is re-run with `--artifacts /tmp/... --cache-dir artifacts/features`
  purely to prove metric reproducibility; the delivered `artifacts/` files are
  not overwritten.

## Scenarios

### S01 — PMID join is complete and lossless
Given `data/final_labels.csv` (974 rows) and `data/fulltext_package/manifest_fulltext.csv` (974 rows),
when `data/labeled_papers.csv` is joined by normalized PMID, then:
974 output rows; `pmid` set equals the labels' PMID set; 0 duplicate `doc_id`;
`doc_id == group_id == pmid` for every row; `label`, `label_reason`, `weak` equal the source values;
0 PMIDs missing from the manifest.

### S02 — Default training subset excludes weak and model-sourced labels
`weak=True` rows are excluded from the default subset; `label_source == "model"` rows are excluded;
the default subset size equals `weak=False AND label_source != model` recomputed independently
(expected 883 of 974 given 91 weak rows). The full-label control is reported separately.

### S03 — `full_text_available` means "non-empty extracted text"
For every row: `full_text_available == (len(text) > 0)`; `text_source` is one of
`txt|pdf|none`; `text_source == "none"` implies `full_text_available == False`.
For a synthetic record with title+abstract but empty text, `document_features` returns
`full_text_available: false`, zero body mean/max blocks, and the title+abstract block is non-zero.
No row may claim full text when the extracted text is empty.

### S04 — Abstracts come from JATS XML; absence is explicit
Non-empty `abstract` count equals `build_stats.json` `with_abstract` (764). For sampled rows with
an XML file, the CSV abstract equals the JATS `<abstract>` extraction of that file.
Rows with empty abstract have no XML file or an XML without `<abstract>`; 210 empty abstracts
are consistent with 206 PDF-only papers + 4 XML files lacking `<abstract>`.

### S05 — Encoder uses the classification adapter; feature dims are 2304 (+2 aux)
`load_encoder()` reports `adapter_loaded=True`, `adapter_error==""`, and an active adapter setup
containing `CLF`; every model parameter has `requires_grad == False`; the model is in eval mode.
Embeddings with the adapter differ materially from base-only embeddings on the same text
(cosine similarity < 0.999) — i.e. the adapter is actually in the forward path.
`document_features` returns 768 (A: title+abstract), 2304 (B: + body mean/max),
2306 (C: + 2 lexical aux). The delivered `config.json` declares `feature_dim == 2306` and
`model.joblib`'s classifier expects 2306 inputs.

### S06 — Chunking respects max_chars/max_chunks and covers head/middle/tail
No chunk is longer than `max_chars=3000`; at most `max_chunks=24` chunks are returned;
for a long synthetic document the first chunk contains the head marker and the last chunk contains
the tail marker; adjacent chunks overlap (a >0-char shared tail/prefix). Applied to all 974
`text` values in `labeled_papers.csv`, the global maximum chunk length is ≤ 3000.

### S07 — Group-aware 64/16/20 split, both classes present
Splits are built with `GroupShuffleSplit` over `group_id`; the same PMID never appears in two of
train/val/test; every split contains both classes; default-subset sizes are 566/140/177 and the
full-label split is 623/156/195 (seed 42). A fresh `train.py` run with the cached features
reproduces those sizes.

### S08 — Threshold selected on validation for recall ≥ 0.95; test metrics are internally consistent
The validation recall at the selected threshold is ≥ 0.95; the threshold was chosen to maximize
precision subject to that recall constraint. Recomputing recall/precision/F1/F2/confusion from
`artifacts/test_predictions.csv` reproduces `train_summary.json`'s test block. Reported test
recall may be below 0.95 (validation→test generalization), but that must be stated, not hidden.

### S09 — Three-band decision uses the configured boundaries
`decide(t) == "relevant"`, `decide(t - 0.15) == "review"`,
`decide(t - 0.15 + 1e-9) == "review"`, `decide(t - 0.15 - 1e-6) == "irrelevant"`.
An end-to-end `predict.py --sample 1` emits `decision`, `relevance_score`, `threshold`,
`review_threshold`, `full_text_available` and a finite score.

### S10 — Prediction degrades honestly without full text, and PDF input works
A title-only record yields `full_text_available == False`, `pdf_used == False`, a finite score,
and a valid decision. A real PDF path yields `full_text_available == True` and `pdf_used == True`.
Missing/absent PDF paths do not crash.

### S11 — FastAPI exposes `/health` and `/predict` (JSON + multipart PDF)
`GET /health` returns `status == "ok"`, `model_loaded == True`, and the threshold/dim matching
`config.json`. `POST /predict` accepts JSON `{title, abstract, text}` and returns a valid decision.
`POST /predict` accepts multipart with a PDF upload and returns `pdf_used == True`.
Empty JSON `{}` returns 422; malformed JSON returns 400. Service is started with `--workers 1`
per the plan.

### S12 — Git hygiene excludes large local payloads
`git check-ignore` matches: `data/fulltext_package/`, `artifacts/features/*.npz`,
`artifacts/*.joblib`, `.venv/`, `.cache/`. `data/final_labels.csv` remains trackable.
Flag (not fail): `data/labeled_papers.csv` (27 MB) is not ignored and is intended to be committed.

### S13 — SPECTER2 was not fine-tuned
No optimizer, no `.backward()`, no `model.train()` in the delivered pipeline; parameters are frozen
(`requires_grad=False`); only scikit-learn `LogisticRegression` is fitted.
`allenai/specter2_base` snapshot files are not modified by the pipeline (no write path).

### S14 — Required artifacts exist with the promised schema
`artifacts/model.joblib` (model, scaler, threshold, feature_mode, encoder),
`artifacts/config.json` (threshold, feature_dim, encoder paths, split, train subset),
`artifacts/metrics.json` (primary + A/B/C ablations + full-label control + full-text stratification),
`artifacts/test_predictions.csv` (177 rows with score/decision),
`artifacts/train_summary.json`. `metrics.json` test block contains Recall, Precision, F1, F2, AP,
ROC-AUC, confusion matrix.

### S15 — README documents the real workflow
README contains install (venv/requirements), dataset build, training, prediction, and API launch
instructions including `uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1`,
plus the local SPECTER2 base path and the no-fine-tuning / research-only limits.

### S16 — Results are reproducible from cached features
Loading the feature cache with `allow_pickle=False` succeeds; cached `doc_ids` equal
`labeled_papers.csv` `doc_id` order; `labels`/`full_text_available` match the CSV; re-running
`train.py` against the cache reproduces the delivered threshold and test metrics bit-for-bit
(≤1e-9 relative difference) and does not require re-encoding.

### S17 — No network full-text retrieval
`build_dataset.py` contains no HTTP/Entrez/eutils fetching; all text comes from the local package.
Encoder/adapter may use Hugging Face caches only.

### S18 — Declared dependencies resolve in the project venv
`torch, transformers, adapters, sklearn, pandas, numpy, joblib, pymupdf, fastapi, uvicorn,
python_multipart` all import; versions recorded.

### Negative constraints (must survive)
- `data/final_labels.csv` is unmodified (mtime/size unchanged during review; content hash stable).
- Review run does not modify business source files or delivered artifacts.
- No scenario may be silently omitted from EVIDENCE; unimplemented or unexecutable scenarios are
  reported as FAIL/SKIP with the exact reason — never as PASS.

### S19 — Delivered offline workflow works from repo state (added during review)
*Appended after the S05 run exposed a cache-resolution failure; the spec is
append-only, so this is a visible addition, not a silent edit.*

With `HF_HOME` unset (fresh shell) and `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`,
`python encoder.py --smoke` must load the SPECTER2 base snapshot and the
classification adapter from caches that ship with the repo (`.cache/huggingface`).
The same must hold for `predict.py`/`api.py`, which load the encoder through the
same path. Deliverable claim under review: dsh-done §3 "已验证完全离线加载".

Status: **FAIL** on the delivered state — the adapter is only present in the
project-local cache, while `encoder.py` prefers a writable `$HOME`/`HF_HOME`
cache and never falls back to `ROOT/.cache/huggingface`. Workaround used for
S03/S05/S09/S10 in this review: `HF_HOME=$PWD/.cache/huggingface`.
