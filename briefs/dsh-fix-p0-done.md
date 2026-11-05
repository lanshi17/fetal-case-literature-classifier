# DSH fix — P0 offline adapter loading

Status: **FIXED / PASS**. The acceptance command from `briefs/omp-review.md`
now exits 0 with the classification adapter active, without setting
`HF_HOME`.

## 1. Root cause

`encoder.py` selected `HF_HOME` (or `$HOME/.cache/huggingface`) whenever that
directory was writable. In a container where `$HOME` is writable but its hub
cache does not contain `allenai/specter2_classification`, the adapter snapshot
under the repository (`.cache/huggingface/hub/models--allenai--specter2_classification/`)
was never considered. `_resolve_adapter()` then fell through to
`snapshot_download()` without `local_files_only`, and with
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` the load raised `RuntimeError`.

In the current DSH runtime `/home` is mounted read-only, so the old code
happened to take the project-cache branch. The defect still reproduced in the
review environment (writable `$HOME`) and was reproduced here by simulating a
writable but empty home cache.

## 2. Changes

### `encoder.py`

- Added `PROJECT_HF_HOME` / `PROJECT_HF_HUB` constants for the repo cache
  (`encoder.py:35-37`).
- Added local probes that do not touch the network:
  - `_repo_cache_dir()` — maps `org/name` to
    `models--<org>--<name>` (`encoder.py:69`).
  - `_adapter_snapshot_in_cached_hub()` — validates a snapshot by
    `adapter_config.json`, preferring `refs/main`, otherwise the newest
    snapshot (`encoder.py:79`).
  - `_local_hub_roots()` — ordered, de-duplicated candidate hubs
    (`encoder.py:124`).
- Added `_configure_hf_cache()` (`encoder.py:146`, invoked at
  `encoder.py:188`). Selection order:
  1. the cache selected by `HF_HOME` / `HF_HUB_CACHE` when it already holds
     the adapter;
  2. the repo-local `.cache/huggingface` when it holds the adapter (the fix
     for a writable-but-empty `$HOME`);
  3. any writable cache, so a first-time download still has a target.
  It exports a consistent `HF_HOME`, `HF_HUB_CACHE`, and (when set)
  `TRANSFORMERS_CACHE`.
- Added `_offline_mode()` (`encoder.py:53`), honouring `HF_HUB_OFFLINE`,
  `TRANSFORMERS_OFFLINE`, and `SPECTER2_LOCAL_FILES_ONLY`.
- Rewrote `_resolve_adapter()` (`encoder.py:322`) to be local-first:
  1. explicit directory;
  2. selected HF hub cache (`local_files_only=True`);
  3. every local hub candidate, including the repo cache and `$HOME`;
  4. network only when not offline/local-only.
  In offline mode it logs the probed paths and never attempts a download.
- `load_encoder()` (`encoder.py:380`) normalises `local_files_only` from the
  offline env and passes it to the base tokenizer/model and adapter loader.

### `README.md`

- Replaced the old "adapter is downloaded on first use; `HF_HOME` only if
  `$HOME` is read-only" note with the documented local-first resolution order
  (`README.md:64`).
- Documented the default offline smoke command that needs no `HF_HOME`
  (`README.md:66-70`) and the override variables (`README.md:72`).
- Updated the self-check section to use the same offline command
  (`README.md:282-283`).

### Not changed

- No threshold, feature layout/dimension, cache format, model bundle, or API
  contract changes.
- No full re-encode. The cached 974-document / 11,275-chunk feature matrix was
  reused; the review re-run was written to a temporary directory and removed.

## 3. Acceptance

### 3.1 Required command

```bash
env -u HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python encoder.py --smoke
```

Observed (exit 0):

```text
INFO encoder: Loaded adapter allenai/specter2_classification
  (.../.cache/huggingface/hub/models--allenai--specter2_classification/snapshots/d843816b...)
device=cpu dim=768 adapter=allenai/specter2_classification
metadata={'adapter_loaded': True, 'adapter_error': '',
          'active_adapter_setup': 'Stack[[CLF]]', ...}
chunks=1 lengths=[192]
embedding_shape=(768,) embedding_norm=1.0000
feature_shape=(2306,) meta={'full_text_available': True, 'n_chunks': 1, 'text_chars': 192}
finite=True
```

`adapter_loaded=True` and `active_adapter_setup='Stack[[CLF]]'` confirm the
adapter is active, not merely present on disk.

### 3.2 S19 scenario

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "briefs")
import review_checks
review_checks.s19_default_offline_load()
PY
```

```text
PASS S19: HF_HOME unset + HF_HUB_OFFLINE=1: base and adapter resolved from repo caches
```

### 3.3 Full checker

```bash
env -u HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python briefs/review_checks.py --rerun-dir "$PWD/.review-rerun"
```

```text
18 scenarios: 18 PASS, 0 SKIP, 0 FAIL
```

The re-run used `--cache-dir artifacts/features` and reproduced the delivered
threshold, split, and test metrics (S16 PASS). S05 still reports
`cos(adapted, base-only)=0.7681`, i.e. the adapter remains in the forward
path.

### 3.4 Downstream entry points

`predict.py --sample 1` under the same offline env:

```text
"decision": "relevant", "relevance_score": 0.99213, "feature_mode": "C"
```

`uvicorn api:app --port 8124` under the same offline env:

```json
{"status":"ok","model_loaded":true,"feature_mode":"C","feature_dim":2306,
 "device":"cpu","adapter":"allenai/specter2_classification"}
```

## 4. Regression matrix for cache selection

| Environment | Adapter in selected cache? | Chosen cache | Result |
|---|---|---|---|
| `HF_HOME` unset, `$HOME` read-only (this runtime) | no | repo `.cache/huggingface` | PASS |
| `HF_HOME` unset, `$HOME` writable but empty (simulated `HOME=/tmp/dsh-fakehome`) | no | repo `.cache/huggingface` | PASS |
| `HF_HOME=/tmp/dsh-emptyhf` writable, no adapter | no | repo cache via fallback | PASS |
| Selected cache contains the adapter (simulated home with snapshot) | yes | selected cache | PASS |

The second row is the exact review failure mode (writable `$HOME`, no adapter
there); the previous code selected `$HOME` and failed offline.

## 5. Files

- `encoder.py` — cache selection + local-first adapter resolution.
- `README.md` — default offline instructions.
- `briefs/dsh-fix-p0-done.md` — this report.

Reviewer-owned files (`briefs/EVIDENCE.md`, `briefs/omp-review.md`,
`briefs/review_checks.py`) were not modified.
