#!/usr/bin/env bash
# AHP wave1 orchestration (light experts: haiku / gpt-6-luna / gemini-3.8-flash)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
ART="${ROOT}/artifacts/ahp_wave1"
mkdir -p "$ART" data/ahp_wave1/cache

echo "== sample =="
$PY src/scripts/ahp_label/sample_wave.py

echo "== corpus =="
$PY src/scripts/ahp_label/build_corpus.py

LIMIT_ARG=()
if [[ "${1:-}" == "--pilot" ]]; then
  LIMIT_ARG=(--limit 100)
  echo "== judge pilot 100 =="
else
  echo "== judge full 3000 =="
fi

$PY src/scripts/ahp_label/judge_models.py --experts claude gpt gemini --workers 3 "${LIMIT_ARG[@]}"

echo "== ahp weights =="
$PY src/scripts/ahp_label/ahp_weights.py --uniform-experts
# if gold overlap exists after judging gold PMIDs, recalibrate:
$PY src/scripts/ahp_label/ahp_weights.py || true

echo "== aggregate =="
$PY src/scripts/ahp_label/aggregate.py

echo "== train =="
$PY src/scripts/ahp_label/merge_and_train.py --device "${DEVICE:-cuda}"

echo "Done. See $ART"
