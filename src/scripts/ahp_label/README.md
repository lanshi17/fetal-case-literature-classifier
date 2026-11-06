# AHP multi-model labeling (wave 1)

Light expert panel (defaults):

| Expert | Model | Credentials |
|---|---|---|
| claude | `claude-haiku-5-5` | `ANTHROPIC_API_KEY` |
| gpt | `gpt-6-luna` (fallback `gpt-4o-mini` on linxi) | `JUDGE_API_KEY` + `JUDGE_API_BASE` or `OPENAI_API_KEY` |
| gemini | `gemini-3.8-flash` | Vertex (`GOOGLE_GENAI_USE_VERTEXAI`) or `GOOGLE_API_KEY` |

Override models with `AHP_CLAUDE_MODEL` / `AHP_GPT_MODEL` / `AHP_GEMINI_MODEL`.  
Wave1 run notes: [`artifacts/ahp_wave1/STATUS.md`](../../../artifacts/ahp_wave1/STATUS.md).

## Pipeline

```bash
cd /data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening

# 1) sample 3000 (excludes existing 974)
.venv/bin/python src/scripts/ahp_label/sample_wave.py

# 2) extract PDF text
.venv/bin/python src/scripts/ahp_label/build_corpus.py

# 3) judge (pilot 100 first)
.venv/bin/python src/scripts/ahp_label/judge_models.py --limit 100 --workers 4
# full wave
.venv/bin/python src/scripts/ahp_label/judge_models.py --workers 4

# 4) AHP weights + aggregate
.venv/bin/python src/scripts/ahp_label/ahp_weights.py          # gold-calibrated if overlap
.venv/bin/python src/scripts/ahp_label/ahp_weights.py --uniform-experts  # until gold seed judged
.venv/bin/python src/scripts/ahp_label/aggregate.py

# 5) retrain LR with AHP hard labels train-only
.venv/bin/python src/scripts/ahp_label/merge_and_train.py --device cuda
```

Outputs under `data/ahp_wave1/` and `artifacts/ahp_wave1/`.
