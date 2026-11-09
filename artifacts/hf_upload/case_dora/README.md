---
license: apache-2.0
base_model: NeuML/pubmedbert-base-embeddings
library_name: pytorch
pipeline_tag: text-classification
tags:
  - pubmedbert
  - dora
  - case-report
  - case-series
  - prenatal-genetics
  - biomedical
language:
  - en
---

# Prenatal case-only DoRA (case report / case series)

Binary classifier for prenatal genetics literature: **case report / case series**
vs other study designs (cohort, screening/validation, review, method, registry).

This is a **research artifact**, not a clinical decision tool.

## Model details

| Component | Setting |
|---|---|
| Base encoder | [`NeuML/pubmedbert-base-embeddings`](https://huggingface.co/NeuML/pubmedbert-base-embeddings) |
| Adapter | DoRA (PEFT), r=16, α=32, target `query` / `value` |
| Document encoding | 480-token chunks, max 12 chunks, keep first 2; `L2(mean_all) ‖ L2(mean_first_2)` → 1536-d |
| Head | linear 1536 → 2 |
| Trainable params | 611,330 / 110,093,570 (0.56%) |
| Decision threshold | **0.3962** (F1-optimal on out-of-fold scores) |
| High-precision threshold | **0.3502** (P ≥ 0.90) |

Training labels: 864 papers with an existing `design` annotation (376 case report / case series,
488 cohort / screening / review / method / validation / registry; 110 missing or “other” excluded).
The design labels come from an LLM + review annotation pipeline and contain some noise
(e.g. large retrospective case series occasionally labeled as cohort).

## Architecture

![Two-pass architecture](architecture.png)

[PDF version](architecture.pdf) · [Design philosophy](DESIGN_PHILOSOPHY.md)

## Evaluation

5-fold stratified cross-validation, each fold with an internal validation split for early
stopping and threshold selection:

| Metric | Result |
|---|---|
| Test average precision (AP) | **0.987 ± 0.009** |
| Test F1 | **0.938 ± 0.021** |
| Test precision (mean) | 0.926 |
| Test recall (mean) | 0.952 |
| Out-of-fold AUC | **0.970** |
| Out-of-fold AP | 0.965 |
| OOF at threshold 0.396 | P 0.942 / R 0.947 / F1 0.944 |
| OOF at threshold 0.350 | P 0.900 / R 0.960 |

### Per-fold test evaluation matrix

Thresholds are selected on each fold's internal validation split, then applied to the held-out fold.

| Fold | Thr | TP | FP | FN | TN | Precision | Recall | F1 | AP | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6465 | 69 | 3 | 6 | 95 | 0.958 | 0.920 | 0.939 | 0.990 | 0.992 |
| 1 | 0.4327 | 70 | 1 | 5 | 97 | 0.986 | 0.933 | 0.959 | 0.993 | 0.994 |
| 2 | 0.4080 | 74 | 5 | 1 | 93 | 0.937 | 0.987 | 0.961 | 0.996 | 0.997 |
| 3 | 0.2112 | 72 | 11 | 4 | 86 | 0.868 | 0.947 | 0.906 | 0.972 | 0.972 |
| 4 | 0.2449 | 73 | 10 | 2 | 87 | 0.880 | 0.973 | 0.924 | 0.984 | 0.984 |
| **Mean** | — | — | — | — | — | **0.926** | **0.952** | **0.938** | **0.987** | **0.988** |

Out-of-fold confusion at the deployed thresholds:

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3962 (F1-optimal) | 356 | 22 | 20 | 466 | 0.942 | 0.947 | 0.944 |
| 0.3502 (P≥0.90 on val) | 361 | 40 | 15 | 448 | 0.900 | 0.960 | 0.929 |

### Baselines and ablations (5-fold)

All methods use identical splits; thresholds are selected on each fold's inner validation split.
AP and F1 are mean ± std across folds.

| Method | Input | AP | AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Majority | — | — | — | 0.000 | 0.000 | 0.000 |
| TF-IDF + LR | title | 0.963 ± 0.007 | 0.964 | 0.876 | 0.894 | 0.882 ± 0.018 |
| TF-IDF + LR | title+abstract | 0.971 ± 0.016 | 0.974 | 0.938 | 0.910 | 0.923 ± 0.027 |
| Frozen PubMedBERT + LR | title | 0.953 ± 0.010 | 0.956 | 0.872 | 0.873 | 0.869 ± 0.029 |
| Frozen PubMedBERT + LR | title+abstract | 0.961 ± 0.020 | 0.965 | 0.910 | 0.896 | 0.902 ± 0.019 |
| Frozen PubMedBERT + LR | full, mean only | 0.968 ± 0.018 | 0.972 | 0.910 | 0.878 | 0.890 ± 0.046 |
| Frozen PubMedBERT + LR | full, single 512 | 0.960 ± 0.016 | 0.961 | 0.892 | 0.899 | 0.894 ± 0.017 |
| Frozen PubMedBERT + LR | full, 12 chunks | 0.971 ± 0.016 | 0.974 | 0.894 | 0.910 | 0.897 ± 0.049 |
| **DoRA** | **full, 12 chunks** | **0.987 ± 0.009** | **0.988** | **0.926** | **0.952** | **0.938 ± 0.021** |
| DoRA | title+abstract | 0.974 ± 0.008 | 0.975 | 0.916 | 0.936 | 0.926 ± 0.024 |

Reading: frozen features already reach AP 0.971; DoRA fine-tuning adds +0.016 AP. Full text adds a
limited but consistent gain over title+abstract (AP +0.013, F1 +0.012); title+abstract TF-IDF is
already strong (F1 0.923), so study design is mostly explicit in the title/abstract. Chunk pooling
over 12 chunks beats mean-only and single-512 truncation.

The final checkpoint was trained on all 864 rows for 2 epochs (median best epoch across folds).

## Usage

```bash
pip install torch transformers peft safetensors huggingface_hub numpy
```

```python
from modeling_case_dora import CaseDoraPredictor

clf = CaseDoraPredictor.from_pretrained("<repo-id>")
out = clf.score(
    title="Prenatal diagnosis of a 22q11.2 deletion in a fetus with tetralogy of Fallot",
    text="Full article text (methods/results are useful; title alone also works).",
)
print(out)
# {'case_score': 0.93, 'case_decision': 'case',
#  'case_decision_p90': 'case', 'threshold': 0.396..., ...}
```

`modeling_case_dora.py` is included in this repository and implements the chunking,
DoRA-wrapped encoder and pooling exactly as used during training.

## Files

| File | Description |
|---|---|
| `model.safetensors` | full model state (encoder + DoRA adapter + classifier) |
| `config.json` | base model, adapter, chunking, thresholds, CV metrics |
| `modeling_case_dora.py` | self-contained inference code |
| `cv_metrics.json`, `report.md` | cross-validation results |
| `tokenizer.json`, `tokenizer_config.json` | tokenizer snapshot |

## Intended use and limitations

- **Use**: screening/candidate prioritization for prenatal genetic variant case reports in
  research workflows; output should be reviewed by a domain expert.
- **Not for**: clinical diagnosis or treatment decisions.
- **Scope**: prenatal genetics literature; performance outside this domain is unknown.
- **Label noise**: study-design labels are not perfect; large case series may be scored
  “non-case” if the annotation treated them as cohorts. In the accompanying corpus pipeline,
  titles explicitly containing `case report` / `case series` (or `report of N<100 cases`)
  are retained by a rule, independent of this model.
- **Truncation**: very long articles are sampled to 12 chunks (first 2 + evenly spaced),
  so evidence in unsampled sections is not seen.

## License

Apache-2.0, following the base model. Training data (PubMed PDFs) is not redistributed.
