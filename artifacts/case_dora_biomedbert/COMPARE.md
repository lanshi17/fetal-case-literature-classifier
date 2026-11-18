# Encoder swap: NeuML vs BiomedBERT

- data: 864 design-labelled papers (376 case / 488 non-case), identical 5-fold split
- base run: `artifacts/case_dora` (NeuML/pubmedbert-base-embeddings)
- new run: `artifacts/case_dora_biomedbert` (.cache/models/biomedbert-base)

## Per-fold test metrics

| Fold | NeuML AP | BiomedBERT AP | ΔAP | NeuML F1 | BiomedBERT F1 | best epoch (base/new) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.990 | 0.980 | -0.010 | 0.939 | 0.932 | 2/2 |
| 1 | 0.993 | 0.998 | +0.006 | 0.959 | 0.980 | 1/4 |
| 2 | 0.996 | 0.968 | -0.028 | 0.961 | 0.954 | 1/1 |
| 3 | 0.972 | 0.978 | +0.007 | 0.906 | 0.907 | 2/3 |
| 4 | 0.984 | 0.986 | +0.002 | 0.924 | 0.941 | 4/4 |
| **mean test AP** | **0.987 ± 0.009** | **0.982 ± 0.010** | **-0.005** | | | |
| **mean test AUC** | **0.988 ± 0.009** | **0.986 ± 0.007** | **-0.002** | | | |

## Out-of-fold

| Metric | NeuML | BiomedBERT | Δ |
|---|---:|---:|---:|
| AP | 0.9650 | 0.9740 | +0.0090 |
| ROC AUC | 0.9699 | 0.9756 | +0.0057 |
| F1 at per-fold val thresholds (primary) | P 0.923 / R 0.952 / F1 0.937 | P 0.928 / R 0.957 / F1 0.942 | F1 +0.0052 |
| F1 at own OOF threshold (NeuML 0.3962 / BiomedBERT 0.4347) | P 0.942 / R 0.947 / F1 0.944 | P 0.944 / R 0.944 / F1 0.944 | F1 -0.0001 |
| F1 at shared threshold 0.3962 | P 0.942 / R 0.947 / F1 0.944 | P 0.906 / R 0.949 / F1 0.927 | F1 -0.0170 |

Paired bootstrap ΔAP (BiomedBERT − NeuML): **+0.0090** (95% CI -0.0018 … +0.0219, 2000 resamples). Per-fold test AP: BiomedBERT wins 3/5.
McNemar at per-fold val thresholds: only-NeuML correct 6, only-BiomedBERT correct 10 (p = 0.454).
McNemar at shared 0.3962: only-NeuML correct 24, only-BiomedBERT correct 10 (p = 0.0243).

## Conclusion

No statistically significant AP difference (CI includes 0). Choose the encoder on operational grounds (license, packaging, reproducibility), not on this delta.

## Caveats

- Same folds, seed, hyperparameters and chunking; only the encoder checkpoint differs.
- Thresholds are selected per fold on the inner validation split, then applied to the held-out fold.
- BiomedBERT weights are the original `pytorch_model.bin` (BertForMaskedLM head discarded by `AutoModel`); NeuML is a SentenceTransformer fine-tune of the same architecture.
