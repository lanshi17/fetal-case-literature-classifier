# Rule layer evaluation (DoRA + rules)

Threshold (gold-recal): **0.7619**

Pipeline = keep if `relevant|review` (FN only when forced irrelevant).

| set | system | R | P | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| test | model_pipeline | 1.000 | 0.774 | 0.872 | 38 | 0 |
| test | rules_pipeline | 0.977 | 0.830 | 0.898 | 26 | 3 |
| test | rules_auto | 0.854 | 0.835 | 0.844 | 22 | 19 |
| gold49 | model_pipeline | 1.000 | 0.486 | 0.655 | 19 | 0 |
| gold49 | rules_pipeline | 1.000 | 0.581 | 0.735 | 13 | 0 |
| gold49 | rules_auto | 0.944 | 0.708 | 0.810 | 7 | 1 |

## Hard negatives (test, label=0 & hard tag; FP = predicted relevant)
- model Spec=0.370 FP=17 / n=27
- +rules Spec=**0.815** FP=5 / n=27

Test overrides: 32 / 177

Rule codes (test):
- DESIGN_SCREENING: 10
- NIPT_SCREENING: 10
- COHORT_LANGUAGE: 8
- REVIEW_OR_GUIDELINE: 7
- DESIGN_HARD_EXCLUDE: 6
- NIPT_SCREENING_COHORT: 6
- PROTECTED_DEMOTE_ONLY: 5
- LARGE_PRENATAL_COHORT: 2
