# Design taxonomy (bootstrap)

The case-only model is binary (`case report / case series` vs other designs). This document
records a finer taxonomy used to test whether the binary task hides distinguishable classes.

## Labels and precedence

`design` strings are mapped with first-match precedence:

```
case_report > case_series > review > method > validation > screening > registry > cohort > unclear
```

| class | rule | n |
|---|---|---:|
| case_report | `^case report` | 279 |
| case_series | `^case series` | 97 |
| review | contains `review` | 9 |
| method | contains `method` | 18 |
| validation | contains `validation` | 49 |
| screening | contains `screening` | 2 |
| registry | contains `registry` | 1 |
| cohort | contains cohort/retrospective/prospective/observational/metabolomics | 409 |
| unclear | missing or `other` | 110 |

## Bootstrap result

Frozen 1536-d document features + multinomial logistic regression, 5-fold stratified CV
(seed=42), `artifacts/design_taxonomy/`:

| metric | value |
|---|---:|
| macro-F1 (9 classes) | **0.419** |
| weighted-F1 | 0.726 |
| binary case vs other (same features) | P 0.838 / R 0.883 / F1 0.860 |

Per class: case_report F1 0.828, cohort 0.844, case_series **0.421**, validation 0.521,
method 0.400, review 0.250, unclear 0.503. `screening` (n=2) and `registry` (n=1) are not
learnable at this support.

## What this means

- `case report` and `cohort` are separable from frozen features; **case series vs cohort is the
  hard boundary**, consistent with the label-noise note (large retrospective series labeled as
  cohort).
- The binary model's 0.94 F1 therefore overstates how well the underlying 9-way design problem is
  solved.
- This taxonomy is **rule-derived, not expert re-annotated**. The next step is a human-audited
  sample of `case_series` / `cohort` boundaries (80–200 cases), then a mixed/unclear label for
  papers that combine designs.

## Files

- `src/scripts/design_taxonomy.py` — mapping + CV
- `artifacts/design_taxonomy/{metrics.json,report.md,taxonomy.csv}`
