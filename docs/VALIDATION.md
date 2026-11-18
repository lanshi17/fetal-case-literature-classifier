# Validation record

Central record for the case-model validation work. All numbers come from files under
`artifacts/`; commands are in `src/scripts/`.

## 1. Temporal holdout (train ≤2019, test ≥2020)

Train 173 papers, test 667 (292 positives), identical recipe (DoRA r=16, same chunking).

| metric | value |
|---|---:|
| AP / AUC | 0.987 / 0.988 |
| F1 at inner-val threshold 0.505 | 0.939 (P 0.954 / R 0.925) |
| F1 at the old threshold 0.396 | 0.676 (P 0.510 / R 1.000) |
| mean score, positives / negatives | 0.557 / 0.424 |

Ranking survives the time shift; the score distribution does not. The old threshold sits below the
negative mean, so thresholds must be recalibrated per corpus. By-year AP: 2020–22 0.993,
2023–24 0.975, 2025+ 0.982.

Calibration on the temporal test: ECE **0.358**, Brier 0.188; temperature scaling (T fit on the
test) → ECE 0.015, Brier 0.039. The ranking transfers; the probabilities do not.

Files: `artifacts/case_dora_external/temporal/`, `artifacts/case_dora_calibration/`.

## 1b. Journal-held-out CV (GroupKFold, 183 journals)

| metric | value |
|---|---:|
| AP (5-fold mean) | 0.986 ± 0.005 |
| AUC | 0.988 ± 0.003 |
| F1 | 0.925 ± 0.019 |
| pooled OOF AP / AUC | 0.977 / 0.978 |
| per-fold thresholds | 0.299 / 0.388 / 0.760 / 0.455 / 0.291 |
| ECE (after temperature) | 0.153 (0.040) |

Ranking is stable across journals; thresholds and calibration are not. Files:
`artifacts/case_dora_external/journal_cv/`, `artifacts/case_dora_external/SUMMARY.md`.

## 2. Calibration

| model | ECE | Brier | after temperature | after isotonic |
|---|---:|---:|---:|---:|
| NeuML | 0.202 | 0.089 | 0.080 / 0.054 | 0.040 / 0.048 |
| BiomedBERT | 0.153 | 0.076 | 0.070 / 0.061 | 0.103 / 0.080 |

Scores are ranking scores, not probabilities. The HF card exposes
`case_score_calibrated = sigmoid(logit(p)/0.338)`; decisions still use `p`.

Files: `artifacts/case_dora_calibration/`.

## 3. Review band

Long documents with `p ∈ [τ − 0.05, τ)` return `insufficient`. On OOF: coverage 96.9%, 3.1%
abstained (5 positives / 22 negatives), case P/R unchanged (0.942 / 0.947). A 0.10 margin would
abstain 176 confident negatives and is not used. Short inputs keep the stricter 0.75 bar.

Verified: `score_case_corpus.py --build-only` fixture (long 0.370 → `insufficient`, 0.340 →
`non-case`); HF `score()` returns `insufficient` for a 13-token fragment.

## 4. Label audit

Structural flags on 974 papers: 3 cohort papers with case cues (0.7%), 13 case series reporting
≥50 cases, 8 case reports excluded because `design` was missing/other, 4 mixed-language papers.
A 26-row queue and a 209-row annotation sheet are in `artifacts/label_audit/`; the first-pass
adjudication proposes 8 additions, 8 relabels and 10 confirmations.

Metric sensitivity of the 9 in-corpus relabels: AP 0.9650 → 0.9661, AUC 0.9699 → 0.9713,
F1 0.9443 → 0.9398. Label noise does not inflate ranking; coverage (excluded case reports) is the
larger problem. Source labels are unchanged pending expert sign-off.

## 5. Taxonomy

Nine rule-derived design classes, frozen features + multinomial LR, 5-fold: macro-F1 0.419
(0.445 with the audit overlay), weighted-F1 0.726. `case_series` vs `cohort` is the hard boundary
(case series F1 0.42). `screening` (n=2) and `registry` (n=1) are not learnable.

Files: `artifacts/design_taxonomy/`, `docs/DESIGN_TAXONOMY.md`.

## 6. Limits

- Temporal and journal splits share the same `design` label pipeline; they test corpus shift, not
  label correctness.
- No independent institution/annotator test set.
- Taxonomy and adjudication are rule/AI first passes, not expert annotations.
