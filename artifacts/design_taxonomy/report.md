# Design taxonomy (bootstrap)

- n = 974; macro-F1 **0.419**, weighted-F1 **0.726** (5-fold, frozen 1536-d features + LR)
- binary case vs other: P 0.838 / R 0.883 / F1 0.860

| class | n | precision | recall | F1 |
|---|---:|---:|---:|---:|
| case_report | 279 | 0.828 | 0.828 | 0.828 |
| case_series | 97 | 0.385 | 0.464 | 0.421 |
| review | 9 | 0.200 | 0.333 | 0.250 |
| method | 18 | 0.312 | 0.556 | 0.400 |
| validation | 49 | 0.398 | 0.755 | 0.521 |
| screening | 2 | 0.000 | 0.000 | 0.000 |
| registry | 1 | 0.000 | 0.000 | 0.000 |
| cohort | 409 | 0.917 | 0.782 | 0.844 |
| unclear | 110 | 0.576 | 0.445 | 0.503 |

Rules: `case_report` > `case_series` > `review` > `method` > `validation` > `screening` > `registry` > `cohort` > `unclear` (missing/other).
