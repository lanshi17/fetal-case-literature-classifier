# Design taxonomy (bootstrap)

- n = 974; macro-F1 **0.445**, weighted-F1 **0.737** (5-fold, frozen 1536-d features + LR)
- binary case vs other: P 0.847 / R 0.907 / F1 0.876

| class | n | precision | recall | F1 |
|---|---:|---:|---:|---:|
| case_report | 285 | 0.868 | 0.832 | 0.849 |
| case_series | 92 | 0.382 | 0.543 | 0.448 |
| review | 9 | 0.364 | 0.444 | 0.400 |
| method | 18 | 0.303 | 0.556 | 0.392 |
| validation | 49 | 0.422 | 0.776 | 0.547 |
| screening | 6 | 0.000 | 0.000 | 0.000 |
| registry | 1 | 0.000 | 0.000 | 0.000 |
| cohort | 412 | 0.918 | 0.760 | 0.831 |
| unclear | 102 | 0.605 | 0.480 | 0.536 |

Rules: `case_report` > `case_series` > `review` > `method` > `validation` > `screening` > `registry` > `cohort` > `unclear` (missing/other).
