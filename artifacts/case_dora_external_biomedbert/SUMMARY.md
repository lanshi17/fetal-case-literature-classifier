# External validity

## temporal (train <=2019, test >2019)

| metric | value |
|---|---:|
| folds | 1 |
| test docs | 667 |
| AP (mean ± std) | 0.965 ± 0.000 |
| AUC (mean ± std) | 0.968 ± 0.000 |
| F1 (mean ± std) | 0.839 ± 0.000 |
| precision / recall (mean) | 0.969 / 0.740 |
| pooled OOF AP / AUC | 0.965 / 0.968 |
| selected thresholds | 0.483 |

## journal GroupKFold (5 folds)

| metric | value |
|---|---:|
| folds | 5 |
| test docs | 864 |
| AP (mean ± std) | 0.985 ± 0.008 |
| AUC (mean ± std) | 0.986 ± 0.004 |
| F1 (mean ± std) | 0.923 ± 0.026 |
| precision / recall (mean) | 0.939 / 0.910 |
| pooled OOF AP / AUC | 0.980 / 0.981 |
| selected thresholds | 0.231, 0.331, 0.703, 0.369, 0.437 |

