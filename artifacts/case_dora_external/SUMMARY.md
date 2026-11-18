# External validity

## temporal (train <=2019, test >2019)

| metric | value |
|---|---:|
| folds | 1 |
| test docs | 667 |
| AP (mean ± std) | 0.987 ± 0.000 |
| AUC (mean ± std) | 0.988 ± 0.000 |
| F1 (mean ± std) | 0.939 ± 0.000 |
| precision / recall (mean) | 0.954 / 0.925 |
| pooled OOF AP / AUC | 0.987 / 0.988 |
| selected thresholds | 0.505 |

## journal GroupKFold (5 folds)

| metric | value |
|---|---:|
| folds | 5 |
| test docs | 864 |
| AP (mean ± std) | 0.986 ± 0.005 |
| AUC (mean ± std) | 0.988 ± 0.003 |
| F1 (mean ± std) | 0.925 ± 0.019 |
| precision / recall (mean) | 0.922 / 0.930 |
| pooled OOF AP / AUC | 0.977 / 0.978 |
| selected thresholds | 0.299, 0.388, 0.760, 0.455, 0.291 |

