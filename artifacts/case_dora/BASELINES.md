# Case-only baselines and ablations

Identical 5-fold splits as the DoRA model; thresholds selected on each fold's inner 12% validation split.

| Method | AP | AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| majority | — | — | 0.000 | 0.000 | 0.000 |
| tfidf_title | 0.963 ± 0.007 | 0.964 | 0.876 | 0.894 | 0.882 ± 0.018 |
| tfidf_title_abstract | 0.971 ± 0.016 | 0.974 | 0.938 | 0.910 | 0.923 ± 0.027 |
| frozen_title | 0.953 ± 0.010 | 0.956 | 0.872 | 0.873 | 0.869 ± 0.029 |
| frozen_title_abstract | 0.961 ± 0.020 | 0.965 | 0.910 | 0.896 | 0.902 ± 0.019 |
| frozen_full_mean | 0.968 ± 0.018 | 0.972 | 0.910 | 0.878 | 0.890 ± 0.046 |
| frozen_single512 | 0.960 ± 0.016 | 0.961 | 0.892 | 0.899 | 0.894 ± 0.017 |
| frozen_full | 0.971 ± 0.016 | 0.974 | 0.894 | 0.910 | 0.897 ± 0.049 |
| dora_title_abstract | 0.974 ± 0.008 | 0.975 | 0.916 | 0.936 | 0.926 ± 0.024 |
| dora_full | 0.987 ± 0.009 | 0.988 | 0.926 | 0.952 | 0.938 ± 0.021 |

OOF metrics for the same methods are in `baselines.json`.
