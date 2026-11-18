# Section-aware pooling (frozen encoder)

| representation | AP | F1 |
|---|---:|---:|
| flat | 0.977 ± 0.013 | 0.918 ± 0.030 |
| section | 0.980 ± 0.012 | 0.917 ± 0.029 |
| attention | 0.969 ± 0.014 | 0.886 ± 0.055 |

5-fold, same splits; thresholds selected on each inner validation split.
The current DoRA model is end-to-end fine-tuned, so this table compares pooling strategies on frozen features, not against DoRA.
