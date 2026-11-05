# Post-review gold-49 re-eval

Gold n=49 (pos=23, neg=26). Thresholds unchanged from each model's val R≥0.95.

| method | gold AP | gold R | gold P | gold F1 | thr |
|---|---:|---:|---:|---:|---:|
| primary model.joblib | 0.484 | 0.652 | 0.500 | 0.566 | 0.192 |
| AHP-LR | 0.600 | 0.739 | 0.486 | 0.586 | 0.492 |
| chunk-fair dora | 0.762 | 0.826 | 0.500 | 0.623 | 0.879 |
| chunk-fair full | 0.608 | 0.783 | 0.529 | 0.632 | 0.790 |
| chunk-fair galore | 0.664 | 0.826 | 0.487 | 0.613 | 0.721 |

Confusion (tn,fp,fn,tp):
- primary model.joblib: {'tn': 11, 'fp': 15, 'fn': 8, 'tp': 15}
- AHP-LR: {'tn': 8, 'fp': 18, 'fn': 6, 'tp': 17}
- chunk-fair dora: {'tn': 7, 'fp': 19, 'fn': 4, 'tp': 19}
- chunk-fair full: {'tn': 10, 'fp': 16, 'fn': 5, 'tp': 18}
- chunk-fair galore: {'tn': 6, 'fp': 20, 'fn': 4, 'tp': 19}
