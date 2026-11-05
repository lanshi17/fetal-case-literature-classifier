# DoRA threshold recalibration on gold-49

- old thr (val R≥0.95): **0.8794**
- new thr (gold R≥0.95): **0.7619** (Δ=-0.1174)
- gold n=49 (pos=23); test n=177

| strategy | thr | gold R | gold P | gold F1 | gold AP | test R | test P | test F1 | test AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_r95_old | 0.8794 | 0.826 | 0.500 | 0.623 | 0.762 | 0.969 | 0.808 | 0.881 | 0.885 |
| gold_r95 ← chosen | 0.7619 | 1.000 | 0.535 | 0.697 | 0.762 | 0.992 | 0.791 | 0.881 | 0.885 |
| gold_f2 | 0.7619 | 1.000 | 0.535 | 0.697 | 0.762 | 0.992 | 0.791 | 0.881 | 0.885 |
| gold_f1 | 0.9354 | 0.783 | 0.692 | 0.735 | 0.762 | 0.838 | 0.832 | 0.835 | 0.885 |
| fixed_0.5 | 0.5000 | 1.000 | 0.489 | 0.657 | 0.762 | 1.000 | 0.760 | 0.864 | 0.885 |

## Chosen operating point
- gold: R=1.000 P=0.535 F1=0.697 CM={'tn': 6, 'fp': 20, 'fn': 0, 'tp': 23}
- test: R=0.992 P=0.791 F1=0.881 CM={'tn': 13, 'fp': 34, 'fn': 1, 'tp': 129}

Artifacts: `threshold.json`, `threshold_recal_gold.json`, `metrics_recal_gold.json`, `gold49_scores_recal.csv`
