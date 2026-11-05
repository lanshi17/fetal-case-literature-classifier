# Fair head_v3 retrain

Train-only fit, C=10.0 (CV AUC=0.869), n_train/val/test=566/140/177

## Test set

| system | thr | Recall | Precision | F1 | AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **fair head_v3 (R≥0.95)** | 0.192 | 0.954 | 0.849 | 0.899 | 0.927 | 0.851 |
| fair head_v3 @0.5 | 0.500 | 0.838 | 0.908 | 0.872 | 0.927 | 0.851 |
| fair head_v3 @bal | 0.610 | 0.769 | 0.901 | 0.830 | 0.927 | 0.851 |
| sklearn balanced+scaler (R≥0.95) | 0.094 | 0.869 | 0.863 | 0.866 | 0.919 | 0.821 |
| shipped head_v3 R95 (**UNFAIR**) | 0.283 | 0.985 | 0.871 | 0.924 | 0.966 | 0.916 |
| shipped head_v3 @0.5 (**UNFAIR**) | 0.500 | 0.900 | 0.914 | 0.907 | 0.966 | 0.916 |

## Gold-40

| system | Recall | Precision | F1 | AP |
|---|---:|---:|---:|---:|
| fair_v3_thr_r95 | 0.722 | 0.500 | 0.591 | 0.491 |
| sklearn_bal_thr_r95 | 0.556 | 0.455 | 0.500 | 0.568 |
| shipped_v3_thr_r95_UNFAIR | 0.833 | 0.577 | 0.682 | 0.559 |

> Shipped rows are an upper bound only: that head was fit on nearly the full corpus.
