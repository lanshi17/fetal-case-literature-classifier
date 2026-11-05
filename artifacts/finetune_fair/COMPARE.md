# Fair chunk-pool fine-tune comparison

Same chunking as frozen encoder (480×≤12 → 1536-d). AHP train-only. Val R≥0.95 thr.

| method | test AP | test R | test P | test F1 | gold AP | gold R |
|---|---:|---:|---:|---:|---:|---:|
| frozen+LR fair | 0.927 | 0.954 | 0.849 | 0.899 | 0.491 | 0.722 |
| frozen+LR +AHP | 0.908 | 0.900 | 0.791 | 0.842 | 0.649 | 0.778 |
| 512-trunc full | 0.907 | 0.938 | 0.813 | 0.871 | 0.617 | 0.722 |
| 512-trunc dora | 0.906 | 0.977 | 0.784 | 0.870 | 0.699 | 0.889 |
| 512-trunc galore | 0.915 | 0.923 | 0.811 | 0.863 | 0.671 | 0.722 |
| chunk-fair full | 0.892 | 0.962 | 0.828 | 0.890 | 0.610 | 0.833 |
| chunk-fair dora | 0.885 | 0.969 | 0.808 | 0.881 | 0.783 | 0.833 |
| chunk-fair galore | 0.845 | 0.977 | 0.789 | 0.873 | 0.675 | 0.833 |
