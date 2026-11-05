# Fair chunk-pool fine-tune — `galore`

- encode: chunk_tokens=480, max_chunks=12, feat=1536
- train=3313 (base 566+ahp 2747), mean_chunks=10.0
- best_epoch=2 val_AP@0.5=0.9190 thr=0.7207
- elapsed=42.7 min

## Test: R=0.977 P=0.789 F1=0.873 AP=**0.845**
## Gold-40: R=0.833 P=0.469 F1=0.600 AP=**0.675**
