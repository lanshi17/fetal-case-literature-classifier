# Fair chunk-pool fine-tune — `dora`

- encode: chunk_tokens=480, max_chunks=12, feat=1536
- train=3313 (base 566+ahp 2747), mean_chunks=10.0
- best_epoch=2 val_AP@0.5=0.9492 thr=0.8794
- elapsed=49.5 min

## Test: R=0.969 P=0.808 F1=0.881 AP=**0.885**
## Gold-40: R=0.833 P=0.484 F1=0.612 AP=**0.783**
