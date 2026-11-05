# Fair chunk-pool fine-tune — `full`

- encode: chunk_tokens=480, max_chunks=12, feat=1536
- train=3313 (base 566+ahp 2747), mean_chunks=10.0
- best_epoch=2 val_AP@0.5=0.9371 thr=0.7903
- elapsed=51.9 min

## Test: R=0.962 P=0.828 F1=0.890 AP=**0.892**
## Gold-40: R=0.833 P=0.484 F1=0.612 AP=**0.610**
