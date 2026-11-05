# Ablation A/B/C/D

Generated: 2026-11-05T02:22:40.025112+00:00

## Encode speed (frozen PubMedBERT, n=974)
- A (title+abstract): **11.0s** (0.011s/doc)
- B (fulltext): **147.9s** (0.152s/doc)
- Speedup A vs B: **13.49×**

## Experiments A/B/C (frozen + LR, val R≥0.95)

| exp | input | test AP | test R | test P | test F1 | gold AP | hard-neg Spec |
|---|---|---:|---:|---:|---:|---:|---:|
| A | title + abstract only | 0.919 | 0.969 | 0.792 | 0.872 | 0.432 | 0.296 |
| B | title + abstract + fulltext | 0.926 | 0.946 | 0.842 | 0.891 | 0.478 | 0.481 |
| C | fulltext + explicit case/variant lexical f | 0.931 | 0.954 | 0.844 | 0.895 | 0.479 | 0.481 |

### Key deltas
- Fulltext recall gain (B−A): **-0.023** (AP +0.007)
- Lexical features precision (C−B): **+0.001**
- Hard-neg specificity (C−B): **+0.000**

## Experiment D — hard negatives
- Catalog size (label=0 & hard tag): **142** (in test: 27)
- See `hard_negatives_catalog.csv`

## DoRA inference A vs B (recal thr=0.762)
- A test: R=0.946 P=0.783 AP=0.836 (0.016s/doc)
- B test: R=0.992 P=0.791 AP=0.885 (0.173s/doc)
- Δ recall B−A = **+0.046**, Δ AP = **+0.049**
- Hard-neg Spec A=0.407 B=0.370 (DoRA 高召回阈值下排除仍偏弱)

## Conclusions
1. **A 速度基线**：仅标题摘要编码比全文快 **~13.5×**（0.011 vs 0.152 s/doc）。
2. **B 全文增益（frozen+LR）**：test AP +0.007、P +0.05；但在 R≥0.95 选阈下 recall **未升反降**（0.969→0.946）。全文主要改善排序与精度，不是抬高召回。
3. **B 全文增益（DoRA）**：召回 **+4.6pt**、AP **+4.9pt** —— 生产 DoRA 必须用全文。
4. **C 显式病例特征**：相对 B 几乎无增益（P +0.001，困难负 Spec 持平）→ 词表密度特征不够，需规则层。
5. **D 困难负样本**（test n=27）：frozen B/C Spec **0.48**（A 仅 0.30）；DoRA Spec ~0.37–0.41。主要漏拦类型：NIPT/筛查队列、综述、CMA 队列。

**推荐配置**：线上 DoRA + 全文（B）；另加规则挡 cohort/review/NIPT；C 的 lexical aux 可省略。

