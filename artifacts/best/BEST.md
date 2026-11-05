# 最优解（冻结）

**DoRA（公平分块） + 金标重校准阈值 + 困难负样本规则层**

| 组件 | 设定 |
|---|---|
| 编码器 | NeuML PubMedBERT，480×≤12 块，`mean_all‖mean_first2` → 1536-d |
| 适配 | DoRA r=16 α=32 on `query`/`value`；分类头可训 |
| 训练 | train=默认子集+AHP 硬标；val/test=seed-42 默认子集；best=val AP@0.5 epoch2 |
| 阈值 | **0.7619**（金标 49 上 R≥0.95） |
| 规则 | NIPT/综述/筛查/GWAS 等硬排除；病例线索保护；默认开启 |

## 怎么用

```bash
# 推荐入口（读 artifacts/best/config.json）
.venv/bin/python predict.py --best --sample 1
.venv/bin/python predict.py --best --pdf paper.pdf --title "..."

# API
MODEL_BACKEND=best USE_RULES=1 .venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

## 关键指标（冻结快照）

- test pipeline +rules：R **0.977** / P **0.830** / F1 **0.898**
- test 困难负 Spec：**0.815**（规则前 0.37）
- gold-49 +rules pipeline：R **1.000** / P **0.581**
- gold-49 DoRA AP：**0.762**（优于 Full/GaLore/AHP-LR）

对照与消融见 `paths` 内报告；旧 frozen+LR `artifacts/model.joblib` 仅作基线，不再作为默认。
