# PubMedBERT 遗传变异病例筛选

产前遗传变异**病例报告 / 病例系列**相关性筛查。输出三档：`relevant` / `review` / `irrelevant`。

**默认最优解**：公平分块 **DoRA** + 金标重校准阈值 + **规则层**（`artifacts/best/`）。

| 组件 | 设定 |
|------|------|
| 编码器 | NeuML `pubmedbert-base-embeddings`，块长 480、最多 12 块，保留前 2 块 |
| 文档特征 | `L2(mean_all) ‖ L2(mean_first_2)` → **1536-d** |
| 适配器 | DoRA（r=16, α=32，`query`/`value`）+ 线性分类头 |
| 阈值 | **0.7619**（金标 49 篇上 R≥0.95 重校准） |
| 规则层 | 硬排除 NIPT/大规模筛查、综述/指南、GWAS/数据库等；病例线索保护 |
| 决策 | `score ≥ t` → relevant；`t−0.15 ≤ score < t` → review；否则 irrelevant；规则可强制排除或降为 review |

旧方案 frozen PubMedBERT + LR：`artifacts/model.joblib`（`--legacy-lr`）。SPECTER2 已弃用（`artifacts/archive_specter2/`）。

---

## 1. 目录

```text
encoder.py              # 分块编码 / 池化
rule_layer.py           # 困难负样本规则层
predict.py / api.py     # 默认走最优解 DoRA+rules
train.py                # frozen+LR 基线训练
scripts/
  finetune_chunk_fair.py    # 公平分块 Full / DoRA / GaLore
  ablation_abcd.py          # 消融 A/B/C/D
  eval_rule_layer.py        # 规则层评测
  ahp_label/                # 多模型 AHP 弱标
  gold_review.py            # 金标审核 UI
data/
  labeled_papers.csv        # 974 篇监督标签
  gold_standard.csv         # 人审金标（当前 49）
  ahp_wave1/                # AHP 弱标与语料
artifacts/
  best/                     # ★ 最优解配置与说明
  finetune_fair/dora/       # DoRA 权重与阈值
  rule_layer/               # 规则评测报告
  ablation_abcd/            # 消融报告
  model.joblib              # 旧 LR 基线
```

---

## 2. 环境

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# GPU 版 torch 按本机 CUDA 安装；另需 peft / accelerate / galore-torch（见 requirements）
```

可选本地快照：

```bash
export PUBMEDBERT_MODEL_PATH=/path/to/NeuML/pubmedbert-base-embeddings/snapshot
```

---

## 3. 预测 / API（默认最优解）

```bash
# 单条 / 抽样
.venv/bin/python predict.py --best --sample 1
.venv/bin/python predict.py --best --pdf paper.pdf --title "Prenatal case report"

# 关闭规则
.venv/bin/python predict.py --best --no-rules --sample 1

# 旧 frozen+LR
.venv/bin/python predict.py --legacy-lr --sample 1

# API（默认 MODEL_BACKEND=best）
MODEL_BACKEND=best USE_RULES=1 \
  .venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

说明见 [`artifacts/best/BEST.md`](artifacts/best/BEST.md)。

---

## 4. 主结果（seed=42，默认子集 test n=177）

Pipeline 口径：`relevant|review` 计为召回保留（仅 `irrelevant` 为漏检）。

| 系统 | thr | Recall | Precision | F1 | 备注 |
|------|----:|-------:|----------:|---:|------|
| **DoRA + rules（最优）** | **0.762** | **0.977** | **0.830** | **0.898** | 困难负 Spec **0.815** |
| DoRA only | 0.762 | 1.000 | 0.774 | 0.872 | 困难负 Spec 0.37 |
| fair head_v3（frozen+LR） | 0.192 | 0.954 | 0.849 | 0.899 | test AP 高，gold 泛化弱 |

金标 49（人审）：DoRA+rules pipeline R=1.000 / P=0.581；DoRA gold AP≈0.76（优于 Full FT / GaLore / AHP-LR）。

更多：[`artifacts/rule_layer/REPORT.md`](artifacts/rule_layer/REPORT.md)、[`artifacts/ablation_abcd/REPORT.md`](artifacts/ablation_abcd/REPORT.md)。

---

## 5. 训练与复现（可选）

```bash
# 数据（全文包就位时）
.venv/bin/python build_dataset.py

# 公平分块微调（DoRA / full / galore）
.venv/bin/python scripts/finetune_chunk_fair.py --method dora --device cuda

# AHP 弱标波次
bash scripts/ahp_label/run_wave.sh

# 消融 A/B/C/D
.venv/bin/python scripts/ablation_abcd.py --device cuda

# 规则层评测（读已导出的 DoRA 分数）
.venv/bin/python scripts/eval_rule_layer.py

# 金标审核 UI
.venv/bin/python scripts/gold_review.py --queue fn_fp --port 8765
```

训练协议摘要：默认子集 train + AHP 硬标入训；val/test 为 seed-42 默认子集；早停看 val AP@0.5；阈值先在 val 上 R≥0.95，再用金标重校准。

---

## 6. 消融结论（简）

| 实验 | 结论 |
|------|------|
| A 仅标题摘要 | 编码约快 13×；可作速度基线 |
| B +全文 | DoRA 召回/AP 明显受益；生产必须全文 |
| C +词表特征 | 相对 B 几乎无增益 |
| D 困难负样本 | 规则层把 Spec 从 ~0.37 提到 ~0.82 |

---

## 7. 限制

- 仅研究用途，非临床决策工具。
- API 默认无鉴权，请单 worker 加载大模型。
- 金标规模仍有限（49）；规则偏保守，边界病例依赖 `review` 人工复核。
