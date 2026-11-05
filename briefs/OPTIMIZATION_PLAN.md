# SPECTER2 筛选基线 — 优化计划（Phase 1）

日期：2026-11-03  
仓库：`/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening`  
基线产物：`artifacts/metrics.json`（主模型 C，阈值 0.1526）  
编排：Cursor 写计划与任务书 → **dsh** 实现 → **omp + old-coder** 审查

## 0. 基线事实（优化起点）

| 集合 | n | Recall | Precision | F1 | AP | 混淆 |
|------|--:|-------:|----------:|---:|---:|------|
| Val（默认子集） | 140 | 0.963 | 0.890 | 0.925 | 0.963 | FN4 / FP13 |
| Test（默认子集） | 177 | 0.923 | 0.800 | 0.857 | 0.931 | **FN10 / FP30** |

消融（同 split，test）：A 仅标题摘要 Recall 最高但 AP 最低；B/C AP≈0.931，**辅助特征几乎无增益**。  
正式编码在 **CPU** 完成；当前机器 GPU 可用，但 Phase 1 **不强制全量重编码**。

## 1. 目标与非目标

**Phase 1 目标（本轮必须完成）**

1. 可复核的 **FN/FP 归因表**（含文献元数据 + 启发式错误标签 + 人工填写列）。
2. **评估硬化**：group bootstrap CI；基于**已缓存特征**的 Group K-fold（不重新 GPU/CPU 编码全文）。
3. **阈值策略对照**：val 上 Recall≥0.95 最大 Precision（现状）vs F2 最优 vs 固定复核预算（如 predicted-positive 上限）下的 Recall。
4. 一份 **下一阶段实验矩阵**（Phase 2 建议，不在本轮实现微调）。

**非目标（本轮不做）**

- 微调 SPECTER2 / 新训 adapter
- 全量强制重编码（除非脚本支持 `--device cuda` 作为可选项且默认仍用 cache）
- 改纳入标签标准或重标全库
- 部署/鉴权大改

## 2. 成功标准

| ID | 标准 |
|----|------|
| O1 | `artifacts/optimization/fn_audit.csv` 与 `fp_audit.csv` 存在；行数分别对应测试 FN/FP；含 pmid/title/score/decision/design/reason/text_source/heuristic_tag/manual_note |
| O2 | `artifacts/optimization/error_summary.json`：按 heuristic_tag 计数；报告 FN/FP 总数与基线混淆矩阵一致 |
| O3 | `artifacts/optimization/bootstrap_ci.json`：至少 Recall/Precision/AP 的 95% CI（按 PMID group resample，≥500 次） |
| O4 | `artifacts/optimization/group_cv.json`：GroupKFold（k=5）均值±std（Recall/Precision/F1/AP）；用现有 feature cache，禁止为 CV 全量重编码 |
| O5 | `artifacts/optimization/threshold_strategies.json`：≥3 种阈值策略在 **同一 val** 选阈、**同一 test** 报告 |
| O6 | `briefs/OPTIMIZATION_PHASE1_DONE.md`：结果摘要 + Phase 2 建议（≤1 页有效信息密度） |
| O7 | 不破坏现有 `artifacts/model.joblib` 主路径；新脚本可独立运行 |

## 3. 交付物与目录

```
briefs/OPTIMIZATION_PLAN.md          # 本计划
scripts/error_audit.py               # FN/FP 导出 + 启发式标签
scripts/eval_hardening.py            # bootstrap CI + GroupKFold + 阈值策略
artifacts/optimization/
  fn_audit.csv
  fp_audit.csv
  error_summary.json
  bootstrap_ci.json
  group_cv.json
  threshold_strategies.json
  phase2_experiment_matrix.md
briefs/OPTIMIZATION_PHASE1_DONE.md
```

README 增加一小节「Optimization Phase 1」指向上述产物（短）。

## 4. 实现要点

### 4.1 错误归因（error_audit.py）

- 输入：`artifacts/test_predictions.csv` + `data/labeled_papers.csv`（及可选 `final_labels` 的 design/quote）。
- FN：`label==1` 且 `decision!='relevant'`（或 score&lt;threshold，与现网决策一致）。
- FP：`label==0` 且 `decision=='relevant'`。
- 启发式 `heuristic_tag`（可多标签 `;` 分隔），例如：
  - `empty_abstract` / `pdf_text_source`
  - `design_cohort_or_screening` / `design_review` / `design_case_report`
  - `low_score_band`（落在 review 灰带）
  - `high_score_confident_error`（|score−0.5| 大仍错）
  - `weak_or_model_label`（若误入测试）
- 保留空列 `manual_tag`,`manual_note` 供人工复核。
- 正文可截断写入 `text_head`（如前 500 字），避免 CSV 爆炸。

### 4.2 评估硬化（eval_hardening.py）

- **必须**加载现有 npz 特征缓存（路径写在 `artifacts/config.json` → `feature_cache`）。
- 复用 `train.py` 中的 scaler+LR、阈值选择逻辑（import 或抽公共函数；允许小幅重构，但主模型文件不强制覆盖）。
- Bootstrap：对 test 的 PMID 有放回抽样；每次重算混淆与指标；输出 mean/ci_low/ci_high。
- GroupKFold：仅在 **default 子集**（weak=False, ≠model）上；每折内再划 val 选阈（目标 Recall 0.95）或明确记录折内协议。
- 阈值策略至少：
  1. `max_precision_at_recall_ge_0.95`（现状）
  2. `max_f2`
  3. `max_recall_at_precision_ge_0.80`（或 predicted_positive 预算，如 ≤120）

### 4.3 Phase 2 矩阵（只写文档）

在 `phase2_experiment_matrix.md` 列出建议实验（不做）：

| 实验 | 假设 | 依赖 |
|------|------|------|
| 困难负样本加标 | 降 FP | 人工/规则挖综述·队列·预测文 |
| 章节感知分块 | 降 FN | PDF/XML 结构 |
| 补 PubMed 摘要 | 提无 abstract 子集 | 网络或已有库 |
| 主模型改 B | 简化特征 | 已有消融 |
| CUDA 重编码 | 加速迭代 | GPU |
| 轻量 adapter 微调 | 最后手段 | 标签质量+CV 仍不达标 |

## 5. 验收命令（dsh 必须实跑）

```bash
cd /data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening
.venv/bin/python scripts/error_audit.py
.venv/bin/python scripts/eval_hardening.py
test -f artifacts/optimization/fn_audit.csv
test -f artifacts/optimization/bootstrap_ci.json
test -f briefs/OPTIMIZATION_PHASE1_DONE.md
```

## 6. 审查（omp）

对照本计划 O1–O7；用 old-coder 产出 `briefs/OPTIMIZATION_REVIEW_SPEC.md`、`briefs/OPTIMIZATION_EVIDENCE.md`、`briefs/omp-optimize-review.md`。  
重点：FN/FP 行数是否与混淆矩阵一致；CV/bootstrap 是否真用 cache 而非重编码；是否误改主模型。
