# Optimization Phase 1 — 完成报告

日期：2026-11-03  
状态：**PASS**（O1–O7 全部满足；无数据缺失，无 FAIL）  
范围：错误归因 + 评估硬化 + 阈值策略 + Phase 2 建议；**未微调、未重编码、未改标签、未抓取 PubMed**。

## 0. 关键数字（先看这里）

- 基线测试集（n=177，130 正/47 负，t=0.152578）：**Recall 0.923 / Precision 0.800 / F1 0.857 / AP 0.931**，FN 10 / FP 30。
- PMID cluster bootstrap（2000 次，95% 百分位 CI）：Recall **0.923 [0.871, 0.963]**；Precision **0.800 [0.734, 0.862]**；AP **0.931 [0.891, 0.963]**；F1 0.857 [0.810, 0.899]。
- GroupKFold k=5（默认子集 883 篇）：Recall **0.933 ± 0.016**；Precision **0.846 ± 0.038**；F1 **0.887 ± 0.026**；AP **0.943 ± 0.014**。
- 错误归因：FN **10/10** 行、FP **30/30** 行均带元数据与 `heuristic_tag`；与 `metrics.json` 混淆矩阵逐项一致（TN17/FP30/FN10/TP120）。
- 阈值策略（同 val 选阈、同 test 报告）：`max_f2` 把 test Recall 提到 **0.962**、Precision 降到 **0.776**（FN 10→5，FP 30→36）；`pp_budget≤120` 得 Recall 0.946 / Precision 0.789（FN7/FP33）。

## 1. O1–O7 验收

| ID | 状态 | 证据 |
|---|---|---|
| O1 | PASS | `artifacts/optimization/fn_audit.csv`（10 行）、`fp_audit.csv`（30 行）；含 pmid/title/score/decision/design/reason/text_source/heuristic_tag/manual_tag/manual_note/text_head 等 21 列 |
| O2 | PASS | `error_summary.json`：按 heuristic_tag 的 FN/FP/合计计数；`confusion_consistent=true`，`checks.all_pass=true` |
| O3 | PASS | `bootstrap_ci.json`：Recall/Precision/F1/F2/AP 的 95% CI，2000 次 PMID 分组重采样（≥500） |
| O4 | PASS | `group_cv.json`：GroupKFold k=5，Recall/Precision/F1/F2/AP 均值±std；仅读 `artifacts/features/*.npz`，`encoder_invoked=false` |
| O5 | PASS | `threshold_strategies.json`：4 种策略，同一 val 选阈、同一 test 报告（含 max_precision@R≥0.95、max_f2、max_recall@P≥0.80、PP 预算≤120） |
| O6 | PASS | 本文件 + `artifacts/optimization/phase2_experiment_matrix.md` |
| O7 | PASS | `model.joblib` / 特征 npz / `test_predictions.csv` 的 MD5 运行前后一致；新脚本独立运行，不覆盖主模型 |

MD5（运行后复核）：`model.joblib 90905b95…`、`specter2_mc24_…npz 3507745f…`、`test_predictions.csv bb079439…`（与运行前一致）。

## 2. 错误归因（FN 10 / FP 30）

启发式标签是多标签，计数不可相加；完整定义在 `error_summary.json.tag_definitions`。

| 组 | 主要标签（计数） | 读法 |
|---|---|---|
| FN（10） | low_score_band 9；high_score_confident_error 8；design_case_report 5；design_cohort_or_screening 4；empty_abstract 2 | 9/10 落在阈值下方 0.15 灰带；病例报告 5 条、队列/筛查 4 条；8/10 分数 ≤0.1，不是“差一点”而是表征/截断问题 |
| FP（30） | high_score_confident_error 17；design_case_report 13；design_cohort_or_screening 12；mid_score_band 9；empty_abstract 4；design_review 3 | 17/30 分数 ≥0.9，属高置信误判；病例报告与队列文是主要来源，提示“遗传病/产前诊断但非变异筛查”的困难负样本缺失 |

逐条明细（含标题、design、reason、500 字正文头、人工留空列）见两个 CSV；`manual_tag/manual_note` 保持空白，未伪造人工结论。

## 3. 评估硬化

**Bootstrap**（`bootstrap_ci.json`）：按 PMID 有放回抽组，2000 次；AP 单类重采样样本被显式排除（`n_valid` 记录）。Precision 的 CI 宽度约 0.13，说明 177 篇测试集上的 0.800 不能当作精确常数。

**GroupKFold**（`group_cv.json`）：默认子集 883 篇（678 正）上 5 折，折内再按 group 划 80/20 选阈（目标 Recall 0.95），外层折只评估。各折阈值为 0.256 / 0.030 / 0.118 / 0.231 / 0.249，**阈值本身跨折波动很大**；因此 pooled OOF 使用逐行折内阈值，并注明不可当作单一部署阈值。Pooled OOF：R 0.934 / P 0.846 / AP 0.940。

**阈值策略**（`threshold_strategies.json`，模型与 val/test 行完全相同）：

| 策略 | 阈值 | val R / P | test R | test P | test F1 | test PP | FN / FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| max_precision@R≥0.95（现状） | 0.152578 | 0.963 / 0.890 | 0.923 | 0.800 | 0.857 | 150 | 10 / 30 |
| max_f2 | 0.015851 | 0.991 / 0.871 | **0.962** | 0.776 | 0.859 | 161 | 5 / 36 |
| max_recall@P≥0.80 | 0.001056 | 1.000 / 0.839 | **1.000** | 0.756 | 0.861 | 172 | 0 / 42 |
| max_recall@PP≤120 | 0.056945 | 0.973 / 0.883 | 0.946 | 0.789 | 0.860 | 156 | 7 / 33 |

解读：提高 Recall 的代价是 FP 增加；`P≥0.80` 策略在 val 上仍满足 0.839，但 test 落到 0.756，且 172/177 判正，接近“全通过”，不建议作为默认。若复核预算有限，`PP≤120`（val 口径）是折中；若目标是少漏，`max_f2` 更稳。

## 4. 复现命令（已实跑）

```bash
cd /data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening
.venv/bin/python scripts/error_audit.py        # ~0.2s；FN=10 FP=30，all_pass=true
.venv/bin/python scripts/eval_hardening.py     # ~23s；bootstrap 2000 + 5-fold CV + 4 策略
test -f artifacts/optimization/fn_audit.csv
test -f artifacts/optimization/bootstrap_ci.json
test -f briefs/OPTIMIZATION_PHASE1_DONE.md
```

复现校验（写入 `bootstrap_ci.json.reproduction`）：加载的 `model.joblib` 对 test 打分与 `test_predictions.csv` 最大绝对差 **2.96e-08**；与 `metrics.json` 的 Recall/Precision/F1/F2/AP 最大差 **0.0**；混淆矩阵一致。脚本无 encoder 调用、无网络请求。

## 5. Phase 2 建议（详见 `artifacts/optimization/phase2_experiment_matrix.md`）

1. **P0 困难负样本加标**：优先补 review / cohort / prediction-only 的高分负样本；验收看 Precision 的 bootstrap CI 下界是否 >0.734。
2. **P0 章节感知分块**：FN 含 4/10 队列/筛查文且分数 ≤0.015，关键筛查指标可能位于被截断的 Results；需重编码（Phase 2 可 GPU），冻结 SPECTER2。
3. **P1 简化主模型为 B + 补摘要 + CUDA 重编码**：B 与 C 的 AP 无实质差异；摘要补齐针对 210 篇无摘要文；CUDA 只加速迭代，不改模型。
4. **P2 轻量 adapter 微调**：仅当上述实验后 GroupKFold AP 仍 <0.95 且人工复核确认标签噪声 <5% 时启动；本轮不做。

## 6. 限制与未做事项

- 未做人工逐条复核；`manual_tag/manual_note` 为空，启发式标签不是因果结论。1 条 FN（PMID 35791503，curated_gold）的 reason/quote 在源数据中为空，CSV 保持空白而非补写。
- Bootstrap/CV 只描述当前模型与当前数据划分的不确定性，不是新模型指标。
- GroupKFold 折内阈值波动大（0.030–0.256），说明“单次 val 选阈”本身不稳定；部署前应固定策略并单独校准。
- 未微调 SPECTER2、未全量重编码、未抓取 PubMed、未改标签定义、未覆盖 `model.joblib`。
- 无 FAIL 项；所有计划要求的文件均已生成并可复跑。
