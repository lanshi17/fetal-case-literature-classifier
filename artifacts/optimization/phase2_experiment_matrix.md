# Phase 2 实验矩阵（建议，不在 Phase 1 实现）

日期：2026-11-03  
依据：`artifacts/optimization/error_summary.json`、`bootstrap_ci.json`、`group_cv.json`、`threshold_strategies.json`  
基线：主模型 C，阈值 0.1526，测试集 Recall 0.923 / Precision 0.800 / F1 0.857 / AP 0.931（FN 10 / FP 30）

## 0. 硬化后的基线（Phase 2 的比较基准）

| 估计 | Recall | Precision | F1 | AP |
|---|---:|---:|---:|---:|
| 单次 test 点估计 | 0.923 | 0.800 | 0.857 | 0.931 |
| PMID cluster bootstrap 95% CI | [0.871, 0.963] | [0.734, 0.862] | [0.810, 0.899] | [0.891, 0.963] |
| GroupKFold k=5（883 篇） | 0.933 ± 0.016 | 0.846 ± 0.038 | 0.887 ± 0.026 | 0.943 ± 0.014 |

测试集只有 177 篇（47 负），Precision 的 CI 宽度约 0.13。**Phase 2 任何“提升”必须超过该区间噪声，并同时报告 bootstrap CI 与 GroupKFold，不能只看单次 test。**  
Phase 1 的 40 条错误中：FN 9/10 落在阈值下方 0.15 灰带，8/10 是高分置信错误（分数 ≤0.1）；FP 17/30 分数 ≥0.9，13 条为 case report/series、12 条为 cohort/screening、3 条为 review（标签计数，非互斥 design 值）。该分布是下列实验排序的主要依据。

## 1. 实验矩阵

| # | 实验 | 假设 | 依赖 | 预期主指标 | 成本 | 优先级 | 验收/护栏 |
|---|---|---|---|---|---|---|---|
| 1 | **困难负样本加标**：从 review / cohort / prediction-only 文中挖 hard negatives | FP 多来自“遗传病/产前诊断但非变异筛查”的高分文；补负样本可压 FP | 规则 + 人工复核（优先用已有 `final_labels` 的 design/reason）；不改纳入标准 | test Precision @ Recall≥0.95；bootstrap CI 下界 > 0.734 | M | **P0** | 新增负样本只进 train；test 不重标；CI 下界提升才算通过 |
| 2 | **章节感知分块**：按 Methods/Results 分块并池化，替代首 3000 字截断 | FN 中 4/10 为 cohort/screening 且分数 ≤0.015；关键筛查指标可能在被截断的 Results | PDF/XML 结构；需重编码（Phase 1 禁止，Phase 2 可用 GPU） | test Recall / F2；FN 从 10 降至 ≤7 | L | **P0** | 冻结 SPECTER2；同一 split 重算；缓存指纹必须变化并可复现 |
| 3 | **补 PubMed 摘要**：仅对 210 篇无摘要文补摘要 | `empty_abstract` 在 FN/FP 均有分布；摘要可稳定标题+摘要通道 | Phase 2 允许网络或已有本地库；需遵守 API 限速 | `empty_abstract` 子集 Recall；全量 AP 非劣 | S–M | P1 | 不抓全文、不改标签；记录 PMID 与抓取日期 |
| 4 | **主模型改 B**：去掉 2 个词密度辅助特征 | C 相对 B 的 AP/ROC 无实质增益，简化可降低过拟合与维护成本 | 已有消融（AP 0.931 vs 0.931） | AP/ROC 非劣（ΔAP ≥ −0.005）；维度 2306→2304 | S | P1 | 非劣才切换；保留 C 作为回退 |
| 5 | **CUDA 重编码**：`--device cuda` 加速特征迭代 | CPU 39.5 min/974 篇，阻塞实验 2/3 | GPU + 与 CPU cache 的数值一致性检查 | 编码耗时（目标 ≤5 min）；特征余弦相似度 ≥0.999 | M | P1 | 只加速，不改模型；CPU/cache 仍为默认可复现路径 |
| 6 | **轻量 adapter 微调**：冻结主干，仅训 adapter/分类头 | 若 1–3 后 GroupKFold AP 仍 <0.95 且标签质量过关，表征层面可能不足 | 标签审计完成 + 1–3 的结论；GPU | GroupKFold AP / Precision @ Recall≥0.95 | XL | **P2（最后手段）** | 固定 test；按 PMID 分组；禁止用 test 选超参；需与无微调基线同 split 对比 |

## 2. 建议执行顺序与门槛

1. **先做 1 + 4（1–2 天）**：困难负样本与去辅助特征都不需要重编码，能最快验证“FP 是否可由数据侧解释”。若 Precision CI 下界仍 ≤0.734，再进入 2。
2. **再做 2 + 5（3–5 天）**：章节分块需要重编码，先用 CUDA 把单次编码压到 5 分钟内；同时补 3（摘要）作为低成本并行项。
3. **最后才考虑 6**：启动条件为——40 条错误完成人工 `manual_tag` 复核、1–3 均未使 GroupKFold AP 达到 0.95、且标签审计估计噪声率 <5%。否则优先修数据而非加参数。

## 3. 统一评估协议（Phase 2 必须复用）

- 固定当前 `data/labeled_papers.csv`、默认子集（weak=False, label_source≠model）、PMID 分组与 test 177 篇；任何实验不得用 test 选阈值或早停。
- 每个实验同时报告：单次 test 点估计、PMID cluster bootstrap 95% CI（≥2000 次）、GroupKFold k=5 均值±std。
- 阈值策略至少报告 `max_precision_at_recall_ge_0.95` 与 `max_f2` 两种；部署策略由复核预算决定，不由 test 决定。
- 变更特征缓存时记录新指纹、编码设备与耗时；旧 `artifacts/model.joblib` 与 `artifacts/features/*.npz` 保留只读，作为回退基线。

## 4. 明确不做

- 不修改纳入/排除标签定义，不重标全库。
- 不抓取 PubMed 全文；摘要补充仅限 Phase 2 且需记录来源。
- 不在 Phase 1 微调 SPECTER2、不新增 adapter、不做全量重编码。
- 不把 bootstrap/CV 的数字当作新模型指标；它们只描述当前基线的不确定性。
