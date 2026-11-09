# Case 类筛选（仅保留病例报告 / 病例系列）

## 口径

- 正类：`design ∈ {case report, case series}`（既有 974 篇人工/LLM 设计标签）
- 负类：cohort / screening / review / method / validation / registry
- 分类器：标题 TF-IDF（word 1–2gram）+ 逻辑回归（class-balanced），仅用标题，不依赖全文

## 5 折交叉验证（训练集）

- 样本 863（正 375 / 负 488），AUC **0.964**
- 阈值 0.50：Precision **0.936**、Recall **0.859**、F1 **0.896**
- 混淆矩阵：TP 322 / FP 22 / FN 53 / TN 466

## 全量筛选结果中的 case 类

| 相关性决策 | 总数 | 其中 case 类 | 占比 |
|---|---:|---:|---:|
| relevant | 11,999 | 6,916 | 57.64% |
| review | 1,753 | 835 | 47.63% |
| irrelevant | 9,626 | 4,213 | 43.77% |
| **保留（relevant+review）** | **13,752** | **7,751** | **56.36%** |

## 产出

- `case_retained.csv`：7,751 篇（relevant + review，且判为 case 类）
- `case_relevant.csv`：6,916 篇
- `case_review.csv`：835 篇
- `results_with_design.csv`：23,378 篇全量结果 + `design_score` / `design_guess`

## 高置信 case 示例（前 10，按相关性分数）

- `27642715` (2016) score=0.9556 design=0.871 — Rodriguez acrofacial dysostosis is caused by apparently de novo heterozygous mutations in the SF3B4 gene
- `33837277` (2021) score=0.9554 design=0.752 — Deletion in COL4A2 is associated with a three-generation variable phenotype: from fetal to adult manifestations
- `29575618` (2018) score=0.9554 design=0.933 — A homozygous TTN gene variant associated with lethal congenital contracture syndrome
- `34821546` (2022) score=0.9553 design=0.849 — Meckel Gruber and Joubert Syndrome Diagnosed Prenatally: Allelism between the Two Ciliopathies, Complexities of Mutation Types and Digenic Inheritance
- `25589041` (2015) score=0.9553 design=0.742 — Agnathia-otocephaly complex and asymmetric velopharyngeal insufficiency due to an in-frame duplication in OTX2
- `32347653` (2020) score=0.9552 design=0.881 — Hydrocephaly associated with compound heterozygous alterations in TRAPPC12
- `25708584` (2015) score=0.9551 design=0.589 — ECEL1 mutation causes fetal arthrogryposis multiplex congenita
- `7784362` (1995) score=0.9551 design=0.681 — An accessory marker derived from chromosome 20 and its co-existence with a mosaic trisomy 20 cell line
- `16941720` (2006) score=0.9551 design=0.872 — Metaphyseal chondrodysplasia McKusick type in a Chinese fetus, caused by novel compound heterozygosity 64T> A and 79G >T in RMRPgene
- `25959837` (2015) score=0.9551 design=0.705 — 5, 4, 3, 2, 1: embryologic variants of pentalogy of Cantrell

## 注意

- 仅用标题判设计；未写“case report/series”的单病例论文可能被漏掉（CV Recall≈0.86），边界条目可结合 `design_score` 和 `review` 档人工复核。
- `design_score` 是设计类型概率，与相关性分数独立；最终集合 = 相关性保留 ∩ case 类。
