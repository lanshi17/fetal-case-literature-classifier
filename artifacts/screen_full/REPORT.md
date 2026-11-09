# 全量文献筛选报告

- 语料：AutomatedPubMed PDF 全量（23,378 篇，按 PMID 去重）
- 已筛选：**23,378** 篇（唯一 PMID 23,378）
- 模型：DoRA + 规则层（阈值 0.7619，review 阈值 0.6119）

## 三档分布

| 决策 | 篇数 | 占比 |
|---|---:|---:|
| relevant | 11,999 | 51.33% |
| review | 1,753 | 7.50% |
| irrelevant | 9,626 | 41.18% |
| **保留（relevant+review）** | **13,752** | **58.82%** |

## 全文与规则

- 成功抽取正文：23,292 篇；正文缺失/过短：86 篇（OCR 疑似需要 114，抽取报错 0）
- 规则层改判：664 篇；病例线索保护：8,516 篇

| 规则 | 命中篇数 |
|---|---:|
| `NIPT_SCREENING` | 540 |
| `REVIEW_OR_GUIDELINE` | 366 |
| `COHORT_LANGUAGE` | 347 |
| `PROTECTED_DEMOTE_ONLY` | 200 |
| `NIPT_SCREENING_COHORT` | 36 |
| `LARGE_PRENATAL_COHORT` | 12 |
| `GWAS_DB_OR_PREDICTION` | 6 |
| `METHOD_TOOL_TITLE` | 4 |

## 分数分布

| min | p05 | p25 | median | p75 | p95 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0513 | 0.0570 | 0.1213 | 0.8455 | 0.9490 | 0.9536 | 0.9556 |

## relevant 年份分布（前 15）

| 年份 | 篇数 |
|---|---:|
| 2025 | 702 |
| 2026 | 649 |
| 2024 | 600 |
| 2022 | 595 |
| 2021 | 594 |
| 2023 | 572 |
| 2020 | 472 |
| 2019 | 408 |
| 2016 | 386 |
| 2017 | 377 |
| 2018 | 366 |
| 2015 | 339 |
| 2014 | 334 |
| 2012 | 316 |
| 2013 | 304 |

## 高置信 relevant 示例（前 10）

- `27642715` (2016) score=0.9556 — Rodriguez acrofacial dysostosis is caused by apparently de novo heterozygous mutations in the SF3B4 gene
- `33837277` (2021) score=0.9554 — Deletion in COL4A2 is associated with a three-generation variable phenotype: from fetal to adult manifestations
- `29575618` (2018) score=0.9554 — A homozygous TTN gene variant associated with lethal congenital contracture syndrome
- `34821546` (2022) score=0.9553 — Meckel Gruber and Joubert Syndrome Diagnosed Prenatally: Allelism between the Two Ciliopathies, Complexities of Mutation Types and Digenic Inheritance
- `25589041` (2015) score=0.9553 — Agnathia-otocephaly complex and asymmetric velopharyngeal insufficiency due to an in-frame duplication in OTX2
- `32347653` (2020) score=0.9552 — Hydrocephaly associated with compound heterozygous alterations in TRAPPC12
- `38278647` (2024) score=0.9552 — Clinical and molecular characteristics of 26 fetuses with lethal multiple congenital contractures
- `25510707` (2014) score=0.9552 — Complete trisomy 9 with unusual phenotypic associations: Dandy-Walker malformation, cleft lip and cleft palate, cardiovascular abnormalities
- `25708584` (2015) score=0.9551 — ECEL1 mutation causes fetal arthrogryposis multiplex congenita
- `7784362` (1995) score=0.9551 — An accessory marker derived from chromosome 20 and its co-existence with a mosaic trisomy 20 cell line

## 文件

- `results.jsonl` / `results.csv`：全部筛选结果（含分数、规则命中、抽取状态）
- `relevant.csv` / `review.csv` / `irrelevant.csv`：三档子集
- `summary.json`：机器可读汇总
