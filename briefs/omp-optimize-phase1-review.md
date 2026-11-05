# OMP 审查：优化 Phase 1（old-coder）

先读并遵循：`/home/yangzs/.agents/skills/old-coder/SKILL.md`

工作目录：`/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening`

对照计划：`briefs/OPTIMIZATION_PLAN.md`（验收 O1–O7）  
实现任务书：`briefs/dsh-optimize-phase1.md`  
完成报告（若有）：`briefs/OPTIMIZATION_PHASE1_DONE.md`

## 产出（绝对路径）

1. `/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening/briefs/OPTIMIZATION_REVIEW_SPEC.md`
2. `/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening/briefs/OPTIMIZATION_EVIDENCE.md`
3. `/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening/briefs/omp-optimize-review.md`

注明：`spec approval: not obtained (autonomous review run)`。

## 必查

- FN/FP CSV 行数是否分别等于基线测试混淆的 FN=10、FP=30（或与 DONE/当前 predictions 一致并解释）
- bootstrap / group_cv 是否读取 feature cache，日志中无全量 encode
- 主模型 `artifacts/model.joblib` 是否被无意覆盖
- 阈值策略是否 ≥3 且 test 指标可复现
- 可复跑检查命令写入 EVIDENCE

发现问题列入回派清单；不要大规模改业务代码。
