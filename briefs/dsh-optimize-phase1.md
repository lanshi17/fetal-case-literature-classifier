# DSH 任务：优化 Phase 1 执行

工作目录：`/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening`

**权威计划（必须逐条满足 O1–O7）：**

`/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening/briefs/OPTIMIZATION_PLAN.md`

基线勿破坏：`artifacts/model.joblib`、`artifacts/features/*.npz` 作为输入只读（可不覆盖主模型）。允许新增 `scripts/` 与 `artifacts/optimization/`。

## 禁止

- 微调 SPECTER2 / 全量重编码（可用 cache；不要 `--force-reencode`）
- PubMed 抓取、改标签定义
- 交互式确认

## 必做

1. 实现 `scripts/error_audit.py` → FN/FP CSV + `error_summary.json`
2. 实现 `scripts/eval_hardening.py` → bootstrap CI + GroupKFold + 阈值策略 JSON
3. 写 `artifacts/optimization/phase2_experiment_matrix.md`
4. 写 `briefs/OPTIMIZATION_PHASE1_DONE.md`（含关键数字）
5. README 短节指向优化产物
6. 按计划第 5 节命令实际跑通

完成后退出。若某步数据缺失，在 DONE 中 FAIL 说明，勿伪造指标。
