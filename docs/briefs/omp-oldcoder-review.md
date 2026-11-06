# OMP 审查任务：old-coder 证据优先审查

你是审查 Agent。必须先阅读并遵循：

`/home/yangzs/.agents/skills/old-coder/SKILL.md`

工作目录：

`/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening`

对照计划：

`/home/yangzs/.cursor/plans/SPECTER2 screening pipeline-e4a3448e.plan.md`

实现任务书（实现方承诺）：

`briefs/dsh-implement.md`

实现完成报告（若存在）：

`briefs/dsh-done.md`

## 你的目标

对 dsh 交付物做 **old-coder 式审查**：信任来自可执行约束与证据，而非泛泛读代码。

1. 写出审查用 SPEC 到绝对路径：
   `/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening/briefs/REVIEW_SPEC.md`
2. 对 SPEC 中每条场景尽量实际跑检查（import、build 统计、predict smoke、metrics 文件存在性等）。
3. 写出 EVIDENCE 到：
   `/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening/briefs/EVIDENCE.md`
   每条场景映射：场景 → 命令 → 结果 → PASS/FAIL/SKIP(+理由)
4. 写出可回派修复清单到：
   `/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening/briefs/omp-review.md`
   （偏离计划、缺文件、冒烟失败、指标缺失、数据泄漏风险等）

## 审查范围（必须覆盖）

- PMID join 覆盖是否完整；`full_text_available` 语义是否正确
- 编码器是否使用 classification adapter；特征是否 2304（+辅助特征时注明）
- 训练是否默认过滤 `weak` 与 `model` 源；是否 Group 切分
- 阈值是否按高召回选择；predict 三段带是否实现
- api `/health` `/predict` 是否存在
- `.gitignore` 是否排除全文包与 artifacts 大文件
- 是否错误地微调了 SPECTER2 Base

## 约束

- 不要大规模重写业务代码；发现问题记入修复清单。
- 若实现未完成，在 EVIDENCE 中明确 FAIL，并停止假装通过。
- 本审查视为 autonomous：在 EVIDENCE 中注明 `spec approval: not obtained (autonomous review run)`。

开始审查并写出上述三个产物文件。
