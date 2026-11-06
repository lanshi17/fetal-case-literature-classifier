# Optimization Phase 1 — Review SPEC（可执行验收契约）

日期：2026-11-03  
范围：只读审查 `scripts/error_audit.py`、`scripts/eval_hardening.py`、`artifacts/optimization/*`、`briefs/OPTIMIZATION_PHASE1_DONE.md`、`README.md` 与受保护基线产物。  
对应计划：`briefs/OPTIMIZATION_PLAN.md`（O1–O7）；实现任务书：`briefs/dsh-optimize-phase1.md`。

**spec approval: not obtained (autonomous review run)**

> 本次为自主审查：本 SPEC 未经人工批准，因此置信度低于“人工批 SPEC + 机器证据”的完整闭环。审查不修改业务代码，只新增审查产物（本文件、`briefs/OPTIMIZATION_EVIDENCE.md`、`briefs/omp-optimize-review.md`、`briefs/optimization_review_checks.py`）。

## 0. 审查主张（被验证的对象）

审查要证明的不是“代码看起来对”，而是下列可判定的主张。每条主张对应一个编号场景，由 `briefs/optimization_review_checks.py` 执行；场景失败即主张不成立。

| 主张 | 内容 | 依据 |
|---|---|---|
| C1 | `fn_audit.csv` / `fp_audit.csv` 的 doc_id 集合与独立重算的 test FN/FP（`label==1 & score<threshold`、`label==0 & score>=threshold`）逐一相等，行数 10 / 30，且分数列与 `test_predictions.csv` 一致；`decision` 列与分数规则一致 | O1 |
| C2 | 两个 CSV 含 O1 要求的列（pmid/title/score/decision/design/reason/text_source/heuristic_tag/manual_note 等），元数据非空；`manual_tag`/`manual_note` 全空；`text_head` ≤500 字符且无换行 | O1 |
| C3 | 每条 `heuristic_tag` 都能由源数据按 `error_summary.json.tag_definitions` 重新推导（empty_abstract、pdf_text_source、design_*、灰带、mid/high 置信、weak_or_model_label），且 tag 计数与 CSV 重算一致 | O2 |
| C4 | `error_summary.json` 的 computed/baseline 混淆矩阵 = `metrics.json` 的 TN17/FP30/FN10/TP120，`checks.all_pass=true`，totals FN/FP = 10/30 | O2 |
| C5 | DONE 声明的唯一“reason/quote 空”FN 就是 `fn_audit.csv` 中唯一空行，PMID 35791503 | DONE §6 |
| C6 | `threshold_strategies.json` ≥3 种策略；每种策略的 test Recall/Precision/F1/F2/PP/混淆可由 `test_predictions.csv` 在**该策略阈值**下重算；AP 与 `metrics.json` 一致；基线策略阈值 = 部署阈值 0.1525784656 | O5 |
| C7 | 每种策略阈值确实由**同一 val**（140 行）按所选规则得出：`max_precision_at_recall_ge_0.95` 与 `max_f2` 用缓存特征+模型重算 val 分数后独立复现阈值；所有策略的 val 指标可重算；test 不参与选阈 | O5 |
| C8 | `bootstrap_ci.json` 为 PMID 分组有放回重采样、≥500 次（实际 2000）、95% 百分位区间；点估计 = `metrics.json`；用独立实现 + 独立种子重算的 CI 端点与记录值差 ≤0.02 | O3 |
| C9 | `group_cv.json` k=5，`n=883/678 正`，`encoder_invoked=false`，每折 n_train+n_val+n_test=883，折内阈值在 val 上达到 Recall≥0.95；均值/std 可由 per_fold 重算；pooled OOF 混淆覆盖 883 行且 R/P 自洽 | O4 |
| C10 | 两个脚本在“encoder 入口被替换为抛错 + socket/urllib audit hook”的子进程中仍成功产出全部文件，`ENCODER_CALLS=0`、`NETWORK_EVENTS=0`；`eval_hardening.py` 运行时间 <180s（全量编码在此机器需 39.4 min） | O4、实现要点 4.2 |
| C11 | 同命令重跑（各 2 次 + 与仓库已提交产物对比）：CSV 字节相同；JSON 除 generated_at/fit_seconds/输出目录外逐字段相同 | O1–O5 可复现 |
| C12 | `artifacts/model.joblib`、`model_all.joblib`、`metrics.json`、`test_predictions.csv`、`config.json`、特征 npz、`data/labeled_papers.csv`、`data/final_labels.csv` 在审查全过程 hash 不变；DONE 记录的 MD5 前缀（90905b95 / 3507745f / bb079439）与实际文件一致；脚本内无 `joblib.dump`/`np.savez`/`torch.save` | O7 |
| C13 | `OPTIMIZATION_PHASE1_DONE.md` 含 O1–O7 与关键结论、`phase2_experiment_matrix.md` 含困难负样本/章节分块/CUDA 等条目、README 有 “Optimization Phase 1” 小节并指向 `artifacts/optimization/` | O6 |
| C14 | DONE/README 中引用的关键数字（bootstrap CI、GroupKFold 均值±std、pooled OOF、折内阈值、4 种策略的 val R/P、test R/P/F1、PP、FN/FP）与 JSON 原始值按 3 位小数四舍五入（ROUND_HALF_UP）后逐字一致 | O6（报告可信） |

## 1. 不变量（必须存活）

- 主模型路径不被覆盖：`artifacts/model.joblib` 仍是 2026-11-02 19:48 的基线产物，内容 hash 不变。
- 缓存只读：`artifacts/features/specter2_mc24_ch3000_n974_339bee2712.npz` hash 不变；不新增 npz。
- 标签口径不变：`data/labeled_papers.csv`、`data/final_labels.csv` hash 不变；未改纳入/排除定义。
- 评估协议不变：test 不用于选阈/早停；val 与 test 行集合与基线 split 相同（140/177）。
- 运行副作用受控：脚本只写 `--out` 指定的优化目录（默认 `artifacts/optimization/`），无网络、无 encoder 调用。

## 2. 判定容差

| 对象 | 容差 | 理由 |
|---|---|---|
| 由 CSV/JSON 重算的指标 | 1e-9 | 同一批分数、同一 sklearn 定义，应精确 |
| val 阈值复现 | 1e-9（同一候选网格规则） | 阈值由 train.select_threshold 的网格定义，须逐位一致 |
| 独立 bootstrap CI 端点 | ±0.02 | 不同 RNG 种子的蒙特卡洛误差 |
| 文档数字 | 3 位小数半进位后逐字匹配 | 报告以 3 位小数呈现；四舍五入口径必须与原始值一致 |
| 受保护产物 | sha256 完全相等 | 二进制产物不应被动过 |

## 3. 环境与副作用计划

- 无新增依赖：使用 `.venv` 现有 pandas / numpy / sklearn / joblib（版本记录在 EVIDENCE）。
- 无 git 提交：仓库 `master` 尚无任何 commit，审查为只读；源状态用“受审查文件 sha256 树哈希”标识，不用 SHA。
- 新增文件仅 4 个审查产物（均在 `briefs/`）；临时目录用 `tempfile`，退出即删。
- 不修改 dsh 的实现脚本、产物或 DONE 报告；发现的问题写入回派清单，不做静默修补。

## 4. 否定对照（checker 必须能失败）

home-grown checker 的通过本身不是证据，需证明其失败路径可达。对输入副本做定向篡改，要求对应场景变红：

| 对照 | 篡改 | 期望 |
|---|---|---|
| N1 | `fp_audit.csv` 删 1 行 | S01 FAIL |
| N2 | 改 1 行 `heuristic_tag` | S03 FAIL |
| N3 | 改 1 个策略的 test precision（+0.05） | S06 FAIL |
| N4 | 改 1 个策略阈值（+0.001） | S07 FAIL |
| N5 | 抬高 bootstrap recall `ci_low`（+0.15） | S08 FAIL |
| N6 | 改 1 折 test recall（+0.01） | S09 FAIL |
| N7 | 改 DONE/README 的 CI 边界（0.963→0.960） | S15 FAIL |
| N8 | 直接调用 `encoder.document_features` | guard 报 `ENCODER_CALLED` |
| N9 | 子进程内 `socket.getaddrinfo('127.0.0.1', 9)` | hook 报 `NETWORK_EVENTS=1` |

## 5. 入口命令（唯一）

```bash
cd /data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening
.venv/bin/python briefs/optimization_review_checks.py --negative-controls \
  --json briefs/optimization_review_checks.json
```

退出码 0 = 全部主张成立且对照全部被检出；非 0 = 至少一条主张不成立或对照未检出。

## 6. 不在本次范围

- 重训/微调、重编码、改标签、抓取 PubMed（计划明确非目标）。
- 主观评价 Phase 2 矩阵的科学价值；只核对其事实与数字来源。
- 人工逐条复核 40 条错误（`manual_tag`/`manual_note` 属人工后续工作）。
- 修复本 SPEC 发现的问题：修正属 dsh/人工，审查侧只给回派清单。

## 7. 已知的 spec 不完备之处（显式声明）

- C14 只覆盖报告里出现的数字，不能证明报告的所有定性表述正确。
- C10 的 encoder guard 只能证明“Python 侧 encoder 入口未被调用”；原生库内部的编码路径无法由此排除（但脚本从未加载 torch/transformers，见 EVIDENCE）。
- 场景由同一 agent 编写，存在同源盲点；否定对照只证明“已知坏输入能触发失败路径”，不证明覆盖所有违规形态。
