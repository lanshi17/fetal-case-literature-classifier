# Optimization Phase 1 — EVIDENCE（审查证据）

审查对象：`scripts/error_audit.py`、`scripts/eval_hardening.py`、`artifacts/optimization/*`、`briefs/OPTIMIZATION_PHASE1_DONE.md`、README 5.2 节、受保护基线产物。  
契约：`briefs/OPTIMIZATION_REVIEW_SPEC.md`（C1–C14 + N1–N9）。

**spec approval: not obtained (autonomous review run)** —— 无人工批准，置信度按“无人值守审查”计。

## 0. 环境与源状态

| 项 | 值 |
|---|---|
| 日期 | 2026-11-03 |
| 解释器 | Python 3.12.13（`.venv`） |
| 依赖 | scikit-learn 1.9.1 / numpy 2.5.4 / pandas 3.1.0 / joblib 1.6.0 |
| 版本控制 | 仓库 `master` **无任何 commit**（`git rev-parse HEAD` 失败）；不用 SHA |
| 源状态标识 | 受审查文件 sha256 树哈希 = `42ffa64c75481128e5614df4b9f28f5ba8e2ebcf98925e53847c88146893617b`（20 个文件，清单见 §6） |
| 审查产物 | 本文件、`OPTIMIZATION_REVIEW_SPEC.md`、`omp-optimize-review.md`、`optimization_review_checks.py`、`optimization_review_checks.json` |
| 未修改 | 实现脚本、DONE 报告、README、所有 artifacts（审查为只读；见 §5 前后 hash 一致） |

## 1. 唯一入口命令与最终结果（最后一次运行，晚于最后一次代码编辑）

```bash
cd /data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening
.venv/bin/python briefs/optimization_review_checks.py --negative-controls \
  --json briefs/optimization_review_checks.json
```

实际输出（2026-11-03T03:54:09Z，wall **54.6 s**，含 S10/S11 子进程）：

```
PASS S01 audit rows == independent FN/FP selection  (0.0s)
PASS S02 audit schema + blank manual columns  (0.0s)
PASS S03 heuristic tags match definitions  (0.1s)
PASS S04 error_summary confusion/totals  (0.0s)
PASS S05 documented empty-reason gap  (0.0s)
PASS S06 threshold strategies reproduce on test  (0.0s)
PASS S07 thresholds re-derived on val  (0.4s)
PASS S08 bootstrap CI reproducible  (7.2s)
PASS S09 GroupKFold protocol and aggregates  (0.0s)
PASS S10 cached features only (encoder guard)  (6.5s)
PASS S11 reruns reproduce committed artifacts  (39.1s)
PASS S12 protected artifacts unchanged  (0.0s)
PASS S13 docs + Phase 2 matrix  (0.0s)
FAIL S15 doc numbers == artifacts  (0.0s)
  CheckFailure: doc numbers disagree with artifacts:
  ['DONE fold threshold: missing 0.119',
   'DONE strategy val: missing 0.838',
   'DONE strategy val: missing 0.972',
   'DONE strategy test: missing 0.788']
PASS S14 protected artifacts unchanged across the suite (8 files)
CTRL-OK S01 / S03 / S06 / S07 / S08 / S09 / S15 / S10(encoder) / S10(network) —— 9/9
14/15 scenarios passed, 0 skipped, 1 failures
```

结论：O1–O7 的**行为**主张全部通过；唯一红灯是 DONE 报告的 4 处 3 位小数舍入与 artifacts 原始值不一致（文档缺陷，P3，见 §4）。退出码 1，因为 S15 是契约中的 C14。

## 2. 主张 → 场景 → 结果

| 场景 | 契约 | 结果（实测值） |
|---|---|---|
| S01 | C1 | FN doc_id 集合 = 独立重算集合，10 行；FP 30 行；FN/FP 分数与 `test_predictions.csv` 逐位相同（atol 1e-9）；`decision!='relevant'` 与 `score<threshold` 在 177 行上**零分歧** |
| S02 | C2 | 两 CSV 均 21 列（doc_id, pmid, title, year, label, score, threshold, decision, design, reason, quote, label_source, confidence, weak, text_source, abstract_available, text_chars, heuristic_tag, manual_tag, manual_note, text_head）；pmid/title/heuristic_tag 全非空；manual 两列全空；`text_head` 最大 500 字符、无换行 |
| S03 | C3 | 40 条记录逐条按定义重推 tag 全部一致（含 design_* 与 1 条 `design_missing`）；CSV 重算 FN tag：low_score_band 9, high_score_confident_error 8, design_case_report 5, design_cohort_or_screening 4, empty_abstract 2, pdf_text_source 2, below_review_band 1, design_missing 1；FP tag：high 17, case_report 13, cohort 12, mid 9, empty_abstract 4, low 4, pdf 4, review 3, other 2 —— 与 `error_summary.json` 完全相同 |
| S04 | C4 | totals n_test 177 / pos 130 / neg 47 / FN 10 / FP 30 / TP 120 / TN 17；computed = baseline cm = `metrics.json`；`checks.all_pass=true` |
| S05 | C5 | 唯一空 reason 的 FN = PMID 35791503，DONE §6 已披露 |
| S06 | C6 | 4 种策略；阈值 0.1525784656 / 0.0158511018 / 0.0010562744 / 0.0569451898；逐策略用 `test_predictions.csv` 重算 R/P/F1/F2/PP/混淆，全部 ≤1e-9 误差；AP 0.9315 与 `metrics.json` 一致；基线策略阈值 = 部署阈值 |
| S07 | C7 | 用缓存 npz（`spectra2_mc24_ch3000_n974_339bee2712.npz`）+ `model.joblib` 重算 val 140 行分数；`max_precision@R≥0.95`→0.1525784656405449、`max_f2`→0.015851101838052273，与记录值逐位相同；4 策略 val 指标全部重算一致；test 未参与 |
| S08 | C8 | n_bootstrap 2000、n_groups 177、95% 百分位；点估计 = metrics.json（≤1e-12）；独立实现（种子 20261103）CI vs 记录 CI：Recall [0.8741,0.9664] vs [0.8712,0.9627]；Precision [0.7361,0.8616] vs [0.7342,0.8618]；F1 [0.8106,0.9000] vs [0.8104,0.8987]；AP [0.8903,0.9609] vs [0.8906,0.9633] |
| S09 | C9 | k=5、n=883（678 正）、groups=883、`encoder_invoked=false`；折阈值 0.256305/0.030307/0.118538/0.231313/0.249127；每折 n_train+n_val+n_test=883，val 处 Recall 0.9528–0.9646 ≥0.95；均值±std 由 per_fold 精确重算（R 0.9333±0.0164、P 0.8460±0.0380、F1 0.8873±0.0262、AP 0.9427±0.0140）；pooled OOF 883 行自洽（R 0.9336/P 0.8463/AP 0.9403，cm TP633/FP115/FN45/TN90） |
| S10 | C10 | 子进程把 `encoder.document_features(_batch)`/`load_encoder`/`SpecterEncoder` 换成抛错实现 + socket/urllib audit hook：`eval_hardening.py` 5.7s 产出 3 个 JSON、`error_audit.py` 0.8s 产出 3 个文件，`ENCODER_CALLS=0`、`NETWORK_EVENTS=0`。（全量编码基线 39.4 min/974 篇，5.7s 不可能包含编码） |
| S11 | C11 | 两脚本各重跑 2 次并对已提交产物：`fn_audit.csv`/`fp_audit.csv` 字节完全相同；`error_summary.json`/`bootstrap_ci.json`/`group_cv.json`/`threshold_strategies.json` 除 generated_at/fit_seconds/输出目录外逐字段相同（含 bootstrap 2000 次与 CV 5 折的每个数值） |
| S12 | C12 | 8 个受保护文件 sha256 一致（§6）；md5 前缀与 DONE 声明一致：model.joblib `90905b95`、npz `3507745f`、test_predictions `bb079439`；两个脚本内无 `joblib.dump`/`np.savez`/`torch.save` |
| S13 | C13 | DONE 7,084 B（含 O1–O7、bootstrap、GroupKFold、max_f2）、matrix 5,163 B（含 困难负样本/章节感知分块/CUDA）、README 有 “Optimization Phase 1” 且指向 `artifacts/optimization/` |
| S15 | C14 | **FAIL**：4 处文档数字与原始值不符（见 §4） |
| S14 | 不变量 | 套件运行前后 8 个受保护文件 hash 不变 |

## 3. 否定对照（checker 失败路径可达）

对输入副本做定向篡改（不触碰仓库文件），逐条要求对应场景变红：

| 对照 | 篡改 | 实际检测输出 |
|---|---|---|
| N1 | 删 1 行 `fp_audit.csv` | `fp_audit.csv doc_ids != score-rule FP` |
| N2 | 改 1 行 `heuristic_tag` | `FP 33785045: empty_abstract mismatch` |
| N3 | 策略 test precision +0.05 | `max_f2: test precision mismatch` |
| N4 | 策略阈值 +0.001 | `max_f2: threshold 0.0168511018... != re-derived 0.0158511018...` |
| N5 | bootstrap recall ci_low +0.15 | `recall: mean outside CI` |
| N6 | 1 折 test recall +0.01 | `recall: mean not reproducible from folds` |
| N7 | DONE/README 0.963→0.960 | `doc numbers disagree ... missing 0.963` |
| N8 | 直接调用 `encoder.document_features` | 子进程 `ENCODER_CALLED` |
| N9 | `socket.getaddrinfo('127.0.0.1', 9)` | hook `NETWORK_EVENTS=1` |

9/9 被检出。注意其边界：这证明“已知坏输入会触发失败路径”，不证明 checker 覆盖该约束的所有违规形态。

## 4. 失败项与发现（诚实记录）

### F1（P3，文档，唯一红灯）DONE 的 4 个数字由“已四舍五入的中间值”再取整

`briefs/OPTIMIZATION_PHASE1_DONE.md` 与 artifacts 原始值对照（3 位小数，ROUND_HALF_UP）：

| 位置 | 写入值 | 原始值 | 正确值 |
|---|---|---|---|
| §0 与策略表：`max_recall@PP≤120` test Precision | 0.789 | 123/156 = 0.7884615… | **0.788** |
| 策略表：`max_recall@P≥0.80` val Precision | 0.839 | 109/130 = 0.8384615… | **0.838** |
| 策略表：`max_recall@PP≤120` val Recall | 0.973 | 106/109 = 0.9724770… | **0.972** |
| §3 GroupKFold 折阈值第 3 折 | 0.118 | 0.11853820… | **0.119** |

影响：不改变任何结论（O1–O7 判定、策略排序、Phase 2 门槛均不变），但 DONE 是决策文档，数字须与 artifacts 对齐。JSON 全部正确，属报告转写错误。

## 5. 跳过/不适用层及原因

| 层 | 状态 | 原因 |
|---|---|---|
| 项目测试套件 | 不适用 | 仓库无 `tests/`、无 pytest 配置；这两个脚本原本无测试。审查用独立 checker 覆盖其输出契约 |
| 类型检查（mypy/pyright） | 未运行 | 仓库无类型配置；审查侧不为只读审查引入新工具链 |
| lint/format | 未运行 | 同上；脚本可 `py_compile`，且审查未改实现 |
| 覆盖率 | 不适用 | 无测试套件可统计；checker 覆盖范围已在 §2 逐条声明 |
| 变异测试工具 | 未使用 | 环境无 mutmut/cosmic-ray；按 old-coder 允许的替代：9 条人工否定对照（§3），每条都记录了检测输出 |
| 人工 SPEC 批准 | 未获得 | 自主审查；置信度相应下调 |
| 人工逐条复核 40 条错误 | 未做 | 属计划内人工后续工作；`manual_tag`/`manual_note` 保持空白是正确行为 |

## 6. 受保护产物（审查前后一致）

| 文件 | sha256 |
|---|---|
| artifacts/model.joblib | `e3b90fb50bb1bfd4fbe505f9701fbad9186b733e0ddc0be630487c20a9afc556` |
| artifacts/model_all.joblib | `518ffc09c92176f29d04c1d63464108e75a417969ac4bea5dddf9c469aa09821` |
| artifacts/metrics.json | `4bf5b583834720b7a812ba775b382d1bad8efb78979558d93a4a4d7ff1493621` |
| artifacts/test_predictions.csv | `73c760dc4c5d2df1aac3b3f3eac13f90085ae612705fc4d9bd7a80fc7a37c201` |
| artifacts/config.json | `5b2be8642df5a1ea0fea9ce1b689fb092959428c193fcbb909270f7bd6284dc3` |
| artifacts/features/specter2_mc24_ch3000_n974_339bee2712.npz | `6ef1ec6d9ea66ce487a8399d7e7b0853aebb29443cdc49e9a2518cb36960d76d` |
| data/labeled_papers.csv | `f70b78522334a4b7279e6fadffa7e0f681b4e7474cec01307910a133a711b30d` |
| data/final_labels.csv | `4e612a1b2731d90426f8f8629442002868a30079f6a3aee9b3fb04b1d571bdad` |

mtime 佐证未被覆盖：model.joblib 2026-11-02 19:48:02、metrics/test_predictions 19:48、特征 npz 19:45；优化产物 2026-11-03 11:34。

受审查输入（树哈希 `42ffa64c…` 的 20 个文件）另含：scripts/error_audit.py `7664c6f2…`、scripts/eval_hardening.py `52bc2c5b…`、artifacts/optimization/{bootstrap_ci,error_summary,group_cv,threshold_strategies}.json、fn/fp_audit.csv、phase2_experiment_matrix.md、briefs/OPTIMIZATION_PHASE1_DONE.md、briefs/OPTIMIZATION_PLAN.md、README.md。

## 7. 已知限制

1. 断言与实现由同一 agent 编写，存在同源盲点；否定对照只覆盖 9 种已知坏输入。
2. 数据集内 PMID 唯一（974/974；test 177/177；default 883/883），因此 GroupKFold 退化为 KFold、PMID cluster bootstrap 退化为行级 bootstrap；CI 不是 cluster-robust 的额外保护。报告未声明该退化，属表述精度问题（P3，未单列红灯）。
3. `max_recall@pp_budget_120` 的预算在 val（n=140）上满足，test（n=177）实际 PP=156；DONE 标注“val 口径”但预算绝对值不可跨集比较（P3 表述建议）。
4. S10 只证明 Python 侧 encoder 入口未被调用；脚本运行环境也未加载 torch/transformers（实测 `'torch' in sys.modules == False`，`import train` 0.58 s），故原生编码路径亦不可达。
5. 本次不评估 Phase 2 矩阵的科学价值，只核对事实与数字来源。
