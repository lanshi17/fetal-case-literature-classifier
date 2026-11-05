# OMP 审查：优化 Phase 1（对照 O1–O7）

审查日期：2026-11-03  
仓库：`/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening`  
契约：`briefs/OPTIMIZATION_REVIEW_SPEC.md` ｜ 证据：`briefs/OPTIMIZATION_EVIDENCE.md` ｜ 复跑：`briefs/optimization_review_checks.py`

**spec approval: not obtained (autonomous review run)** —— 本次为无人值守审查，置信度按此计。

## 结论

**O1–O7 全部满足（行为与产物层面）；唯一红灯是 DONE 报告的 4 处小数舍入（P3 文档缺陷）。**  
未发现需要回派重做的 P0/P1/P2 问题；`artifacts/model.joblib`、特征 cache、`metrics.json`、`test_predictions.csv`、标签数据在审查全过程 hash 不变。复跑套件：**14/15 场景通过，9/9 否定对照被检出**（唯一失败项 S15 即下列文档问题）。

## 四项必查

| 必查项 | 结论 | 证据 |
|---|---|---|
| FN/FP CSV 行数 = 基线混淆 FN=10 / FP=30 | **是**。`fn_audit.csv` 10 行、`fp_audit.csv` 30 行，doc_id 集合与独立重算完全一致；`error_summary.json` computed = baseline = TN17/FP30/FN10/TP120；`decision` 列与 `score>=threshold` 在 177 行零分歧 | S01、S04 |
| bootstrap / group_cv 读 feature cache、无全量 encode | **是**。两脚本在“encoder 入口抛错 + socket audit hook”子进程中正常产出，`ENCODER_CALLS=0`、`NETWORK_EVENTS=0`；`eval_hardening.py` 5.7 s（全量编码基线 39.4 min）；`group_cv.json` `encoder_invoked=false`；重跑 2 次数值逐位复现 | S08–S11 |
| 主模型 `artifacts/model.joblib` 未被无意覆盖 | **是**。sha256 `e3b90fb5…` 恒定，mtime 仍是 2026-11-02 19:48:02，与 DONE 记录的 MD5 前缀 `90905b95` 一致；脚本内无 `joblib.dump`/`np.savez` | S12、S14 |
| 阈值策略 ≥3 且 test 指标可复现 | **是**。4 种策略，全部同一 val 选阈、同一 test 报告；逐策略用 `test_predictions.csv` 重算 R/P/F1/F2/PP/混淆（≤1e-9），且 `max_precision@R≥0.95` 与 `max_f2` 的阈值由缓存 val 分数独立复现 | S06、S07 |

## O1–O7 判定

| ID | 结论 | 关键实测 |
|---|---|---|
| O1 | PASS | 两个 CSV 各 21 列、10/30 行，含 pmid/title/score/decision/design/reason/text_source/heuristic_tag/manual_note/text_head；manual 列刻意留空 |
| O2 | PASS | `error_summary.json`：tag 计数可由 CSV 逐条重推，confusion 与基线一致，`checks.all_pass=true` |
| O3 | PASS | 2000 次 PMID 分组 bootstrap，95% CI：R [0.871,0.963]、P [0.734,0.862]、AP [0.891,0.963]；独立实现复核端点差 ≤0.005 |
| O4 | PASS | GroupKFold k=5，883 篇/678 正，R 0.933±0.016、P 0.846±0.038、F1 0.887±0.026、AP 0.943±0.014；仅读 npz |
| O5 | PASS | 4 策略 `threshold_strategies.json`；基线策略阈值 = 部署 0.1525784656 |
| O6 | PASS | DONE 7.1 KB + Phase 2 矩阵 5.2 KB + README 5.2 节 |
| O7 | PASS | 主模型/特征/预测/标签 hash 不变；新脚本独立运行；回滚点多 |

## 回派清单（全部 P3，不阻塞结论）

1. **修正 DONE 的 4 个数字**（唯一红灯 S15；JSON 正确，属报告转写错误）：
   - `§0` 与策略表：`max_recall@PP≤120` test Precision `0.789` → **`0.788`**（123/156）。
   - 策略表：`max_recall@P≥0.80` val Precision `0.839` → **`0.838`**（109/130）。
   - 策略表：`max_recall@PP≤120` val Recall `0.973` → **`0.972`**（106/109）。
   - `§3` 折阈值 `0.118` → **`0.119`**（0.118538）。
   修完 `S15` 即转绿，无需改代码。
2. **注明分组退化**：数据集 PMID 唯一（974/974；test 177/177；default 883/883），故 GroupKFold ≡ KFold、PMID cluster bootstrap ≡ 行级 bootstrap。建议在 DONE/README 加一句“CI 为行级重采样口径，无重复 PMID 可聚类”。
3. **PP 预算口径**：`pp_budget≤120` 在 val（n=140）选出，test（n=177）实际 PP=156。建议表述为比率（val PP/n≈0.86）或在评估集上重新施加预算，避免跨集绝对数比较。
4. 可选清理：`eval_hardening.py` 的 `--jobs` 实为 GroupKFold 折数（建议 `--folds`）；`error_audit.py` 的 `checks` 可增加“decision 与 score 规则一致”一条（当前数据一致，但缺自检）。

## 方法学观察（非缺陷）

- `max_recall@P≥0.80` 在 test 上 172/177 判正（接近全通过），DONE 已明确不建议默认使用——判断正确。
- GroupKFold 折阈值 0.030–0.256 波动大，DONE 已声明不可当单一部署阈值，并给出 pooled OOF 的逐行阈值口径——处理正确。
- 复现校验（test 分数最大差 2.96e-08）来自 `test_predictions.csv` 的 float32 落盘精度，非模型不一致。

## 置信度与残余风险

- 行为主张（O1–O5、O7）**高**：全部有独立重算 + 重跑确定性 + 否定对照。
- 报告文字（O6）**中**：4 处数字需按第 1 条修正；其余定性表述未逐句核验。
- 残余风险：审查断言与实现同源；无人工 SPEC 批准；CI 与 CV 的分组结构退化（第 2 条），因此 bootstrap 区间不应被理解为“聚类稳健”的额外保证。
