# PubMedBERT 遗传变异病例筛选

产前遗传变异**病例报告 / 病例系列**相关性筛查。输出三档：`relevant` / `review` / `irrelevant`。

**默认最优解**：公平分块 **DoRA** + 金标重校准阈值 + **规则层**（`artifacts/best/`）。

| 组件 | 设定 |
|------|------|
| 编码器 | NeuML `pubmedbert-base-embeddings`，块长 480、最多 12 块，保留前 2 块 |
| 文档特征 | `L2(mean_all) ‖ L2(mean_first_2)` → **1536-d** |
| 适配器 | DoRA（r=16, α=32，`query`/`value`）+ 线性分类头 |
| 阈值 | **0.7619**（金标 49 篇上 R≥0.95 重校准） |
| 规则层 | 硬排除 NIPT/大规模筛查、综述/指南、GWAS/数据库等；病例线索保护 |
| 决策 | `score ≥ t` → relevant；`t−0.15 ≤ score < t` → review；否则 irrelevant；规则可强制排除或降为 review |

旧方案 frozen PubMedBERT + LR：`artifacts/model.joblib`（`--legacy-lr`）。SPECTER2 已弃用（`artifacts/archive_specter2/`）。

---

## 1. 目录

```text
src/
  encoder.py                 # 分块编码 / 池化（冻结 PubMedBERT）
  rule_layer.py              # 困难负样本规则层
  predict.py                 # 推理：默认 DoRA+rules，--legacy-lr 走旧基线
  api.py                     # FastAPI 服务
  mcp_server.py              # MCP 入口（stdio / sse / streamable-http）
  train.py                   # frozen + LR 基线训练
  build_dataset.py           # data/labeled_papers.csv 构建
  scripts/
    finetune_chunk_fair.py   # 公平分块微调 Full / DoRA / GaLore（★ 最优解训练器）
    ablation_abcd.py         # 消融 A/B/C/D
    eval_rule_layer.py       # 规则层评测
    gold_review.py           # 金标审核（Web / 终端）
    export_dora_review.py    # 导出 DoRA 分数与 FN/FP 复核队列
    ahp_label/               # 多模型 AHP 弱标流水线
    <历史基线脚本>            # compare_* / retrain_* / eval_hardening / error_audit 等
docs/
  ARCHITECTURE.md            # 整体架构、算法设计、数据结构、目录结构
  briefs/                    # 任务书、评审规格、实验与复盘记录
data/                        # 标签、语料、全文包（大文件已 gitignore）
artifacts/                   # 模型权重、阈值、评测产物（脚本写出，勿手工编辑）
```

> 旧代 `pubmedbert/`（独立打包的 `casefilter` 子项目：规则层 + 现成 head_v3 权重）已移除，
> 本仓库只保留最终代 **DoRA + 规则层**。源码存档在 git tag `archive/pre-dora-legacy`：
> `git checkout archive/pre-dora-legacy -- pubmedbert`

代码根 `src/` 不是包：模块之间按扁平名互相 import（`from encoder import ...`），脚本用
`ROOT = Path(__file__).resolve().parents[2]` 定位仓库根，并把 `ROOT/"src"` 加入 `sys.path`。
因此**所有命令都在仓库根目录执行**。也可 `export PYTHONPATH=src` 后直接 `python -c "import train"`。

---

## 2. 环境

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# GPU 版 torch 按本机 CUDA 安装；另需 peft / accelerate / galore-torch（见 requirements）
```

可选本地快照：

```bash
export PUBMEDBERT_MODEL_PATH=/path/to/NeuML/pubmedbert-base-embeddings/snapshot
```

---

## 3. 预测 / API（默认最优解）

```bash
# 单条 / 抽样
.venv/bin/python src/predict.py --best --sample 1
.venv/bin/python src/predict.py --best --pdf paper.pdf --title "Prenatal case report"

# 关闭规则
.venv/bin/python src/predict.py --best --no-rules --sample 1

# 旧 frozen+LR
.venv/bin/python src/predict.py --legacy-lr --sample 1

# API（默认 MODEL_BACKEND=best）
MODEL_BACKEND=best USE_RULES=1 \
  .venv/bin/uvicorn --app-dir src api:app --host 0.0.0.0 --port 8000 --workers 1

# MCP（默认 stdio；Cursor 见 .cursor/mcp.json）
MODEL_BACKEND=best USE_RULES=1 .venv/bin/python src/mcp_server.py
# 可选 HTTP：.venv/bin/python src/mcp_server.py --transport streamable-http --port 8765
```

MCP 工具：`get_screener_info` / `screen_paper` / `screen_pdf` / `screen_batch`。说明见 [`artifacts/best/BEST.md`](artifacts/best/BEST.md)，架构与算法见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

### 3.1 最佳模型用法示例（逐条可跑）

以下输出为本机实跑（`device=cuda`，`thr=0.7619`），可直接对照。

**① 只给标题 + 摘要（一条）**

```bash
.venv/bin/python src/predict.py --best \
  --title "Prenatal diagnosis of a 22q11.2 deletion in a fetus with tetralogy of Fallot" \
  --abstract "A 28-year-old gravida 2 woman whose fetus showed tetralogy of Fallot and increased nuchal translucency at 13 weeks. Amniocentesis was performed; chromosomal microarray revealed a 2.5 Mb deletion at 22q11.2."
```

```jsonc
{
  "doc_id": null,
  "title": "Prenatal diagnosis of a 22q11.2 deletion in a fetus with tetralogy of Fallot",
  "decision": "relevant",
  "model_decision": "relevant",
  "relevance_score": 0.948522,
  "threshold": 0.7619361281394958,
  "review_threshold": 0.611936,
  "full_text_available": false,
  "pdf_used": false,
  "feature_mode": "dora_chunk_1536",
  "encoder_family": "pubmedbert_dora",
  "backend": "dora_chunk",
  "rules_enabled": true,
  "rules_fired": [],
  "rules_overridden": false,
  "rules_protected_by_case": true      // 命中病例线索保护（不改变结论，只阻止误排除）
}
```

**② 给全文 PDF（推荐，召回明显更高）**

```bash
.venv/bin/python src/predict.py --best --pdf paper.pdf --title "Prenatal case report"
```

摘要常省略用了哪种检测；有全文时 `full_text_available=true`，DoRA 能在 Methods/Results 里找到检测与表型证据。消融 B 显示全文对召回/AP 有明显提升，**生产环境请传全文**。

**③ 批量：CSV / TSV / JSON**

```bash
# 输入列：doc_id(pmid), title, abstract，可选 text / pdf_path / design
.venv/bin/python src/predict.py --best --input papers.csv --output out/pred.json
```

输出形态：**多条 → JSON 数组；结果只剩一条时折叠为单个对象**（`--limit 1` 也如此），下游解析请兼容两种。JSON 输入支持对象数组或 `{"records": [...]}`。也可用 `--json '<单条 JSON>'` 直接喂一条：

```bash
.venv/bin/python src/predict.py --best \
  --json '{"doc_id":"x1","title":"Fetal ultrasound anomaly cohort","abstract":"Retrospective cohort of 120 fetuses with increased nuchal translucency; whole exome sequencing performed."}' \
  --no-json-lines
```

这条是**规则层生效的典型**：模型给 `0.95116`（判 `relevant`），但命中 `LARGE_PRENATAL_COHORT`（大规模产前筛查队列，非单病例报告），最终：

```jsonc
{
  "decision": "irrelevant",          // 规则强制改判
  "model_decision": "relevant",      // 模型原始判断
  "relevance_score": 0.95116,
  "rules_fired": ["LARGE_PRENATAL_COHORT"],
  "rules_overridden": true
}
```

`decision` 与 `model_decision` 不一致 = 规则层改判了，用 `rules_fired` 查原因。

**④ Python 调用（集成到自己的流程）**

```python
import sys
sys.path.insert(0, "src")          # 在仓库根目录执行
from predict import load_predictor

predictor = load_predictor(best=True, use_rules=True)   # device 默认 cuda→cpu
print(predictor.backend, predictor.threshold)           # dora_chunk 0.7619361281394958

out = predictor.predict_record({
    "doc_id": "demo-1",
    "title": "Prenatal diagnosis of a 22q11.2 deletion in a fetus with tetralogy of Fallot",
    "abstract": "…chromosomal microarray revealed a 2.5 Mb deletion at 22q11.2.",
    # "text": "...",          # 可选：全文
    # "pdf_path": "paper.pdf",  # 可选：自动抽文本（相对路径以仓库根为基准）
})
print(out["decision"], out["relevance_score"], out["rules_fired"])
```

常用方法：

| 方法 | 输入 | 返回 |
|---|---|---|
| `predict_record(dict)` | `doc_id` / `title` / `abstract` / `text` / `pdf_path` / `design` 任一组合 | 单条结果 dict |
| `predict_pdf(path, title="", abstract="")` | PDF 路径 | 单条结果 dict（自动抽全文） |
| `predict_input_file(path, limit=None)` | `.json` / `.csv` / `.tsv` | 结果 dict 列表 |

**⑤ 决策规则（读输出时的关键）**

```text
score ≥ t (0.7619)                 → relevant
t − 0.15 (0.6119) ≤ score < t      → review
score < t − 0.15                   → irrelevant
之后规则层可强制 irrelevant，或把 relevant 降为 review（病例线索可保护，见 rules_protected_by_case）
```

评分口径：`relevant` 与 `review` 都算"召回保留"，只有 `irrelevant` 是漏检。下游只想要高置信集合 → 取 `decision == "relevant"`；要保召回 → 取 `decision != "irrelevant"`，再把 `review` 送人工。

**⑥ 常用开关**

```text
--no-rules         # 关闭规则层，只看模型原始分（对比 / 调试）
--device cpu       # 强制 CPU（无 GPU 或显存不足）
--sample 5         # 从 data/labeled_papers.csv 抽前 5 条，冒烟测试
--limit 100        # 配合 --input，只跑前 N 条
--output out.json  # 结果落盘（批量建议必带）
```

> **前提**：`artifacts/finetune_fair/dora/checkpoint/model.pt`（约 440 MB，gitignore，需训练或另行获取）、`artifacts/best/config.json` 与阈值文件，以及编码器快照 `NeuML/pubmedbert-base-embeddings`（可用 `PUBMEDBERT_MODEL_PATH` 指向本地快照）。
>
> **耗时参考**（本机 RTX 4060 + cuda）：进程启动+加载约 4 s（三次实测 3.6–4.6 s，含 CUDA 初始化），之后单条标题摘要推理约 6 ms；有全文时按块数线性增长（≤12 块）。CPU 会明显更慢，建议 `--device cpu` 仅用于调试。

---

## 4. 主结果（seed=42，默认子集 test n=177）

Pipeline 口径：`relevant|review` 计为召回保留（仅 `irrelevant` 为漏检）。

| 系统 | thr | Recall | Precision | F1 | 备注 |
|------|----:|-------:|----------:|---:|------|
| **DoRA + rules（最优）** | **0.762** | **0.977** | **0.830** | **0.898** | 困难负 Spec **0.815** |
| DoRA only | 0.762 | 1.000 | 0.774 | 0.872 | 困难负 Spec 0.37 |
| fair head_v3（frozen+LR） | 0.192 | 0.954 | 0.849 | 0.899 | test AP 高，gold 泛化弱 |

金标 49（人审）：DoRA+rules pipeline R=1.000 / P=0.581；DoRA gold AP≈0.76（优于 Full FT / GaLore / AHP-LR）。

更多：[`artifacts/rule_layer/REPORT.md`](artifacts/rule_layer/REPORT.md)、[`artifacts/ablation_abcd/REPORT.md`](artifacts/ablation_abcd/REPORT.md)。

---

## 5. 训练与复现（可选）

```bash
# 数据（全文包就位时）
.venv/bin/python src/build_dataset.py

# 公平分块微调（DoRA / full / galore；device 自动 cuda→cpu）
.venv/bin/python src/scripts/finetune_chunk_fair.py --method dora

# AHP 弱标波次
bash src/scripts/ahp_label/run_wave.sh

# 消融 A/B/C/D
.venv/bin/python src/scripts/ablation_abcd.py --device cuda

# 规则层评测（读已导出的 DoRA 分数）
.venv/bin/python src/scripts/eval_rule_layer.py

# 金标审核 UI
.venv/bin/python src/scripts/gold_review.py --queue fn_fp --port 8765
```

训练协议摘要：默认子集 train + AHP 硬标入训；val/test 为 seed-42 默认子集；早停看 val AP@0.5；阈值先在 val 上 R≥0.95，再用金标重校准。

---

## 6. 消融结论（简）

| 实验 | 结论 |
|------|------|
| A 仅标题摘要 | 编码约快 13×；可作速度基线 |
| B +全文 | DoRA 召回/AP 明显受益；生产必须全文 |
| C +词表特征 | 相对 B 几乎无增益 |
| D 困难负样本 | 规则层把 Spec 从 ~0.37 提到 ~0.82 |

---

## 7. 限制

- 仅研究用途，非临床决策工具。
- API 默认无鉴权，请单 worker 加载大模型。
- 金标规模仍有限（49）；规则偏保守，边界病例依赖 `review` 人工复核。
