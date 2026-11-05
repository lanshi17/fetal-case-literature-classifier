# DSH 实现任务：SPECTER2 遗传变异病例筛选基线

你是实现 Agent。工作目录必须是：

`/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening`

权威计划（必须遵守）：

`/home/yangzs/.cursor/plans/SPECTER2 screening pipeline-e4a3448e.plan.md`

完成后在仓库根写 `briefs/dsh-done.md`，列出已创建文件、关键命令输出摘要、已知限制。

## 禁止

- 不要微调 SPECTER2 / 不要改 Base 权重。
- 不要从 PubMed 抓取网络全文（全文包已有）。
- 不要把 `data/fulltext_package/`、大特征缓存、模型大文件 commit 进 git（可写 `.gitignore`）。
- 不要依赖交互式确认。

## 1. 恢复全文包

源文件（回收站）：

`/data/.Trash-1000/files/fulltext_package.tar.gz`

解压到：

`data/fulltext_package/`（内含 `manifest_fulltext.csv`、`txt/`、`xml/`、`pdfs/`）

若已存在且完整可跳过。

## 2. 环境

- 创建项目 `.venv`
- 安装 GPU 版 torch（本机有 RTX 4060）及：`transformers`、`adapters`、`scikit-learn`、`pandas`、`numpy`、`joblib`、`pymupdf`、`fastapi`、`uvicorn[standard]`、`python-multipart`
- SPECTER2 Base 本地路径（优先离线加载）：

`/data/yangzs/.cache/huggingface/hub/models--allenai--specter2_base/snapshots/3447645e1def9117997203454fa4495937bfbd83`

- Classification Adapter：`allenai/specter2_classification`（首次可下载并缓存）

## 3. 必须交付的文件

```
build_dataset.py
encoder.py
train.py
predict.py
api.py
requirements.txt
README.md
.gitignore
data/labeled_papers.csv   # 或等价 parquet + 说明；推荐 CSV/parquet 均可，README 写明
```

可选：`artifacts/` 下特征缓存、模型、指标。

## 4. 数据构建（build_dataset.py）

输入：

- `data/final_labels.csv`（BOM 注意 utf-8-sig；列含 pmid,title,year,label,label_source,confidence,weak,design,reason,...）
- `data/fulltext_package/manifest_fulltext.csv`（列 PMID,PMCID,...,TXT,PDF,XML,TXT_file,...）

规则：

- 按 PMID 对齐（应 974/974）
- `doc_id` = PMID；`group_id` = PMID
- `abstract`：优先从对应 XML 抽 JATS `<abstract>`；无则空字符串
- `text`：优先 TXT；否则 PyMuPDF 抽 PDF；仍无则空，并 `full_text_available=false`
- `label_reason` ← `reason`
- 保留 `label_source,confidence,weak,design`
- 输出 `data/labeled_papers.csv`（或 parquet）及简短构建统计（行数、有摘要数、有全文数）

## 5. encoder.py

- `load_encoder()`：Base + classification adapter，eval，cuda if available
- `chunk_text`：按段落/句子，约 300–500 tokens 量级（可用 max_chars≈3000 作基线），相邻重叠，`max_chunks=24`，尽量覆盖前中后
- `encode_texts` / `document_features`：
  - 标题+摘要向量 (768)
  - 正文块 mean (768)
  - 正文块 per-dim max (768)
  - 拼接 → 2304
  - 无正文时 mean/max 用零向量，且不要假装有全文
- 可选 2 个廉价辅助特征（病例词、变异/检测词命中），拼到特征末尾；train 消融可开关

## 6. train.py

- 特征缓存到 `artifacts/features/`（按 doc_id 或矩阵文件），避免反复 GPU 编码
- 默认训练子集：`weak=False` 且 `label_source != model`；另跑或另存「全量对照」指标
- `GroupShuffleSplit`：约 64/16/20（train/val/test），保证各组两类尽量都有
- LogisticRegression（class_weight 可 balanced）
- 验证集选阈值：目标 recall ≈ 0.95（`select_threshold`）
- 测试集报告：Recall, Precision, F1, F2, AP, confusion matrix
- 消融至少：A 仅标题摘要；B +全文；C +辅助特征（若实现）
- 按 `full_text_available` 分层报告
- 写出 `artifacts/model.joblib`、`artifacts/config.json`（含 threshold、feature 维、模型路径）、`artifacts/metrics.json`

若全量编码过久：至少完成数据集构建 + 小样本（如 32–64 篇）编码与训练冒烟，并在 `dsh-done.md` 说明全量训练状态；优先尽力跑完全量过滤后集合。

## 7. predict.py

- 支持 PDF 路径或 JSON/CSV 记录
- `score >= t` → relevant；`t-0.15 <= score < t` → review；否则 irrelevant
- 输出含 `decision,relevance_score,threshold,full_text_available`

## 8. api.py

- FastAPI：`GET /health`，`POST /predict`（接受元数据 + 可选上传 PDF）
- 单进程加载模型；README 写明 `uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1`

## 9. 验收（你必须实际跑）

1. `build_dataset.py` 成功，打印 overlap/全文覆盖统计
2. `python -c "from encoder import load_encoder; ..."` 或等价冒烟能加载编码器
3. `train.py` 至少冒烟完成并写出 artifacts（全量更好）
4. `predict.py` 对 1 篇样例输出三段决策之一
5. 写好 README：如何安装、构建数据、训练、预测、启动 API

开始实现。完成后写 `briefs/dsh-done.md`。
