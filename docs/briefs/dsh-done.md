# DSH 完成报告：SPECTER2 遗传变异病例筛选基线

日期：2026-11-02  
工作目录：`/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening`  
权威计划：`/home/yangzs/.cursor/plans/SPECTER2 screening pipeline-e4a3448e.plan.md`

## 1. 状态

全部必交文件已创建；数据构建、编码器加载、**全量 974 篇特征编码 + 全量训练**、单篇预测、FastAPI 冒烟均已实际跑通。测试集 177 篇（默认子集）指标已写入 `artifacts/metrics.json`。

## 2. 已创建文件

| 文件 | 说明 | 大小 |
|---|---|---|
| `build_dataset.py` | PMID 对齐、JATS 摘要、TXT→PDF 全文回退、统计 | 13 KB |
| `encoder.py` | SPECTER2 Base + classification adapter、分块、2304/2306 维特征 | 20 KB |
| `train.py` | 特征缓存、GroupShuffleSplit、阈值选择、A/B/C 消融、全量对照 | 28 KB |
| `predict.py` | PDF/JSON/CSV/样例预测、三段决策 | 10 KB |
| `api.py` | FastAPI `/health`、`/predict`（JSON + multipart PDF） | 6 KB |
| `requirements.txt` | 依赖清单 | 0.5 KB |
| `README.md` | 安装/构建/训练/预测/API/指标/限制 | 14 KB |
| `.gitignore` | 忽略全文包、特征缓存、模型、venv、cache | 0.5 KB |
| `data/labeled_papers.csv` | **974 行 × 21 列**，CSV 交付 | 27 MB |
| `data/build_stats.json` | 构建统计 | 0.5 KB |
| `artifacts/model.joblib` | 默认子集模型 bundle | 65 KB |
| `artifacts/model_all.joblib` | 全量对照 bundle | 65 KB |
| `artifacts/config.json` | 阈值、特征维、模型/编码器配置 | 2.1 KB |
| `artifacts/metrics.json` | 主指标 + 消融 + 分层 + 全量对照 | 12 KB |
| `artifacts/test_predictions.csv` | 测试集逐篇分数/决策 | 34 KB |
| `artifacts/features/specter2_mc24_ch3000_n974_*.npz` | 特征缓存（8 MB，不提交） | 8 MB |
| `briefs/dsh-done.md` | 本报告 | — |

## 3. 环境

```text
Python 3.12.13 (uv venv)
torch 2.15.0+cu132
transformers 4.57.6
adapters 1.3.0
scikit-learn 1.9.1
pandas 3.1.0 / numpy 2.5.4
PyMuPDF 2.0
fastapi 0.162.0 / uvicorn 0.54.0
```

- 项目 `.venv` 已创建；`uv pip check`：`All installed packages are compatible`。
- **本机 GPU 不可用**：`/dev/nvidia*` 不存在、`nvidia-smi` 失败，`torch.cuda.is_available() == False`。安装的是 CUDA 13.2 版 torch，代码自动回退 CPU；全量编码在 CPU 完成（2,366.8 秒，约 39.5 分钟，0.41 docs/s）。
- Base 模型从本地 snapshot 加载：
  `/data/yangzs/.cache/huggingface/hub/models--allenai--specter2_base/snapshots/3447645e1def9117997203454fa4495937bfbd83`
- Classification adapter 已下载到项目 `.cache/huggingface/hub/models--allenai--specter2_classification/...`。由于 `$HOME` 只读，`encoder.py` 自动把 `HF_HOME`/`TORCH_HOME` 落到项目 `.cache/`（已 gitignore）。
- 未微调 SPECTER2；Base 和 adapter 参数全部 `requires_grad=False`。
- 已验证完全离线加载：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 encoder.py --smoke` 成功，Base/adapter 均来自本地缓存。

## 4. 全文包恢复

```bash
tar -xzf /data/.Trash-1000/files/fulltext_package.tar.gz -C data
```

结果：`data/fulltext_package/` 338 MB，含 `manifest_fulltext.csv`、`txt/`、`xml/`、`pdfs/`。压缩包内含 macOS AppleDouble `._*` 文件（txt/xml 各 768 个），构建脚本已跳过，未计入统计。

## 5. 数据构建（实际输出）

```bash
.venv/bin/python build_dataset.py
```

```json
{
  "labels_rows": 974,
  "output_rows": 974,
  "manifest_rows": 974,
  "overlap_with_manifest": 974,
  "missing_manifest": [],
  "with_abstract": 764,
  "with_fulltext": 974,
  "without_fulltext": 0,
  "text_source": {"txt": 768, "pdf": 206, "none": 0},
  "labels": {"1": 699, "0": 275},
  "mean_text_chars": 26616.2,
  "mean_abstract_chars": 1383.4,
  "pdf_errors": []
}
```

审计：`doc_id == group_id == pmid` 全量成立；label / weak / reason 与 `final_labels.csv` 对齐；974 篇全文均非空；无重复 doc_id。

**与计划的差异（重要）**：manifest 标记 `TXT=Y/XML=Y` 929 篇，但 tar 内实际只有 **768** 套 txt/xml；另外 **161** 篇的 TXT/XML 不在包内，但都有 PDF。脚本按“TXT 优先、PDF 回退”处理，因此 974 篇全部有全文（768 TXT + 206 PDF），PDF 抽取无错误。210 篇仅 PDF、无 XML，`abstract` 为空字符串（符合规则）。

## 6. 编码器冒烟

```bash
.venv/bin/python encoder.py --smoke
```

```text
device=cpu dim=768 adapter=allenai/specter2_classification
metadata={'adapter_loaded': True, 'adapter_error': '', 'torch_version': '2.15.0+cu132',
          'cuda_available': False, 'device_name': 'cpu'}
chunks=1 lengths=[192]
embedding_shape=(768,) embedding_norm=1.0000
feature_shape=(2306,) meta={'full_text_available': True, 'n_chunks': 1, 'text_chars': 192}
finite=True
```

另验证：停用 adapter 后同一文本余弦相似度从 1.0 降到 **0.769**，确认 adapter 权重确实参与前向；`document_features` 无正文时 mean/max 为零、`full_text_available=false`。分块边界测试：全量 974 篇所有 chunk 最大长度 3000；长文生成 24 块并等距覆盖头/中/尾。

## 7. 全量训练

```bash
.venv/bin/python train.py --force-reencode --batch-size 16
```

- 编码 974 篇 → **11,275** 个正文块（平均 11.58 块/篇），耗时 **2,366.8 秒**（CPU，约 39.5 分钟，0.41 docs/s）。
- 特征缓存：`artifacts/features/specter2_mc24_ch3000_n974_339bee2712.npz`（8 MB）。
- 划分：`GroupShuffleSplit`（group_id=PMID），全量 623/156/195；默认子集 **566/140/177**，两边均含两类。
- 阈值：验证集目标 Recall≥0.95 → **t = 0.152578**（val Recall 0.9633 / Precision 0.8898）。
- 主模型：`StandardScaler + LogisticRegression(class_weight="balanced", C=1.0)`，模式 C（2306 维）。

**测试集（默认子集，n=177，130 正 / 47 负）：**

| 指标 | 值 |
|---|---:|
| Recall | 0.9231 |
| Precision | 0.8000 |
| F1 | 0.8571 |
| F2 | 0.8955 |
| AP | 0.9315 |
| ROC-AUC | 0.8280 |
| 混淆矩阵 | TN 17 / FP 30 / FN 10 / TP 120 |

**消融（同一 split，测试集 n=177）：**

| 模式 | 维度 | 阈值 | Recall | Precision | F1 | F2 | AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A 仅标题+摘要 | 768 | 0.0373 | 0.9692 | 0.7826 | 0.8660 | 0.9251 | 0.8998 | 0.7637 |
| B +全文 mean/max | 2304 | 0.2425 | 0.9077 | 0.8138 | 0.8582 | 0.8872 | 0.9308 | 0.8265 |
| **C +2 辅助特征** | **2306** | **0.1526** | **0.9231** | **0.8000** | **0.8571** | **0.8955** | **0.9315** | **0.8280** |
| 全量对照（974） | 2306 | 0.1017 | 0.9338 | 0.7888 | 0.8552 | 0.9007 | 0.9319 | 0.8547 |

解读：A 的 Recall 最高但 Precision/AP 最低；加入全文提升 AP/ROC-AUC，但同目标下测试 Recall 略降；C 相对 B 的 2 个辅助特征几乎无增益（AP 0.9315 vs 0.9308），保留用于可开关消融。

**分层报告**：测试集 177/177 均有全文，`abstract_only` 层为 0，因此该层没有测试指标；A 消融可作为“只有摘要”场景的近似参考。`metrics.json` 中已显式记录 `abstract_only: {"n": 0, "note": "empty stratum"}`。

缓存复用已验证：第二次 `train.py` 直接 `Cache hit`，约数秒完成并写出相同阈值/指标。期间修复了两个问题：
1. 旧 npz 的 `doc_ids` 为 object dtype，改为固定宽度 unicode，`allow_pickle=False` 可加载；
2. 分块重叠在超长段落上可能超过 `max_chars`，已加保护并**重新全量编码**；最终 974 篇所有 chunk 最大长度 = 3000（11,275 块）。

## 8. 预测冒烟

```bash
.venv/bin/python predict.py --sample 1
```

```json
{
  "doc_id": "11175621",
  "title": "Nuchal translucency: an ultrasound marker for fetal chromosomal abnormalities.",
  "decision": "relevant",
  "relevance_score": 0.99213,
  "threshold": 0.1525784656405449,
  "review_threshold": 0.002578,
  "full_text_available": true,
  "pdf_used": false,
  "feature_mode": "C"
}
```

另测：

- PDF 路径输入：`--pdf data/fulltext_package/pdfs/32282607_...pdf --title "PDF smoke"`（最终模型）→ `relevant`，`score 0.999707`，`full_text_available=true`，`pdf_used=true`。
- 内联 JSON、JSON 文件 `--input` 均跑通；标题-only 记录 → `full_text_available=false`，`score 0.0`，`irrelevant`（证明不会假装有全文）。
- 决策边界单元检查：`score=t` → relevant；`t-0.01` → review；`t-0.15` → review；`t-0.150001` → irrelevant。

测试集决策分布：relevant 150 / review 19 / irrelevant 8（共 177）；review 组正例率 0.474。

## 9. API 冒烟

```bash
MODEL_PATH=$PWD/artifacts/model.joblib \
  .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8014 --workers 1
```

`GET /health`：

```json
{
  "status": "ok",
  "model_loaded": true,
  "threshold": 0.1525784656405449,
  "review_threshold": 0.002578465640544897,
  "feature_mode": "C",
  "feature_dim": 2306,
  "device": "cpu",
  "adapter": "allenai/specter2_classification"
}
```

`POST /predict`（JSON）：

```json
{"decision":"relevant","relevance_score":0.999892,"threshold":0.152578,
 "review_threshold":0.002578,"full_text_available":true,"pdf_used":false,"feature_mode":"C"}
```

`POST /predict`（multipart，上传 PDF）：

```json
{"decision":"relevant","relevance_score":0.998069,"threshold":0.152578,
 "review_threshold":0.002578,"full_text_available":true,"pdf_used":true,"feature_mode":"C"}
```

API 不接受服务端本地路径（避免任意文件读取）；PDF 通过上传或 `pdf_base64` 传入，默认上限 30 MB。

## 10. 已知限制

1. **无可用 GPU**：容器未暴露 `/dev/nvidia*`，`torch.cuda.is_available()=False`；全量编码/训练在 CPU 完成。换到有 GPU 的机器同代码自动用 CUDA。
2. **不微调**：Base + adapter 冻结，只训练线性头；adapter 仓库不含分类 head 权重（日志 `No matching prediction head found`），本基线只用 CLS 表示，不需要该 head。
3. **摘要覆盖 764/974**：210 篇仅 PDF、无 XML，`abstract` 为空；4 篇 XML 抽取不到 abstract。全文特征仍可用。
4. **manifest 与实际包不一致**：161 篇标记 TXT/XML 但文件缺失，已用 PDF 回退；统计以实际 `text_source` 为准。
5. **单次划分**：仅一个 64/16/20 split，未做交叉验证；测试集 177 篇，指标有方差，`metrics.json` 同时给 AP/ROC-AUC。
6. **阈值目标**：验证集 Recall≈0.95，实际测试 Recall 0.923；灰带宽度 0.15 为固定启发式，未调优。
7. **abstract-only 层为空**：974 篇全部有全文，无法在真实测试集上评估“无全文”分层；只能用 A 消融近似。
8. **标签噪声**：默认排除 91 条 `weak=True` 或 `label_source=model` 的样本；全量对照仍含这些标签。标签是 LLM/人工混合流程，不是临床金标准。
9. **仅研究用途**：不是临床诊断/筛查工具。
10. **API 无鉴权/限流**：单 worker 设计，生产需自行加认证、限流和 PDF 大小限制。
11. **扫描版 PDF 无 OCR**：本次 206 个 PDF 均抽到文本；未来扫描件可能为空。

## 11. 验收命令清单（均已实际执行）

```bash
# 1 恢复
tar -xzf /data/.Trash-1000/files/fulltext_package.tar.gz -C data

# 2 数据
.venv/bin/python build_dataset.py

# 3 编码器
.venv/bin/python encoder.py --smoke

# 4 全量训练
.venv/bin/python train.py --batch-size 16

# 5 预测（样例 + PDF + JSON）
.venv/bin/python predict.py --sample 1
.venv/bin/python predict.py --pdf data/fulltext_package/pdfs/32282607_PMC7170435_Fetal_Sex_Results_of_Noninvasive_Prenata.pdf --title "PDF smoke"
.venv/bin/python predict.py --input /tmp/records.json

# 6 API
MODEL_PATH=$PWD/artifacts/model.joblib .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8013 --workers 1
curl -s http://127.0.0.1:8013/health
curl -s -X POST http://127.0.0.1:8013/predict -H 'Content-Type: application/json' -d '{"title":"...","abstract":"...","text":"..."}'
curl -s -X POST http://127.0.0.1:8013/predict -F 'title=test' -F 'pdf=@paper.pdf'
```

## 12. Git 卫生

`.gitignore` 已忽略：

```text
.venv/  .uv-cache/  .uv-python/  .cache/
data/fulltext_package/          # 338 MB
artifacts/features/             # 8 MB 特征缓存
artifacts/**/*.joblib           # 模型
artifacts/smoke/                # 冒烟产物
*.log
```

`git status` 中仅源码、README、`.gitignore`、`briefs/`、`data/final_labels.csv`、`data/labeled_papers.csv`、`artifacts/{config,metrics,test_predictions,train_summary}` 等文本产物会进入版本控制；未提交全文包、特征缓存或模型权重。

## 13. 建议的后续（非本次范围）

- 用 `--no-aux` 训练 B 作为更简基线；C 的辅助特征在本 split 无增益。
- 做 5-fold group CV 或 bootstrap CI，降低 177 篇测试集的方差。
- 若需要“无全文”鲁棒性，可在训练时随机丢弃正文块或用 A 模型作为 fallback。
- 上 GPU 后可用更大 `max_chunks` / 更长 `max_length`，并尝试 SPECTER2 的 retrieval adapter 作为对照（仍不微调）。
