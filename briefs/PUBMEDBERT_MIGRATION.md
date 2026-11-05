# SPECTER2 → PubMedBERT 迁移（2026-11-03）

## 决策

对比 `weights.rar`（casefilter-head-v3）后，**舍弃 SPECTER2-C**，主链路改为冻结 PubMedBERT 嵌入 + 线性头。

## 主模型选择

| 候选 | 测试 AP | 测试 R@协议 | 结论 |
|---|---:|---:|---|
| shipped head + val R≥0.95 标定 | **0.966** | **0.985** | **采用为 `model.joblib`** |
| 本仓库标签重训 LR（B, C=1） | 0.919 | 0.892 | 对照 `model_retrained.joblib` |
| 网格最优（C=10, mode C） | 0.922 | 0.938 | 对照 `model_opt_C10.joblib` |
| SPECTER2-C | 0.931 | 0.923 | 归档 `artifacts/archive_specter2/` |

## 代码变更

- `encoder.py`：SentenceTransformer PubMedBERT；1536-d pooling
- `train.py` / `predict.py` / `api.py`：主链路切换
- `scripts/export_shipped_bundle.py`：现成权重 → joblib + 阈值标定
- `scripts/optimize_pubmedbert.py`：C × mode 网格
- `scripts/retrain_with_gold.py`：读 PubMedBERT cache，默认 mode B
- `requirements.txt`：`sentence-transformers`；移除 `adapters`

## 下一步

见 README §5。
