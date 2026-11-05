# DSH 回派修复：P0 离线 adapter 加载

工作目录：`/data/yangzs/Projects/02_Reproduction/10_genetic-variant-screening`

审查结论见：`briefs/omp-review.md`、`briefs/EVIDENCE.md`（S19 FAIL）。

## 缺陷

在默认环境（未设置 `HF_HOME`）且 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 时：

```bash
env -u HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python encoder.py --smoke
```

会因找不到 `allenai/specter2_classification` 而 `RuntimeError`。adapter 实际在仓库 `./.cache/huggingface/hub/...`，但 `_resolve_adapter` / HF_HOME 逻辑在 `$HOME` 可写时不回退到该路径。

## 要求

1. 修复 `encoder.py`：离线或本地优先解析时，若默认 HF hub 找不到 adapter，回退到项目 `.cache/huggingface`（或文档化并默认使用该路径）。
2. 同步修正 README / 启动说明，使默认离线冒烟无需手工 `HF_HOME=$PWD/.cache/huggingface`（或明确写死推荐环境变量）。
3. 不要改业务阈值、特征缓存格式或重跑全量编码（除非修复迫使）。
4. 验收：上面的 `env -u HF_HOME HF_HUB_OFFLINE=1 ... encoder.py --smoke` 必须成功，且 adapter 激活。
5. 可选：再跑 `briefs/review_checks.py` 中与 S19 相关的检查，确认 PASS。
6. 完成后写 `briefs/dsh-fix-p0-done.md`。

开始修复。
