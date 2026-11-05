"""Summary of predictions_v2.pkl (after retraining with Claude labels) -> summary_v2.md"""
import json
from pathlib import Path

import pandas as pd

out = Path("out")
df = pd.read_pickle(out / "predictions_v2.pkl")
m = json.loads((out / "metrics_v2.json").read_text())
pk = df[df.in_package].copy()
pk["year"] = pd.to_numeric(pk.year, errors="coerce")
pk["period"] = pd.cut(pk.year, [1994, 2005, 2015, 2026], labels=["1995-2005", "2006-2015", "2016-2026"])
n, pos = len(pk), int(pk.final.sum())
old = int(pk.pred.sum())


def t(g):
    r = pk.groupby(g, observed=True).final.agg(n="count", positive="sum")
    r["rate"] = (r.positive / r.n).round(3)
    return r.to_markdown()


nv, ov = m["new_model"], m["old_model_same_papers"]
L = ["# Classification summary (v2: retrained with Claude labels)\n",
     f"Papers classified: **{n}**. Final label 1 (qualifies): **{pos} ({pos / n:.1%})**; label 0: **{n - pos} ({(n - pos) / n:.1%})**. "
     f"(v1 model said {old} positives, {old / n:.1%}.)\n",
     "## Where the final labels come from\n", pk.groupby(["final_src", "final"]).size().unstack(fill_value=0).to_markdown() + "\n",
     f"Model-labelled papers still near the threshold (|p-0.5|<0.15): **{int(pk.uncertain_v2.sum())}**.\n",
     "## Validation (30% of Claude-labelled papers, never seen by the model that scored them)\n",
     f"n={nv['n']} (positives {nv['n_pos']}, negatives {nv['n'] - nv['n_pos']}). AUC={nv['auc']:.3f}, precision={nv['precision']:.3f}, "
     f"recall={nv['recall']:.3f}, F1={nv['f1']:.3f}; confusion TP={nv['tp']} FP={nv['fp']} FN={nv['fn']} TN={nv['tn']}.\n",
     f"Reference: labelling everything 1 gives accuracy {m['majority_baseline_acc']:.3f} on this set, precision {nv['n_pos'] / nv['n']:.3f}, recall 1.0. "
     "With only 7 negatives in the validation set the metrics have wide error bars; they do not show that the head beats 'always 1'.\n",
     f"v1 model on the same papers: AUC={ov['auc']:.3f}, recall={ov['recall']:.3f} (these papers were selected because v1 was unsure, so its AUC here is not meaningful).\n",
     "## By period\n", t("period") + "\n", "## By text source\n", t("source") + "\n",
     "## Methods in label-1 papers\n"]
cnt = pd.Series([x for s in pk[pk.final == 1].methods for x in s.split(";") if x]).value_counts()
L += [cnt.rename("papers").to_markdown() + "\n", "## Change vs v1\n",
      pd.crosstab(pk.pred.rename("v1"), pk.final.rename("v2")).to_markdown() + "\n"]
(out / "summary_v2.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
