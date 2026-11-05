"""Statistical summary of predictions.pkl -> summary.md + summary.png (only papers in the fulltext package)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def md(df):
    return df.to_markdown() if hasattr(df, "to_markdown") else df.to_string()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = Path(a.out)
    df = pd.read_pickle(out / "predictions.pkl")
    m = json.loads((out / "metrics.json").read_text())
    pk = df[df.in_package].copy()
    pk["year"] = pd.to_numeric(pk.year, errors="coerce")
    pk["period"] = pd.cut(pk.year, [1994, 2005, 2015, 2026], labels=["1995-2005", "2006-2015", "2016-2026"])
    n, pos = len(pk), int(pk.pred.sum())
    L = [f"# Classification summary\n",
         f"Papers classified (fulltext package): **{n}**; predicted **1 (qualifies)**: **{pos} ({pos / n:.1%})**; "
         f"predicted 0: **{n - pos} ({(n - pos) / n:.1%})**; uncertain (|p-{m['threshold']}|<0.15): "
         f"**{int(pk.uncertain.sum())}** (need manual check).\n",
         "## Model validation (5-fold CV on the training set)\n",
         f"Training set: {m['n_pos']} curated positives vs {m['n_neg']} rule-confirmed negatives. "
         f"AUC={m['auc']:.3f}, precision={m['precision']:.3f}, recall={m['recall']:.3f}, F1={m['f1']:.3f} "
         f"(C={m['C']}, threshold={m['threshold']}). Negatives are rule-derived, not human-labelled, so precision/AUC are optimistic.\n",
         "## By period\n"]
    t = pk.groupby("period", observed=True).pred.agg(n="count", positive="sum")
    t["rate"] = (t.positive / t.n).round(3)
    L.append(md(t) + "\n")
    L.append("## By text source used\n")
    t = pk.groupby("source").pred.agg(n="count", positive="sum")
    t["rate"] = (t.positive / t.n).round(3)
    L.append(md(t) + "\n")
    L.append("## Detection methods mentioned in predicted-positive papers\n")
    cnt = pd.Series([x for s in pk[pk.pred == 1].methods for x in s.split(";") if x]).value_counts()
    L.append(md(cnt.rename("papers")) + "\n")
    L.append("## Agreement with rule stage (full text)\n")
    L.append(md(pd.crosstab(pk.rule, pk.pred, margins=True)) + "\n")
    L.append("## Probability bands\n")
    bands = pd.cut(pk.prob, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
    L.append(md(bands.value_counts().sort_index().rename("papers")) + "\n")
    L.append("## Lowest-probability papers (most likely negatives)\n")
    L.append(md(pk.nsmallest(15, "prob")[["pmid", "year", "prob", "title"]].assign(title=lambda d: d.title.str[:90])) + "\n")
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(pk.prob, bins=20, color="#3b6ea5")
    ax[0].axvline(m["threshold"], color="r", ls="--")
    ax[0].set(title="P(qualifies)", xlabel="probability", ylabel="papers")
    pk.groupby("period", observed=True).pred.value_counts().unstack(fill_value=0).plot.bar(ax=ax[1], stacked=True, rot=0)
    ax[1].set(title="Predictions by period")
    cnt.plot.barh(ax=ax[2], color="#5a9")
    ax[2].set(title="Methods in positives")
    plt.tight_layout()
    plt.savefig(out / "summary.png", dpi=130)
    print("\n".join(L)[:3000])


if __name__ == "__main__":
    main()
