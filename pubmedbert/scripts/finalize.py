"""Merge every label source into out/final_labels.csv (+ summary_final.md).

Priority per paper: round-3 full-read recheck > round-3 first pass > round-1 Claude label (with its recheck)
> curated gold label > v2 model prediction.
"""
import glob
import json
import os
from pathlib import Path

import pandas as pd

out = Path("out")


def load_json(f):
    raw = open(f, "rb").read()
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw.decode("gb18030")
    return json.loads(txt)


def table(files, tag):
    rows = []
    for f in files:
        for r in load_json(f):
            r["pmid"] = str(r["pmid"])
            r["src_file"] = os.path.basename(f)
            rows.append(r)
    d = pd.DataFrame(rows).drop_duplicates("pmid", keep="last")
    d["tag"] = tag
    return d.set_index("pmid")


df = pd.read_pickle(out / "predictions_v2.pkl")
df = df[df.in_package].copy().set_index("pmid")
r1 = pd.read_csv(out / "llm_labels_final.csv", dtype={"pmid": str}, encoding="utf-8-sig").set_index("pmid")
p1 = table(sorted(glob.glob(str(out / "llm_labels3" / "batch_*.json"))), "claude_round3")
p2 = table(sorted(glob.glob(str(out / "llm_labels3" / "p2_*.json"))), "claude_round3_fullread")

df["label"] = df.final
df["label_source"] = df.final_src.map({"claude": "claude_round1", "gold": "curated_gold", "model": "model"})
df["confidence"] = pd.NA
df["reason"] = ""
df["quote"] = ""
df["design"] = ""
df["tests"] = ""


def apply(src, tag):
    for pm, r in src.iterrows():
        if pm not in df.index:
            continue
        df.loc[pm, ["label", "label_source", "confidence", "reason", "quote", "design"]] = [
            int(r["label"]), tag, r.get("confidence"), r.get("reason", ""), r.get("quote", ""), r.get("design", "")]
        df.at[pm, "tests"] = ";".join(r.get("tests") or [])


for pm, r in r1.iterrows():
    if pm in df.index:
        df.loc[pm, ["label", "label_source", "confidence", "reason", "quote", "design"]] = [
            int(r.label_llm), "claude_round1", r.confidence, r.reason, r.quote, r.design]
        df.at[pm, "tests"] = str(r.tests) if pd.notna(r.tests) else ""
apply(p1, "claude_round3")
apply(p2, "claude_round3_fullread")
df["label"] = df.label.astype(int)
df["confidence"] = pd.to_numeric(df.confidence, errors="coerce")
df["weak"] = (df.confidence < 0.6) | (df.label_source == "model")
df = df.reset_index()
cols = ["pmid", "title", "year", "label", "label_source", "confidence", "weak", "design", "tests", "methods", "reason",
        "quote", "prob_v2"]
df[cols].to_csv(out / "final_labels.csv", index=False, encoding="utf-8-sig")

pk = df.copy()
pk["year"] = pd.to_numeric(pk.year, errors="coerce")
pk["period"] = pd.cut(pk.year, [1994, 2005, 2015, 2026], labels=["1995-2005", "2006-2015", "2016-2026"])
n, pos = len(pk), int(pk.label.sum())


def t(g):
    r = pk.groupby(g, observed=True).label.agg(n="count", positive="sum")
    r["rate"] = (r.positive / r.n).round(3)
    return r.to_markdown()


L = [f"# Final classification summary\n",
     f"Papers: **{n}**. Label 1 (qualifies): **{pos} ({pos / n:.1%})**; label 0: **{n - pos} ({(n - pos) / n:.1%})**.\n",
     "## Label source\n", pk.groupby(["label_source", "label"]).size().unstack(fill_value=0).to_markdown() + "\n",
     f"Weak labels (Claude confidence < 0.6, or model-only): **{int(pk.weak.sum())}** "
     f"(of which model-only {int((pk.label_source == 'model').sum())}).\n",
     "## By period\n", t("period") + "\n", "## By text source\n", t("source") + "\n",
     "## Methods mentioned in label-1 papers\n"]
cnt = pd.Series([x for s in pk[pk.label == 1].methods for x in s.split(";") if x]).value_counts()
L += [cnt.rename("papers").to_markdown() + "\n", "## Study design of Claude-labelled papers\n",
      pk[pk.design != ""].groupby(["design", "label"]).size().unstack(fill_value=0).to_markdown() + "\n"]
(out / "summary_final.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
