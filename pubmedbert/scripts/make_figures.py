"""Figures for the classifier slide deck / report. Output: <outdir>/*.png (style follows the lab slide template)."""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casefilter.embed import pool  # noqa: E402

BLUE, RED, GRAY, LBLUE, LRED, GREEN = "#0F4C81", "#C8423B", "#8A8F98", "#8DB4DC", "#E8A39F", "#4C9A6A"
plt.rcParams.update({"font.family": ["Microsoft YaHei", "Arial"], "axes.unicode_minus": False, "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555", "figure.dpi": 100})
out = Path(sys.argv[1] if len(sys.argv) > 1 else "ppt_work/figs")
out.mkdir(parents=True, exist_ok=True)
R = Path("out")


def save(fig, name):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="white", ec=BLUE, tc="black", fs=12, bold=False, lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=BLUE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color=color, lw=1.8))


fl = pd.read_csv(R / "final_labels.csv", dtype={"pmid": str}, encoding="utf-8-sig")
v3 = pd.read_csv(R / "predictions_v3.csv", dtype={"pmid": str}, encoding="utf-8-sig")
met = json.loads((R / "metrics_v3.json").read_text())
val = v3[v3.split == "validation"]
yv, pv = val.label.values.astype(int), val.prob_v3.values

# ---- F1 pipeline overview --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
box(ax, 0.1, 4.6, 2.2, 1.2, "974 篇全文\nxml / txt / pdf", fc="#EAF1F8", bold=True)
box(ax, 3.1, 4.6, 2.4, 1.2, "① 文本提取\n取 xml / txt / pdf\n中最完整的一份", fc="white")
box(ax, 6.3, 4.6, 2.2, 1.2, "② 规则预筛\n(可解释, 离线)", fc="white")
box(ax, 9.0, 4.6, 1.9, 1.2, "候选文献", fc="#EAF1F8", bold=True)
box(ax, 0.1, 2.5, 2.8, 1.3, "③ 切块 + PubMedBERT\n逐块向量化\n(每篇 ≤12 块 × 480 token)", fc="white")
box(ax, 3.7, 2.5, 2.6, 1.3, "④ 池化\n全文均值 ⊕ 首段均值\n(1536 维)", fc="white")
box(ax, 7.1, 2.5, 3.8, 1.3, "⑤ 逻辑回归分类头 → P(入库)\n来源均衡权重, 阈值 0.5", fc=BLUE, ec=BLUE, tc="white", bold=True)
box(ax, 0.1, 0.3, 4.0, 1.3, "⑥ Claude 逐篇判定\n模型拿不准 (0.35~0.65) 的 635 篇\n摘要+方法初判 → 低置信度读全文复核", fc="#FBEDEC", ec=RED)
box(ax, 4.9, 0.3, 3.0, 1.3, "最终标签 0 / 1\n(含来源、置信度、理由)", fc="#EAF1F8", bold=True)
box(ax, 8.6, 0.3, 2.3, 1.3, "人工抽查\n弱标签 91 篇", fc="white", ec=GRAY)
for a in [(2.3, 5.2, 3.1, 5.2), (5.5, 5.2, 6.3, 5.2), (8.5, 5.2, 9.0, 5.2)]: arrow(ax, *a)
arrow(ax, 9.95, 4.6, 1.5, 3.85); arrow(ax, 2.9, 3.15, 3.7, 3.15); arrow(ax, 6.3, 3.15, 7.1, 3.15)
arrow(ax, 9.0, 2.5, 3.0, 1.65, RED); arrow(ax, 4.1, 0.95, 4.9, 0.95); arrow(ax, 7.9, 0.95, 8.6, 0.95)
save(fig, "f1_pipeline")

# ---- F2 admission gates -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
gates = [("人类胎儿 / 妊娠", "fetus · prenatal ·\n妊娠 / 羊水穿刺等"), ("产前超声表型异常", "结构畸形或软指标\n(脑室增宽、NT 增厚…)"),
         ("11 种检测之一", "核型 · CMA · FISH · CNV-seq\nWES · Panel · MLPA · Sanger\nWGS · PCR · mtDNA-seq"),
         ("与该胎儿关联", "同一病例的检测结果\n(阳性或阴性均可)")]
for i, (t, s) in enumerate(gates):
    x = 0.1 + i * 2.5
    box(ax, x, 2.2, 2.2, 1.6, t, fc=BLUE, ec=BLUE, tc="white", bold=True, fs=13)
    ax.text(x + 1.1, 1.65, s, ha="center", va="top", fontsize=10, linespacing=1.45)
    if i: ax.text(x - 0.15, 3.0, "∧", ha="center", va="center", fontsize=20, color=RED, fontweight="bold")
ax.text(5, 4.2, "同时满足 (≥1 例胎儿)  →  文献入库", ha="center", fontsize=14, color=RED, fontweight="bold")
ax.text(5, 0.25, "排除: 综述 / Meta 分析 / 指南 · 动物或细胞研究 · 无 PMID · 仅产后或仅父母 / 母血检测",
        ha="center", fontsize=11, color=GRAY)
save(fig, "f2_gates")

# ---- F3 dataset: text source + rule-stage outcome ---------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
src = fl.assign(year=pd.to_numeric(fl.year, errors="coerce"))
v2 = pd.read_pickle(R / "predictions_v2.pkl")
v2 = v2[v2.in_package]
cnt = v2.source.value_counts().reindex(["xml", "txt", "pdf"])
b = axs[0].bar(["JATS XML", "TXT", "PDF"], cnt.values, color=[BLUE, LBLUE, RED])
axs[0].bar_label(b, fontsize=12); axs[0].set_title("全文来源 (n=974)", fontweight="bold"); axs[0].set_ylabel("篇数")
per = pd.cut(src.year, [1994, 2005, 2015, 2026], labels=["1995-2005", "2006-2015", "2016-2026"]).value_counts().reindex(
    ["1995-2005", "2006-2015", "2016-2026"])
b = axs[1].bar(per.index.astype(str), per.values, color=[LBLUE, LBLUE, BLUE])
axs[1].bar_label(b, fontsize=12); axs[1].set_title("发表年份段", fontweight="bold")
fig.tight_layout(); save(fig, "f3_dataset")

# ---- F4 label composition ---------------------------------------------------------------
names = {"curated_gold": "精选文献", "claude_round1": "Claude 第一轮", "claude_round3": "Claude 第三轮首遍",
         "claude_round3_fullread": "Claude 全文复核", "model": "分类器"}
ct = fl.groupby(["label_source", "label"]).size().unstack(fill_value=0).reindex(list(names))
fig, ax = plt.subplots(figsize=(9.5, 4.6))
y = np.arange(len(ct))
ax.barh(y, ct[1], color=BLUE, label="标签 1 (入库)"); ax.barh(y, ct[0], left=ct[1], color=RED, label="标签 0")
for i, (a_, b_) in enumerate(zip(ct[1], ct[0])):
    ax.text(a_ / 2, i, str(a_), ha="center", va="center", color="white", fontsize=12)
    if b_: ax.text(a_ + b_ / 2, i, str(b_), ha="center", va="center", color="white", fontsize=12)
ax.set_yticks(y, [names[k] for k in ct.index]); ax.invert_yaxis(); ax.set_xlabel("篇数")
ax.legend(frameon=False, loc="lower right"); ax.set_title("最终标签的来源构成 (n=974)", fontweight="bold")
save(fig, "f4_label_sources")

# ---- F5 Claude confidence ---------------------------------------------------------------
cl = fl[fl.label_source.str.startswith("claude")]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
axs[0].hist([cl[cl.label == 1].confidence, cl[cl.label == 0].confidence], bins=np.arange(0.5, 1.01, 0.05), stacked=True,
            color=[BLUE, RED], label=["标签 1", "标签 0"])
axs[0].axvline(0.7, color=GRAY, ls="--"); axs[0].text(0.705, axs[0].get_ylim()[1] * 0.9, "<0.7 → 全文复核", color=GRAY, fontsize=11)
axs[0].set_xlabel("Claude 置信度"); axs[0].set_ylabel("篇数"); axs[0].legend(frameon=False, loc="upper left")
axs[0].set_title("Claude 标注置信度 (n=884)", fontweight="bold")
st = cl.groupby("label_source").label.agg(["count", "sum"]).reindex(["claude_round1", "claude_round3", "claude_round3_fullread"])
b = axs[1].bar(["第一轮\n(249)", "第三轮首遍\n(521)", "全文复核\n(114)"], (st["sum"] / st["count"]).values, color=[LBLUE, BLUE, RED])
axs[1].bar_label(b, labels=[f"{x:.0%}" for x in (st["sum"] / st["count"]).values], fontsize=12)
axs[1].set_ylim(0, 1); axs[1].set_title("各轮判为 1 的比例", fontweight="bold")
fig.tight_layout(); save(fig, "f5_claude")

# ---- F6 source-shortcut leakage (numbers from the first training runs) ------------------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
before = {"精选文献\n(清洗 txt)": 96.2, "PDF": 37.4, "TXT": 22.8, "XML": 1.5}
after = {"PDF": 42.2, "TXT": 42.4, "XML": 29.3}
b = axs[0].bar(list(before), list(before.values()), color=[RED, LBLUE, LBLUE, LBLUE])
axs[0].bar_label(b, fmt="%.1f%%"); axs[0].set_title("修正前: CV AUC = 0.99", fontweight="bold", color=RED)
axs[0].set_ylabel("判为 1 的比例 (%)")
b = axs[1].bar(list(after), list(after.values()), color=BLUE)
axs[1].bar_label(b, fmt="%.1f%%"); axs[1].set_title("修正后: CV AUC = 0.83", fontweight="bold", color=BLUE)
axs[0].set_ylim(0, 110)
fig.suptitle("文本格式捷径: 模型把“来源格式”当成了标签", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout(); save(fig, "f6_leakage")

# ---- F7 AUC vs C ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
Cs, au = [0.1, 1, 10, 100], [0.813, 0.851, 0.847, 0.829]
ax.plot(range(4), au, "o-", color=BLUE, lw=2.5, ms=9)
ax.plot([1], [0.851], "o", color=RED, ms=14, mfc="none", mew=2.5)
for i, a_ in enumerate(au): ax.text(i, a_ + 0.004, f"{a_:.3f}", ha="center", fontsize=12)
ax.set_xticks(range(4), [str(c) for c in Cs]); ax.set_xlabel("正则强度参数 C"); ax.set_ylabel("5 折交叉验证 AUC")
ax.set_ylim(0.79, 0.87); ax.set_title("在训练部分内选择 C (最优 C=1)", fontweight="bold")
save(fig, "f7_cv_C")

# ---- F8 ROC + PR ------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
fpr, tpr, _ = roc_curve(yv, pv)
axs[0].plot(fpr, tpr, color=BLUE, lw=2.8, label=f"分类器  AUC = {roc_auc_score(yv, pv):.3f}")
axs[0].plot([0, 1], [0, 1], "--", color=GRAY, label="随机"); axs[0].plot([1 - 0.8125], [0.8515], "o", color=RED, ms=10, label="阈值 0.5")
axs[0].set_xlabel("1 - 特异度 (假阳性率)"); axs[0].set_ylabel("召回率"); axs[0].legend(frameon=False, loc="lower right")
axs[0].set_title("ROC (验证集 n=266)", fontweight="bold")
pr, rc, _ = precision_recall_curve(yv, pv)
axs[1].plot(rc, pr, color=BLUE, lw=2.8, label="分类器"); axs[1].axhline(yv.mean(), color=GRAY, ls="--", label=f"全判 1 基线 ({yv.mean():.2f})")
axs[1].set_xlabel("召回率"); axs[1].set_ylabel("精确率"); axs[1].set_ylim(0.6, 1.02); axs[1].legend(frameon=False, loc="lower left")
axs[1].set_title("精确率-召回率", fontweight="bold")
fig.tight_layout(); save(fig, "f8_roc_pr")

# ---- F9 threshold sweep -----------------------------------------------------------------
th = np.linspace(0.05, 0.95, 91)
def rates(t):
    p = (pv >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yv, p, labels=[0, 1]).ravel()
    return tp / (tp + fn), tn / (tn + fp), (tp / (tp + fp) if tp + fp else np.nan)
rr = np.array([rates(t) for t in th])
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(th, rr[:, 0], color=BLUE, lw=2.5, label="召回率"); ax.plot(th, rr[:, 1], color=RED, lw=2.5, label="特异度 (识别出 0)")
ax.plot(th, rr[:, 2], color=GREEN, lw=2.5, label="精确率")
ax.axvline(0.5, color=GRAY, ls="--"); ax.text(0.51, 0.05, "默认 0.5", color=GRAY); ax.set_xlabel("判定阈值"); ax.set_ylim(0, 1.03)
ax.set_ylim(0, 1.22); ax.legend(frameon=False, loc="upper center", ncol=3); ax.set_title("阈值的取舍 (验证集)", fontweight="bold")
save(fig, "f9_threshold")

# ---- F10 confusion matrix + metrics vs all-ones baseline --------------------------------
m5, bl = met["val_at_0_5"], met["baseline_all_ones"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
cm = np.array([[m5["tn"], m5["fp"]], [m5["fn"], m5["tp"]]])
axs[0].imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.1)
for (i, j), v in np.ndenumerate(cm): axs[0].text(j, i, str(v), ha="center", va="center", fontsize=22, fontweight="bold",
                                                   color="white" if v > cm.max() * 0.5 else "black")
axs[0].set_xticks([0, 1], ["预测 0", "预测 1"]); axs[0].set_yticks([0, 1], ["真实 0", "真实 1"])
axs[0].set_title("混淆矩阵 (阈值 0.5)", fontweight="bold")
for s in axs[0].spines.values(): s.set_visible(False)
labs = ["精确率", "召回率", "特异度", "F1", "平衡准确率"]
mine = [m5["precision"], m5["recall"], m5["specificity"], m5["f1"], m5["balanced_acc"]]
base = [bl["precision"], bl["recall"], bl["specificity"], bl["f1"], bl["balanced_acc"]]
x = np.arange(5); w = 0.38
b1 = axs[1].bar(x - w / 2, mine, w, color=BLUE, label="分类器"); b2 = axs[1].bar(x + w / 2, base, w, color=LRED, label="全判 1 基线")
axs[1].bar_label(b1, fmt="%.2f", fontsize=10); axs[1].bar_label(b2, fmt="%.2f", fontsize=10)
axs[1].set_xticks(x, labs); axs[1].set_ylim(0, 1.3); axs[1].legend(frameon=False, ncol=2, loc="upper center")
axs[1].set_title("验证指标 vs 基线", fontweight="bold")
fig.tight_layout(); save(fig, "f10_confusion_metrics")

# ---- F11 probability distribution by true label -----------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bins = np.linspace(0, 1, 21)
ax.hist(pv[yv == 1], bins=bins, color=BLUE, alpha=0.85, label=f"Claude 标签 1 (n={int((yv == 1).sum())})")
ax.hist(pv[yv == 0], bins=bins, color=RED, alpha=0.85, label=f"Claude 标签 0 (n={int((yv == 0).sum())})")
ax.axvline(0.5, color=GRAY, ls="--"); ax.set_xlabel("分类器输出 P(入库)"); ax.set_ylabel("篇数")
ax.legend(frameon=False, loc="upper left"); ax.set_title("验证集上的概率分布", fontweight="bold")
save(fig, "f11_prob_hist")

# ---- F12 PCA of document embeddings -----------------------------------------------------
pred = pd.read_pickle(R / "predictions.pkl").reset_index(drop=True)
X = pool(joblib.load(R / "chunk_embeddings.pkl"))[:, :768]
Z = PCA(2, random_state=0).fit_transform(X)
lab = pred.pmid.map(fl.set_index("pmid").label)
fig, ax = plt.subplots(figsize=(7.2, 5.4))
na = lab.isna().values
ax.scatter(Z[na, 0], Z[na, 1], s=14, c="#CCCCCC", label="不在 974 篇内 (精选)")
for v_, c_, n_ in ((0, RED, "最终标签 0"), (1, BLUE, "最终标签 1")):
    mk = (lab == v_).values
    ax.scatter(Z[mk, 0], Z[mk, 1], s=16, c=c_, alpha=0.65, label=f"{n_} (n={int(mk.sum())})")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(frameon=False, loc="best", fontsize=11)
ax.set_title("文献向量的 PCA 投影", fontweight="bold")
save(fig, "f12_pca")

# ---- F13 results by period and by method ------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})
d = fl.assign(year=pd.to_numeric(fl.year, errors="coerce"))
d["period"] = pd.cut(d.year, [1994, 2005, 2015, 2026], labels=["1995-2005", "2006-2015", "2016-2026"])
g = d.groupby("period", observed=True).label.agg(["count", "sum"])
axs[0].bar(g.index.astype(str), g["count"] - g["sum"], color=RED, label="标签 0")
axs[0].bar(g.index.astype(str), g["sum"], bottom=g["count"] - g["sum"], color=BLUE, label="标签 1")
for i, (c_, s_) in enumerate(zip(g["count"], g["sum"])): axs[0].text(i, c_ + 8, f"{s_ / c_:.0%}", ha="center", fontsize=12)
axs[0].legend(frameon=False); axs[0].set_title("各年份段分类结果", fontweight="bold"); axs[0].set_ylabel("篇数")
pos = fl[fl.label == 1]
mc = pd.Series([x_ for s in pos.methods.fillna("") for x_ in s.split(";") if x_]).value_counts()
b = axs[1].barh(mc.index[::-1], mc.values[::-1], color=BLUE); axs[1].bar_label(b, fontsize=10)
axs[1].set_title("入库文献 (标签 1) 中出现的检测方案", fontweight="bold"); axs[1].set_xlabel("篇数")
fig.tight_layout(); save(fig, "f13_results")
print("figures ->", out)
