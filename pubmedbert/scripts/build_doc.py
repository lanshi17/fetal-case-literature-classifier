"""Word report that merges the slide content and every figure (same numbering as the deck)."""
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FIGS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
BLUE = RGBColor(0x0F, 0x4C, 0x81)
doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(1.0)
st = doc.styles["Normal"]
st.font.name = "Arial"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
for name, size in (("Heading 1", 18), ("Heading 2", 14)):
    h = doc.styles[name]; h.font.name = "Arial"; h.font.size = Pt(size); h.font.bold = True; h.font.color.rgb = BLUE
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
fig_no = [0]


def para(text, bold_prefix=None, style=None):
    p = doc.add_paragraph(style=style)
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(4)
    return p


def bullets(items):
    for it in items:
        if isinstance(it, tuple):
            para(it[1], bold_prefix=it[0], style="List Bullet")
        else:
            para(it, style="List Bullet")


def figure(name, caption, width=6.2):
    fig_no[0] += 1
    doc.add_picture(str(FIGS / f"{name}.png"), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(f"图 {fig_no[0]}　{caption}")
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.runs[0].font.size = Pt(9.5); c.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def table(rows, widths, caption=None):
    if caption:
        c = doc.add_paragraph(caption); c.runs[0].font.size = Pt(9.5); c.runs[0].bold = True
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = t.cell(i, j); cell.text = str(v); cell.width = Inches(widths[j])
            for r in cell.paragraphs[0].runs:
                r.font.size = Pt(10); r.bold = i == 0
    doc.add_paragraph()


# ---------------------------------------------------------------- title
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("UltraPhenoCards 入库文献分类器\n设计原理与训练过程"); r.bold = True; r.font.size = Pt(24); r.font.color.rgb = BLUE
s = doc.add_paragraph("汇报人：王喜弟　　2026年11月\n本文档整合了同名汇报 PPT 的全部文字与图表。"); s.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_heading("摘要", 1)
para("本项目要从上万篇文献中筛出“至少能抽出 1 例符合数据库入库要求的胎儿病例”的文献。分类器由四部分组成：规则预筛、"
     "本地 PubMedBERT 向量化加逻辑回归分类头、Claude 逐篇判定、结果合并。974 篇全文最终有 699 篇判为 1（71.8%），275 篇判为 0。"
     "在留出的 266 篇验证文献上，分类头 AUC 为 0.89，F1 为 0.89，高于“全判 1”基线的 0.86，并能识别 81% 的不合格文献。"
     "验证用的标签主要来自 Claude 而非人工，因此这些指标反映的是与 Claude 判断的一致程度。")

# ---------------------------------------------------------------- 1
doc.add_heading("1　任务与入库标准", 1)
bullets([("目标：", "从上万篇文献中筛出至少含 1 例“产前超声异常 + 遗传检测”胎儿病例的文献。"),
         ("入库要求：", "产前超声表型异常，加 11 种遗传检测之一，二者关联到同一胎儿；非综述、非动物研究。"),
         ("依据：", "PubMed 检索策略中的纳入/排除标准，以及数据结构文档（D01、D15）的必填约束。")])
figure("f2_gates", "入库判定的四个条件与排除项")
doc.add_heading("1.1　数据概况", 2)
bullets(["974 篇全文：JATS XML 464 篇、TXT 304 篇、PDF 206 篇；发表于 2016–2026 年的占 873 篇。",
         "只有 90 篇精选文献带有人工正标签，没有人工负标签。",
         "xml/txt 过短（仅有标题或存根）时回退到 PDF，三种格式取最完整的一份。"])
figure("f3_dataset", "全文来源与发表年份分布")

# ---------------------------------------------------------------- 2
doc.add_heading("2　分类器设计", 1)
doc.add_heading("2.1　整体架构", 2)
figure("f1_pipeline", "分类器整体流程：规则预筛 → 向量化与分类头 → Claude 判定 → 最终标签")
bullets(["规则预筛：可解释、离线，给出原因代码和证据句。", "向量化加分类头：输出每篇的入库概率。",
         "Claude 判定：处理分类器拿不准的文献。", "合并为最终标签 0 / 1，并保留来源、置信度和理由。"])
doc.add_heading("2.2　第一阶段：规则预筛", 2)
para("对标题和全文做 5 道门检查，输出 INCLUDE / REVIEW / EXCLUDE。设计偏重召回：拿不准的进 REVIEW，不直接排除。"
     "在全文上规则偏宽松（848 篇 INCLUDE），所以只把 EXCLUDE 当作可靠负样本使用。")
table([["门", "含义"], ["硬排除", "综述 / Meta / 指南 / 社论类型或标题；无人类背景的动物或细胞研究；无 PMID"],
       ["胎儿", "fetus / prenatal / pregnancy 等词出现"], ["表型", "超声模态词，或已知超声异常词（脑室增宽、NT 增厚、先心病等）"],
       ["方法", "命中 11 种检测之一（只命中 PCR / Sanger 属弱证据）"], ["病例 + 关联", "病例报告 / 个体胎儿描述 / 带例数的队列；同一句话同时提到胎儿与检测"]],
      [1.3, 5.2], "表 1　规则预筛的 5 道门")
doc.add_heading("2.3　第二阶段：向量化与分类头", 2)
bullets([("向量化：", "本地 PubMedBERT 句向量；全文切成 480 token 的块，每篇最多 12 块，必保留前 2 块。"),
         ("池化：", "全文块均值与首段（标题、摘要）均值拼接，共 1536 维。"),
         ("分类头：", "L2 逻辑回归，输出 P(入库)，阈值 0.5。"),
         ("来源均衡权重：", "每个（文本来源，标签）组合的总权重相同，防止模型靠文件格式作答。"),
         ("选参：", "C 只在训练部分内做 5 折交叉验证选择，取 C=1。")])
figure("f7_cv_C", "正则强度 C 的选择（训练部分内 5 折 CV AUC）", 4.6)
doc.add_heading("2.4　第三阶段：Claude 逐篇判定", 2)
bullets(["对象：先行 249 篇，加分类器拿不准（0.35~0.65）的 635 篇；标准与入库标准一致，要求引用原文。",
         "两遍流程：首遍读摘要和方法；置信度低于 0.7 的 114 篇再读全文复核，并覆盖首遍结果。",
         "输出：标签、置信度、研究类型、引文、理由，便于人工抽查。"])
figure("f5_claude", "Claude 标注的置信度分布与各轮判为 1 的比例")

# ---------------------------------------------------------------- 3
doc.add_heading("3　训练过程", 1)
doc.add_heading("3.1　训练数据与标签构成", 2)
bullets(["训练集：884 篇 Claude 标签加 90 篇精选文献；规则自动生成的负样本不再使用。",
         "按 70/30 分层划分（标签 × 文本来源）：训练 699 篇，验证 266 篇，其中 64 篇为 0。",
         "最终标签中仍由分类器决定的有 71 篇（概率离阈值较远，其中 63 篇为 0）。"])
figure("f4_label_sources", "最终标签的来源构成（n=974）")
table([["来源", "标签 0", "标签 1"], ["精选文献", 0, 19], ["Claude 第一轮", 25, 224], ["Claude 第三轮首遍", 158, 363],
       ["Claude 全文复核", 29, 85], ["分类器", 63, 8]], [2.4, 1.2, 1.2], "表 2　最终标签来源")
doc.add_heading("3.2　避坑：文本来源格式捷径", 2)
para("第一次训练的交叉验证 AUC 高达 0.99，但此时精选文献用的是清洗过的 txt，其余是 xml 或 pdf，模型学到的是文件格式：xml 文献只有 1.5% 被判为 1。"
     "修正办法是对所有文献用同一流程提取文本，并加入来源均衡权重。修正后 AUC 回落到 0.83，各来源的判为 1 比例回到 29%–42%。")
figure("f6_leakage", "修正前后各文本来源被判为 1 的比例")

# ---------------------------------------------------------------- 4
doc.add_heading("4　结果与讨论", 1)
doc.add_heading("4.1　验证集表现", 2)
para("验证集 266 篇（202 篇标签 1，64 篇标签 0）。训练内 5 折 CV AUC 为 0.85，验证 AUC 为 0.89。")
figure("f8_roc_pr", "验证集 ROC 曲线与精确率-召回率曲线")
figure("f10_confusion_metrics", "混淆矩阵（阈值 0.5）与各指标相对“全判 1”基线的比较")
table([["指标", "分类器", "全判 1 基线"], ["精确率", "0.93", "0.76"], ["召回率", "0.85", "1.00"], ["特异度", "0.81", "0.00"],
       ["F1", "0.89", "0.86"], ["平衡准确率", "0.83", "0.50"], ["AUC", "0.89", "—"], ["PR-AUC（标签 1 / 标签 0）", "0.96 / 0.73", "—"]],
      [2.4, 1.5, 1.5], "表 3　验证指标（阈值 0.5）")
para("验证标签来自 Claude，其中部分只读了摘要和方法，因此这些指标反映与 Claude 判断的一致程度，不是人工金标准。")
doc.add_heading("4.2　阈值与概率分布", 2)
figure("f9_threshold", "阈值对召回率、特异度、精确率的影响", 5.2)
figure("f11_prob_hist", "验证集上分类器输出的概率分布", 5.2)
bullets(["阈值 0.5（默认）：召回 0.85，精确 0.93。", "阈值提高到 0.64：精确 0.97，召回降到 0.72。",
         "要少漏收就保持 0.5 或再调低；要少误收，用 0.64 并配合人工复核。"])
doc.add_heading("4.3　全库分类结果", 2)
figure("f13_results", "各年份段的分类结果与入库文献中出现的检测方案")
bullets(["974 篇中 699 篇为 1（71.8%）、275 篇为 0；2016–2026 年段占 873 篇，入库比例 75%。",
         "入库文献中提到最多的检测：核型 625 篇、CMA 549 篇、WES 385 篇；mtDNA-seq 仅 11 篇。",
         "91 篇为弱标签（Claude 置信度低于 0.6 或仅有分类器判断），建议优先人工抽查。"])
figure("f12_pca", "文献向量的 PCA 投影（标签 1 与 0 在二维上重叠，需要 Claude 判定补充）", 5.0)

doc.add_heading("5　总结", 1)
doc.add_heading("分类器设计", 2)
bullets(["规则预筛、本地 PubMedBERT 向量、逻辑回归分类头和 Claude 逐篇判定，各阶段输出可追溯的原因与证据。",
         "来源均衡权重去掉了文件格式捷径；验证 AUC 0.89、F1 0.89，优于“全判 1”基线（0.86），并能识别 81% 的不合格文献。"])
doc.add_heading("局限与下一步", 2)
bullets(["标签主要来自 Claude 而非人工；全文被截断在 22000 字符；71 篇仅靠分类器判定。",
         "下一步：人工抽查弱标签（91 篇），用抽查结果校准阈值，并扩充人工负样本。"])
doc.add_heading("附：文件与复现", 2)
table([["内容", "位置"], ["最终标签（974 篇）", "literature-classifier/out/final_labels.csv"], ["验证指标", "out/metrics_v3.json"],
       ["逐篇概率", "out/predictions_v3.csv"], ["训练脚本", "scripts/retrain_v3.py"], ["图表脚本", "scripts/make_figures.py"],
       ["PPT 与本文档生成脚本", "scripts/build_deck.py、scripts/build_doc.py"]], [2.4, 4.2])
doc.save(str(OUT))
print("saved", OUT, "figures:", fig_no[0])
