"""Build the classifier-design slide deck on top of the lab's reference .pptx (template, header, footer, styles)."""
import copy
import re
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

REF, FIGS, OUT = (Path(a) for a in sys.argv[1:4])
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main", "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
A = "{%s}" % NS["a"]
prs = Presentation(str(REF))
orig = list(prs.slides)
T_TITLE, T_TOC, T_CALL, T_BODY, T_SUM, T_THANKS = orig[0], orig[1], orig[8], orig[9], orig[25], orig[26]


def find(shapes, name):
    for s in shapes:
        if s.name == name:
            return s
        if s.shape_type == 6:
            r = find(s.shapes, name)
            if r is not None:
                return r


def clone(tpl, keep=None):
    """New slide on tpl's layout with deep copies of tpl's non-picture, non-placeholder shapes (optionally only `keep` names)."""
    s = prs.slides.add_slide(tpl.slide_layout)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    for sh in tpl.shapes:
        if sh.shape_type in (13, 14) or (keep and sh.name not in keep):
            continue
        el = copy.deepcopy(sh._element)
        for bad in el.xpath(".//p:custDataLst"):  # tags rel (rIdN) would dangle on the new slide
            bad.getparent().remove(bad)
        s.shapes._spTree.append(el)
    return s


def para_xml(text, level=0, size=18, bullet=True, spacing=125, bold=False, color=None):
    """Paragraph XML in the template's body style (Wingdings arrow bullets, YaHei). **x** marks bold runs."""
    p = etree.SubElement(etree.Element("x"), A + "p")
    ppr = etree.SubElement(p, A + "pPr")
    if bullet:
        ppr.set("marL", str(285750 + level * 285750)); ppr.set("indent", "-285750")
    ln = etree.SubElement(etree.SubElement(ppr, A + "lnSpc"), A + "spcPct"); ln.set("val", str(spacing * 1000))
    if bullet:
        bf = etree.SubElement(ppr, A + "buFont")
        bf.set("typeface", "Wingdings" if level == 0 else "Arial")
        if level == 0:
            bf.set("pitchFamily", "2"); bf.set("charset", "2")
        etree.SubElement(ppr, A + "buChar").set("char", "Ø" if level == 0 else "•")
    else:
        etree.SubElement(ppr, A + "buNone")
    for i, part in enumerate(re.split(r"\*\*", text)):
        if not part:
            continue
        r = etree.SubElement(p, A + "r")
        rp = etree.SubElement(r, A + "rPr"); rp.set("lang", "zh-CN"); rp.set("sz", str(int(size * 100 * (0.88 if level else 1))))
        if bold or i % 2:
            rp.set("b", "1")
        if color:
            etree.SubElement(etree.SubElement(rp, A + "solidFill"), A + "srgbClr").set("val", color)
        for tag in ("latin", "ea"):
            etree.SubElement(rp, A + tag).set("typeface", "微软雅黑")
        etree.SubElement(r, A + "t").text = part
    return p


def bullets(slide, x, y, w, items, size=18, spacing=125, h=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h or 1))
    tf = tb.text_frame; tf.word_wrap = True
    body = tf._txBody
    for p in body.findall(A + "p"):
        body.remove(p)
    for it in items:
        if isinstance(it, tuple):
            txt, lvl = it
        else:
            txt, lvl = it, 0
        head = txt.startswith("##")
        body.append(para_xml(txt.lstrip("# "), lvl, size, bullet=not head, spacing=spacing, bold=head))
    return tb


def set_title(slide, text):
    sh = find(slide.shapes, "文本框 1")
    p = sh.text_frame.paragraphs[0]
    p.runs[0].text = text
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)


def pic(slide, name, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(FIGS / f"{name}.png"), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def set_runs(shape, text):
    p = shape.text_frame.paragraphs[0]
    p.runs[0].text = text
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)


def callouts(slide, texts, x=8.92, y0=1.55, step=1.05):
    """Numbered right-hand callouts copied from the template's workflow slide (absolute positions)."""
    for i, t in enumerate(texts):
        top = y0 + i * step
        for nm, dx, dy in (("矩形 17", 0.0, 0.0), ("文本框 19", 0.19, 0.09), ("椭圆 29", -0.2, 0.10)):
            el = copy.deepcopy(find(T_CALL.shapes, nm)._element)
            for bad in el.xpath(".//p:custDataLst"):
                bad.getparent().remove(bad)
            slide.shapes._spTree.append(el)
            sh = slide.shapes[-1]
            sh.left, sh.top = Inches(x + dx), Inches(top + dy)
            if nm == "文本框 19":
                set_runs(sh, t)
            if nm == "椭圆 29":
                set_runs(sh, str(i + 1))


def toc(active):
    s = clone(T_TOC)
    names = ["任务与入库标准", "分类器设计", "训练过程", "结果与总结"]
    rects = [find(s.shapes, n) for n in ("矩形 19", "矩形 26", "矩形 33", "矩形 61")]
    act_rpr = copy.deepcopy(find(T_TOC.shapes, "矩形 19").text_frame.paragraphs[0].runs[0]._r.find(A + "rPr"))
    off_rpr = copy.deepcopy(find(T_TOC.shapes, "矩形 26").text_frame.paragraphs[0].runs[0]._r.find(A + "rPr"))
    for i, (r, n) in enumerate(zip(rects, names)):
        run = r.text_frame.paragraphs[0].runs[0]
        run.text = n
        old = run._r.find(A + "rPr"); run._r.replace(old, copy.deepcopy(act_rpr if i == active else off_rpr))
    return s


def body(title):
    s = clone(T_BODY, keep={"组合 10", "文本框 1"})
    set_title(s, title)
    return s


def table(slide, x, y, w, rows, col_w, size=13, row_h=0.42):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shp.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.text = str(val)
            para = c.text_frame.paragraphs[0]
            for r in para.runs:
                r.font.size = Pt(size); r.font.name = "微软雅黑"; r.font.bold = i == 0
                r.font.color.rgb = RGBColor(255, 255, 255) if i == 0 else RGBColor(0x17, 0x21, 0x2B)
            c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0x0F, 0x4C, 0x81) if i == 0 else (RGBColor(0xEA, 0xF1, 0xF8) if i % 2 else RGBColor(255, 255, 255))
    return tbl


# ---------- slide 1: title (edit the template's title slide in place) ----------
for sh in T_TITLE.shapes:
    if sh.is_placeholder and sh.placeholder_format.idx == 0:
        p0 = sh.text_frame.paragraphs[0]
        for r in p0.runs[1:]:
            r._r.getparent().remove(r._r)
        r0 = p0.runs[0]
        r0.text = "UltraPhenoCards 入库文献分类器"
        r1 = copy.deepcopy(r0._r)
        r1.find(A + "t").text = "设计原理与训练过程"
        r0._r.addnext(r1)
        r0._r.addnext(etree.Element(A + "br"))
    elif sh.is_placeholder:
        ps = sh.text_frame.paragraphs
        ps[0].runs[0].text = "汇报人：王喜弟"
        for r in ps[0].runs[1:]:
            r._r.getparent().remove(r._r)
        if len(ps) > 1:
            ps[1].runs[0].text = "2026年11月"
            for r in ps[1].runs[1:]:
                r._r.getparent().remove(r._r)
    elif sh.has_text_frame and sh.text_frame.text.strip():
        ps = sh.text_frame.paragraphs
        for p_, t_ in zip(ps, ["向量化 + 分类头", "Claude 逐篇判定", "验证 AUC = 0.89"]):
            if p_.runs:
                p_.runs[0].text = t_
                for r in p_.runs[1:]:
                    r._r.getparent().remove(r._r)

# ---------- section 1 ----------
toc(0)
s = body("任务与入库标准")
bullets(s, 0.83, 1.15, 11.6, ["**目标**：从上万篇文献中筛出“至少能抽出 1 例符合入库要求的胎儿病例”的文献",
                              "**入库要求**：产前超声表型异常 + 11 种遗传检测之一 + 二者关联到同一胎儿；非综述、非动物研究"], size=19)
pic(s, "f2_gates", 2.0, 2.95, w=9.2)
s = body("数据概况")
pic(s, "f3_dataset", 0.6, 1.5, w=8.1)
callouts(s, ["974 篇全文 (xml/txt/pdf)", "仅 90 篇精选文献有人工正标签", "没有人工负标签", "xml/txt 过短时回退 PDF"], x=9.0, y0=1.75)

# ---------- section 2: design ----------
toc(1)
s = body("整体架构")
pic(s, "f1_pipeline", 0.6, 1.2, w=7.9)
callouts(s, ["规则预筛：可解释、离线", "向量化 + 分类头：给出概率", "Claude 判定：处理拿不准的", "合并为最终标签 0 / 1"], x=9.2, y0=1.7, step=1.15)
s = body("第一阶段：规则预筛")
bullets(s, 0.83, 1.15, 11.6, ["对标题+全文做 5 道门检查，输出 INCLUDE / REVIEW / EXCLUDE，并附原因代码和证据句",
                              "偏重召回：拿不准的进 REVIEW，不直接排除；全文上规则偏宽松 (848 篇 INCLUDE)，所以只把 EXCLUDE 当可靠负样本"], size=18)
table(s, 0.83, 3.35, 11.6, [["门", "含义"], ["硬排除", "综述 / Meta / 指南 / 社论类型或标题；无人类背景的动物或细胞研究；无 PMID"],
                           ["胎儿", "fetus / prenatal / pregnancy 等词出现"], ["表型", "超声模态词，或已知超声异常词 (脑室增宽、NT 增厚、先心病等)"],
                           ["方法", "命中 11 种检测之一 (只命中 PCR / Sanger 属弱证据)"], ["病例 + 关联", "病例报告 / 个体胎儿描述 / 带例数的队列；同一句话同时提到胎儿与检测"]],
      [2.2, 9.4], size=16, row_h=0.6)
s = body("第二阶段：向量化 + 分类头")
bullets(s, 0.83, 1.25, 6.4, ["**向量化**：本地 PubMedBERT 句向量，全文切成 480 token 的块，每篇最多 12 块 (必保留前 2 块)",
                             "**池化**：全文块均值 ⊕ 首段 (标题/摘要) 均值，共 1536 维",
                             "**分类头**：L2 逻辑回归，输出 P(入库)，阈值 0.5",
                             "**来源均衡权重**：每个 (文本来源, 标签) 组合总权重相同，防止模型靠文件格式作答",
                             "**选参**：C 只在训练部分内做 5 折交叉验证选择，取 C=1"], size=18, spacing=120)
pic(s, "f7_cv_C", 7.5, 1.6, w=5.3)
s = body("第三阶段：Claude 逐篇判定")
bullets(s, 0.83, 1.15, 11.8, ["对象：先行 249 篇 + 分类器拿不准 (0.35~0.65) 的 635 篇，标准与入库标准一致，要求引用原文",
                              "两遍流程：首遍读摘要+方法；置信度 <0.7 的 114 篇再读全文复核并覆盖首遍结果",
                              "输出：标签、置信度、研究类型、引文、理由 (可追溯，便于人工抽查)"], size=18, spacing=115)
pic(s, "f5_claude", 2.4, 3.8, w=8.2)

# ---------- section 3: training ----------
toc(2)
s = body("训练数据与标签构成")
pic(s, "f4_label_sources", 0.5, 1.5, w=7.9)
callouts(s, ["884 篇 Claude + 90 篇精选", "规则负样本不再使用", "70/30 分层划分: 标签×来源", "验证集 266 篇 (其中 64 篇为 0)"], x=9.0, y0=1.6, step=1.1)
s = body("避坑：来源格式捷径")
pic(s, "f6_leakage", 0.6, 1.45, w=8.0)
callouts(s, ["首训 CV AUC=0.99 过高", "精选文献是清洗过的 txt", "XML 仅 1.5% 被判为 1", "同流程提取 + 来源均衡权重"], x=9.0, y0=1.6, step=1.1)

# ---------- section 4: results ----------
toc(3)
s = body("验证结果：ROC 与 PR 曲线")
pic(s, "f8_roc_pr", 0.5, 1.5, w=8.4)
callouts(s, ["验证 AUC = 0.89", "训练内 5 折 CV AUC = 0.85", "PR-AUC：标签 1 为 0.96", "PR-AUC：标签 0 为 0.73"], x=9.2, y0=1.7, step=1.1)
s = body("验证结果：与“全判 1”基线对比")
pic(s, "f10_confusion_metrics", 1.0, 1.3, w=9.3)
bullets(s, 0.83, 5.2, 11.8, ["F1 0.89 vs 基线 0.86；能识别出 81% 的不合格文献 (基线为 0)；精确率 0.93，召回率 0.85",
                             "验证标签来自 Claude，其中部分只读了摘要和方法；指标反映与 Claude 判断的一致程度，非人工金标准"], size=17, spacing=115)
s = body("阈值与概率分布")
pic(s, "f9_threshold", 0.5, 1.4, w=6.0)
pic(s, "f11_prob_hist", 6.8, 1.4, w=6.0)
bullets(s, 0.83, 5.85, 11.8, ["阈值 0.5 (默认)：召回 0.85 / 精确 0.93；提高到 0.64：精确 0.97 / 召回降到 0.72",
                             "要少漏收就保持 0.5 或再调低；要少误收，用 0.64 并配合人工复核"], size=17, spacing=115)
s = body("全库分类结果")
pic(s, "f13_results", 0.5, 1.4, w=9.6)
bullets(s, 0.83, 5.2, 11.8, ["974 篇中 699 篇为 1 (71.8%)、275 篇为 0；2016–2026 年段占 873 篇，入库比例 75%",
                             "入库文献中提到最多的检测：核型 625 篇、CMA 549 篇、WES 385 篇；mtDNA-seq 仅 11 篇",
                             "91 篇为弱标签 (Claude 置信度 <0.6 或仅有分类器判断)，建议优先人工抽查"], size=17, spacing=115)
s = body("文献向量的分布")
pic(s, "f12_pca", 0.6, 1.3, w=6.9)
callouts(s, ["标签 1 与 0 在二维投影上重叠", "单靠整体语义难以完全分开", "所以需要 Claude 判定补充", "投影只用于直观查看"], x=8.6, y0=1.7, step=1.1)

# ---------- summary ----------
s = body("总结")
bullets(s, 0.65, 1.2, 12.0, ["##分类器设计",
                             "规则预筛 + 本地 PubMedBERT 向量 + 逻辑回归分类头 + Claude 逐篇判定，各阶段输出可追溯的原因与证据",
                             "**来源均衡权重**去掉了文件格式捷径，验证 AUC 0.89，F1 0.89，优于“全判 1”基线 (0.86)，并能识别 81% 的不合格文献",
                             "##局限与下一步",
                             "标签主要来自 Claude 而非人工；全文被截断在 22000 字符；71 篇仅靠分类器判定",
                             "下一步：人工抽查弱标签 (91 篇)，用抽查结果校准阈值，并扩充人工负样本"], size=21, spacing=140)

# ---------- thank-you slide: keep the template's, move to the end; delete the reference deck's other slides ----------
ids = prs.slides._sldIdLst
for sld in orig:
    if sld is T_TITLE or sld is T_THANKS:
        continue
    for sid in list(ids):
        if prs.part.related_part(sid.rId) is sld.part:
            prs.part.drop_rel(sid.rId); ids.remove(sid)
thanks = [sid for sid in ids if prs.part.related_part(sid.rId) is T_THANKS.part][0]
ids.remove(thanks); ids.append(thanks)
prs.save(str(OUT))
print("slides:", len(prs.slides), "->", OUT)
