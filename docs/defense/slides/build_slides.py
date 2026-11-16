"""用 docs/defense/slides/模板.pptx 生成答辩 PPT：docs/defense/slides/B03-答辩.pptx（约 10 分钟，每页带演讲备注）。

运行环境：独立的 conda 环境 slides（python-pptx），不需要 torch：
    conda env create -f docs/defense/slides/environment.yml
    conda run -n slides python docs/defense/slides/build_slides.py

做法：保留模板的封面、目录、节标题、封底四页并替换文字；其余内容页都用模板的"内容页5"版式新建，
图表用 PowerPoint 原生图表（可在 PowerPoint 里改数据），实验曲线用 docs/figures/ 里的 png。
所有数字来自 docs/report/ 下的 experiments.md、data.md、model.md。
"""

from __future__ import annotations

import math
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "docs/defense/slides/模板.pptx"
OUT = ROOT / "docs/defense/slides/B03-答辩.pptx"
FIG = ROOT / "docs/figures"

# 模板色彩规范：主色深蓝、平衡色金色、对比色交大红
NAVY = RGBColor(0x06, 0x3A, 0x65)
BLUE = RGBColor(0x1F, 0x4D, 0x78)
GOLD = RGBColor(0xBD, 0x9F, 0x68)
RED = RGBColor(0xC8, 0x16, 0x1E)
TINT = RGBColor(0xF4, 0xEF, 0xE6)  # 金色的浅色版，做卡片底色
NAVY_TINT = RGBColor(0xE8, 0xEE, 0xF4)
TEXT = RGBColor(0x26, 0x26, 0x26)
GRAY = RGBColor(0x59, 0x59, 0x59)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "微软雅黑"  # 模板字体规范：中文微软雅黑，英文 Arial
MODEL_COLORS = [GOLD, BLUE, NAVY]  # s10m / s50m / s100m，全篇一致


# ---------------------------------------------------------------- 文字工具


def style_run(run, size: float, color: RGBColor = TEXT, bold: bool = False,
              latin: str = FONT) -> None:
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    f.name = latin
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):  # 中文字体要单独设 ea，否则中文会落到主题字体
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
        el.set("typeface", FONT)


def fill_tf(tf, paras: list, size: float, color: RGBColor = TEXT, bold: bool = False,
            align=PP_ALIGN.LEFT, space_after: float = 4, line: float | None = None) -> None:
    """paras：每项是 str，或 [(文字, {size/color/bold}), ...] 的若干段 run。"""
    tf.clear()
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        if line:
            p.line_spacing = line
        runs = [(para, {})] if isinstance(para, str) else para
        for text, opt in runs:
            r = p.add_run()
            r.text = text
            style_run(r, opt.get("size", size), opt.get("color", color), opt.get("bold", bold),
                      opt.get("latin", FONT))


def text(slide, x, y, w, h, paras, size=16, color=TEXT, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name="文本", margin=0.0, **kw):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    fill_tf(tf, paras if isinstance(paras, list) else [paras], size, color, bold, align, **kw)
    return box


def shape(slide, kind, x, y, w, h, fill=TINT, line=None, name="形状", radius=0.08):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.name = name
    s.shadow.inherit = False  # 去掉主题默认阴影
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def card(slide, x, y, w, h, paras, fill=TINT, size=14, color=TEXT, name="卡片", pad=0.18,
         anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, **kw):
    """圆角卡片 + 内部文字（文字直接写在形状里，随形状一起移动）。"""
    s = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill, name=name)
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(pad))
    fill_tf(tf, paras, size, color, align=align, **kw)
    return s


def arrow(slide, x1, y1, x2, y2, color=GOLD, width=2.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                   Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def set_title(slide, title: str) -> None:
    """内容页标题用版式自带的标题占位符，字号、位置全篇一致。"""
    ph = slide.shapes.title
    fill_tf(ph.text_frame, [title], 28, NAVY, bold=True)


def lead(slide, s: str) -> None:
    """标题下的一句话结论（每页一个核心信息）。右上角有模板的引号装饰，宽度到 10.6 英寸为止。"""
    text(slide, 1.35, 1.32, 9.6, 0.5, [s], size=17, color=GRAY, name="核心信息")


def notes(slide, s: str) -> None:
    slide.notes_slide.notes_text_frame.text = s


def replace_runs(shape_, new_text: str) -> None:
    """保留第一段第一个 run 的格式，替换整个文本框的文字（模板页用）。"""
    tf = shape_.text_frame
    p0 = tf.paragraphs[0]
    runs = p0.runs
    first = runs[0]
    first.text = new_text
    for r in runs[1:]:
        p0._p.remove(r._r)
    for p in tf.paragraphs[1:]:
        tf._txBody.remove(p._p)


# ---------------------------------------------------------------- 图表工具


def style_chart(chart, legend: bool = True, size: float = 12) -> None:
    chart.font.size = Pt(size)
    chart.font.name = FONT
    chart.font.color.rgb = GRAY
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xE3, 0xE3, 0xE3)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size)
    ca = chart.category_axis
    ca.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    ca.tick_labels.font.size = Pt(size)


def color_series(chart, colors) -> None:
    for s, c in zip(chart.plots[0].series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = c


def bar_labels(chart, fmt: str = "0.00", size: float = 11) -> None:
    plot = chart.plots[0]
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(size)
    dl.font.color.rgb = TEXT


# ---------------------------------------------------------------- 模板页


def edit_cover(slide) -> None:
    by_name = {s.name: s for s in slide.shapes}
    replace_runs(by_name["Titre 11"], "从零训练迷你语言模型")
    replace_runs(by_name["Espace réservé du texte 12"],
                 "手写 Transformer · 训练并对比 0.01B / 0.05B / 0.1B")
    replace_runs(by_name["文本框 19"], "汇报人：（组员姓名）")
    replace_runs(by_name["文本框 20"], "2026 年 11 月")


def edit_toc(slide) -> None:
    items = [("任务与数据", "next-token prediction · BPE 词元化"),
             ("模型与训练", "手写 GPT · 训练设置"),
             ("对比实验", "模型规模 · 采样参数 · 分词方式"),
             ("现场演示", "Gradio 交互界面")]
    groups = [s for s in slide.shapes if s.shape_type == 6]  # GROUP
    for g, (head, sub) in zip(groups, items):
        boxes = list(g.shapes)
        replace_runs(boxes[0], head)
        replace_runs(boxes[1], sub)
        boxes[1].width = Inches(4.2)  # 副标题比模板的 XXX 长，加宽避免换行
        boxes[0].width = Inches(4.2)


def edit_demo_divider(slide) -> None:
    by_name = {s.name: s for s in slide.shapes}
    replace_runs(by_name["Titre 11"], "现场演示")
    body = by_name["Espace réservé du texte 14"]
    tf = body.text_frame
    fill_tf(tf, [[("① ", {"color": GOLD, "bold": True}), ("三模型对比：同一 prompt，三个规模并排", {})],
                 [("② ", {"color": GOLD, "bold": True}), ("调 temperature：0 → 0.8 → 1.5", {})],
                 [("③ ", {"color": GOLD, "bold": True}), ("训练过程：loss 曲线", {})]],
            20, NAVY, align=PP_ALIGN.LEFT, space_after=12)
    for p in tf.paragraphs:  # 模板这一框是全大写 + 渐变色的大字，换成普通深蓝正文
        for r in p.runs:
            rpr = r._r.get_or_add_rPr()
            rpr.set("cap", "none")
    body.top = Inches(3.0)
    replace_runs(by_name["文本框 13"], "04")


# ---------------------------------------------------------------- 内容页


def slide_task(s) -> None:
    set_title(s, "任务：让模型学会预测下一个词")
    lead(s, "给定前文，输出下一个 token 的概率分布；训练目标是让真实的下一个 token 概率尽量大")
    # 左：x / y 错位一格的示意
    words_x = ["Once", " upon", " a", " time"]
    words_y = [" upon", " a", " time", ","]
    x0, w, gap = 0.8, 1.2, 0.15
    text(s, x0, 2.25, 5.8, 0.4, ["输入 x（模型看到的）"], size=14, color=GRAY)
    text(s, x0, 4.05, 5.8, 0.4, ["目标 y（每个位置要预测的下一个 token）"], size=14, color=GRAY)
    for i, (a, b) in enumerate(zip(words_x, words_y)):
        xi = x0 + i * (w + gap)
        card(s, xi, 2.7, w, 0.62, [a], fill=NAVY, color=WHITE, size=16, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, pad=0.05, name=f"x{i}")
        arrow(s, xi + w / 2, 3.38, xi + w / 2, 3.98)
        card(s, xi, 4.48, w, 0.62, [b], fill=TINT, color=NAVY, size=16, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, pad=0.05, name=f"y{i}")
    text(s, x0, 5.35, 5.6, 1.2,
         ["y 就是 x 右移一位：一段 512 个 token 的文本同时提供 512 道"
          "“猜下一个词”的题。损失 = 正确 token 的 −log p 的平均（cross entropy）。"],
         size=14, color=TEXT, line=1.15)
    # 右：三个规模
    rows = [("模型", "层 / 维 / 头", "总参数", "训练 token"),
            ("s10m", "6 / 384 / 6", "13.99M", "3 亿"),
            ("s50m", "10 / 640 / 10", "54.81M", "10 亿"),
            ("s100m", "14 / 768 / 12", "105.92M", "15 亿")]
    table(s, 6.6, 2.3, 5.1, rows, col_w=[1.0, 1.5, 1.25, 1.35])
    text(s, 6.6, 4.55, 5.1, 1.9,
         [[("课程建议约 0.01B；", {"bold": True, "color": NAVY}),
           ("我们做了 0.01B / 0.05B / 0.1B 三个规模，同一份数据、同一套设置，比较规模带来的差别。", {})],
          [("token 预算约 14–21 token/参数", {"bold": True, "color": NAVY}),
           ("（接近 Chinchilla 的计算最优比例）。", {})]],
         size=14, line=1.15, space_after=8)
    notes(s, "【约 50 秒】我们的任务是 next-token prediction：给前文，预测下一个 token。"
             "左边这个例子：输入 Once upon a time，每个位置的目标就是它后面那个 token，所以 y 是 x 右移一位，"
             "一段文本可以同时训练很多个位置。损失函数是 cross entropy，也就是正确 token 概率的负对数。"
             "课程建议 0.01B，我们在此基础上训练了三个规模，用同一份数据和同一套训练设置，"
             "这样就能比较规模本身带来的差别。")


def table(slide, x, y, w, rows, col_w, size=14, row_h=0.46):
    gf = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                Inches(row_h * len(rows)))
    gf.name = "表格"
    tbl = gf.table
    # 去掉默认表格样式的条纹，自己上色
    tbl_pr = gf._element.graphic.graphicData.tbl.tblPr
    tbl_pr.set("bandRow", "0")
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (TINT if i % 2 else WHITE)
            c.margin_left = c.margin_right = Inches(0.08)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill_tf(c.text_frame, [val], size, WHITE if i == 0 else TEXT, bold=(i == 0),
                    align=PP_ALIGN.CENTER, space_after=0)
    return gf


def slide_data(s) -> None:
    set_title(s, "数据：16.3 亿 token 的混合英文语料")
    lead(s, "难的网页教科书 + 简单的儿童故事，分来源统计 val loss，看规模在难/易文本上的收益")
    stats = [("16.3 亿", "训练 token"), ("70 : 30", "SmolLM : TinyStories"),
             ("8192", "自训练 BPE 词表"), ("3.9", "平均字符 / token")]
    for i, (big, small) in enumerate(stats):
        x = 0.95 + i * 2.75
        text(s, x, 2.2, 2.7, 0.8, [big], size=36, color=NAVY, bold=True, name=f"数字{i}")
        text(s, x, 3.0, 2.7, 0.4, [small], size=13, color=GRAY, name=f"说明{i}")
    card(s, 0.95, 3.65, 5.25, 2.05, [
        [("SmolLM-Corpus 1B 子集", {"bold": True, "color": NAVY, "size": 17})],
        "FineWeb-Edu 教育网页 87% + Cosmopedia 合成教科书 13%",
        "没有官方验证集：按文档内容哈希切出 0.5%，同一篇文章不会两边都有",
        [("许可 ODC-By 1.0（需署名）", {"color": GRAY, "size": 12})],
    ], size=14, space_after=6, name="SmolLM 卡片")
    card(s, 6.45, 3.65, 5.25, 2.05, [
        [("TinyStories", {"bold": True, "color": NAVY, "size": 17})],
        "只用 3–4 岁儿童词汇的短故事（GPT-3.5/4 生成）",
        "约 6% 的故事引号乱码（’ → â€™）：先修复再过滤，丢弃从 3.2 万篇降到 406 篇",
        [("许可 CDLA-Sharing-1.0", {"color": GRAY, "size": 12})],
    ], size=14, space_after=6, name="TinyStories 卡片")
    notes(s, "【约 1 分钟】数据有两个来源。SmolLM 子集是教育类网页和合成教科书，词汇、知识面都很广；"
             "TinyStories 是只用儿童词汇的短故事，结构简单。合在一起 16.3 亿 token，按自然比例 70:30 混合。"
             "清洗时我们发现 TinyStories 有 6% 的故事引号是乱码，直接丢掉会损失 3 万多篇，所以先修复再过滤，"
             "最后只丢了 406 篇。SmolLM 没有验证集，我们按文档内容的哈希切 0.5%，保证同一篇文章不会一半在训练集一半在验证集。"
             "词表是自己训练的 BPE，8192 个，平均一个 token 3.9 个字符；词表小，embedding 参数就少。")


def slide_model(s) -> None:
    set_title(s, "模型：手写 GPT（Decoder-only Transformer）")
    lead(s, "只用 nn.Linear / Embedding / LayerNorm 和 SDPA，从零写出 src/model.py")
    # 左：整体结构竖排流程
    steps = [("token id", "(B, T)", NAVY_TINT, NAVY),
             ("wte + wpe（词向量 + 位置编码）", "(B, T, C)", NAVY_TINT, NAVY),
             ("Block × L（注意力 + MLP，Pre-LN 残差）", "(B, T, C)", NAVY, WHITE),
             ("LayerNorm → lm_head（与 wte 共享）", "(B, T, V)", NAVY_TINT, NAVY),
             ("cross entropy（训练）/ 抽样（生成）", "", TINT, NAVY)]
    y = 2.2
    for i, (a, b, fill, col) in enumerate(steps):
        card(s, 0.95, y, 5.4, 0.62,
             [[(a, {"color": col, "bold": i == 2}), (("   " + b) if b else "", {"color": GOLD if fill == NAVY else GRAY, "size": 13})]],
             fill=fill, size=14, anchor=MSO_ANCHOR.MIDDLE, pad=0.15, name=f"结构{i}")
        if i < len(steps) - 1:
            arrow(s, 3.65, y + 0.64, 3.65, y + 0.84)
        y += 0.86
    # 右：注意力内部张量形状
    text(s, 6.85, 2.2, 4.85, 0.4, [[("因果自注意力的张量形状", {"bold": True, "color": NAVY})]], size=16)
    shapes_ = [("(B, T, C)", "c_attn 一次算出 q、k、v"),
               ("(B, nh, T, hs)", "拆成 nh 个头，每头 hs = 64 维"),
               ("(B, nh, T, T)", "q·kᵀ / √hs，因果掩码，softmax"),
               ("(B, nh, T, hs)", "加权求和 v"),
               ("(B, T, C)", "合并多头 → c_proj")]
    for i, (shp, desc) in enumerate(shapes_):
        yy = 2.7 + i * 0.6
        card(s, 6.85, yy, 1.75, 0.46, [shp], fill=TINT, color=NAVY, size=13,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, pad=0.04, name=f"形状{i}",
             )
        text(s, 8.75, yy, 2.95, 0.46, [desc], size=13, color=TEXT, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 6.85, 5.8, 4.85, 0.8,
         [[("每层 12C² + 13C 个参数；", {"bold": True, "color": NAVY}),
           ("s100m：7.09M × 14 层 + embedding 6.68M = 105.92M", {})]],
         size=13, line=1.1)
    notes(s, "【约 1 分 10 秒】模型是手写的 GPT。token id 先查词向量表、加上位置编码，"
             "然后过 L 个 Block，每个 Block 是注意力和 MLP，前面各有一个 LayerNorm，外面套残差连接。"
             "最后的输出层和输入的词向量表共享同一个矩阵，省掉 V×C 个参数。"
             "右边是注意力里的形状变化：先一次算出 q、k、v，拆成多个头；q 乘 k 的转置得到 T×T 的打分，除以根号 hs "
             "防止 softmax 太尖；因果掩码让每个位置只能看前面，否则训练时就是在偷看答案；再对 v 加权求和、合并多头。"
             "每层参数约 12C²：注意力 4C²，MLP 8C²。")


def slide_train(s) -> None:
    set_title(s, "训练：三个模型，同一套设置")
    lead(s, "只有结构、学习率和训练步数不同；其余设置完全一致，对比才公平")
    # 左：学习率调度（原生折线图）。横轴用训练进度，三条曲线等长，方便比较峰值和 warmup 占比
    cd = CategoryChartData()
    cd.categories = [f"{p}%" for p in range(0, 101, 5)]
    for name, lr, iters in [("s10m", 1e-3, 2290), ("s50m", 6e-4, 7630), ("s100m", 4e-4, 11450)]:
        cd.add_series(name, [get_lr(round(p / 100 * iters), lr, lr / 10, 200, iters) * 1e4
                             for p in range(0, 101, 5)])
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(0.8), Inches(2.05),
                            Inches(5.9), Inches(4.5), cd)
    gf.name = "学习率图"
    ch = gf.chart
    style_chart(ch)
    ch.has_title = True
    ch.chart_title.text_frame.text = "学习率（×10⁻⁴）vs 训练进度"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = GRAY
    for ser, c in zip(ch.plots[0].series, MODEL_COLORS):
        ser.format.line.color.rgb = c
        ser.format.line.width = Pt(2.5)
        ser.smooth = False
    ch.value_axis.minimum_scale = 0
    ch.value_axis.maximum_scale = 10
    # 横轴每 4 个类别（20%）标一次
    cat_ax = ch.category_axis._element
    skip = etree.SubElement(cat_ax, qn("c:tickLblSkip"))
    skip.set("val", "4")
    nm = cat_ax.find(qn("c:noMultiLvlLbl"))
    if nm is not None:
        nm.addprevious(skip)
    # 右：设置要点
    rows = [("AdamW", "β = (0.9, 0.95)，wd 0.1"),
            ("warmup+cosine", "200 步升温，余弦降到 1/10"),
            ("梯度累积", "有效 batch 13.1 万 token/步"),
            ("bf16 + 裁剪", "无需 GradScaler；裁剪 1.0"),
            ("可复现", "seed 1337，全程留档")]
    for i, (k, v) in enumerate(rows):
        yy = 2.15 + i * 0.66
        text(s, 7.0, yy, 1.95, 0.6, [[(k, {"bold": True, "color": NAVY})]], size=14)
        text(s, 8.95, yy, 2.75, 0.6, [v], size=13, color=TEXT, line=1.05)
    card(s, 7.0, 5.55, 4.7, 0.95,
         [[("RTX 5090：", {"bold": True, "color": NAVY}),
           ("4.6 min / 45 min / 116 min", {})],
          [("显存峰值 9.2 / 10.4 / 16.5 GB，合计约 2.8 小时", {"color": GRAY, "size": 12})]],
         size=14, anchor=MSO_ANCHOR.MIDDLE, space_after=2, name="资源卡片")
    notes(s, "【约 50 秒】三个模型除了结构、学习率和训练步数，其余设置完全一样。"
             "左边是学习率曲线（横轴是训练进度）：先 warmup 200 步线性升温，因为一开始参数是随机的、梯度不可靠；然后按余弦降到峰值的十分之一。"
             "模型越大学习率越小，越大的模型对步长越敏感。"
             "有效 batch 是每步 13 万 token，本地 8GB 显卡放不下就用梯度累积凑。用 bf16 混合精度，它和 fp32 的数值范围一样，所以不需要 GradScaler。"
             "正式训练在云端 5090 上，三个模型合计约 2.8 小时。")


def get_lr(it: int, lr: float, min_lr: float, warmup: int, max_iters: int) -> float:
    """与 src/utils.py:get_lr 相同（这里复制一份，避免 slides 环境依赖 torch）。"""
    if it < warmup:
        return lr * (it + 1) / warmup
    if it >= max_iters:
        return min_lr
    progress = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (lr - min_lr)


def slide_scale_curves(s) -> None:
    set_title(s, "实验 A：规模越大 loss 越低，但收益递减")
    lead(s, "s10m → s50m 降 0.555，s50m → s100m 只降 0.134")
    s.shapes.add_picture(str(FIG / "val_overlay_tokens.png"), Inches(0.8), Inches(2.0),
                         height=Inches(3.65)).name = "val loss 叠图"
    s.shapes.add_picture(str(FIG / "scaling.png"), Inches(6.75), Inches(2.0),
                         height=Inches(3.65)).name = "scaling 图"
    stats = [("3.098 → 2.543 → 2.409", "最终 val loss（加权）"),
             ("3.082 / 2.843 / 2.810", "都训练到 3 亿 token 时"),
             ("±0.08", "train − val，没有过拟合")]
    for i, (big, small) in enumerate(stats):
        x = 0.95 + i * 4.0
        text(s, x, 5.8, 3.8, 0.45, [big], size=20, color=NAVY, bold=True)
        text(s, x, 6.25, 3.8, 0.35, [small], size=12, color=GRAY)
    notes(s, "【约 1 分钟】左图是三个模型的验证集 loss，横轴是已训练 token 数。三条线都一路下降，没有过拟合。"
             "最终 loss 从 3.10 到 2.54 到 2.41：从 0.01B 到 0.05B 降了 0.55，再到 0.1B 只降了 0.13，收益递减。"
             "右图是 scaling：参数量和 loss 的关系，越往右越平，和 scaling law 的幂律形状一致。"
             "有人会问三个模型训练的 token 数不一样是否公平，所以我们把三个模型都在 3 亿 token 处的 checkpoint "
             "在同一批验证数据上重新评估：大模型仍然更低，而那时大模型的学习率还没降下来，这个比较对它们是不利的。")


def slide_scale_human(s) -> None:
    set_title(s, "实验 A：0.01B 能写故事，写不了说明文")
    lead(s, "五人盲评（1–5 分）：故事在 0.05B 到顶，说明文到 0.1B 还在进步")
    cd = CategoryChartData()
    cd.categories = ["故事·语法", "故事·连贯", "说明文·语法", "说明文·连贯", "说明文·事实"]
    cd.add_series("s10m", (3.36, 2.20, 2.24, 1.40, 1.44))
    cd.add_series("s50m", (4.28, 3.32, 3.04, 2.04, 1.84))
    cd.add_series("s100m", (4.32, 3.04, 3.56, 2.28, 2.80))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8), Inches(2.0),
                            Inches(6.3), Inches(4.6), cd)
    gf.name = "盲评图"
    ch = gf.chart
    style_chart(ch)
    color_series(ch, MODEL_COLORS)
    bar_labels(ch, "0.0", size=10)
    ch.value_axis.maximum_scale = 5
    ch.value_axis.minimum_scale = 0
    ch.plots[0].gap_width = 60
    text(s, 7.4, 2.1, 4.3, 0.4,
         [[("“The French Revolution began …”", {"bold": True, "color": NAVY, "latin": "Arial"})]], size=15)
    card(s, 7.4, 2.6, 4.3, 1.3, [
        [("s10m  ", {"bold": True, "color": GOLD}),
         ("in the aftermath of the Revolution, and the Roman Empire was the first settlement of the Roman Empire…", {"latin": "Arial"})]],
        size=13, line=1.1, name="样例 s10m")
    card(s, 7.4, 4.1, 4.3, 1.3, [
        [("s100m  ", {"bold": True, "color": NAVY}),
         ("in the aftermath of the Revolutionary War, and on 17 September 1789, the French Revolution was declared…", {"latin": "Arial"})]],
        fill=NAVY_TINT, size=13, line=1.1, name="样例 s100m")
    text(s, 7.4, 5.6, 4.3, 0.45, ["大模型更像样，但事实仍经常是编的"], size=13, color=GRAY)
    notes(s, "【约 50 秒】loss 之外我们还做了人工盲评：30 条生成样本打乱顺序、隐藏模型名，五位打分人独立打分。"
             "故事类从 0.01B 到 0.05B 提升很明显，但 0.05B 和 0.1B 几乎一样——写简单故事 0.05B 就够了，"
             "这和 TinyStories 上 loss 只差 0.1 一致。说明文一直到 0.1B 都在进步，事实性在 0.05B 到 0.1B 这一步提升最大。"
             "右边是例子：小模型反复说 Roman Empire，大模型出现了 1789 这样的正确年份，但后面的内容还是编的。"
             "（如果被问：有一位打分人把“重复”一项打反了，和自动指标负相关，我们按 6 减分数换算；另一位整体打分偏松，但去掉这一位不影响排序。文档里如实记录。）")


def slide_sampling(s) -> None:
    set_title(s, "实验 B：temperature 控制“保守”与“胡说”")
    lead(s, "低温复读，高温乱码，top-k 能兜底；推荐 T = 0.7–0.8 + top-k 50")
    cd = CategoryChartData()
    cd.categories = ["greedy", "T=0.3", "T=0.7", "T=1.0", "T=1.5"]
    cd.add_series("不限 top-k", (0.529, 0.510, 0.757, 0.961, 1.000))
    cd.add_series("top-k = 50", (0.529, 0.510, 0.705, 0.940, 0.984))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8), Inches(2.0),
                            Inches(5.9), Inches(4.6), cd)
    gf.name = "distinct-2 图"
    ch = gf.chart
    style_chart(ch)
    color_series(ch, [GOLD, NAVY])
    bar_labels(ch, "0.00", size=10)
    ch.value_axis.maximum_scale = 1.15
    ch.value_axis.minimum_scale = 0
    ch.value_axis.has_major_gridlines = True
    ch.plots[0].gap_width = 70
    ch.has_title = True
    ch.chart_title.text_frame.text = "distinct-2（越高越不重复）"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = GRAY
    examples = [("T = 0", "复读", "…What is the difference between photosynthesis and photosynthesis? Photosynthesis is the process of…", TINT),
                ("T = 0.8", "正常", "…converts light energy (solar) into chemical energy… with carbon dioxide (CO2) and water…", NAVY_TINT),
                ("T = 1.5", "乱码", "…perform itself ASE Washington IT Dangerers Just bood for these sorts…", TINT)]
    text(s, 7.0, 2.05, 4.7, 0.4, [[("“Photosynthesis is the process …”", {"bold": True, "color": NAVY, "latin": "Arial"})]], size=15)
    for i, (t, tag, ex, fill) in enumerate(examples):
        card(s, 7.0, 2.6 + i * 1.33, 4.7, 1.18,
             [[(f"{t}  ", {"bold": True, "color": NAVY}), (tag, {"bold": True, "color": RED if tag != "正常" else GOLD})],
              [(ex, {"latin": "Arial", "size": 12})]],
             fill=fill, size=14, space_after=2, line=1.05, name=f"采样样例{i}", pad=0.14)
    notes(s, "【约 50 秒】采样参数用 0.1B 模型比较。distinct-2 衡量重复程度，越高越不重复。"
             "greedy 和低温只有 0.5 左右，整句复读：每步都选最可能的词，一旦进入循环，循环里每一步都是最可能的，就出不来。"
             "温度 1.5 时 distinct-2 接近 1，但文本已经是乱码，所以这个指标高不代表好，要结合人工阅读。"
             "temperature 的作用是 softmax 前把 logits 除以 T：T 大分布变平，尾部不可能的词也会被抽到。"
             "加上 top-k=50 只在前 50 个词里抽，高温下拼写就正常了。我们推荐 0.7 到 0.8 加 top-k 50，也是界面的默认值。")


def slide_tokenizer(s) -> None:
    set_title(s, "实验 C：比较不同分词器要看 bits-per-byte")
    lead(s, "字符级的 loss 看起来低得多，换算成每字节的信息量后，BPE 反而好 17%–36%")
    cd = CategoryChartData()
    cd.categories = ["SmolLM", "TinyStories"]
    cd.add_series("字符级 s10m（词表 336）", (1.602, 0.929))
    cd.add_series("BPE s10m（词表 8192）", (1.367, 0.683))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8), Inches(2.0),
                            Inches(5.6), Inches(4.6), cd)
    gf.name = "bpb 图"
    ch = gf.chart
    style_chart(ch)
    color_series(ch, [GOLD, NAVY])
    bar_labels(ch, "0.000", size=11)
    ch.value_axis.minimum_scale = 0
    ch.value_axis.maximum_scale = 1.9
    ch.plots[0].gap_width = 80
    ch.has_title = True
    ch.chart_title.text_frame.text = "bits-per-byte（越低越好）"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = GRAY
    card(s, 6.75, 2.1, 4.95, 1.35, [
        [("为什么不能直接比 loss", {"bold": True, "color": NAVY})],
        "字符级 loss 1.11、BPE 3.67——但前者每次只猜 1 个字符，后者猜约 3.9 个字符",
    ], size=14, space_after=4, line=1.05, name="卡片1")
    card(s, 6.75, 3.6, 4.95, 0.95, [
        [("bpb = 总 NLL ÷ ln2 ÷ 字节数", {"bold": True, "color": NAVY})],
        "在同一段原文上分别用两种分词器计算",
    ], fill=NAVY_TINT, size=14, space_after=4, name="卡片2")
    card(s, 6.75, 4.7, 4.95, 1.85, [
        [("差距从哪来", {"bold": True, "color": NAVY})],
        "同样 3 亿 token：字符级只看了 3 亿字节，BPE 看了 11.6 亿字节",
        "上下文 512：字符级约 100 个词，BPE 约 400 个词",
    ], size=14, space_after=4, line=1.05, name="卡片3")
    notes(s, "【约 40 秒】第三个实验比较分词方式。字符级模型的 loss 是 1.1，BPE 是 3.7，看起来字符级好得多，"
             "但字符级每次只猜一个字符，BPE 一次猜大约四个字符，任务难度不一样，不能直接比。"
             "我们换算成 bits-per-byte，也就是每个字节平均需要多少比特，在同一段原文上比较：BPE 在两个语料上都更好。"
             "原因是同样 3 亿 token，BPE 看过的文本是字符级的 4 倍，同样 512 的上下文也能装下 4 倍的内容。")


def slide_summary(s) -> None:
    set_title(s, "总结与展望")
    lead(s, "从数据到界面全部自己实现；三组实验各得到一个结论")
    items = [("01", "规模", "参数越多 loss 越低，但收益递减；写故事 0.05B 就够，说明文到 0.1B 还在进步"),
             ("02", "采样", "低温复读、高温乱码；T = 0.7–0.8 + top-k 50 最均衡"),
             ("03", "分词", "跨分词器比较要用 bits-per-byte；相同 token 预算下 BPE 优于字符级")]
    for i, (num, head, body) in enumerate(items):
        y = 2.2 + i * 1.3
        text(s, 0.95, y, 1.0, 1.0, [[(num, {"latin": "Arial"})]], size=36, color=GOLD, bold=True)
        text(s, 2.0, y + 0.05, 5.0, 0.45, [[(head, {"bold": True, "color": NAVY})]], size=18)
        text(s, 2.0, y + 0.5, 5.0, 0.75, [body], size=14, color=TEXT, line=1.1)
    card(s, 7.4, 2.2, 4.3, 2.75, [
        [("下一步", {"bold": True, "color": NAVY, "size": 18})],
        [("更多数据：", {"bold": True}), ("s100m 还在欠拟合", {})],
        [("RoPE：", {"bold": True}), ("可外推的位置编码", {})],
        [("KV cache：", {"bold": True}), ("缓存 k、v，生成提速", {})],
        [("指令微调：", {"bold": True}), ("从续写到回答问题", {})],
    ], fill=TINT, size=14, space_after=10, line=1.1, name="下一步", pad=0.25)
    notes(s, "【约 30 秒】总结三点：规模带来稳定但递减的收益；采样参数决定了保守和胡说之间的平衡；"
             "比较不同分词器要用 bits-per-byte。下一步，s100m 的 loss 还在降，最直接的是加数据；"
             "其次可以换 RoPE 位置编码、加 KV cache 提高生成速度，或者做指令微调。谢谢大家，欢迎提问。")


def slide_backup(s) -> None:
    set_title(s, "附录：演示备用样例（seed 42）")
    lead(s, "未注明时 T = 0.8、top-k 50；完整样例见 docs/results/samples_A.md、samples_B.md")
    rows = [("prompt", "对照 1", "对照 2"),
            ("Once upon a time",
             "s10m：, there was a little boy named Timmy. Timmy loved to play with his toy car all day long. One day, Timmy's mommy gave him a big package of a new toy car…",
             "s100m：… One day, Timmy's mom told him it was time to take a bath. Timmy didn't want to stop playing, but he knew he had to listen to his mom."),
            ("Lily wanted to",
             "s10m：play with her friends, but they had to be careful and not go near the pond. \"Go away, hay!\" it said…",
             "s100m：play with the ball, but Lily said, \"No, it's mine. You have your own toys.\" Lily felt sad and went to sit on a bench…"),
            ("Photosynthesis is the process",
             "s100m，T=0：of converting organic matter into chemical energy. What is the difference between photosynthesis and photosynthesis? …",
             "s100m，T=1.5，top-k 50：by which bacteria perform this type of chemiluminescence. The food for these two types of bacteria…")]
    gf = table(s, 0.8, 2.05, 10.9, rows, col_w=[2.1, 4.4, 4.4], size=12, row_h=1.0)
    for i in range(1, len(rows)):
        for j in range(3):
            c = gf.table.cell(i, j)
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.name = "Arial"
    gf.table.rows[0].height = Inches(0.45)
    notes(s, "备用页：正常情况下不讲。现场演示失败时，用这一页代替第 1、2 步。")


# ---------------------------------------------------------------- 组装


def build() -> None:
    prs = Presentation(str(TEMPLATE))
    tpl = list(prs.slides)
    cover, toc, demo, thanks = tpl[0], tpl[4], tpl[6], tpl[23]
    edit_cover(cover)
    edit_toc(toc)
    edit_demo_divider(demo)

    layout = next(l for l in prs.slide_layouts if l.name == "内容页5")
    builders = [slide_task, slide_data, slide_model, slide_train, slide_scale_curves,
                slide_scale_human, slide_sampling, slide_tokenizer, slide_summary, slide_backup]
    new = []
    for b in builders:
        s = prs.slides.add_slide(layout)
        b(s)
        new.append(s)

    notes(cover, "【约 10 秒】各位老师好，我们组的题目是 B03：从零训练一个迷你语言模型。")
    notes(toc, "【约 10 秒】分四部分：任务与数据、模型与训练、三组对比实验，最后现场演示界面。")
    notes(demo, "【约 2 分钟】切到浏览器（http://127.0.0.1:7860，提前启动 python app/app.py --preload all）。"
                "按 docs/defense/demo.md 第三节的脚本演示；出问题按第四节兜底，或翻到最后的附录页。")
    notes(thanks, "谢谢，欢迎提问。问答准备见 docs/defense/qa.md。")

    task, data, model, train, curves, human, sampling, tokenizer, summary, backup = new
    order = [cover, toc, task, data, model, train, curves, human, sampling, tokenizer, demo,
             summary, thanks, backup]

    # 重排并删除不用的模板页：sldIdLst 只保留 order 里的页，其余页的关系一并去掉，保存时不再写入
    sld_lst = prs.slides._sldIdLst
    id_of = {int(el.get("id")): el for el in sld_lst}
    keep_ids = [s.slide_id for s in order]
    for el in list(sld_lst):
        sld_lst.remove(el)
    for sid in keep_ids:
        sld_lst.append(id_of[sid])
    keep_parts = {s.part for s in order}
    for s in tpl:
        if s.part not in keep_parts:
            rid = next(r.rId for r in prs.part.rels.values() if r.target_part is s.part)
            prs.part.drop_rel(rid)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"保存：{OUT}（{len(order)} 页）")


if __name__ == "__main__":
    build()
