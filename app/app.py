"""Gradio 交互界面：输入 prompt，选 checkpoint，调采样参数，看续写结果。

用法：
    python app/app.py                    # 浏览器打开 http://127.0.0.1:7860
    python app/app.py --preload all      # 答辩用：启动时就把三个正式模型都加载好

三个标签页：单模型生成 / 三模型对比 / 训练过程。
界面只负责"调用"：加载模型和生成全部复用 src/sample.py 的 load_model / generate_text。
外观：Soft 主题 + 下面的 CSS 常量；颜色全部用 Gradio 主题变量，深色模式自动适配。
"""

from __future__ import annotations

import argparse
import html
import time

import gradio as gr
import torch

from common import (COMPARE_RUNS, DEFAULT_CKPT, FIGURES, RESULT_TABLES, ROOT, RUNS_DIR, get_model,
                    list_checkpoints, list_runs, model_stats, preload, resolve_seed, run_curve)
from eval_prompts import PROMPTS  # noqa: E402  （common 已把 src/ 加进 sys.path）
from sample import generate_text  # noqa: E402

# 采样预设：名字 -> (temperature, top_k, top_p)。top_k=0 / top_p=1.0 表示不限
PRESETS: dict[str, tuple[float, int, float]] = {
    "Greedy": (0.0, 0, 1.0),
    "保守": (0.5, 20, 1.0),
    "默认": (0.8, 50, 1.0),
    "创意": (1.2, 0, 0.95),
}


# ---------------------------------------------------------------- 外观

# 系统字体：答辩现场不联网，不用 Google Fonts
FONT = ["Inter", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "sans-serif"]
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate",
                       font=FONT, radius_size="lg")

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* 头部横幅 */
.hero { padding: 28px 32px; border-radius: 18px; color: #fff;
        background: linear-gradient(120deg, #4338ca 0%, #6366f1 45%, #0891b2 100%);
        box-shadow: 0 10px 30px -12px rgba(67, 56, 202, .55); }
.hero h1 { margin: 0 0 6px; font-size: 30px; font-weight: 750; letter-spacing: -.01em; color: #fff; }
.hero p { margin: 0 0 14px; opacity: .9; font-size: 15px; }
.hero .chip { display: inline-block; margin: 0 6px 6px 0; padding: 3px 11px; border-radius: 999px;
              font-size: 12.5px; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); }

/* 指标卡 */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 10px; }
.stat { padding: 12px 14px; border-radius: 12px; background: var(--background-fill-secondary);
        border: 1px solid var(--border-color-primary); }
.stat .k { font-size: 12px; color: var(--body-text-color-subdued); }
.stat .v { font-size: 20px; font-weight: 700; margin-top: 2px; color: var(--body-text-color); }
.stat .s { font-size: 11.5px; color: var(--body-text-color-subdued); }
.ckpt-path { font-size: 12px; color: var(--body-text-color-subdued); margin-bottom: 8px;
             font-family: ui-monospace, Menlo, Consolas, monospace; }

/* 生成结果卡 */
.gen-card { padding: 18px 20px; border-radius: 14px; min-height: 120px;
            background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
.gen-text { white-space: pre-wrap; font-size: 17px; line-height: 1.75; color: var(--body-text-color);
            font-family: Georgia, "Noto Serif CJK SC", "Songti SC", serif; }
.gen-prompt { background: var(--color-accent-soft); color: var(--color-accent); font-weight: 650;
              padding: 1px 4px; border-radius: 5px; }
.gen-empty { color: var(--body-text-color-subdued); font-size: 14px; text-align: center; padding-top: 38px; }
.gen-meta { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 6px; }
.pill { font-size: 12px; padding: 2px 10px; border-radius: 999px; color: var(--body-text-color-subdued);
        background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); }

/* 三模型对比列头 */
.cmp-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.cmp-name { font-size: 18px; font-weight: 700; }
.badge { font-size: 12px; font-weight: 600; padding: 2px 10px; border-radius: 999px; color: #fff; }
.tier-0 { background: #22d3ee; } .tier-1 { background: #6366f1; } .tier-2 { background: #4338ca; }

/* 预设按钮、结果表 */
.preset-row button { min-width: 0 !important; }
.result-tables table { border-collapse: collapse; width: 100%; font-size: 14px; }
.result-tables th, .result-tables td { border: 1px solid var(--border-color-primary) !important; padding: 6px 10px; }
.result-tables tr:nth-child(even) td { background: var(--background-fill-secondary); }
.app-foot { text-align: center; font-size: 12.5px; color: var(--body-text-color-subdued); padding: 8px 0 4px; }
"""

HERO = """
<div class="hero">
  <h1>迷你语言模型 · Tiny LLM</h1>
  <p>从零实现并训练的 Decoder-only Transformer —— 输入一段开头，看模型如何逐个 token 续写。</p>
  <span class="chip">Decoder-only Transformer</span><span class="chip">BPE 词表 8192</span>
  <span class="chip">上下文 512</span><span class="chip">SmolLM-Corpus + TinyStories</span>
  <span class="chip">0.01B / 0.05B / 0.1B</span>
</div>
"""

EMPTY_OUTPUT = '<div class="gen-card"><div class="gen-empty">输入 prompt，点击「生成」查看续写结果</div></div>'


# ---------------------------------------------------------------- 模型


def _stat(key: str, value: str, sub: str = "") -> str:
    """一张指标卡：小标题 / 大数字 / 灰色注释。"""
    sub_html = f'<div class="s">{html.escape(sub)}</div>' if sub else ""
    return (f'<div class="stat"><div class="k">{html.escape(key)}</div>'
            f'<div class="v">{html.escape(value)}</div>{sub_html}</div>')


def model_cards(ckpt_path: str) -> str:
    """模型结构、参数量、已训练 token 数、best val loss，渲染成一排指标卡（HTML）。"""
    if not ckpt_path:
        return '<div class="ckpt-path">没有选择 checkpoint</div>'
    m = model_stats(ckpt_path)
    cards = [
        _stat("结构", f"{m['n_layer']}/{m['n_embd']}/{m['n_head']}", "层 / 维 / 头"),
        _stat("总参数", f"{m['params'] / 1e6:.1f}M", f"非 emb {m['params_non_emb'] / 1e6:.1f}M"),
        _stat("分词器", f"{m['vocab_size']}", m["tokenizer"]),
        _stat("已训练", f"{m['tokens'] / 1e6:.0f}M", f"token · {m['iters']} 步"),
        _stat("best val loss", f"{m['best_val']:.3f}", f"上下文 {m['block_size']}"),
    ]
    return f'<div class="ckpt-path">{html.escape(ckpt_path)}</div><div class="stats">{"".join(cards)}</div>'


# ---------------------------------------------------------------- 生成


def _run(ckpt: str, prompt: str, max_new_tokens: int, temperature: float,
         top_k: int, top_p: float, seed: int) -> tuple[str, int, float]:
    """生成一条，返回 (续写文本, 新 token 数, 用时秒)。界面上的 0 / 1.0 / -1 表示"不限 / 随机"。"""
    model, tok, _ = get_model(ckpt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    text = generate_text(model, tok, prompt, num_samples=1, max_new_tokens=int(max_new_tokens),
                         temperature=float(temperature), top_k=int(top_k) or None,
                         top_p=float(top_p) if top_p < 1.0 else None, seed=int(seed))[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return text, len(tok.encode(text)), time.time() - t0


def _render(prompt: str, text: str, notes: list[str]) -> str:
    """输出卡：prompt 高亮、续写正常显示，底部一排灰色胶囊。全部 HTML 转义，保留换行。"""
    pills = "".join(f'<span class="pill">{html.escape(n)}</span>' for n in notes)
    return (f'<div class="gen-card"><div class="gen-text"><span class="gen-prompt">{html.escape(prompt)}</span>'
            f'{html.escape(text)}</div><div class="gen-meta">{pills}</div></div>')


def generate_single(ckpt: str, prompt: str, max_new_tokens: int, temperature: float,
                    top_k: int, top_p: float, seed: float) -> str:
    seed = resolve_seed(seed)
    text, n, dt = _run(ckpt, prompt, max_new_tokens, temperature, top_k, top_p, seed)
    return _render(prompt, text, [f"seed {seed}", f"{n} token", f"{dt:.2f} s", f"{n / max(dt, 1e-6):.0f} tok/s"])


def generate_compare(prompt: str, max_new_tokens: int, temperature: float,
                     top_k: int, top_p: float, seed: float) -> tuple[str, str, str]:
    """同一 prompt、同一组参数、同一 seed，依次跑三个规模的模型。"""
    seed = resolve_seed(seed)
    outs = []
    for run in COMPARE_RUNS:
        ckpt = f"runs/{run}/ckpt_best.pt"
        if not (ROOT / ckpt).exists():
            outs.append(f'<div class="gen-card"><div class="gen-empty">找不到 {html.escape(ckpt)}</div></div>')
            continue
        text, n, dt = _run(ckpt, prompt, max_new_tokens, temperature, top_k, top_p, seed)
        outs.append(_render(prompt, text, [f"seed {seed}", f"{n} token", f"{dt:.2f} s"]))
    return outs[0], outs[1], outs[2]


def _compare_head(run: str, tier: int) -> str:
    """对比列的列头：模型名 + 参数量徽章（颜色由浅到深对应 0.01B → 0.1B）。"""
    ckpt = f"runs/{run}/ckpt_best.pt"
    size = f"{get_model(ckpt)[0].num_params(False) / 1e6:.1f}M 参数" if (ROOT / ckpt).exists() else "缺失"
    return (f'<div class="cmp-head"><span class="cmp-name">{html.escape(run)}</span>'
            f'<span class="badge tier-{tier}">{size}</span></div>')


# ---------------------------------------------------------------- 训练过程


def results_tables() -> str:
    """docs/figures/ 下已有的结果表，拼成一段 Markdown。"""
    parts = []
    for name, title in RESULT_TABLES:
        p = FIGURES / name
        if p.exists():
            parts.append(f"### {title}\n\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) or "还没有结果表：先运行 `python src/plot_compare.py ...`"


# ---------------------------------------------------------------- 界面


def _sampling_controls() -> list[gr.components.Component]:
    """预设按钮 + 长度、temperature、top-k、top-p、seed 五个控件（两个标签页各建一套）。"""
    with gr.Row(elem_classes="preset-row"):
        buttons = {name: gr.Button(name, size="sm", variant="secondary") for name in PRESETS}
    n = gr.Slider(10, 500, value=200, step=10, label="生成长度", info="max_new_tokens")
    t = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="temperature", info="越大越随机；0 = greedy")
    k = gr.Slider(0, 200, value=50, step=1, label="top-k", info="只在概率最高的 k 个里采样；0 = 不限")
    p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top-p", info="累计概率截断；1.0 = 不限")
    s = gr.Number(value=42, precision=0, label="seed", info="-1 = 随机（结果里会显示实际 seed）")
    for name, btn in buttons.items():
        btn.click(lambda v=PRESETS[name]: v, None, [t, k, p])  # 默认参数绑定当前预设，避免闭包晚绑定
    return [n, t, k, p, s]


def _tab_single(ckpts: list[str], default: str | None, examples: list[str]) -> None:
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300, variant="panel"):
            with gr.Row(equal_height=True):
                ckpt = gr.Dropdown(ckpts, value=default, label="checkpoint", scale=5)
                refresh = gr.Button("🔄", size="sm", scale=0, min_width=44)
            gr.Markdown("**采样预设**")
            controls = _sampling_controls()
        with gr.Column(scale=2, min_width=420):
            info = gr.HTML(model_cards(default) if default else "")
            prompt = gr.Textbox(value="Once upon a time", label="prompt", lines=3,
                                placeholder="输入英文开头，例如 Once upon a time")
            gr.Examples(examples, inputs=prompt, label="示例 prompt（故事 3 个 + 说明文 3 个）")
            with gr.Row():
                btn = gr.Button("✨ 生成", variant="primary", size="lg", scale=4)
                clear = gr.Button("清空", variant="secondary", size="lg", scale=1)
            out = gr.HTML(EMPTY_OUTPUT)

    ckpt.change(model_cards, ckpt, info)
    refresh.click(lambda: gr.Dropdown(choices=list_checkpoints()), None, ckpt)
    btn.click(generate_single, [ckpt, prompt, *controls], out)
    prompt.submit(generate_single, [ckpt, prompt, *controls], out)
    clear.click(lambda: EMPTY_OUTPUT, None, out)


def _tab_compare(examples: list[str]) -> None:
    gr.Markdown("同一个 prompt、同一组采样参数、同一个 seed，比较 **0.01B / 0.05B / 0.1B** 三个规模。")
    with gr.Row(equal_height=True):
        prompt = gr.Textbox(value="Once upon a time", label="prompt", lines=1, scale=5)
        btn = gr.Button("✨ 三个模型一起生成", variant="primary", size="lg", scale=2)
    gr.Examples(examples, inputs=prompt, label="示例 prompt")
    with gr.Accordion("采样参数", open=False):
        controls = _sampling_controls()
    cols = []
    with gr.Row(equal_height=False):
        for i, run in enumerate(COMPARE_RUNS):
            with gr.Column(min_width=260):
                gr.HTML(_compare_head(run, i))
                cols.append(gr.HTML(EMPTY_OUTPUT))
    btn.click(generate_compare, [prompt, *controls], cols)
    prompt.submit(generate_compare, [prompt, *controls], cols)


def _tab_training() -> None:
    runs = list_runs()
    with gr.Column(variant="panel"):
        run = gr.Dropdown(runs, value="s100m" if "s100m" in runs else (runs[0] if runs else None),
                          label="run")
        curve = gr.Image(run_curve(run.value) if run.value else None,
                         label="loss 曲线（左：step，右：token）", show_label=True)
    run.change(run_curve, run, curve)
    with gr.Accordion("对比图（docs/figures/）", open=True):
        gr.Gallery([str(p) for p in sorted(FIGURES.glob("*.png")) if not p.name.startswith("app_")],
                   columns=3, height="auto", show_label=False)
    with gr.Accordion("结果表", open=True):
        gr.Markdown(results_tables(), elem_classes="result-tables")


def build_ui() -> gr.Blocks:
    ckpts = list_checkpoints()
    default = DEFAULT_CKPT if DEFAULT_CKPT in ckpts else (ckpts[0] if ckpts else None)
    examples = PROMPTS["story"][:3] + PROMPTS["expository"][:3]

    with gr.Blocks(title="Tiny LLM") as demo:
        gr.HTML(HERO)
        with gr.Tab("💬 单模型生成"):
            _tab_single(ckpts, default, examples)
        with gr.Tab("⚖️ 三模型对比"):
            _tab_compare(examples)
        with gr.Tab("📈 训练过程"):
            _tab_training()
        gr.HTML('<div class="app-foot">上海交通大学 计算机科学导论 · B03 从零训练一个迷你语言模型</div>')
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gradio 交互界面")
    ap.add_argument("--preload", choices=["none", "default", "all"], default="default")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    preload(args.preload)
    # Gradio 默认只允许返回"当前工作目录 / 系统临时目录"里的文件；从 app/ 等其他目录启动时，
    # runs/*/loss_curve.png 会被拒绝（InvalidPathError）。显式放行这两个只读的图片目录。
    # Gradio 6 起 theme / css 在 launch() 上传，不在 gr.Blocks() 上。
    build_ui().launch(server_name="127.0.0.1", server_port=args.port, theme=THEME, css=CSS,
                      allowed_paths=[str(RUNS_DIR), str(FIGURES)])
