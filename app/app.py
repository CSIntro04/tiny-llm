"""Gradio 交互界面：输入 prompt，选 checkpoint，调采样参数，看续写结果。

用法：
    python app/app.py                    # 浏览器打开 http://127.0.0.1:7860
    python app/app.py --preload all      # 答辩用：启动时就把三个正式模型都加载好

三个标签页：单模型生成 / 三模型对比 / 训练过程。
界面只负责"调用"：加载模型和生成全部复用 src/sample.py 的 load_model / generate_text。
"""

from __future__ import annotations

import argparse
import functools
import html
import random
import sys
import time
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 让 app/ 下的脚本能 import src/ 里的模块

from eval_prompts import PROMPTS  # noqa: E402
from model import GPT  # noqa: E402
from sample import Tokenizer, generate_text, load_model  # noqa: E402

RUNS_DIR = ROOT / "runs"
MAIN_RUNS = ["s100m", "s50m", "s10m", "s10m_char"]  # 下拉框里排在最前面
COMPARE_RUNS = ["s10m", "s50m", "s100m"]
DEFAULT_CKPT = "runs/s100m/ckpt_best.pt"
FIGURES = ROOT / "docs/figures"


# ---------------------------------------------------------------- 模型


def list_checkpoints(runs_dir: Path = RUNS_DIR) -> list[str]:
    """扫描 runs/*/ckpt_*.pt，返回相对项目根目录的路径。正式 run 在前，smoke/overfit 在后。"""
    def order(p: Path) -> tuple[int, str]:
        run = p.parent.name
        return (MAIN_RUNS.index(run) if run in MAIN_RUNS else len(MAIN_RUNS), str(p))
    paths = sorted(runs_dir.glob("*/ckpt_*.pt"), key=order)
    return [p.relative_to(ROOT).as_posix() for p in paths]


@functools.lru_cache(maxsize=4)
def get_model(ckpt_path: str) -> tuple[GPT, Tokenizer, dict]:
    """加载并缓存模型。参数是字符串（可哈希），同一个 checkpoint 只加载一次。"""
    p = ROOT / ckpt_path
    return load_model(str(p.parent), p.name)


@functools.lru_cache(maxsize=32)
def ckpt_meta(ckpt_path: str) -> dict:
    """读 checkpoint 里的训练步数和 best_val。mmap=True：不把整个文件（含优化器状态）读进内存。"""
    state = torch.load(ROOT / ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    return {"iter": state["iter"], "best_val": state["best_val"]}


def model_info(ckpt_path: str) -> str:
    """模型结构、参数量、已训练 token 数、best val loss（Markdown）。"""
    if not ckpt_path:
        return "没有选择 checkpoint"
    model, tok, cfg = get_model(ckpt_path)
    meta = ckpt_meta(ckpt_path)
    c, tc = model.cfg, cfg["train"]
    tokens = meta["iter"] * tc["batch_size"] * tc["grad_accum"] * c.block_size
    return (
        f"**{ckpt_path}**\n\n"
        f"| 结构（层/维/头） | 总参数 | 非 embedding | 分词器（词表） | 上下文 | 已训练 | best val loss |\n"
        f"|---|---|---|---|---|---|---|\n"
        f"| {c.n_layer} / {c.n_embd} / {c.n_head} | {model.num_params(False) / 1e6:.2f}M "
        f"| {model.num_params(True) / 1e6:.2f}M | {type(tok).__name__}（{tok.vocab_size}） | {c.block_size} "
        f"| {meta['iter']} 步 / {tokens / 1e6:.0f}M token | {meta['best_val']:.3f} |"
    )


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


def _render(prompt: str, text: str, note: str) -> str:
    """prompt 加粗、续写正常显示；都做 HTML 转义，保留换行。"""
    return (f'<div style="white-space: pre-wrap; line-height: 1.6; font-size: 15px">'
            f'<b style="color: #2563eb">{html.escape(prompt)}</b>{html.escape(text)}</div>'
            f'<p style="color: gray; font-size: 12px">{html.escape(note)}</p>')


def _resolve_seed(seed: float) -> int:
    """seed = -1 时随机取一个，并显示出来，方便复现这次结果。"""
    return random.randint(0, 2**31 - 1) if int(seed) == -1 else int(seed)


def generate_single(ckpt: str, prompt: str, max_new_tokens: int, temperature: float,
                    top_k: int, top_p: float, seed: float) -> str:
    seed = _resolve_seed(seed)
    text, n, dt = _run(ckpt, prompt, max_new_tokens, temperature, top_k, top_p, seed)
    return _render(prompt, text, f"seed={seed} · 生成 {n} 个 token · {dt:.2f}s · {n / max(dt, 1e-6):.0f} tok/s")


def generate_compare(prompt: str, max_new_tokens: int, temperature: float,
                     top_k: int, top_p: float, seed: float) -> tuple[str, str, str]:
    """同一 prompt、同一组参数、同一 seed，依次跑三个规模的模型。"""
    seed = _resolve_seed(seed)
    outs = []
    for run in COMPARE_RUNS:
        ckpt = f"runs/{run}/ckpt_best.pt"
        if not (ROOT / ckpt).exists():
            outs.append(f"<p>找不到 {ckpt}</p>")
            continue
        text, n, dt = _run(ckpt, prompt, max_new_tokens, temperature, top_k, top_p, seed)
        outs.append(_render(prompt, text, f"seed={seed} · {n} token · {dt:.2f}s"))
    return outs[0], outs[1], outs[2]


def _compare_title(run: str) -> str:
    ckpt = f"runs/{run}/ckpt_best.pt"
    if not (ROOT / ckpt).exists():
        return f"### {run}"
    return f"### {run}（{get_model(ckpt)[0].num_params(False) / 1e6:.1f}M 参数）"


# ---------------------------------------------------------------- 训练过程


def list_runs() -> list[str]:
    return [p.parent.name for p in sorted(RUNS_DIR.glob("*/log.csv"))
            if (p.parent / "loss_curve.png").exists()]


def run_curve(run: str) -> str | None:
    p = RUNS_DIR / run / "loss_curve.png"
    return str(p) if p.exists() else None


def results_tables() -> str:
    """docs/figures/ 下已有的结果表，拼成一段 Markdown。"""
    parts = []
    for name, title in [("results.md", "最终结果（实验 A）"),
                        ("ckpt_eval.md", "相同已训练 token 数下的 val loss"),
                        ("bpb.md", "字符级 vs BPE（bits-per-byte，实验 C）")]:
        p = FIGURES / name
        if p.exists():
            parts.append(f"### {title}\n\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) or "还没有结果表：先运行 `python src/plot_compare.py ...`"


# ---------------------------------------------------------------- 界面


def _sampling_controls() -> list[gr.components.Component]:
    """长度、temperature、top-k、top-p、seed 五个控件（两个标签页各建一套）。"""
    with gr.Row():
        n = gr.Slider(10, 500, value=200, step=10, label="max_new_tokens（生成长度）")
        t = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="temperature（0 = greedy）")
    with gr.Row():
        k = gr.Slider(0, 200, value=50, step=1, label="top_k（0 = 不限）")
        p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top_p（1.0 = 不限）")
        s = gr.Number(value=42, precision=0, label="seed（-1 = 随机）")
    return [n, t, k, p, s]


def build_ui() -> gr.Blocks:
    ckpts = list_checkpoints()
    default = DEFAULT_CKPT if DEFAULT_CKPT in ckpts else (ckpts[0] if ckpts else None)
    examples = PROMPTS["story"][:3] + PROMPTS["expository"][:3]

    with gr.Blocks(title="Tiny LLM") as demo:
        gr.Markdown("# 迷你语言模型：从零训练的 Decoder-only Transformer\n"
                    "SmolLM-Corpus 子集 + TinyStories 训练，BPE 词表 8192，上下文 512。")

        with gr.Tab("单模型生成"):
            with gr.Row():
                ckpt = gr.Dropdown(ckpts, value=default, label="checkpoint", scale=5)
                refresh = gr.Button("刷新列表", scale=1)
            info = gr.Markdown(model_info(default) if default else "")
            prompt = gr.Textbox(value="Once upon a time", label="prompt", lines=2)
            controls = _sampling_controls()
            btn = gr.Button("生成", variant="primary")
            out = gr.HTML(label="续写")
            gr.Examples(examples, inputs=prompt, label="示例 prompt（故事 3 个 + 说明文 3 个）")

            ckpt.change(model_info, ckpt, info)
            refresh.click(lambda: gr.Dropdown(choices=list_checkpoints()), None, ckpt)
            btn.click(generate_single, [ckpt, prompt, *controls], out)

        with gr.Tab("三模型对比"):
            gr.Markdown("同一个 prompt、同一组采样参数、同一个 seed，比较 0.01B / 0.05B / 0.1B。")
            prompt2 = gr.Textbox(value="Once upon a time", label="prompt", lines=2)
            controls2 = _sampling_controls()
            btn2 = gr.Button("三个模型一起生成", variant="primary")
            cols = []
            with gr.Row():
                for run in COMPARE_RUNS:
                    with gr.Column():
                        gr.Markdown(_compare_title(run))
                        cols.append(gr.HTML())
            gr.Examples(examples, inputs=prompt2)
            btn2.click(generate_compare, [prompt2, *controls2], cols)

        with gr.Tab("训练过程"):
            runs = list_runs()
            run = gr.Dropdown(runs, value="s100m" if "s100m" in runs else (runs[0] if runs else None),
                              label="run")
            curve = gr.Image(run_curve(run.value) if run.value else None, label="loss 曲线（左：step，右：token）")
            run.change(run_curve, run, curve)
            gr.Markdown("### 对比图（`docs/figures/`）")
            gr.Gallery([str(p) for p in sorted(FIGURES.glob("*.png"))], columns=3, height="auto")
            gr.Markdown(results_tables())
    return demo


def preload(mode: str) -> None:
    """启动时预加载模型，第一次点"生成"不用等。default 只加载默认模型；all 加载三模型对比用到的全部。"""
    wanted = {"none": [], "default": [DEFAULT_CKPT],
              "all": [DEFAULT_CKPT] + [f"runs/{r}/ckpt_best.pt" for r in COMPARE_RUNS]}[mode]
    available = list_checkpoints()
    for ckpt in dict.fromkeys(wanted):  # 去重并保持顺序
        if ckpt in available:
            t0 = time.time()
            get_model(ckpt)
            print(f"预加载 {ckpt}（{time.time() - t0:.1f}s）")
        else:
            print(f"跳过 {ckpt}：文件不存在")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gradio 交互界面")
    ap.add_argument("--preload", choices=["none", "default", "all"], default="default")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    preload(args.preload)
    # Gradio 默认只允许返回"当前工作目录 / 系统临时目录"里的文件；从 app/ 等其他目录启动时，
    # runs/*/loss_curve.png 会被拒绝（InvalidPathError）。显式放行这两个只读的图片目录。
    build_ui().launch(server_name="127.0.0.1", server_port=args.port,
                      allowed_paths=[str(RUNS_DIR), str(FIGURES)])
