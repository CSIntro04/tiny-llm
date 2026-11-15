"""两套界面（Gradio 的 app.py、React 前端的 server.py）共用的部分：找 checkpoint、加载并缓存模型、读训练结果。

这里不 import gradio / fastapi，只依赖 src/ 下的模块。
"""

from __future__ import annotations

import functools
import random
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # 让 app/ 下的脚本能 import src/ 里的模块

from model import GPT  # noqa: E402
from sample import Tokenizer, load_model  # noqa: E402

RUNS_DIR = ROOT / "runs"
FIGURES = ROOT / "docs/figures"
MAIN_RUNS = ["s100m", "s50m", "s10m", "s10m_char"]  # 下拉框里排在最前面
COMPARE_RUNS = ["s10m", "s50m", "s100m"]
DEFAULT_CKPT = "runs/s100m/ckpt_best.pt"
RESULT_TABLES = [("results.md", "最终结果（实验 A）"),
                 ("ckpt_eval.md", "相同已训练 token 数下的 val loss"),
                 ("bpb.md", "字符级 vs BPE（bits-per-byte，实验 C）")]

# 同一时刻只让一个请求用 GPU 生成：set_seed 是全局的，两个请求交错会互相打乱随机数序列
GPU_LOCK = threading.Lock()


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


def model_stats(ckpt_path: str) -> dict:
    """模型结构、参数量、已训练 token 数、best val loss。"""
    model, tok, cfg = get_model(ckpt_path)
    meta = ckpt_meta(ckpt_path)
    c, tc = model.cfg, cfg["train"]
    return {
        "ckpt": ckpt_path,
        "n_layer": c.n_layer, "n_embd": c.n_embd, "n_head": c.n_head, "block_size": c.block_size,
        "params": model.num_params(False),               # 总参数（含 embedding）
        "params_non_emb": model.num_params(True),        # 不含 embedding
        "tokenizer": type(tok).__name__, "vocab_size": tok.vocab_size,
        "iters": meta["iter"],
        "tokens": meta["iter"] * tc["batch_size"] * tc["grad_accum"] * c.block_size,
        "best_val": meta["best_val"],
    }


def resolve_seed(seed: float) -> int:
    """seed = -1 时随机取一个，并显示出来，方便复现这次结果。"""
    return random.randint(0, 2**31 - 1) if int(seed) == -1 else int(seed)


def list_runs() -> list[str]:
    """有训练日志和 loss 曲线的 run。"""
    return [p.parent.name for p in sorted(RUNS_DIR.glob("*/log.csv"))
            if (p.parent / "loss_curve.png").exists()]


def run_curve(run: str) -> str | None:
    p = RUNS_DIR / run / "loss_curve.png"
    return str(p) if p.exists() else None


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
