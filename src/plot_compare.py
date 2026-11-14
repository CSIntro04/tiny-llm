"""实验 A：把多个 run 的训练日志画成对比图，并输出最终结果表。

用法：
    python src/plot_compare.py --runs runs/s10m runs/s50m runs/s100m --out docs/figures/

输出（都在 --out 下）：
    val_overlay_tokens.png    加权 val loss 叠图，横轴已训练 token 数（主图）
    val_overlay_time.png      同上，横轴训练时间（辅图）
    val_smollm_overlay.png    SmolLM 验证集 loss 叠图
    val_tinystories_overlay.png
    train_val_gap.png         train loss − val loss（看过拟合）
    scaling.png               参数量 vs 最终 val loss（分来源）
    results.md                最终结果表（Markdown）
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import build_model  # noqa: E402

# 每个 eval 行的列（训练行这些列是空的）
VAL_COLS = ["val_loss", "val_smollm", "val_tinystories"]
LABELS = {"val_loss": "val (mix)", "val_smollm": "val SmolLM", "val_tinystories": "val TinyStories"}


def load_logs(run_dirs: list[str]) -> dict[str, list[dict[str, float]]]:
    """run 名 -> log.csv 的每一行（字符串转 float，空值去掉）。"""
    logs: dict[str, list[dict[str, float]]] = {}
    for d in run_dirs:
        rows = []
        with open(Path(d) / "log.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({k: float(v) for k, v in row.items() if v != ""})
        logs[Path(d).name] = rows
    return logs


def column(rows: list[dict[str, float]], x: str, y: str) -> tuple[list[float], list[float]]:
    """取出同时有 x 和 y 的行，返回 (xs, ys)。"""
    pts = [(r[x], r[y]) for r in rows if x in r and y in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_overlay(logs: dict[str, list[dict[str, float]]], x: str = "tokens",
                 y: str = "val_loss", out: str = "docs/figures/val_overlay_tokens.png") -> None:
    """多个 run 的同一条曲线画在一张图上。x 为 tokens 或 elapsed_s。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scale, xlabel = (1e6, "tokens seen (M)") if x == "tokens" else (60, "training time (min)")
    for name, rows in logs.items():
        xs, ys = column(rows, x, y)
        xs, ys = xs[1:], ys[1:]  # 去掉 step 0（loss≈9，会把纵轴压扁）
        ax.plot([v / scale for v in xs], ys, label=name, lw=1.8)
    if x == "tokens":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(LABELS.get(y, y))
    ax.set_title(f"{LABELS.get(y, y)} vs {xlabel}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_gap(logs: dict[str, list[dict[str, float]]], out: str) -> None:
    """train loss − val loss。eval 行只有 val；用 eval 之前约 30 步的 train loss 平均来配对。
    窗口不能太长：loss 下降快的早期，长窗口的平均会偏高（滞后），看起来像 train > val。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, rows in logs.items():
        train = column(rows, "tokens", "train_loss")
        xs, gaps = [], []
        for tok, val in zip(*column(rows, "tokens", "val_loss")):
            recent = [l for t, l in zip(*train) if tok - 4e6 < t <= tok]  # 约前 30 步（每步 131072 token）
            if recent and tok > 0:
                xs.append(tok / 1e6)
                gaps.append(sum(recent) / len(recent) - val)
        ax.plot(xs, gaps, label=name, lw=1.5)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("tokens seen (M)")
    ax.set_ylabel("train loss − val loss")
    ax.set_title("train/val gap (≈0: no overfitting)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_scaling(final: dict[str, dict[str, float]], out: str) -> None:
    """横轴参数量（对数），纵轴最终 val loss，每个来源一条线。"""
    names = sorted(final, key=lambda n: final[n]["params"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for col, marker in [("val_smollm", "o"), ("val_tinystories", "s"), ("val_loss", "^")]:
        xs = [final[n]["params"] / 1e6 for n in names]
        ys = [final[n][col] for n in names]
        ax.plot(xs, ys, marker=marker, label=LABELS[col])
        for n, xv, yv in zip(names, xs, ys):
            ax.annotate(f"{yv:.3f}", (xv, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("total params (M, log scale)")
    ax.set_ylabel("final val loss")
    ax.set_title("scaling: params vs final val loss")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def final_stats(run_dir: str, rows: list[dict[str, float]]) -> dict[str, float]:
    """一个 run 的最终结果：最后一次 eval 的 loss、参数量、用时、吞吐、显存、ckpt 大小。"""
    run = Path(run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(cfg)  # 只为数参数，不加载权重
    last_eval = [r for r in rows if "val_loss" in r][-1]
    speeds = [r["tokens_per_sec"] for r in rows if "tokens_per_sec" in r]
    speeds = speeds[len(speeds) // 10:]  # 去掉前 10%（含 torch.compile 编译和 warmup）
    ckpt = run / "ckpt_best.pt"
    return {
        "params": model.num_params(False),
        "params_nonemb": model.num_params(True),
        "tokens": last_eval["tokens"],
        **{c: last_eval[c] for c in VAL_COLS},
        "ppl": math.exp(last_eval["val_loss"]),
        "minutes": last_eval["elapsed_s"] / 60,
        "tok_s": sum(speeds) / len(speeds),
        "mem_gb": max(r.get("mem_gb", 0.0) for r in rows),
        "ckpt_mb": ckpt.stat().st_size / 1e6 if ckpt.exists() else float("nan"),
        "weights_mb": model.num_params(False) * 4 / 1e6,  # fp32 权重，不含优化器状态
    }


def results_markdown(final: dict[str, dict[str, float]]) -> str:
    head = ("| 模型 | 总参数 | 非 emb 参数 | 训练 token | val（加权） | val SmolLM | val TinyStories "
            "| 困惑度 | 训练时长 | 平均 tok/s | 显存峰值 | ckpt 文件 / 仅权重 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for n, s in sorted(final.items(), key=lambda kv: kv[1]["params"]):
        lines.append(
            f"| {n} | {s['params'] / 1e6:.2f}M | {s['params_nonemb'] / 1e6:.2f}M | {s['tokens'] / 1e9:.2f}B "
            f"| {s['val_loss']:.3f} | {s['val_smollm']:.3f} | {s['val_tinystories']:.3f} | {s['ppl']:.1f} "
            f"| {s['minutes']:.1f} min | {s['tok_s'] / 1e3:.0f}k | {s['mem_gb']:.1f} GB "
            f"| {s['ckpt_mb']:.0f} / {s['weights_mb']:.0f} MB |")
    note = ("\n\nval 取 log.csv 最后一次评估（训练结束时）；困惑度 = exp(val 加权)；"
            "训练时长只算训练步（云端 RTX 5090）；tok/s 去掉了前 10% 的步；"
            "ckpt 文件含 AdamW 状态，约为仅权重大小的 3 倍。\n")
    return head + "\n".join(lines) + note


def main() -> None:
    p = argparse.ArgumentParser(description="多 run 对比画图")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="docs/figures/")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    logs = load_logs(args.runs)
    plot_overlay(logs, "tokens", "val_loss", str(out / "val_overlay_tokens.png"))
    plot_overlay(logs, "elapsed_s", "val_loss", str(out / "val_overlay_time.png"))
    plot_overlay(logs, "tokens", "val_smollm", str(out / "val_smollm_overlay.png"))
    plot_overlay(logs, "tokens", "val_tinystories", str(out / "val_tinystories_overlay.png"))
    plot_gap(logs, str(out / "train_val_gap.png"))

    final = {Path(d).name: final_stats(d, logs[Path(d).name]) for d in args.runs}
    plot_scaling(final, str(out / "scaling.png"))
    md = results_markdown(final)
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"图和表已写入 {out}")


if __name__ == "__main__":
    main()
