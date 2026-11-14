"""训练和评估共用的小工具：数据加载、配置读取、随机种子、学习率调度、日志、画图。"""

from __future__ import annotations

import csv
import math
import os
import random

import numpy as np
import torch
import yaml


def get_batch(
    sources: dict[str, str],
    weights: dict[str, float],
    batch_size: int,
    block_size: int,
    device: str,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机取一个训练 batch。返回 x, y，形状都是 (B, T)，dtype int64。

    sources: 来源名 -> .bin 文件路径（np.uint16 的 token id 序列）
    weights: 来源名 -> 抽样权重（不必归一化）；只在 weights 里出现的来源会被抽到

    每个样本：先按 weights 抽一个来源，再在该来源里随机取 block_size+1 个连续 token：
        t = data[i : i+T+1]
        x = t[0 : T]      输入
        y = t[1 : T+1]    目标 = x 右移一位，即"每个位置都去预测下一个 token"
    一个样本只来自一个来源，不跨来源拼接。

    为什么每次调用都重新创建 memmap：np.memmap 只把用到的部分读进内存，
    但一直持有同一个 memmap 时，被访问过的页会累积，长时间训练内存会慢慢涨（nanoGPT 的经验）。
    """
    rng = rng if rng is not None else np.random.default_rng()
    names = list(weights)
    p = np.array([weights[n] for n in names], dtype=np.float64)
    choice = rng.choice(len(names), size=batch_size, p=p / p.sum())  # (B,) 每个样本的来源下标

    rows: list[np.ndarray] = []
    for k in choice:
        data = np.memmap(sources[names[k]], dtype=np.uint16, mode="r")
        i = int(rng.integers(0, len(data) - block_size))  # 保证 i+T+1 <= len(data)
        rows.append(data[i : i + block_size + 1].astype(np.int64))  # (T+1,)
    t = torch.from_numpy(np.stack(rows))  # (B, T+1)；nn.Embedding 需要 int64
    x, y = t[:, :-1], t[:, 1:]  # 都是 (B, T)

    if device.startswith("cuda"):
        # pin_memory：放进锁页内存，之后 non_blocking 拷贝可以和 GPU 计算重叠
        x = x.contiguous().pin_memory().to(device, non_blocking=True)
        y = y.contiguous().pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.contiguous().to(device), y.contiguous().to(device)
    return x, y


# ---------------------------------------------------------------- 配置


def load_config(path: str, overrides: list[str]) -> dict:
    """读 yaml 配置，再用命令行覆盖。overrides 形如 ["--max_iters", "50", "--name", "x"]。

    键名先在顶层找，再在 train / model / data 小节里找；值用 yaml.safe_load 转类型（"50" -> 50）。
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert len(overrides) % 2 == 0, f"覆盖参数要成对出现：{overrides}"
    for flag, raw in zip(overrides[::2], overrides[1::2]):
        key = flag.lstrip("-")
        value = yaml.safe_load(raw)
        if isinstance(value, str):  # PyYAML 遵循 YAML 1.1："3e-4"（没有小数点）会被当成字符串
            try:
                value = float(value)
            except ValueError:
                pass
        if key in cfg and not isinstance(cfg[key], dict):
            cfg[key] = value
            continue
        for section in ("train", "model", "data"):
            if key in cfg.get(section, {}):
                cfg[section][key] = value
                break
        else:
            raise KeyError(f"配置里没有键 {key}")
    return cfg


def set_seed(seed: int) -> None:
    """固定所有随机源。GPU 上部分算子不保证逐位复现，但曲线应大致一致。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_mix(cfg: dict, meta: dict) -> dict[str, float]:
    """混合权重（归一化）。mix: natural -> 按各来源训练 token 数的比例（prepare.py 写在 meta.json）。"""
    mix = cfg["data"]["mix"]
    weights = meta["mix_weights"] if mix == "natural" else mix
    w = {s: float(weights[s]) for s in cfg["data"]["sources"]}
    total = sum(w.values())
    return {s: v / total for s, v in w.items()}


# ---------------------------------------------------------------- 学习率


def get_lr(it: int, cfg: dict) -> float:
    """warmup + cosine 学习率。cfg 是配置里的 train 小节。

    - 前 warmup_iters 步：从 0 线性升到 lr。刚开始参数是随机的、梯度方向不可靠，
      Adam 的二阶矩估计也还不准，一上来就用大步长容易发散。
    - 之后：按余弦曲线从 lr 平滑降到 min_lr。后期用小步长在低谷附近精细收敛。
    """
    lr, min_lr = cfg["lr"], cfg["min_lr"]
    warmup, max_iters = cfg["warmup_iters"], cfg["max_iters"]
    if it < warmup:
        return lr * (it + 1) / warmup
    if it >= max_iters:
        return min_lr
    progress = (it - warmup) / max(1, max_iters - warmup)  # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------- 日志与画图


class CSVLogger:
    """逐行追加写 CSV。文件已存在（续训）时不重复写表头。"""

    def __init__(self, path: str, fields: list[str]) -> None:
        self.path = path
        self.fields = fields
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(
                {k: row.get(k, "") for k in self.fields}
            )


def _read_log(log_path: str) -> dict[str, list[tuple[float, float, float]]]:
    """读 log.csv，返回 列名 -> [(step, tokens, 值)]，跳过空值。"""
    cols = ["train_loss", "val_loss", "val_smollm", "val_tinystories"]
    out: dict[str, list[tuple[float, float, float]]] = {c: [] for c in cols}
    with open(log_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for c in cols:
                if row.get(c):
                    out[c].append((float(row["step"]), float(row["tokens"]), float(row[c])))
    return out


def plot_loss(log_path: str, out_png: str) -> None:
    """画 loss 曲线：左图横轴 step，右图横轴已训练 token 数。"""
    import matplotlib

    matplotlib.use("Agg")  # 无界面后端，WSL / 服务器上也能存图
    import matplotlib.pyplot as plt

    data = _read_log(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    styles = {
        "train_loss": dict(label="train", alpha=0.5, lw=1),
        "val_loss": dict(label="val (mix)", lw=2),
        "val_smollm": dict(label="val smollm", ls="--"),
        "val_tinystories": dict(label="val tinystories", ls="--"),
    }
    for ax, xi, xlabel in [(axes[0], 0, "step"), (axes[1], 1, "tokens")]:
        for col, st in styles.items():
            pts = data[col]
            if pts:
                ax.plot([p[xi] for p in pts], [p[2] for p in pts], **st)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(os.path.basename(os.path.dirname(os.path.abspath(log_path))))
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
