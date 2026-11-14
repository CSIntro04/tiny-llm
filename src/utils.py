"""训练和评估共用的小工具。目前只有数据加载 get_batch。"""

from __future__ import annotations

import numpy as np
import torch


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
