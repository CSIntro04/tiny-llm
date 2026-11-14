"""环境自检脚本：确认 PyTorch 能在 RTX 5060 Ti（sm_120）上正常使用 bf16、SDPA 和 torch.compile。

用法：python src/check_env.py
每一项打印 [OK] 或 [FAIL]，任一项失败时退出码为 1。
"""

import math
import sys
from typing import Callable

import torch
import torch.nn.functional as F


def _version_tuple(v: str) -> tuple[int, ...]:
    """'2.9.1+cu129' -> (2, 9, 1)，只取前面的数字部分，方便比较大小。"""
    parts = []
    for p in v.split("+")[0].split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_torch() -> str:
    """检查 torch 版本、CUDA 版本，以及编译时是否包含 sm_120 的 kernel。"""
    assert _version_tuple(torch.__version__) >= (2, 7), f"torch {torch.__version__} < 2.7"
    assert torch.version.cuda is not None, "这是 CPU 版 PyTorch（torch.version.cuda 为 None）"
    assert _version_tuple(torch.version.cuda) >= (12, 8), f"CUDA {torch.version.cuda} < 12.8"
    assert torch.cuda.is_available(), "torch.cuda.is_available() 为 False，检查驱动"
    # arch list 是 PyTorch 编译时生成 kernel 的目标架构；没有 sm_120 会报 "no kernel image"
    archs = torch.cuda.get_arch_list()
    assert "sm_120" in archs, f"arch list 不含 sm_120：{archs}"
    return f"torch {torch.__version__}, CUDA {torch.version.cuda}, arch list contains sm_120"


def check_gpu() -> str:
    """检查显卡名称、显存、compute capability（Blackwell 消费卡是 12.0）。"""
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    mem_gb = props.total_memory / 1024**3
    assert cap == (12, 0), f"compute capability {cap} != (12, 0)"
    return f"{props.name}, {mem_gb:.1f} GB, capability {cap}"


def check_bf16() -> str:
    """bf16 矩阵乘法：autocast 下结果应与 fp32 接近。

    bf16 有 8 位指数（和 fp32 一样，数值范围大，不容易溢出），但只有 7 位尾数，
    所以精度较低。这里用相对误差判断。
    """
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    ref = a @ b  # fp32 参考结果
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = a @ b  # autocast 自动把 matmul 转成 bf16 计算
    assert out.dtype == torch.bfloat16, f"autocast 输出类型是 {out.dtype}"
    rel_err = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel_err < 1e-2, f"相对误差 {rel_err:.2e} 过大"
    return f"bf16 matmul, relative error vs fp32 = {rel_err:.2e}"


def _naive_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """手写因果注意力，用来核对 SDPA。输入输出形状都是 (B, nh, T, hs)。"""
    T, hs = q.size(-2), q.size(-1)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)  # (B, nh, T, T)
    # 因果掩码：位置 i 只能看到 j <= i，上三角（未来）填 -inf，softmax 后权重为 0
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    att = att.masked_fill(~mask, float("-inf"))
    att = F.softmax(att, dim=-1)
    return att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)


def check_sdpa() -> str:
    """F.scaled_dot_product_attention(is_causal=True) 能跑、结果正确、满足因果性。"""
    torch.manual_seed(0)
    B, nh, T, hs = 2, 4, 128, 64
    q, k, v = (torch.randn(B, nh, T, hs, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert out.shape == (B, nh, T, hs), f"输出形状 {tuple(out.shape)}"

    # 1) 与手写版本（fp32 计算）对比
    ref = _naive_causal_attention(q.float(), k.float(), v.float())
    max_diff = (out.float() - ref).abs().max().item()
    assert max_diff < 5e-2, f"与手写注意力最大差 {max_diff:.2e}"

    # 2) 因果性：改动最后一半位置的 k、v，前一半位置的输出不应改变
    k2, v2 = k.clone(), v.clone()
    k2[:, :, T // 2 :] = torch.randn_like(k2[:, :, T // 2 :])
    v2[:, :, T // 2 :] = torch.randn_like(v2[:, :, T // 2 :])
    out2 = F.scaled_dot_product_attention(q, k2, v2, is_causal=True)
    assert torch.equal(out[:, :, : T // 2], out2[:, :, : T // 2]), "未来位置影响了过去的输出"
    return f"SDPA causal attention, max diff vs naive = {max_diff:.2e}, causality OK"


def check_compile() -> str:
    """可选：torch.compile 一个小函数，结果与 eager 模式一致（需要 Triton，WSL2/Linux 可用）。"""

    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    compiled = torch.compile(silu)
    x = torch.randn(4096, device="cuda")
    max_diff = (compiled(x) - silu(x)).abs().max().item()
    assert max_diff < 1e-5, f"compile 结果与 eager 最大差 {max_diff:.2e}"
    return "torch.compile"


def main() -> None:
    checks: list[Callable[[], str]] = [check_torch, check_gpu, check_bf16, check_sdpa, check_compile]
    failed = 0
    for check in checks:
        try:
            print(f"[OK] {check()}")
        except Exception as e:  # 自检脚本：捕获所有错误，继续跑后面的项
            failed += 1
            print(f"[FAIL] {check.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{failed} 项失败")
        sys.exit(1)
    print("\n全部通过")


if __name__ == "__main__":
    main()
