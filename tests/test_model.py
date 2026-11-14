"""阶段 3 的测试：参数量、形状、初始 loss、注意力等价、因果性、权重共享、梯度检查点、过拟合。

运行：pytest tests/test_model.py -v
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import GPT, MODEL_PRESETS, GPTConfig  # noqa: E402

V = 8192


def _small_cfg(**kw) -> GPTConfig:
    """测试用的小模型：debug 结构，上下文 64，足够快。"""
    return GPTConfig(**MODEL_PRESETS["debug"], block_size=64, **kw)


# 文档 docs/03-model.md 里的期望值（单位 M）：非 embedding, 总参数
EXPECTED = {
    "debug": (3.16, 5.39),
    "s10m": (10.65, 13.99),
    "s50m": (49.24, 54.81),
    "s100m": (99.23, 105.92),
}


@pytest.mark.parametrize("name", list(MODEL_PRESETS))
def test_param_count(name: str) -> None:
    with torch.device("meta"):
        model = GPT(GPTConfig(**MODEL_PRESETS[name]))
    non_emb, total = EXPECTED[name]
    assert abs(model.num_params(non_embedding=True) / 1e6 - non_emb) < 0.01
    assert abs(model.num_params(non_embedding=False) / 1e6 - total) < 0.01
    assert model.num_params(non_embedding=False) < 1.1e8  # CLAUDE.md：不超过 0.1B


def test_shapes() -> None:
    torch.manual_seed(0)
    model = GPT(_small_cfg())
    idx = torch.randint(0, V, (2, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, V) and loss is not None and loss.dim() == 0
    logits, loss = model(idx)
    assert logits.shape == (2, 1, V) and loss is None


def test_initial_loss() -> None:
    """随机初始化时模型接近均匀猜测，loss ≈ ln V。"""
    torch.manual_seed(0)
    model = GPT(_small_cfg())
    idx = torch.randint(0, V, (4, 64))
    tgt = torch.randint(0, V, (4, 64))
    _, loss = model(idx, tgt)
    assert abs(loss.item() - math.log(V)) < 0.3


def test_manual_attn_matches_sdpa() -> None:
    torch.manual_seed(0)
    sdpa = GPT(_small_cfg()).eval()
    manual = GPT(_small_cfg(manual_attn=True)).eval()
    manual.load_state_dict(sdpa.state_dict())  # 同一套权重
    idx = torch.randint(0, V, (2, 32))
    with torch.no_grad():
        a, _ = sdpa(idx, idx)
        b, _ = manual(idx, idx)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("manual", [False, True])
def test_causality(manual: bool) -> None:
    """修改位置 t 之后的 token，位置 0..t 的 logits 不应改变。"""
    torch.manual_seed(0)
    model = GPT(_small_cfg(manual_attn=manual)).eval()
    idx = torch.randint(0, V, (1, 32))
    t = 10
    idx2 = idx.clone()
    idx2[0, t + 1:] = torch.randint(0, V, (32 - t - 1,))
    with torch.no_grad():
        a, _ = model(idx, idx)
        b, _ = model(idx2, idx2)
    assert torch.equal(a[:, : t + 1], b[:, : t + 1])
    assert not torch.allclose(a[:, t + 1:], b[:, t + 1:])  # 后面确实变了，说明测试有效


def test_weight_tying() -> None:
    model = GPT(_small_cfg())
    assert model.lm_head.weight is model.wte.weight
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()


def test_grad_ckpt_same_grads() -> None:
    torch.manual_seed(0)
    a = GPT(_small_cfg())
    b = GPT(_small_cfg(grad_ckpt=True))
    b.load_state_dict(a.state_dict())
    idx = torch.randint(0, V, (2, 32))
    _, la = a(idx, idx)
    _, lb = b(idx, idx)
    la.backward()
    lb.backward()
    assert torch.allclose(la, lb)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6)


def test_overfit_single_batch() -> None:
    """CLAUDE.md 要求的过拟合测试：固定一个小 batch，loss 应该能降到接近 0。"""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(_small_cfg()).to(device)
    data = torch.randint(0, V, (4, 65), device=device)
    x, y = data[:, :-1], data[:, 1:]  # (4, 64)
    opt = model.configure_optimizers(weight_decay=0.0, lr=1e-3, betas=(0.9, 0.95))
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"final loss {loss.item():.4f}"
