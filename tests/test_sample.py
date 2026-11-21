"""阶段 5 的测试：top-k / top-p 只抽允许的 token、greedy 确定性、seed 可复现、长 prompt 截断、eot 停止。

运行：pytest tests/test_sample.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import GPT, MODEL_PRESETS, GPTConfig  # noqa: E402
from sample import generate, sample_next  # noqa: E402

V = 512


@pytest.fixture(scope="module")
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(**MODEL_PRESETS["debug"], vocab_size=V, block_size=32)).eval()


def test_top_k_only_allowed() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, V)
    allowed = torch.topk(logits, 5, dim=-1).indices  # (4, 5)
    for _ in range(200):
        nxt = sample_next(logits, temperature=1.5, top_k=5, top_p=None)  # (4, 1)
        assert (nxt == allowed).any(dim=-1).all()


def test_top_p_only_nucleus() -> None:
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    # p=0.7：0.5 不够，加上 0.3 后超过 → nucleus = {0, 1}
    seen = {int(sample_next(logits, 1.0, None, 0.7)) for _ in range(500)}
    assert seen == {0, 1}
    # p 极小：至少保留概率最高的那个
    seen = {int(sample_next(logits, 1.0, None, 1e-6)) for _ in range(100)}
    assert seen == {0}


def test_greedy_equals_top_k_1(model: GPT) -> None:
    idx = torch.randint(0, V, (2, 8))
    torch.manual_seed(1)
    a = generate(model, idx, 20, temperature=0)
    torch.manual_seed(2)
    b = generate(model, idx, 20, temperature=0)
    torch.manual_seed(3)
    c = generate(model, idx, 20, temperature=1.0, top_k=1)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_same_seed_same_output(model: GPT) -> None:
    idx = torch.randint(0, V, (3, 4))
    torch.manual_seed(42)
    a = generate(model, idx, 20, temperature=1.0, top_k=50, top_p=0.9)
    torch.manual_seed(42)
    b = generate(model, idx, 20, temperature=1.0, top_k=50, top_p=0.9)
    assert torch.equal(a, b)


def test_long_prompt_truncated(model: GPT) -> None:
    idx = torch.randint(0, V, (1, 100))  # 远超 block_size=32
    out = generate(model, idx, 10, temperature=1.0)
    assert out.shape == (1, 110)


def test_stops_at_eot(model: GPT) -> None:
    idx = torch.randint(0, V, (1, 4))
    first = int(generate(model, idx, 1, temperature=0)[0, -1])  # greedy 的第一个 token 当作 eot
    out = generate(model, idx, 50, temperature=0, eot_id=first)
    assert out.shape == (1, 5)


@pytest.mark.parametrize("t0", [1, 8])
def test_generate_cache_matches_no_cache(model: GPT, t0: int) -> None:
    """有/无 KV cache 的 greedy 输出必须逐 token 相同；t0=1 覆盖单 token prompt。"""
    idx = torch.randint(0, V, (2, t0))
    no_cache = generate(model, idx, 20, temperature=0, use_cache=False)
    cached = generate(model, idx, 20, temperature=0, use_cache=True)
    assert torch.equal(no_cache, cached)


def test_generate_cache_long_prompt(model: GPT) -> None:
    """prompt 超过 block_size=32：cache 要重建窗口，结果仍与无 cache 一致。"""
    idx = torch.randint(0, V, (1, 100))
    no_cache = generate(model, idx, 20, temperature=0, use_cache=False)
    cached = generate(model, idx, 20, temperature=0, use_cache=True)
    assert cached.shape == (1, 120)
    assert torch.equal(no_cache, cached)


def test_generate_cache_batch_stops_at_eot(model: GPT) -> None:
    """B>1 且提前遇到 eot 时，cache 和 no-cache 的停止位置一致。"""
    idx = torch.randint(0, V, (3, 4))
    first = int(generate(model, idx, 1, temperature=0)[0, -1])
    no_cache = generate(model, idx, 30, temperature=0, eot_id=first, use_cache=False)
    cached = generate(model, idx, 30, temperature=0, eot_id=first, use_cache=True)
    assert torch.equal(no_cache, cached)


def test_same_seed_same_output_with_cache(model: GPT) -> None:
    idx = torch.randint(0, V, (2, 4))
    torch.manual_seed(42)
    a = generate(model, idx, 20, temperature=1.0, top_k=50, top_p=0.9, use_cache=True)
    torch.manual_seed(42)
    b = generate(model, idx, 20, temperature=1.0, top_k=50, top_p=0.9, use_cache=True)
    assert torch.equal(a, b)
