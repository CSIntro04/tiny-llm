"""阶段 2 的测试：get_batch 的形状/偏移/比例，以及分词器的往返一致性。

运行：pytest tests/test_data.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenizer import BPETokenizer, CharTokenizer, train_bpe_toy  # noqa: E402
from utils import get_batch  # noqa: E402


def _make_sources(tmp_path: Path) -> dict[str, str]:
    """两个假来源：a 的 token 是 0..9999，b 是 20000..29999，便于区分来源和检查偏移。"""
    paths = {}
    for name, start in [("a", 0), ("b", 20000)]:
        p = tmp_path / f"{name}.bin"
        np.arange(start, start + 10000, dtype=np.uint16).tofile(p)
        paths[name] = str(p)
    return paths


def test_get_batch_shape_and_shift(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    x, y = get_batch(sources, {"a": 1, "b": 1}, batch_size=8, block_size=16, device="cpu",
                     rng=np.random.default_rng(0))
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    # 数据是连续整数，所以"右移一位"等价于 y == x + 1
    assert torch.equal(y, x + 1)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_get_batch_mix_ratio(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    x, _ = get_batch(sources, {"a": 0.7, "b": 0.3}, batch_size=4000, block_size=8, device="cpu",
                     rng=np.random.default_rng(0))
    frac_a = (x[:, 0] < 20000).float().mean().item()
    assert abs(frac_a - 0.7) < 0.03
    # 一个样本只来自一个来源
    row_is_a = x < 20000
    assert torch.all(row_is_a.all(dim=1) | (~row_is_a).all(dim=1))


def test_get_batch_single_source(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    x, _ = get_batch(sources, {"b": 1.0}, batch_size=16, block_size=8, device="cpu")
    assert torch.all(x >= 20000)


CORPUS = [
    "Once upon a time, there was a little girl named Lily.",
    "The mitochondria is the powerhouse of the cell. Energy is stored as ATP.",
    "Café, naïve, “quotes” — and 中文字符 also work.\n\nNew paragraph.",
] * 50


def test_bpe_roundtrip_and_eot() -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=400)
    assert tok.eot_id == 0
    assert tok.vocab_size <= 400
    for s in CORPUS[:3] + ["unseen wörds ✓ 🙂"]:  # ByteLevel：没见过的字符也能往返
        assert tok.decode(tok.encode(s)) == s


def test_bpe_save_load(tmp_path: Path) -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=400)
    tok.save(tmp_path / "tok.json")
    tok2 = BPETokenizer.load(tmp_path / "tok.json")
    assert tok2.encode(CORPUS[1]) == tok.encode(CORPUS[1])


def test_char_tokenizer() -> None:
    tok = CharTokenizer.build(CORPUS, min_freq=10)
    assert tok.eot_id == 0
    assert tok.decode(tok.encode(CORPUS[0])) == CORPUS[0]
    assert tok.encode("☃")[0] == tok.unk_id  # 没见过的字符 -> <unk>


def test_train_bpe_toy() -> None:
    merges = train_bpe_toy("the the the then there", 3)
    assert merges[0] in {(b"t", b"h"), (b"h", b"e")}
    assert len(merges) == 3
