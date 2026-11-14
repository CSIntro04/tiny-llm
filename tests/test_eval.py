"""阶段 6 的测试：distinct-n、bpb 换算、整段 NLL 统计、盲评表可还原。

运行：pytest tests/test_eval.py -v
"""

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_prompts import distinct_n, export_blind_sheet  # noqa: E402
from evaluate import bits_per_byte, total_nll  # noqa: E402
from model import GPT, MODEL_PRESETS, GPTConfig  # noqa: E402


def test_distinct_n() -> None:
    assert distinct_n("a b c d e", 2) == 1.0  # 4 个 bigram 全不同
    assert distinct_n("a a a a a", 2) == 0.25  # 4 个 bigram 只有 1 种
    assert distinct_n("one", 2) == 0.0  # 不够组成 bigram


def test_bits_per_byte_uniform() -> None:
    # 对 256 个字节均匀猜：每个字节 ln(256) nats = 8 bits
    n = 1000
    assert math.isclose(bits_per_byte(n * math.log(256), n), 8.0)


def test_total_nll_counts_every_prediction() -> None:
    torch.manual_seed(0)
    model = GPT(GPTConfig(**MODEL_PRESETS["debug"], vocab_size=64, block_size=16)).eval()
    ids = torch.randint(0, 64, (100,)).tolist()
    nll, count = total_nll(model, ids, batch_size=4)
    assert count == len(ids) - 1  # 除第一个 token 外每个都被预测恰好一次
    # 随机初始化的模型接近均匀分布：平均 loss ≈ ln(64)
    assert abs(nll / count - math.log(64)) < 0.3


def test_blind_sheet_hides_and_restores(tmp_path: Path) -> None:
    rows = [{"run": f"m{i % 3}", "ckpt": "ckpt_best.pt", "kind": "story",
             "prompt": f"p{i}", "text": f" text {i}"} for i in range(6)]
    md, key = tmp_path / "sheet.md", tmp_path / "key.json"
    export_blind_sheet(rows, str(md), str(key))
    sheet = md.read_text(encoding="utf-8")
    assert all(f"m{j}" not in sheet for j in range(3))  # 表里看不到模型名
    mapping = json.loads(key.read_text(encoding="utf-8"))
    assert sorted(v["prompt"] for v in mapping.values()) == sorted(r["prompt"] for r in rows)
    for v in mapping.values():  # 每个编号都能还原到正确的模型
        assert v["run"] == f"m{int(v['prompt'][1:]) % 3}"
