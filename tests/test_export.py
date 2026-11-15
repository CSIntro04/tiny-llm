"""阶段 8 的测试：导出的仅权重 checkpoint 能被 load_model 加载，输出与原 checkpoint 逐位相同。

运行：pytest tests/test_export.py -v
"""

import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from export import KEEP_KEYS, export_run  # noqa: E402
from sample import load_model  # noqa: E402
from train import build_model  # noqa: E402
from utils import load_config  # noqa: E402


def _fake_run(run_dir: Path) -> None:
    """造一个 debug 规模的 run：随机权重 + 假的优化器状态，结构与 train.save_checkpoint 相同。"""
    cfg = load_config(str(ROOT / "configs/debug.yaml"), [])
    cfg["data"]["dir"] = str(ROOT / "data/processed")
    model = build_model(cfg)
    run_dir.mkdir(parents=True)
    meta = json.loads((ROOT / "data/processed/meta.json").read_text(encoding="utf-8"))
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    torch.save({"model": model.state_dict(), "optimizer": {"state": {"m": torch.zeros(10)}},
                "iter": 7, "best_val": 3.21, "config": cfg, "rng": {}, "meta": meta,
                "data_rng": {}, "elapsed_s": 1.0}, run_dir / "ckpt_best.pt")


def test_export_roundtrip(tmp_path: Path) -> None:
    run = tmp_path / "runs/fake"
    _fake_run(run)
    out = export_run(run, tmp_path / "release")

    slim = torch.load(out / "ckpt_best.pt", map_location="cpu", weights_only=False)
    assert set(slim) <= set(KEEP_KEYS) and "optimizer" not in slim
    assert slim["iter"] == 7 and slim["best_val"] == 3.21
    tok_name = slim["meta"]["tokenizer_file"]
    assert (out / tok_name).exists()  # 分词器一起导出，目录可单独使用

    # 导出前后同一输入的 logits 逐位相同（权重保持 fp32，没有任何转换）
    m1, tok1, _ = load_model(str(run), device="cpu")
    m2, tok2, _ = load_model(str(out), device="cpu")
    idx = torch.tensor([tok1.encode("Once upon a time")])
    with torch.no_grad():
        assert torch.equal(m1(idx)[0], m2(idx)[0])
    assert tok1.encode("hello world") == tok2.encode("hello world")
    shutil.rmtree(run)  # 原 run 删掉后导出目录仍能加载
    load_model(str(out), device="cpu")
