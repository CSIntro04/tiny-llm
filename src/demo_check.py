"""答辩前自检：确认三个正式模型都在、能加载、生成结果和实验记录一致，并测生成速度。

用法：
    python src/demo_check.py                 # 有 GPU 用 GPU，没有自动用 CPU
    python src/demo_check.py --device cpu    # 有 GPU 也强制用 CPU，测兜底方案的速度

检查项：
1. CUDA 是否可用、显卡型号；
2. runs/{s10m,s50m,s100m}/ckpt_best.pt 和分词器是否存在；
3. 每个模型用 docs/results/samples_A.jsonl 里同样的 prompt、采样参数和 seed 生成，
   前 N 个 token 应与当时记录的样例相同——确认没有拿错 checkpoint（例如拿了 ckpt_tok100M）。
   对比时必须优先使用 no-cache：历史样例是整段前向生成的。cache 的增量计算在 bf16 下可能
   产生极小数值差；换显卡/换机器时 no-cache 也可能因内核不同而分叉。因此只要 no-cache
   或 cache 任一路径能复现参考样例，就说明 checkpoint 没拿错；两条都分叉才算失败。
   cache 同时用于单独测速。CPU 的浮点运算顺序不同，分叉时只提示不算失败。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample import generate_text, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["s10m", "s50m", "s100m"]
PROMPT = "Once upon a time"
_loaded: list = []  # 保留模型引用，最后统计三个模型同时在显存里的占用（和界面 --preload all 一样）


def reference(run: str) -> dict:
    """samples_A.jsonl 里该模型、该 prompt 的记录（T=0.8, top-k=50, seed=42）。"""
    for line in (ROOT / "docs/results/samples_A.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["run"] == run and r["prompt"] == PROMPT:
            return r
    raise KeyError(f"samples_A.jsonl 里没有 {run} / {PROMPT!r}")


def check_run(run: str, device: str, n_tokens: int) -> bool:
    ckpt = ROOT / "runs" / run / "ckpt_best.pt"
    if not ckpt.exists():
        print(f"[失败] {ckpt} 不存在：从网盘下载 tiny-llm-weights.tar 后 tar xf ... -C runs/")
        return False
    t0 = time.time()
    model, tok, _ = load_model(str(ckpt.parent), device=device)
    load_s = time.time() - t0
    _loaded.append(model)
    ref = reference(run)
    # 两条路径都预热一次：第一次调用含 CUDA 初始化，不能算进任何一种计时
    generate_text(model, tok, "warm up", max_new_tokens=5, use_cache=False)
    generate_text(model, tok, "warm up", max_new_tokens=5, use_cache=True)

    # 和 samples_A.jsonl 对比必须用 no-cache，复现当时“整段前向”的生成结果
    t0 = time.time()
    text_ref = generate_text(model, tok, PROMPT, 1, n_tokens, ref["temperature"], ref["top_k"],
                             ref["top_p"], ref["seed"], use_cache=False)[0]
    ref_s = time.time() - t0

    # cache 只用于单独测速；它的输出可能因 bf16 数值差和参考在个别 token 分叉
    t0 = time.time()
    text_cache = generate_text(model, tok, PROMPT, 1, n_tokens, ref["temperature"], ref["top_k"],
                               ref["top_p"], ref["seed"], use_cache=True)[0]
    cache_s = time.time() - t0

    # 比较 token 而不是字符：生成 n 个 token 再解码，前缀应和参考样例的前 n 个 token 一致
    want = tok.encode(ref["text"])[:n_tokens]
    same_ref = tok.encode(text_ref)[:n_tokens] == want
    same_cache = tok.encode(text_cache)[:n_tokens] == want
    same = same_ref or same_cache
    if same_ref and same_cache:
        status = "一致（no-cache 和 cache）"
    elif same_ref:
        status = "一致（no-cache；cache 数值分叉）"
    elif same_cache:
        status = "一致（cache；no-cache 数值分叉）"
    else:
        status = "不一致（CPU 上可能正常分叉）" if device == "cpu" else "不一致！"
    print(f"[{'通过' if same or device == 'cpu' else '失败'}] {run:<6} 加载 {load_s:4.1f}s | "
          f"参考(no cache) {n_tokens} token {ref_s:4.1f}s（{n_tokens / ref_s:5.0f} tok/s） | "
          f"cache {cache_s:4.1f}s（{n_tokens / cache_s:5.0f} tok/s） | 与样例 {status}")
    preview = (PROMPT + text_cache[:90]).replace("\n", " ")
    print(f"         {preview}…")
    return same or device == "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="答辩前自检")
    # 默认：有 GPU 用 GPU，没有就用 CPU（Mac、无独显的电脑）
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_tokens", type=int, default=50)
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[失败] CUDA 不可用：检查 conda activate tiny-llm、nvidia-smi；"
                  "或改用 --device cpu 走兜底方案")
            sys.exit(1)
        print(f"GPU：{torch.cuda.get_device_name(0)}，torch {torch.__version__}")
    ok = all([check_run(r, args.device, args.n_tokens) for r in RUNS])
    if args.device.startswith("cuda"):
        print(f"显存占用（三个模型都在显存里）：{torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("全部通过，可以启动：python app/server.py --preload all（备用：python app/app.py --preload all）" if ok else "有检查项失败，见上")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
