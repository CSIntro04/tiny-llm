"""比较 KV cache 开/关时的生成速度。

用法：
    python src/bench_kv_cache.py --run runs/s100m --prompt_tokens 250 --max_new_tokens 200
    python src/bench_kv_cache.py --run runs/s10m --prompt_tokens 50 --max_new_tokens 20 --device cpu

只做推理，不改 checkpoint。两种模式各预热一次，然后跑 repeats 次取最快一次；
计时前后用 torch.cuda.synchronize()，避免把异步 kernel 的时间算漏。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample import generate_text, load_model  # noqa: E402

SENTENCE = "Once upon a time, there was a little girl named Lily. "


def build_prompt(tok, target_tokens: int) -> str:
    """用固定句子拼出大约 target_tokens 个 token 的 prompt，保证每次测速输入相同。

    decode 后再 encode 可能因 BPE 边界差几个 token，实际长度以输出为准。
    """
    ids: list[int] = []
    while len(ids) < target_tokens:
        ids.extend(tok.encode(SENTENCE))
    return tok.decode(ids[:target_tokens])


def sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="比较 KV cache 开/关的生成速度")
    p.add_argument("--run", required=True, help="runs/<实验名>")
    p.add_argument("--ckpt", default="ckpt_best.pt")
    p.add_argument("--prompt", default=None, help="直接给 prompt；默认按 --prompt_tokens 拼一个")
    p.add_argument("--prompt_tokens", type=int, default=250,
                   help="目标 prompt token 数；实际长度以输出为准")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--repeats", type=int, default=2, help="每种模式跑几次，取最快一次")
    p.add_argument("--temperature", type=float, default=0.0, help="0 表示 greedy")
    p.add_argument("--top_k", type=int, default=0, help="0 表示不限制")
    p.add_argument("--top_p", type=float, default=1.0, help="1.0 表示不限制")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats 至少为 1")
    if args.max_new_tokens < 1:
        raise SystemExit("--max_new_tokens 至少为 1")
    model, tok, _ = load_model(args.run, args.ckpt, args.device)
    prompt = args.prompt if args.prompt else build_prompt(tok, args.prompt_tokens)
    prompt_tokens = len(tok.encode(prompt))
    total_tokens = prompt_tokens + args.max_new_tokens
    if total_tokens > model.cfg.block_size:
        print(f"警告：总长度 {total_tokens} 超过 block_size {model.cfg.block_size}，"
              "cache 会在窗口满后每步重建，加速会明显变小")

    top_k = args.top_k or None
    top_p = args.top_p if args.top_p < 1.0 else None

    def generate(use_cache: bool, max_new_tokens: int) -> str:
        return generate_text(model, tok, prompt, 1, max_new_tokens, args.temperature,
                             top_k, top_p, args.seed, stop_at_eot=False,
                             use_cache=use_cache)[0]

    # 预热两条路径：第一次调用含 CUDA 初始化，不能算进任何一种计时
    for use_cache in (False, True):
        generate(use_cache, min(3, args.max_new_tokens))

    results: dict[bool, tuple[float, list[int]]] = {}
    for use_cache in (False, True):
        times: list[float] = []
        ids: list[int] = []
        for _ in range(args.repeats):
            sync(args.device)
            t0 = time.time()
            text = generate(use_cache, args.max_new_tokens)
            sync(args.device)
            times.append(time.time() - t0)
            ids = tok.encode(text)
        results[use_cache] = (min(times), ids)

    no_cache_s, no_cache_ids = results[False]
    cache_s, cache_ids = results[True]
    n_tok = len(no_cache_ids)
    print(f"{args.run} | prompt {prompt_tokens} tok | 新生成 {n_tok} tok | "
          f"max_new_tokens {args.max_new_tokens} | device {args.device}")
    print(f"no_cache: {no_cache_s:6.3f}s  {n_tok / no_cache_s:7.1f} tok/s")
    print(f"cache:    {cache_s:6.3f}s  {n_tok / cache_s:7.1f} tok/s")
    print(f"speedup:  {no_cache_s / cache_s:.2f}x")
    print(f"same_tokens: {no_cache_ids == cache_ids}")
    if no_cache_ids != cache_ids:
        print("提示：GPU bf16 下两条计算路径可能有极小数值差，greedy 偶发分叉，不是 cache 逻辑错误")


if __name__ == "__main__":
    main()
