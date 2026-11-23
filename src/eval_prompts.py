"""固定 prompt 批量生成，用于实验 A（模型规模）、学习过程演变、实验 B（采样参数）。

用法：
    python src/eval_prompts.py --runs runs/s10m runs/s50m runs/s100m --grid scale --out docs/results/samples_A.jsonl
    python src/eval_prompts.py --runs runs/s100m --grid progress --out docs/results/samples_progress.jsonl
    python src/eval_prompts.py --runs runs/s100m --grid sampling --out docs/results/samples_B.jsonl

每种 grid 输出：一个 jsonl（每行一条生成）+ 同名 .md（样例表和汇总）。
scale 还会输出盲评表 docs/results/blind_eval/sheet.md 和答案 docs/results/blind_eval/key.json。
生成逻辑全部复用 sample.py 的 load_model / generate_text，这里只负责组织网格和统计。

默认关闭 KV cache，保证重跑历史样例时结果可复现；只想测生成速度时加 --use_cache。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample import generate_text, load_model  # noqa: E402

PROMPTS: dict[str, list[str]] = {
    "story": [
        "Once upon a time",
        "Lily wanted to",
        "The little dog was sad because",
        "One day, Tom and his mom went to",
        "There was a big tree in the park",
    ],
    "expository": [
        "Photosynthesis is the process",
        "The water cycle describes",
        "In mathematics, a prime number",
        "The French Revolution began",
        "To stay healthy, people should",
    ],
}

# 实验 B：每个设置一个名字 -> (temperature, top_k, top_p)；top_k/top_p 为 None 表示不限
SAMPLING_GRID: dict[str, tuple[float, int | None, float | None]] = {
    "greedy": (0.0, None, None),
    **{f"T={t}": (t, None, None) for t in (0.3, 0.7, 1.0, 1.5)},
    **{f"T={t},k=50": (t, 50, None) for t in (0.3, 0.7, 1.0, 1.5)},
    "T=1.0,p=0.9": (1.0, None, 0.9),
}


# ---------------------------------------------------------------- 自动指标


def distinct_n(text: str, n: int = 2) -> float:
    """不同的 n-gram 数 / n-gram 总数（按空格切词）。越接近 1 越不重复；复读机会很低。"""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


# ---------------------------------------------------------------- 生成


def run_generation_grid(specs: list[tuple[str, str]], prompts: list[tuple[str, str]],
                        settings: dict[str, tuple[float, int | None, float | None]],
                        seeds: list[int], max_new_tokens: int = 200,
                        use_cache: bool = False) -> list[dict]:
    """specs: [(run_dir, ckpt 文件名)]；prompts: [(类别, prompt)]。
    每个 checkpoint 只加载一次，遍历 prompt × 设置 × seed。

    use_cache 默认 False：历史样例是整段前向生成的，关闭 cache 才能原样复现。
    需要测加速效果时显式传 True。
    """
    rows = []
    for run_dir, ckpt in specs:
        model, tok, _ = load_model(run_dir, ckpt)
        n_params = model.num_params(False)
        # 第一次调用含 CUDA 初始化，不计时；预热和实际生成用同一条路径
        generate_text(model, tok, "warm up", max_new_tokens=5, use_cache=use_cache)
        for kind, prompt in prompts:
            for name, (t, k, p) in settings.items():
                for seed in seeds:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.time()
                    text = generate_text(model, tok, prompt, 1, max_new_tokens, t, k, p, seed,
                                         use_cache=use_cache)[0]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dt = time.time() - t0
                    n_tok = len(tok.encode(text))
                    rows.append({
                        "run": Path(run_dir).name, "ckpt": ckpt, "params": n_params,
                        "kind": kind, "prompt": prompt, "setting": name,
                        "temperature": t, "top_k": k, "top_p": p, "seed": seed,
                        "text": text, "new_tokens": n_tok, "seconds": round(dt, 3),
                        "tok_per_s": round(n_tok / dt, 1) if dt > 0 else None,
                        "distinct_2": round(distinct_n(text, 2), 4),
                        "use_cache": use_cache,
                    })
                    print(f"{Path(run_dir).name}/{ckpt} | {name} | seed {seed} | {prompt!r} -> "
                          f"{n_tok} tok, {dt:.2f}s, distinct-2 {rows[-1]['distinct_2']:.3f}")
        del model
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- 导出


def _cell(text: str, limit: int = 600) -> str:
    """放进 Markdown 表格：去掉换行和竖线，过长截断。"""
    s = text.replace("|", "\\|").replace("\r", " ").replace("\n", " ⏎ ")
    return s if len(s) <= limit else s[:limit] + " …"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def export_markdown(rows: list[dict], out_md: str, grid: str) -> None:
    """样例表 + 自动指标汇总。"""
    lines = [f"# 生成样例（grid = {grid}）\n", "由 `src/eval_prompts.py` 自动生成，最长 200 个新 token，"
             "遇到 `<|endoftext|>` 提前结束。`⏎` 表示换行。\n"]
    if grid == "sampling":
        lines += ["## 各采样设置的自动指标（4 个 prompt × 3 个 seed 平均）\n",
                  "| 设置 | 平均 distinct-2 | 平均生成长度（token） |", "|---|---|---|"]
        for name in dict.fromkeys(r["setting"] for r in rows):
            rs = [r for r in rows if r["setting"] == name]
            lines.append(f"| {name} | {_mean([r['distinct_2'] for r in rs]):.3f} "
                         f"| {_mean([r['new_tokens'] for r in rs]):.0f} |")
        lines.append("")
    else:
        lines += ["## 各模型的自动指标\n",
                  "| run / checkpoint | 类别 | 平均 distinct-2 | 平均生成长度 | 平均生成速度 (tok/s) |",
                  "|---|---|---|---|---|"]
        for key in dict.fromkeys((r["run"], r["ckpt"]) for r in rows):
            for kind in PROMPTS:
                rs = [r for r in rows if (r["run"], r["ckpt"]) == key and r["kind"] == kind]
                if rs:
                    lines.append(f"| {key[0]} / {key[1]} | {kind} | {_mean([r['distinct_2'] for r in rs]):.3f} "
                                 f"| {_mean([r['new_tokens'] for r in rs]):.0f} "
                                 f"| {_mean([r['tok_per_s'] for r in rs]):.0f} |")
        lines.append("")

    lines += ["## 全部样例\n"]
    for prompt in dict.fromkeys(r["prompt"] for r in rows):
        lines += [f"### `{prompt}`\n", "| 模型 / 设置 | seed | distinct-2 | 续写 |", "|---|---|---|---|"]
        for r in rows:
            if r["prompt"] == prompt:
                who = f"{r['run']} / {r['ckpt']}" if grid != "sampling" else r["setting"]
                lines.append(f"| {who} | {r['seed']} | {r['distinct_2']:.2f} | {_cell(r['text'])} |")
        lines.append("")
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def export_blind_sheet(rows: list[dict], out_md: str, key_json: str, seed: int = 2026) -> None:
    """盲评表：打乱顺序、去掉模型名，只留编号。答案（编号 -> 模型）单独存，打分完再看。"""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    key, lines = {}, [
        "# 实验 A 人工盲评表\n",
        "每条按 1–5 分打分（5 最好）：**语法**通顺度、内容**连贯**性（前后一致、不跑题）、"
        "**事实**合理性（只给说明文类打，故事类填 -）、**重复**程度（越少越高分）。",
        "三位组员各自独立填写一份，**打分前不要打开同目录下的 `key.json`**。\n",
    ]
    for i, idx in enumerate(order, 1):
        r = rows[idx]
        sid = f"S{i:02d}"
        key[sid] = {"run": r["run"], "ckpt": r["ckpt"], "prompt": r["prompt"]}
        lines += [f"## {sid}（{'故事' if r['kind'] == 'story' else '说明文'}）\n",
                  f"> **{r['prompt']}**{r['text']}\n",
                  "| 语法 | 连贯 | 事实 | 重复 |", "|---|---|---|---|", "|   |   |   |   |\n"]
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    Path(key_json).write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 命令行


def main() -> None:
    p = argparse.ArgumentParser(description="固定 prompt 批量生成")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--grid", choices=["scale", "progress", "sampling"], default="scale")
    p.add_argument("--out", required=True, help="输出 jsonl 路径；同名 .md 一起生成")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--use_cache", action="store_true",
                   help="开启 KV cache（更快；默认关闭，保证和历史样例可复现）")
    args = p.parse_args()

    if args.use_cache and args.grid == "scale":
        print("警告：--use_cache 可能改变生成文本，不要覆盖已有的 samples_A / blind_eval 结果")

    all_prompts = [(k, s) for k, ps in PROMPTS.items() for s in ps]
    default = {"T=0.8,k=50": (0.8, 50, None)}
    if args.grid == "scale":
        specs = [(r, "ckpt_best.pt") for r in args.runs]
        rows = run_generation_grid(specs, all_prompts, default, [42], args.max_new_tokens,
                                   use_cache=args.use_cache)
    elif args.grid == "progress":
        # 同一个模型训练过程中的几个 checkpoint：看"乱码 → 有词 → 有句子 → 有内容"
        run = args.runs[0]
        names = sorted(Path(run).glob("ckpt_tok*.pt"), key=lambda f: int(f.stem[8:-1]))
        specs = [(run, f.name) for f in names] + [(run, "ckpt_best.pt")]
        prompts = [("story", PROMPTS["story"][0]), ("expository", PROMPTS["expository"][0])]
        rows = run_generation_grid(specs, prompts, default, [42], args.max_new_tokens,
                                   use_cache=args.use_cache)
    else:
        prompts = [("story", s) for s in PROMPTS["story"][:2]] + \
                  [("expository", s) for s in PROMPTS["expository"][:2]]
        rows = run_generation_grid([(args.runs[0], "ckpt_best.pt")], prompts, SAMPLING_GRID,
                                   [1, 2, 3], args.max_new_tokens, use_cache=args.use_cache)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    export_markdown(rows, str(out.with_suffix(".md")), args.grid)
    if args.grid == "scale":
        blind = out.parent / "blind_eval"  # 盲评表和答案放在 jsonl 同级的 blind_eval/ 里
        blind.mkdir(exist_ok=True)
        export_blind_sheet(rows, str(blind / "sheet.md"), str(blind / "key.json"))
    print(f"共 {len(rows)} 条，写入 {out} 和 {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
