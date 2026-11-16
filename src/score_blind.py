"""汇总实验 A 的人工盲评：读组员填好的打分表（人数不限），按答案还原模型，求平均。

用法：
    python src/score_blind.py --sheets docs/results/blind_eval/R*.md --flip_repeat R3 --out docs/results/blind_eval/summary.md

--flip_repeat：某位打分人把"重复"一项理解反了（重复越多分越高）时，把他的这一项换算成 6 − x。
判断依据是这一项和自动指标 distinct-2 的相关系数：正常应为正，反了会是明显的负数（脚本会打印出来）。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

DIMS = ["语法", "连贯", "事实", "重复"]
RUNS = ["s10m", "s50m", "s100m"]


def parse_sheet(path: Path) -> dict[str, list[float | None]]:
    """编号 -> 四项分数。不在 1–5 范围内的（空、"-"、0 等）记为 None。"""
    out: dict[str, list[float | None]] = {}
    cur = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"## (S\d\d)", line)
        if m:
            cur = m.group(1)
            continue
        if cur is None or not line.startswith("|") or "语法" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-:") for c in cells):  # 表格分隔行 |---|---|
            continue
        vals = []
        for c in cells[:4]:
            v = float(c) if re.fullmatch(r"\d+(\.\d+)?", c) else None
            vals.append(v if v is not None and 1 <= v <= 5 else None)
        out[cur] = vals
    return out


def corr(a: list[float], b: list[float]) -> float:
    """皮尔逊相关系数。"""
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den


def main() -> None:
    p = argparse.ArgumentParser(description="汇总人工盲评")
    p.add_argument("--sheets", nargs="+", required=True)
    p.add_argument("--key", default="docs/results/blind_eval/key.json")
    p.add_argument("--samples", default="docs/results/samples_A.jsonl")
    p.add_argument("--flip_repeat", nargs="*", default=[], help="需要把'重复'换算成 6−x 的打分人（文件名不含 .md）")
    p.add_argument("--out", default="docs/results/blind_eval/summary.md")
    args = p.parse_args()

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    samples = {(r["run"], r["prompt"]): r for r in map(json.loads, open(args.samples, encoding="utf-8"))}
    kind = {sid: samples[(v["run"], v["prompt"])]["kind"] for sid, v in key.items()}
    d2 = {sid: samples[(v["run"], v["prompt"])]["distinct_2"] for sid, v in key.items()}

    scores: dict[str, dict[str, list[float | None]]] = {}
    lines = ["# 实验 A 人工盲评汇总\n", "由 `src/score_blind.py` 生成。\n", "## 打分质量检查\n",
             "| 打分人 | 有效条数 | '重复'分与 distinct-2 的相关系数（原始） | 处理 |", "|---|---|---|---|"]
    for path in map(Path, args.sheets):
        r = path.stem
        s = parse_sheet(path)
        ids = [i for i in sorted(key) if s.get(i) and s[i][3] is not None]
        c = corr([s[i][3] for i in ids], [d2[i] for i in ids])
        note = "—"
        if r in args.flip_repeat:
            for v in s.values():
                v[3] = 6 - v[3] if v[3] is not None else None
            note = "'重复'一项按 6−x 换算"
        missing = sum(v is None for i in key for j, v in enumerate(s.get(i, [None] * 4))
                      if not (j == 2 and kind[i] == "story"))
        if missing:
            note += f"；{missing} 个格子缺失或越界，不计入"
        lines.append(f"| {r} | {len(s)} | {c:+.2f} | {note} |")
        print(f"{r}: 相关系数 {c:+.2f}，{note}")
        scores[r] = s

    # 每条样本先对所有打分人取平均，再按 模型 × 类别 平均
    def avg(run: str, k: str, dim: int) -> tuple[float, int]:
        per_item = []
        for sid, v in key.items():
            if v["run"] != run or kind[sid] != k:
                continue
            vals = [scores[r][sid][dim] for r in scores if scores[r].get(sid) and scores[r][sid][dim] is not None]
            if vals:
                per_item.append(st.mean(vals))
        return (st.mean(per_item) if per_item else float("nan")), len(per_item)

    lines += [f"\n## 平均分（1–5，越高越好；每条先对 {len(scores)} 人取平均，再对 5 个 prompt 平均）\n",
              "| 模型 | 故事：语法 | 故事：连贯 | 故事：重复 | 说明文：语法 | 说明文：连贯 | 说明文：事实 | 说明文：重复 |",
              "|---|---|---|---|---|---|---|---|"]
    cols = [("story", 0), ("story", 1), ("story", 3),
            ("expository", 0), ("expository", 1), ("expository", 2), ("expository", 3)]
    for run in RUNS:
        lines.append(f"| {run} | " + " | ".join(f"{avg(run, k, d)[0]:.2f}" for k, d in cols) + " |")

    # 打分人之间的一致性：每两人在同一项上的相关系数
    pairs = [(a, b) for i, a in enumerate(scores) for b in list(scores)[i + 1:]]
    lines += ["\n## 打分人之间的一致性（同一项分数的相关系数，换算后）\n",
              "| 项目 | " + " | ".join(f"{a}–{b}" for a, b in pairs) + " |",
              "|---|" + "---|" * len(pairs)]
    for j, dim in enumerate(DIMS):
        row = []
        for a, b in pairs:
            ids = [s for s in key if not (j == 2 and kind[s] == "story")
                   and scores[a][s][j] is not None and scores[b][s][j] is not None]
            row.append(f"{corr([scores[a][s][j] for s in ids], [scores[b][s][j] for s in ids]):+.2f}")
        lines.append(f"| {dim} | " + " | ".join(row) + " |")
    lines.append("\n故事类不评'事实'（表上要求填 -），有人填了也不计入。")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
