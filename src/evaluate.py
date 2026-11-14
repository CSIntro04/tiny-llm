"""对 checkpoint 做统一评估（实验 A 的"相同 token 数"比较、实验 C 的 bits-per-byte）。

用法：
    # 1) 同一批验证数据上评估多个 checkpoint 的 loss（只适用于同一个分词器的 run）
    python src/evaluate.py loss --ckpts runs/s10m/ckpt_tok300M.pt runs/s50m/ckpt_tok300M.pt ... \
        --out docs/figures/ckpt_eval.md
    # 2) 不同分词器之间比较：bits-per-byte
    python src/evaluate.py bpb --runs runs/s10m runs/s10m_char --out docs/figures/bpb.md

为什么要单独评估：训练时每 200 步才 eval 一次，3 亿 token 处最近的一次 eval 在 2.88 亿；
而 ckpt_tok300M.pt 正好存在 3 亿 token 处。重新在完全相同的数据上算 loss，比较才公平。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample import load_model  # noqa: E402
from tokenizer import BPETokenizer, load_tokenizer  # noqa: E402
from utils import get_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["smollm", "tinystories"]


# ---------------------------------------------------------------- 1. 相同数据上的 val loss


@torch.no_grad()
def val_loss(model: torch.nn.Module, data_dir: Path, n_batches: int, batch_size: int,
             seed: int = 0) -> dict[str, float]:
    """每个来源的 val.bin 上取 n_batches 个随机窗口求平均 loss。
    rng 用固定种子 → 所有 checkpoint 看到的是同一批窗口。"""
    device = next(model.parameters()).device.type
    T = model.cfg.block_size
    out = {}
    for s in SOURCES:
        rng = np.random.default_rng(seed)
        path = {s: str(data_dir / f"{s}_val.bin")}
        total = 0.0
        for _ in range(n_batches):
            x, y = get_batch(path, {s: 1.0}, batch_size, T, device, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            total += loss.item()
        out[f"val_{s}"] = total / n_batches
    return out


def run_loss(ckpts: list[str], n_batches: int, batch_size: int, mix: dict[str, float]) -> list[dict]:
    rows = []
    for c in ckpts:
        path = Path(c)
        model, _, cfg = load_model(str(path.parent), path.name)
        state_iter = torch.load(path, map_location="cpu", weights_only=False)["iter"]
        tokens = state_iter * cfg["train"]["batch_size"] * cfg["train"]["grad_accum"] * cfg["model"]["block_size"]
        r = val_loss(model, ROOT / cfg["data"]["dir"], n_batches, batch_size)
        r["val"] = sum(mix[s] * r[f"val_{s}"] for s in SOURCES)
        rows.append({"run": path.parent.name, "ckpt": path.name, "tokens": tokens,
                     "params": model.num_params(False), **r})
        print(f"{c}: tokens={tokens / 1e6:.0f}M " + " ".join(f"{k}={v:.4f}" for k, v in r.items()))
        del model
        torch.cuda.empty_cache()
    return rows


def loss_markdown(rows: list[dict], n_tokens: int) -> str:
    head = ("| run | checkpoint | 已训练 token | val（加权） | val SmolLM | val TinyStories |\n"
            "|---|---|---|---|---|---|\n")
    body = "\n".join(
        f"| {r['run']} ({r['params'] / 1e6:.0f}M) | {r['ckpt']} | {r['tokens'] / 1e6:.0f}M "
        f"| {r['val']:.3f} | {r['val_smollm']:.3f} | {r['val_tinystories']:.3f} |" for r in rows)
    return head + body + f"\n\n每个来源在同一批 {n_tokens / 1e6:.2f}M 个验证 token 上评估（固定随机窗口）。\n"


# ---------------------------------------------------------------- 2. bits-per-byte


def val_texts(n_tokens: int) -> dict[str, list[str]]:
    """从 BPE 版的 val.bin 取前 n_tokens 个 token，按 <|endoftext|> 切成文档并解码回原文。
    两种分词器都在这同一段原文上评估，才能比较。"""
    bpe_dir = ROOT / "data/processed"
    tok = load_tokenizer(bpe_dir / "meta.json")
    assert isinstance(tok, BPETokenizer)
    out = {}
    for s in SOURCES:
        ids = np.memmap(bpe_dir / f"{s}_val.bin", dtype=np.uint16, mode="r")[:n_tokens].tolist()
        docs, cur = [], []
        for i in ids:
            if i == tok.eot_id:
                if cur:
                    docs.append(tok.decode(cur))
                cur = []
            else:
                cur.append(i)
        out[s] = docs  # 丢掉末尾不完整的那篇
    return out


def bits_per_byte(total_nll: float, n_bytes: int) -> float:
    """bpb = 总 NLL（nats）/ ln2 / 字节数。
    loss 是"每个 token 平均多少 nats"，除以 ln2 变成 bits；再按字节数平摊，
    就和每个 token 代表多少文本无关了——字符级和 BPE 可以直接比。"""
    return total_nll / math.log(2) / n_bytes


@torch.no_grad()
def total_nll(model: torch.nn.Module, ids: list[int], batch_size: int = 16) -> tuple[float, int]:
    """把 ids 切成不重叠的 block_size+1 长窗口，返回 (所有预测位置的 NLL 之和, 预测次数)。"""
    device = next(model.parameters()).device
    T = model.cfg.block_size
    chunks = [ids[i:i + T + 1] for i in range(0, len(ids) - 1, T)]  # 相邻窗口共享 1 个 token
    full = [c for c in chunks if len(c) == T + 1]
    rest = [c for c in chunks if 1 < len(c) < T + 1]
    nll, count = 0.0, 0
    groups = [full[i:i + batch_size] for i in range(0, len(full), batch_size)] + [[c] for c in rest]
    for g in groups:
        t = torch.tensor(g, dtype=torch.long, device=device)  # (B, L+1)
        x, y = t[:, :-1], t[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, y)  # loss 是平均值，乘回元素个数得到总和
        nll += loss.float().item() * y.numel()
        count += y.numel()
    return nll, count


def run_bpb(run_dirs: list[str], n_tokens: int) -> list[dict]:
    texts = val_texts(n_tokens)
    rows = []
    for d in run_dirs:
        model, tok, cfg = load_model(d, "ckpt_best.pt")
        row = {"run": Path(d).name, "tokenizer": f"{type(tok).__name__} ({tok.vocab_size})"}
        for s, docs in texts.items():
            ids = [tok.eot_id]  # 和训练时一样：每篇文档前有一个分隔符
            for doc in docs:
                ids += tok.encode(doc) + [tok.eot_id]
            nll, count = total_nll(model, ids)
            n_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
            row[f"loss_{s}"] = nll / count
            row[f"bpb_{s}"] = bits_per_byte(nll, n_bytes)
            row[f"bytes_per_token_{s}"] = n_bytes / count
        rows.append(row)
        print(row)
        del model
        torch.cuda.empty_cache()
    return rows


def bpb_markdown(rows: list[dict]) -> str:
    head = ("| run | 分词器（词表） | SmolLM loss | SmolLM 字节/token | **SmolLM bpb** "
            "| TinyStories loss | TinyStories 字节/token | **TinyStories bpb** |\n"
            "|---|---|---|---|---|---|---|---|\n")
    body = "\n".join(
        f"| {r['run']} | {r['tokenizer']} | {r['loss_smollm']:.3f} | {r['bytes_per_token_smollm']:.2f} "
        f"| **{r['bpb_smollm']:.3f}** | {r['loss_tinystories']:.3f} | {r['bytes_per_token_tinystories']:.2f} "
        f"| **{r['bpb_tinystories']:.3f}** |" for r in rows)
    return head + body + "\n\nbpb = 总 NLL / ln2 / UTF-8 字节数，所有 run 用同一段验证原文。\n"


def main() -> None:
    p = argparse.ArgumentParser(description="checkpoint 统一评估")
    sub = p.add_subparsers(dest="mode", required=True)
    pl = sub.add_parser("loss")
    pl.add_argument("--ckpts", nargs="+", required=True)
    pl.add_argument("--n_batches", type=int, default=100)
    pl.add_argument("--batch_size", type=int, default=16)
    pl.add_argument("--out", default="docs/figures/ckpt_eval.md")
    pb = sub.add_parser("bpb")
    pb.add_argument("--runs", nargs="+", required=True)
    pb.add_argument("--n_tokens", type=int, default=300_000, help="每个来源取多少个 BPE token 的原文")
    pb.add_argument("--out", default="docs/figures/bpb.md")
    args = p.parse_args()

    if args.mode == "loss":
        import json
        mix = json.loads((ROOT / "data/processed/meta.json").read_text(encoding="utf-8"))["mix_weights"]
        rows = run_loss(args.ckpts, args.n_batches, args.batch_size, mix)
        md = loss_markdown(rows, args.n_batches * args.batch_size * 512)
    else:
        md = bpb_markdown(run_bpb(args.runs, args.n_tokens))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
