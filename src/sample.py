"""自回归生成：从 checkpoint 加载模型，按 prompt 续写。

用法：
    python src/sample.py --run runs/s10m --prompt "Once upon a time" \
        --max_new_tokens 200 --temperature 0.8 --top_k 50 --top_p 0.95 \
        --num_samples 3 --seed 42 [--ckpt ckpt_tok300M.pt]

load_model / generate_text 也给 app/app.py（Gradio 界面）直接调用；generate_stream 给 app/server.py（网页界面）流式输出用。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import GPT  # noqa: E402
from tokenizer import BPETokenizer, CharTokenizer, load_tokenizer  # noqa: E402
from train import build_model  # noqa: E402
from utils import set_seed  # noqa: E402

Tokenizer = BPETokenizer | CharTokenizer
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 加载


def _find_meta(run_dir: Path, cfg: dict) -> Path:
    """找一个旁边有分词器文件的 meta.json。
    runs/<name>/ 里只复制了 meta.json，分词器文件还在 data/processed/，所以要退回数据目录找。"""
    data_dir = Path(cfg["data"]["dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    for meta in (run_dir / "meta.json", data_dir / "meta.json"):
        if meta.exists():
            name = json.loads(meta.read_text(encoding="utf-8"))["tokenizer_file"]
            if (meta.parent / name).exists():
                return meta
    raise FileNotFoundError(f"在 {run_dir} 和 {data_dir} 都找不到分词器文件")


def load_model(run_dir: str, ckpt: str = "ckpt_best.pt",
               device: str = "cuda") -> tuple[GPT, Tokenizer, dict]:
    """返回 (eval 模式的模型, 分词器, 训练配置)。没有 GPU 时自动用 CPU。"""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    run = Path(run_dir)
    # map_location="cpu"：先整体读到内存，只把模型权重搬上 GPU。
    # checkpoint 里还有 AdamW 的 m、v（是权重的 2 倍大），直接读到 GPU 会白占显存
    # weights_only=False：checkpoint 里还有 python 随机数状态等非张量对象（文件是自己训练产生的，可信）
    state = torch.load(run / ckpt, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = build_model(cfg)
    model.load_state_dict(state["model"])
    model.to(device).eval()  # eval()：关掉 dropout
    tok = load_tokenizer(_find_meta(run, cfg))
    return model, tok, cfg


# ---------------------------------------------------------------- 采样


def sample_next(logits: torch.Tensor, temperature: float,
                top_k: int | None, top_p: float | None) -> torch.Tensor:
    """logits: (B, V) 最后一个位置的打分 -> (B, 1) 抽中的下一个 token id。

    顺序：temperature -> top-k -> top-p -> softmax -> 按概率抽样。
    """
    if temperature == 0:
        # greedy：直接取最大值。T -> 0 时 softmax(logits/T) 趋向 one-hot，这是它的极限；单独写避免除零
        return logits.argmax(dim=-1, keepdim=True)

    # temperature：softmax(z / T)。T<1 放大差距，分布更尖；T>1 缩小差距，分布更平
    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, [-1]]  # (B, 1) 第 k 大的值
        logits = logits.masked_fill(logits < kth, float("-inf"))  # 其余的概率变成 0

    if top_p is not None and 0 < top_p < 1:
        # nucleus：从高到低累加概率，保留累计刚达到 p 的最小集合
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)  # (B, V)
        probs = F.softmax(sorted_logits, dim=-1)
        # cum_before[i] = 排在 i 前面的 token 的概率和。"前面已经够 p 了"才去掉 i，
        # 所以第一个让累计超过 p 的 token 会被保留，排第一的 token（cum_before=0）永远保留
        cum_before = torch.cumsum(probs, dim=-1) - probs
        remove_sorted = cum_before > top_p
        # 把排序后的掩码放回原来的 token 位置：remove[b, sorted_idx[b, i]] = remove_sorted[b, i]
        remove = remove_sorted.scatter(1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)  # (B, V)，被屏蔽的位置概率为 0
    return torch.multinomial(probs, num_samples=1)  # (B, 1)


def _cache_len(caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None) -> int:
    """当前 cache 已覆盖多少个 token；还没开始缓存时返回 0。"""
    return 0 if caches is None or caches[0] is None else caches[0][0].size(2)


def _cached_logits(model: GPT, idx: torch.Tensor,
                   caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None
                   ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """返回 (最后一个位置的 logits (B,1,V), 新 cache)。

    首轮或上下文窗口已满时，直接 prefill 完整窗口并复用其最后位置 logits；
    否则只把上一轮刚采出的新 token 送进 cached decode 一次。
    """
    block_size = model.cfg.block_size
    if caches is None or _cache_len(caches) >= block_size:
        # prefill：位置从 0 开始，cache 覆盖完整窗口
        return model.forward_cached(idx[:, -block_size:], [None] * model.cfg.n_layer)
    # decode：cache 已覆盖 idx[:-1]，只喂 idx[-1] 这一个新 token
    return model.forward_cached(idx[:, -1:], caches)


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, eot_id: int | None = None,
             use_cache: bool = True) -> torch.Tensor:
    """idx: (B, T0) -> (B, T0 + 新生成长度)。

    自回归：每一步把刚生成的 token 接到输入末尾，再预测下一个。
    给了 eot_id 时，batch 里每一行都生成过 eot 就提前停止（各行在 eot 之后的内容由调用方截掉）。
    use_cache=True 时复用每层 K/V；False 时每步重算完整前缀（用于对比和测试）。
    """
    block_size = model.cfg.block_size
    use_bf16 = idx.is_cuda
    done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)  # (B,)
    caches = None
    for _ in range(max_new_tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            if use_cache:
                logits, caches = _cached_logits(model, idx, caches)
            else:
                # 位置编码只有 block_size 个，超出的部分只保留最后 block_size 个 token
                logits, _ = model(idx[:, -block_size:])  # (B, 1, V)
        logits = logits[:, -1, :].float()  # (B, V)；转回 fp32 再做 softmax，精度更稳
        next_id = sample_next(logits, temperature, top_k, top_p)  # (B, 1)
        idx = torch.cat([idx, next_id], dim=1)  # (B, T+1)
        if eot_id is not None:
            done |= next_id.squeeze(1) == eot_id
            if bool(done.all()):
                break
    return idx


def generate_text(model: GPT, tok: Tokenizer, prompt: str, num_samples: int = 1,
                  max_new_tokens: int = 200, temperature: float = 0.8,
                  top_k: int | None = 50, top_p: float | None = None,
                  seed: int | None = None, stop_at_eot: bool = True,
                  use_cache: bool = True) -> list[str]:
    """返回 num_samples 条续写文本（不含 prompt）。遇到 <|endoftext|> 就截断在它之前。"""
    if seed is not None:
        set_seed(seed)
    device = next(model.parameters()).device
    # 空 prompt：以文档分隔符开头，相当于"从一篇新文档的开头写起"
    ids = tok.encode(prompt) if prompt else [tok.eot_id]
    idx = torch.tensor([ids] * num_samples, dtype=torch.long, device=device)  # (N, T0)
    out = generate(model, idx, max_new_tokens, temperature, top_k, top_p,
                   eot_id=tok.eot_id if stop_at_eot else None, use_cache=use_cache)
    texts = []
    for row in out[:, len(ids):].tolist():  # 只取新生成的部分
        if stop_at_eot and tok.eot_id in row:
            row = row[: row.index(tok.eot_id)]
        texts.append(tok.decode(row))
    return texts


@torch.no_grad()
def generate_stream(model: GPT, tok: Tokenizer, prompt: str, max_new_tokens: int = 200,
                    temperature: float = 0.8, top_k: int | None = 50, top_p: float | None = None,
                    seed: int | None = None, use_cache: bool = True) -> Iterator[str]:
    """流式生成（给网页界面用）：每生成一个 token 就 yield 新增的那段文本。

    循环和 generate() 一样（B=1），随机数调用顺序也一样，所以同一个 seed 的结果与 generate_text() 相同。
    增量解码：每步把"全部新 token"重新 decode，再减去上次的文本。不能只 decode 单个 token，
    因为 BPE 是按字节合并的，一个汉字 / emoji 可能被拆在两个 token 里，单独解码会出现乱码。
    """
    if seed is not None:
        set_seed(seed)
    device = next(model.parameters()).device
    idx = torch.tensor([tok.encode(prompt) if prompt else [tok.eot_id]], dtype=torch.long, device=device)
    new_ids: list[int] = []
    sent = ""  # 已经发给前端的文本
    caches = None
    for _ in range(max_new_tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=idx.is_cuda):
            if use_cache:
                logits, caches = _cached_logits(model, idx, caches)  # (1, 1, V)
            else:
                logits, _ = model(idx[:, -model.cfg.block_size:])  # (1, 1, V)
        next_id = sample_next(logits[:, -1, :].float(), temperature, top_k, top_p)  # (1, 1)
        if next_id.item() == tok.eot_id:  # 文档结束符：停止，不输出它
            break
        idx = torch.cat([idx, next_id], dim=1)  # (1, T+1)
        new_ids.append(int(next_id.item()))
        text = tok.decode(new_ids)
        if text.endswith("�"):  # 末尾是半个多字节字符，等下一个 token 补齐再发
            continue
        yield text[len(sent):]
        sent = text
    text = tok.decode(new_ids)
    if len(text) > len(sent):  # 最后剩下没发出去的部分（比如被截断的半个字符）
        yield text[len(sent):]


# ---------------------------------------------------------------- 命令行


def main() -> None:
    p = argparse.ArgumentParser(description="从训练好的 checkpoint 生成文本")
    p.add_argument("--run", required=True, help="runs/<实验名>")
    p.add_argument("--ckpt", default="ckpt_best.pt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50, help="0 表示不用 top-k")
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_cache", action="store_true",
                   help="关闭 KV cache，每步重算完整前缀（用于对比生成速度）")
    args = p.parse_args()

    model, tok, _ = load_model(args.run, args.ckpt, args.device)
    print(f"模型 {args.run}/{args.ckpt}：{model.num_params(False) / 1e6:.2f}M 参数")

    t0 = time.time()
    texts = generate_text(model, tok, args.prompt, args.num_samples, args.max_new_tokens,
                          args.temperature, args.top_k or None, args.top_p, args.seed,
                          use_cache=not args.no_cache)
    dt = time.time() - t0

    n_tokens = 0
    for i, text in enumerate(texts):
        n_tokens += len(tok.encode(text))
        print(f"\n========== 样本 {i + 1} ==========")
        print(f"[prompt] {args.prompt}")
        print("----------")
        print(text)
    print(f"\n共生成约 {n_tokens} 个 token，用时 {dt:.2f}s，{n_tokens / dt:.1f} tokens/s")


if __name__ == "__main__":
    main()
