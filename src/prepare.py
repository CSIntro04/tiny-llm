"""数据准备：下载 → 清洗 → 切分训练/验证集 → 训练分词器 → 编码成 .bin。

用法：
    python src/prepare.py --datasets smollm,tinystories --tokenizer bpe --vocab_size 8192 --shards 1
    python src/prepare.py --datasets smollm,tinystories --tokenizer bpe --vocab_size 8192 --shards all

产物（data/processed/）：
    tokenizer.json                      分词器
    <来源>_train.bin / <来源>_val.bin   np.uint16 的 token id 序列，文档之间插 <|endoftext|>
    meta.json                           词表、各来源统计、混合权重
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from tokenizer import BPETokenizer, CharTokenizer

ROOT = Path(__file__).resolve().parents[1]

# 每个来源：HF 仓库、训练分片、官方验证分片（None 表示没有，需要自己切）、文本列名
SOURCES: dict[str, dict] = {
    "smollm": {
        "repo": "ecreeth/1b-smollm-corpus",
        "train": [f"train-0000{i}-of-00009.parquet" for i in range(9)],
        "val": None,
        "column": "text",
    },
    "tinystories": {
        "repo": "roneneldan/TinyStories",
        "train": [
            "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
            "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
            "data/train-00002-of-00004-a26307300439e943.parquet",
            "data/train-00003-of-00004-d243063613e5a057.parquet",
        ],
        "val": "data/validation-00000-of-00001-869c898b519ad725.parquet",
        "column": "text",
    },
}

# 分词器训练语料里各来源的字符占比（与预计 token 比例 2:1 一致）
TOK_SAMPLE_RATIO = {"smollm": 2 / 3, "tinystories": 1 / 3}


# ----------------------------------------------------------------------------
# 下载与读取
# ----------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(source: str, shards: str, out_dir: str | Path) -> dict[str, list[Path]]:
    """按文件下载（已下载的会跳过，中断后可续传）。shards='1' 只下第一个训练分片。

    返回 {"train": [...], "val": [...]}，val 为空列表表示该来源没有官方验证集。
    """
    cfg = SOURCES[source]
    train_files = cfg["train"][:1] if shards == "1" else cfg["train"]
    files = {"train": train_files, "val": [cfg["val"]] if cfg["val"] else []}
    local_dir = Path(out_dir) / source
    out: dict[str, list[Path]] = {}
    for split, names in files.items():
        out[split] = [
            Path(hf_hub_download(cfg["repo"], name, repo_type="dataset", local_dir=local_dir))
            for name in names
        ]
    return out


def iter_docs(paths: list[Path], column: str = "text") -> Iterator[str]:
    """流式读 parquet：每次读一个 batch，不把整个文件转成 pandas（文件有几百 MB）。"""
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=4096, columns=[column]):
            yield from batch.column(0).to_pylist()


# ----------------------------------------------------------------------------
# 清洗
# ----------------------------------------------------------------------------

_MANY_NEWLINES = re.compile(r"\n{3,}")  # 3 个以上换行 = 2 个以上空行
# TinyStories 里的弯引号、破折号等统一成 ASCII
_ASCII_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
})
# TinyStories 里约 6% 的故事有乱码：UTF-8 的弯引号被错当成 Windows-1252 解码，
# 例如 ’ 变成 "â€™"。这些模式固定，可以直接还原成 ASCII（按长度从长到短替换）。
# 右双引号 ” 的第三个字节在 Windows-1252 里没有定义、已经丢失，只剩 "â€"，所以放最后。
_TS_MOJIBAKE_FIX = [
    ("\u00e2\u20ac\u0160", " "),  # 细空格
    ("\u00e2\u20ac\u2039", ""),   # 零宽空格
    ("\u00e2\u20ac\u0153", '"'), ("\u00e2\u20ac\u2122", "'"), ("\u00e2\u20ac\u02dc", "'"),
    ("\u00e2\u20ac\u201c", "-"), ("\u00e2\u20ac\u201d", "-"), ("\u00e2\u20ac\u00a6", "..."),
    ("\u00c2\u00a0", " "),
    ("\u00e2\u20ac", '"'),
]
_MOJIBAKE = ("�", "â€", "Ã")  # 替换字符、UTF-8 被当成 Latin-1 解码的典型痕迹

# 丢弃原因计数，键为 "来源/原因"；在编码阶段统计
DROPS: Counter[str] = Counter()


def _non_ascii_frac(s: str) -> float:
    return sum(ord(c) > 127 for c in s) / max(len(s), 1)


def clean(doc: str | None, source: str) -> str | None:
    """清洗一篇文档，返回 None 表示丢弃（原因记在 DROPS）。"""
    if doc is None:
        DROPS[f"{source}/empty"] += 1
        return None
    # 通用：统一换行和 Unicode 形式（NFC：把"e + 组合重音"合成单个字符 é），合并多余空行，去首尾空白
    doc = unicodedata.normalize("NFC", doc.replace("\r\n", "\n").replace("\r", "\n"))
    doc = _MANY_NEWLINES.sub("\n\n", doc).strip()
    if not doc:
        DROPS[f"{source}/empty"] += 1
        return None
    if source == "tinystories":
        for bad, good in _TS_MOJIBAKE_FIX:
            doc = doc.replace(bad, good)
    if any(m in doc for m in _MOJIBAKE):
        DROPS[f"{source}/mojibake"] += 1
        return None

    if source == "smollm":
        if len(doc) < 200:
            DROPS[f"{source}/too_short"] += 1
            return None
        if _non_ascii_frac(doc) > 0.10:  # 多为非英文或乱码
            DROPS[f"{source}/non_ascii"] += 1
            return None
    elif source == "tinystories":
        doc = doc.translate(_ASCII_MAP)
        if len(doc) < 20:
            DROPS[f"{source}/too_short"] += 1
            return None
        if _non_ascii_frac(doc) > 0:  # 儿童故事应当是纯 ASCII，剩下的非 ASCII 字符多为乱码
            DROPS[f"{source}/non_ascii"] += 1
            return None
    return doc


# ----------------------------------------------------------------------------
# 训练/验证切分
# ----------------------------------------------------------------------------


def is_val(doc: str, val_frac: float, seed: int) -> bool:
    """按文档切验证集：用 (seed, 文档内容) 的哈希决定去向。

    为什么用哈希而不是随机数：结果只取决于文档本身，与读取顺序、分片数量无关，
    M1（1 个分片）和全量运行时同一篇文档的去向一致；重复文档也一定落在同一边，不会泄漏。
    按文档切（而不是按 token 切）可以避免同一篇的前后半段分别进训练集和验证集。
    """
    h = hashlib.blake2b(f"{seed}:{doc}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2**64 < val_frac


def split_docs(source: str, files: dict[str, list[Path]], val_frac: float,
               seed: int) -> dict[str, Iterator[str]]:
    """返回 {"train": 清洗后训练文档迭代器, "val": 清洗后验证文档迭代器}。

    有官方验证集（TinyStories）就直接用；没有（SmolLM）就从训练文件里按 is_val 切出 val_frac。
    注意：没有官方验证集时，两个迭代器各自完整读一遍训练文件（用时间换内存）。
    """
    col = SOURCES[source]["column"]

    def cleaned(paths: list[Path]) -> Iterator[str]:
        for d in iter_docs(paths, col):
            if (c := clean(d, source)) is not None:
                yield c

    if files["val"]:
        return {"train": cleaned(files["train"]), "val": cleaned(files["val"])}
    return {
        "train": (d for d in cleaned(files["train"]) if not is_val(d, val_frac, seed)),
        "val": (d for d in cleaned(files["train"]) if is_val(d, val_frac, seed)),
    }


# ----------------------------------------------------------------------------
# 分词器训练语料
# ----------------------------------------------------------------------------


def sample_for_tokenizer(train_files: dict[str, list[Path]], n_chars: int, val_frac: float,
                         seed: int) -> Iterator[str]:
    """按 TOK_SAMPLE_RATIO 从各来源训练集抽约 n_chars 个字符的文本，用来训练 BPE。

    数据集作者已全局打乱，所以从每个分片开头各取一段即可近似随机抽样；
    只取训练集文档（排除 SmolLM 切出的验证文档），避免数据泄漏。
    """
    for source, paths in train_files.items():
        ratio = TOK_SAMPLE_RATIO[source] / sum(TOK_SAMPLE_RATIO[s] for s in train_files)
        per_file = int(n_chars * ratio / len(paths))
        for p in paths:
            got = 0
            for d in iter_docs([p], SOURCES[source]["column"]):
                c = clean(d, source)
                if c is None or (SOURCES[source]["val"] is None and is_val(c, val_frac, seed)):
                    continue
                yield c
                got += len(c)
                if got >= per_file:
                    break
    DROPS.clear()  # 抽样时的丢弃不计入统计，编码阶段会重新统计


# ----------------------------------------------------------------------------
# 编码
# ----------------------------------------------------------------------------


class Reservoir:
    """蓄水池抽样：流式数据里等概率保留 k 个样本，用于事后随机检查往返一致性。"""

    def __init__(self, k: int, seed: int) -> None:
        self.k, self.n, self.items = k, 0, []
        self.rng = random.Random(seed)

    def add(self, x: str) -> None:
        self.n += 1
        if len(self.items) < self.k:
            self.items.append(x)
        elif (j := self.rng.randrange(self.n)) < self.k:
            self.items[j] = x


def _batched(it: Iterator[str], n: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_to_bin(docs: Iterator[str], tok: BPETokenizer | CharTokenizer, out_path: str | Path,
                  reservoir: Reservoir | None = None, chunk: int = 20000) -> dict[str, int]:
    """分块编码并追加写入 .bin（np.uint16），每篇文档后接一个 <|endoftext|>。

    不把十几亿个 id 放进 Python list：每次编码 chunk 篇，写完就释放。
    返回 {"tokens": 总 token 数（含 eot）, "docs": 文档数, "chars": 总字符数}。
    """
    assert tok.vocab_size <= 65535, "词表超过 uint16 能表示的范围"
    stats = {"tokens": 0, "docs": 0, "chars": 0}
    t0 = time.time()
    with open(out_path, "wb") as f:
        for batch in _batched(docs, chunk):
            ids = tok.encode_batch(batch)
            flat = np.fromiter(
                (i for seq in ids for i in (*seq, tok.eot_id)), dtype=np.uint16,
                count=sum(len(s) + 1 for s in ids),
            )
            flat.tofile(f)
            stats["tokens"] += len(flat)
            stats["docs"] += len(batch)
            stats["chars"] += sum(len(d) for d in batch)
            if reservoir is not None:
                for d in batch:
                    reservoir.add(d)
            print(f"  {Path(out_path).name}: {stats['docs']:,} 篇, {stats['tokens']:,} tokens, "
                  f"{time.time() - t0:.0f}s", end="\r", flush=True)
    print()
    return stats


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="smollm,tinystories")
    ap.add_argument("--tokenizer", choices=["bpe", "char"], default="bpe")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--char_min_freq", type=int, default=100)
    ap.add_argument("--shards", choices=["1", "all"], default="1")
    ap.add_argument("--val_frac", type=float, default=0.005)
    ap.add_argument("--tok_chars", type=float, default=4e8, help="分词器训练语料的字符数")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--raw_dir", default=str(ROOT / "data/raw"))
    ap.add_argument("--out_dir", default=str(ROOT / "data/processed"))
    return ap.parse_args()


def build_tokenizer(args: argparse.Namespace, files: dict[str, dict[str, list[Path]]],
                    out_dir: Path) -> tuple[BPETokenizer | CharTokenizer, str]:
    train_files = {s: f["train"] for s, f in files.items()}
    texts = sample_for_tokenizer(train_files, int(args.tok_chars), args.val_frac, args.seed)
    t0 = time.time()
    if args.tokenizer == "bpe":
        tok: BPETokenizer | CharTokenizer = BPETokenizer.train(texts, args.vocab_size)
        name = "tokenizer.json"
    else:
        tok = CharTokenizer.build(texts, args.char_min_freq)
        name = "char_vocab.json"
    tok.save(out_dir / name)
    print(f"分词器训练完成：vocab_size={tok.vocab_size}, eot_id={tok.eot_id}, 用时 {time.time() - t0:.0f}s")
    return tok, name


def show_examples(tok: BPETokenizer | CharTokenizer, samples: dict[str, Reservoir]) -> dict:
    """打印验收项：往返一致性、切分示例。返回写进 meta.json 的示例。"""
    print("\n== 往返检查 decode(encode(s)) == s（每来源随机 5 篇）==")
    for source, r in samples.items():
        print(f"  {source}: {[tok.decode(tok.encode(s)) == s for s in r.items]}")
    examples = {
        "Once upon a time": "Once upon a time",
        "edu": "Photosynthesis converts light energy into chemical energy stored in glucose.",
    }
    print("\n== 切分示例 ==")
    out = {}
    for text in examples.values():
        toks = tok.tokens(text)
        out[text] = toks
        print(f"  {text!r}\n    -> {toks}  ({len(toks)} tokens)")
    return out


def main() -> None:
    args = parse_args()
    sources = args.datasets.split(",")
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 下载
    files = {s: download(s, args.shards, raw_dir) for s in sources}
    file_info = {
        s: [{"file": p.relative_to(raw_dir / s).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in f["train"] + f["val"]]
        for s, f in files.items()
    }
    print("下载完成：", {s: len(v) for s, v in file_info.items()}, "个文件")

    # 2. 训练分词器
    tok, tok_file = build_tokenizer(args, files, out_dir)

    # 3. 清洗 + 切分 + 编码（每个来源一个 train.bin、一个 val.bin）
    meta_sources: dict[str, dict] = {}
    samples: dict[str, Reservoir] = {}
    for s in sources:
        DROPS.clear()
        splits = split_docs(s, files[s], args.val_frac, args.seed)
        samples[s] = Reservoir(5, args.seed)
        tr = encode_to_bin(splits["train"], tok, out_dir / f"{s}_train.bin", samples[s])
        drops_train = {k.split("/")[1]: v for k, v in DROPS.items()}
        DROPS.clear()
        va = encode_to_bin(splits["val"], tok, out_dir / f"{s}_val.bin")
        # 没有官方验证集时，val 迭代器把训练文件又读了一遍，丢弃数是重复统计，不计入
        drops_val = {k.split("/")[1]: v for k, v in DROPS.items()} if SOURCES[s]["val"] else {}
        meta_sources[s] = {
            "repo": SOURCES[s]["repo"],
            "files": file_info[s],
            "train_tokens": tr["tokens"], "val_tokens": va["tokens"],
            "train_docs": tr["docs"], "val_docs": va["docs"],
            "dropped": {"train": drops_train, "val": drops_val},
            "chars_per_token": round((tr["chars"]) / (tr["tokens"] - tr["docs"]), 3),
        }

    # 4. 自然比例混合权重：各来源训练 token 数占比，训满一轮正好每个 token 看一遍
    total = sum(m["train_tokens"] for m in meta_sources.values())
    mix = {s: round(m["train_tokens"] / total, 4) for s, m in meta_sources.items()}

    examples = show_examples(tok, samples)
    meta = {
        "tokenizer": args.tokenizer,
        "tokenizer_file": tok_file,
        "vocab_size": tok.vocab_size,
        "eot_id": tok.eot_id,
        "dtype": "uint16",
        "shards": args.shards,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "tok_train_chars": int(args.tok_chars),
        "sources": meta_sources,
        "mix_weights": mix,
        "examples": examples,
        "samples": {s: [d[:300] for d in r.items[:2]] for s, r in samples.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== 统计 ==")
    for s, m in meta_sources.items():
        print(f"  {s}: train {m['train_docs']:,} 篇 / {m['train_tokens']:,} tokens, "
              f"val {m['val_docs']:,} 篇 / {m['val_tokens']:,} tokens, "
              f"chars/token {m['chars_per_token']}, 丢弃 {m['dropped']}")
    print(f"  总训练 token：{total:,}   混合权重：{mix}")
    print(f"已写入 {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
