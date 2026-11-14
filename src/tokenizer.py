"""词元化（tokenization）：把文本变成整数 id 序列，以及反过来。

本文件提供三样东西：
1. BPETokenizer：正式训练用。包装 HuggingFace `tokenizers` 库的 ByteLevel BPE（词表 8192）。
2. CharTokenizer：字符级分词器，供可选的对比实验使用。
3. train_bpe_toy：手写的极简 BPE 训练，只用来理解原理和答辩演示，不用于正式训练。

BPE（Byte Pair Encoding）的核心思想：
    从最小单位（字节）开始，反复统计"哪两个相邻单位一起出现得最多"，把它们合并成一个新单位，
    直到词表达到指定大小。高频词（如 " the"）最终会变成一个 token，罕见词则被拆成几个子词。
    ByteLevel 意味着最小单位是字节（256 种），任何文本都能表示，永远不会出现未登录词（OOV）。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"  # 文档结束标记：插在两篇文档之间，生成时遇到它就停止


class Tokenizer(Protocol):
    """所有分词器的统一接口，训练和采样代码只依赖这几个成员。"""

    vocab_size: int
    eot_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


# ----------------------------------------------------------------------------
# 1. 正式使用的 ByteLevel BPE
# ----------------------------------------------------------------------------


class BPETokenizer:
    """ByteLevel BPE 分词器（包装 HF tokenizers）。

    处理流程：文本 → 预切分（按 GPT-2 的正则把文本切成"单词"，空格并入下一个词的开头，
    显示为 'Ġ'）→ 每个单词转成字节 → 按训练得到的合并规则合并 → id。
    没有 normalizer（不做大小写/Unicode 变换），因此 decode(encode(s)) == s 严格成立。
    """

    def __init__(self, hf: HFTokenizer) -> None:
        self.hf = hf
        self.vocab_size: int = hf.get_vocab_size()
        eot_id = hf.token_to_id(EOT)
        assert eot_id is not None, "分词器里缺少 <|endoftext|>"
        self.eot_id: int = eot_id

    @staticmethod
    def _empty() -> HFTokenizer:
        hf = HFTokenizer(models.BPE())
        # add_prefix_space=False：不在文本开头额外加空格，保证往返一致
        hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        hf.decoder = decoders.ByteLevel()
        return hf

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 8192) -> BPETokenizer:
        """在一批文本上训练 BPE。special token 最先加入，所以 <|endoftext|> 的 id 是 0。"""
        hf = cls._empty()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[EOT],
            # 先把 256 个字节全部放进词表，保证任何输入都能编码
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            min_frequency=2,
            show_progress=True,
        )
        hf.train_from_iterator(texts, trainer=trainer)
        return cls(hf)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        return cls(HFTokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        self.hf.save(str(path))

    def encode(self, text: str) -> list[int]:
        return self.hf.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """一次编码多篇文档，HF 内部用 Rust 多线程并行，比逐篇 encode 快很多。"""
        return [e.ids for e in self.hf.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        # skip_special_tokens=False：把 <|endoftext|> 也还原出来，方便观察文档边界
        return self.hf.decode(ids, skip_special_tokens=False)

    def tokens(self, text: str) -> list[str]:
        """返回切分后的 token 字符串（如 ['Once', 'Ġupon']），用于展示。"""
        return self.hf.encode(text).tokens


# ----------------------------------------------------------------------------
# 2. 字符级分词器（可选实验 C）
# ----------------------------------------------------------------------------


class CharTokenizer:
    """每个字符一个 token。id 0 = <|endoftext|>，id 1 = <unk>（低频字符），其余按频率排序。"""

    UNK = "<unk>"

    def __init__(self, chars: list[str]) -> None:
        self.itos: list[str] = [EOT, self.UNK] + chars
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size: int = len(self.itos)
        self.eot_id: int = 0
        self.unk_id: int = 1

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 100) -> CharTokenizer:
        """统计字符频率，出现次数少于 min_freq 的字符不进词表（编码时变成 <unk>）。"""
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(t)
        chars = [c for c, n in counter.most_common() if n >= min_freq]
        return cls(chars)

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        itos = json.loads(Path(path).read_text(encoding="utf-8"))["itos"]
        return cls(itos[2:])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.unk_id) for c in text]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def tokens(self, text: str) -> list[str]:
        return [self.itos[i] for i in self.encode(text)]


# ----------------------------------------------------------------------------
# 3. 手写极简 BPE（只用于理解原理）
# ----------------------------------------------------------------------------

# 和 GPT-2 类似的预切分：空格跟在下一个词前面（" the"），标点、空白单独成段
_TOY_SPLIT = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


def _count_pairs(words: dict[tuple[bytes, ...], int]) -> Counter[tuple[bytes, bytes]]:
    """统计所有相邻 token 对的出现次数（按单词频率加权）。"""
    pairs: Counter[tuple[bytes, bytes]] = Counter()
    for word, freq in words.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """把一个单词里所有出现的 pair 合并成一个 token，例如 (t,h,e) + (t,h) -> (th,e)。"""
    out: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_bpe_toy(text: str, num_merges: int) -> list[tuple[bytes, bytes]]:
    """手写 BPE 训练：统计相邻对 → 合并最高频对，重复 num_merges 次。返回合并规则列表。

    步骤：
    1. 预切分成单词并统计词频，例如 {" the": 120, " cat": 3, ...}。
       合并只在单词内部进行，不会跨越单词边界。
    2. 每个单词先拆成单字节序列：" the" -> (b' ', b't', b'h', b'e')。
    3. 循环：数出最常见的相邻对 -> 记录这条合并规则 -> 在所有单词里执行合并。
    真实的 BPE 训练（HF tokenizers）做的是同一件事，只是用 Rust 写并做了大量加速。
    """
    word_freq = Counter(_TOY_SPLIT.findall(text))
    words: dict[tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in w.encode("utf-8")): n for w, n in word_freq.items()
    }
    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        pairs = _count_pairs(words)
        if not pairs:
            break  # 所有单词都已经合并成一个 token
        best = max(pairs, key=pairs.__getitem__)
        merges.append(best)
        words = {_merge_word(w, best): n for w, n in words.items()}
    return merges


# ----------------------------------------------------------------------------
# 加载
# ----------------------------------------------------------------------------


def load_tokenizer(meta_path: str | Path) -> BPETokenizer | CharTokenizer:
    """根据 meta.json 里的 tokenizer 字段加载对应分词器（文件与 meta.json 在同一目录）。"""
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    file = meta_path.parent / meta["tokenizer_file"]
    if meta["tokenizer"] == "bpe":
        return BPETokenizer.load(file)
    if meta["tokenizer"] == "char":
        return CharTokenizer.load(file)
    raise ValueError(f"未知分词器类型：{meta['tokenizer']}")


if __name__ == "__main__":
    # 演示：在一小段文本上手写 BPE，打印前 20 次合并
    demo = (
        "Once upon a time, there was a little girl named Lily. She loved to play outside. "
        "One day, the girl saw the cat in the garden. The cat was there, and the girl was happy. "
        "They played together all day, and then they went home. The end. "
    ) * 20
    for i, (a, b) in enumerate(train_bpe_toy(demo, 20), 1):
        print(f"{i:2d}. {a!r} + {b!r} -> {a + b!r}")
