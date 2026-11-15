# 阶段 2 · 数据与词元化

## 目标

把两个英文语料变成模型能直接读取的 token id 文件，按来源分开存，训练时按比例混合采样：

```
data/raw/smollm/*.parquet               SmolLM 1B 子集原始数据（不进 git）
data/raw/tinystories/*.parquet          TinyStories 原始数据（不进 git）
data/processed/tokenizer.json           在混合语料上训练的 BPE（词表 8192）
data/processed/smollm_train.bin  smollm_val.bin
data/processed/tinystories_train.bin  tinystories_val.bin     均为 np.uint16 的 token id 序列
data/processed/meta.json                词表、分词器、各来源 token 数、混合比例
docs/report/data.md                            数据集说明（交付物）
```

## 前置条件

- 阶段 1 环境通过自检。
- 磁盘剩余 **≥ 12GB**（原始 parquet 约 3.7GB + 编码后 bin 约 3–3.5GB + 分词器训练用的临时文本 + 余量）。

## 数据来源

| | SmolLM-Corpus 1B 子集 | TinyStories |
|---|---|---|
| HuggingFace | `ecreeth/1b-smollm-corpus` | `roneneldan/TinyStories` |
| 内容 | 从 HuggingFaceTB/smollm-corpus 抽样：FineWeb-Edu（去重，教育类网页）87% + Cosmopedia v2（合成教科书/故事/博客）13% | GPT-3.5/4 生成的、只用 3–4 岁儿童词汇的英文短故事 |
| 规模 | 1,011,466 篇文档，数据集作者统计约 10 亿 token（用的不是我们的分词器） | 约 210 万个训练故事 + 约 2.2 万个验证故事，约 5 亿 token |
| 文件 | `train-0000{0..8}-of-00009.parquet`，单列 `text`，**共约 2.67GB** | `data/train-0000{0..3}-of-00004-*.parquet` 共约 0.99GB + `data/validation-*.parquet` 约 10MB |
| 官方验证集 | **无**，需自己切 | 有 |
| 许可 | ODC-By（需署名） | CDLA-Sharing-1.0 |
| 引用 | Ben Allal et al. 2024（SmolLM-Corpus）；Lozhkov et al. 2024（FineWeb-Edu） | Eldan & Li 2023 |

**为什么混合两个语料**：SmolLM 子集词汇和知识面广，接近真实网页和教科书；TinyStories 结构简单，小模型也能写出连贯的故事。混合后，0.01B 模型至少能在故事上表现像样，0.1B 模型在更难的教育文本上的优势也能体现出来。分来源统计 val loss，可以看出"模型规模的收益在难/易文本上是否不同"。

### 下载

总下载量约 **3.7GB**。这两个数据集的指定文件已经过用户确认（见 CLAUDE.md 的数据下载规则），按下表分两步下载，不用再问；下载前检查磁盘剩余空间 ≥ 12GB。

| 时机 | 下载内容 | 大小 | 用途 |
|---|---|---|---|
| M1 | SmolLM `train-00000` 1 个分片 + TinyStories 1 个训练分片 + 验证集 | ~0.57GB | 流程验证、debug 模型 |
| M2 前 | 两个数据集的全部训练分片 | ~3.7GB（含 M1 已下的） | 正式模型 |

- 用 `huggingface_hub.hf_hub_download` 按文件下载（可断点续传）。只下 parquet，**不要**下 TinyStories 仓库里的 `TinyStories-train.txt`、`TinyStoriesV2-GPT4-train.txt`、`TinyStories_all_data.tar.gz`、`MLP_input_output.npy`（内容重复或无关，合计 6GB+）。
- 国内访问 HuggingFace 较慢时，可设环境变量 `HF_ENDPOINT` 指向镜像站（先确认镜像可用）。
- 下载后记录每个文件的大小和 sha256 到 `docs/report/data.md`。

## 要写的文件与函数签名

```python
# src/tokenizer.py
class Tokenizer(Protocol):
    vocab_size: int
    eot_id: int                                   # <|endoftext|> 的 id
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...

class BPETokenizer:                               # 包装 HF tokenizers 的 ByteLevel BPE
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 8192) -> "BPETokenizer": ...
    @classmethod
    def load(cls, path: str) -> "BPETokenizer": ...
    def save(self, path: str) -> None: ...

class CharTokenizer:                              # 字符级，供可选实验 C
    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 100) -> "CharTokenizer": ...

def train_bpe_toy(text: str, num_merges: int) -> list[tuple[bytes, bytes]]:
    """手写极简 BPE（约 60 行）：只在几 KB 文本上演示"统计相邻对 → 合并最高频对"，答辩用。"""

def load_tokenizer(meta_path: str) -> Tokenizer: ...
```

```python
# src/prepare.py
SOURCES = {"smollm": ..., "tinystories": ...}      # repo id、文件名列表、文本列名
def download(source: str, shards: str, out_dir: str) -> list[str]: ...
def iter_docs(paths: list[str]) -> Iterator[str]: ...
def clean(doc: str, source: str) -> str | None: ...                 # 返回 None 表示丢弃
def is_val(doc: str, val_frac: float, seed: int) -> bool: ...                  # 按 (seed, 文档内容) 哈希决定去向
def split_docs(source, files, val_frac, seed) -> dict[str, Iterator[str]]: ...  # {"train": ..., "val": ...}
def sample_for_tokenizer(train_files, n_chars, val_frac, seed) -> Iterator[str]: ...  # 按混合比例抽样
def encode_to_bin(docs: Iterable[str], tok: Tokenizer, out_path: str) -> int: ...  # 返回 token 数
def main() -> None: ...
# --datasets smollm,tinystories --tokenizer bpe|char --vocab_size 8192 --shards 1|all --val_frac 0.005
```

```python
# src/utils.py（数据加载部分）
def get_batch(sources: dict[str, str], weights: dict[str, float], batch_size: int,
              block_size: int, device: str, rng: np.random.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """每个样本先按 weights 选来源，再在该来源里随机取 block_size+1 个连续 token。返回 x, y，形状 (B, T)。"""
```

## 实现要点

### 清洗

通用：去首尾空白、合并 3 个以上连续空行、统一 Unicode 为 NFC、丢弃空文档。

| | SmolLM | TinyStories |
|---|---|---|
| 过短 | 少于 200 字符丢弃 | 少于 20 字符丢弃 |
| 过长 | 不截断（`get_batch` 随机切片即可） | — |
| 字符 | 保留常见 Unicode 标点；非 ASCII 比例超过 10% 的文档丢弃（多为乱码或非英文） | 弯引号等统一为 ASCII；含乱码的故事丢弃 |

- 文档之间插 `<|endoftext|>`，让模型知道"一篇结束了"，生成时也用它判断停止。
- 统计并记录每一步丢弃的数量。

### 验证集切分

- TinyStories：用官方 validation 分片。
- SmolLM：无官方验证集，按文档以固定种子随机切出 **0.5%**（约 5000 篇、约 500–700 万 token）作为 `smollm_val.bin`。**按文档切，不按 token 切**，避免同一篇文档的前后半段分别落进训练集和验证集。
- 实现上用 `blake2b(seed + 文档内容)` 的哈希决定去向（`is_val`），结果与读取顺序和分片数无关：M1 和全量运行时同一篇文档去向一致，重复文档也一定落在同一边。

### BPE 词元化

- 用 HF `tokenizers` 训练 **ByteLevel BPE，`vocab_size=8192`**，`special_tokens=["<|endoftext|>"]`。ByteLevel 先把文本拆成字节，**永远不会有未登录词（OOV）**。
- 训练语料：从两个来源的**训练集**按 2:1（与 token 比例一致）抽约 300–500MB 文本，不用验证集。
- 为什么从 4096 改成 8192：FineWeb-Edu 是真实网页和教育文本，词汇比 TinyStories 丰富得多。词表太小时一个词会被切成很多碎片，序列变长、512 的上下文装下的内容变少。8192 仍在 CLAUDE.md 规定的 4k–8k 范围内，嵌入层只有 `8192 × C`（s10m 315 万、s100m 629 万），相比 GPT-2 的 5 万词表小得多。
- 8192 < 65536，token id 仍可用 `np.uint16` 存。
- **手写 `train_bpe_toy`** 只用来理解和展示：打印前 20 次合并，如 `(b't', b'h') -> b'th'`、`(b'th', b'e') -> b'the'`。

### 编码与保存

- 分块编码（每次几万篇，`encode_batch` 多线程），边编码边追加写 `.bin`，不要把十几亿个 id 放进 Python list。预计全量编码需要 20–40 分钟（CPU）。
- `meta.json` 示例：

```json
{
  "tokenizer": "bpe",
  "vocab_size": 8192,
  "eot_id": 0,
  "sources": {
    "smollm":      {"repo": "ecreeth/1b-smollm-corpus", "train_tokens": 0, "val_tokens": 0, "docs": 0, "dropped": 0, "chars_per_token": 0.0},
    "tinystories": {"repo": "roneneldan/TinyStories",   "train_tokens": 0, "val_tokens": 0, "docs": 0, "dropped": 0, "chars_per_token": 0.0}
  },
  "mix_weights": {"smollm": 0.0, "tinystories": 0.0}
}
```

### 数据混合与加载：`get_batch`

- **默认按自然比例混合**：`mix_weights` = 各来源训练 token 数占总数的比例（预计约 2:1）。按自然比例时，训练满一轮正好每个 token 看一遍。
- 每个样本先按 `mix_weights` 抽来源，再在该来源的 memmap 里随机取起点：

```python
data = np.memmap(path, dtype=np.uint16, mode="r")
i = random_start(len(data) - block_size - 1)
t = torch.from_numpy(data[i : i + block_size + 1].astype(np.int64))
#   x = t[0 : T]      输入           (T,)
#   y = t[1 : T+1]    目标=右移一位  (T,)   ← next-token prediction
# 拼成 batch 后 x, y 形状均为 (B, T)，pin_memory 后 non_blocking 传到 GPU
```

- 一个样本只来自一个来源，不跨来源拼接。
- 为什么不用 DataLoader：数据就是几条长 token 序列，随机切片几乎零开销；memmap 只读用到的部分，3GB 的 bin 不用全部进内存。每次调用重新创建 memmap，避免长时间运行内存增长。
- 验证时**分来源**评估：`val_smollm`、`val_tinystories`，以及按 `mix_weights` 加权的 `val`。

### 训练 token 预算与数据量的关系

| 模型 | 预算 | 约占全量 | 说明 |
|---|---|---|---|
| s10m | 3 亿 | ~18% | 随机采样，大部分数据只会看到一次或看不到 |
| s50m | 10 亿 | ~61% | |
| s100m | 15 亿 | ~0.92 轮 | 实测全量训练集 16.3 亿 token（见 `data.md`） |

## 验收标准

```bash
# M1：各 1 个分片
python src/prepare.py --datasets smollm,tinystories --tokenizer bpe --vocab_size 8192 --shards 1
```

预期打印：

- 每个来源的文档数、丢弃数、训练/验证 token 数、混合权重
- 每个来源的压缩率 `chars_per_token`（预计 SmolLM 约 3.5–4，TinyStories 约 4 以上；以实测为准）
- 每个来源随机 5 篇 `decode(encode(s)) == s` 全部为 True
- 示例切分，如 `Once upon a time` → `['Once', 'Ġupon', 'Ġa', 'Ġtime']`，以及一段教育文本的切分

```bash
pytest tests/test_data.py   # get_batch 形状 (B,T)、y == x 右移一位、dtype int64、来源比例与 weights 接近
```

M2 时用 `--shards all` 重跑（复用 M1 训练好的分词器还是重训，统一决定并记录；建议全量时重训一次，并重新训练 debug 模型），把全量 token 数更新到 [00-roadmap.md](00-roadmap.md)。

## 常见坑

- `uint16` 溢出：词表必须 ≤ 65535，`prepare.py` 里加 `assert`。
- 读 bin 时 dtype 忘写 `np.uint16`。
- 喂给 `nn.Embedding` 前要转 `int64`。
- SmolLM 验证集按 token 切 → 同一文档跨训练/验证集，val loss 偏乐观。
- 用验证集训练分词器 → 轻微数据泄漏。
- 分词器只在 TinyStories 上训练 → 网页文本被切得很碎（压缩率明显下降就要检查）。
- SmolLM parquet 的列类型是 `large_string`，用 pyarrow 按 batch 读，不要一次 `to_pandas()` 整个文件。
- WSL2 下数据放在 `/mnt/c` 会非常慢。

## 答辩可能被问的问题

- 两个数据集分别是什么？为什么要混合？比例怎么定的？
- 什么是 token？为什么用 BPE？BPE 怎么训练（现场演示几次合并）？
- 词表为什么选 8192？embedding 占多少参数？
- SmolLM 的验证集怎么切的？为什么按文档切？
- `x` 和 `y` 的关系？为什么叫 next-token prediction？
- 数据许可分别是什么？使用时要注意什么？

## 产出：`docs/report/data.md` 模板

```markdown
# 数据集说明
- 两个来源：仓库、论文/引用、许可、内容描述
- 下载的文件、大小、sha256
- 清洗规则与每步丢弃统计（分来源）
- 验证集切分方法
- 分词器：类型、词表大小、训练语料构成、各来源压缩率
- 各来源训练/验证 token 数、混合权重
- 每个来源 2 个样例 + 切分示例
```
