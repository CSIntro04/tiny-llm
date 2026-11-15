# 数据集说明

> 由 `src/prepare.py --shards all` 生成，所有数字取自 `data/processed/meta.json`（2026-11-14）。

## 1. 数据来源

| | SmolLM-Corpus 1B 子集 | TinyStories |
|---|---|---|
| HuggingFace 仓库 | [`ecreeth/1b-smollm-corpus`](https://huggingface.co/datasets/ecreeth/1b-smollm-corpus) | [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) |
| 内容 | 从 HuggingFaceTB/smollm-corpus 抽样：FineWeb-Edu（去重的教育类网页）87% + Cosmopedia v2（合成教科书/故事/博客）13% | GPT-3.5/4 生成的、只用 3–4 岁儿童词汇的英文短故事 |
| 许可 | ODC-By 1.0（使用时需署名） | CDLA-Sharing-1.0（再分发需同样许可） |
| 引用 | Ben Allal et al., 2024, *SmolLM-Corpus*；Lozhkov et al., 2024, *FineWeb-Edu* | Eldan & Li, 2023, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* |
| 官方验证集 | 无（自己切） | 有 |

为什么混合：SmolLM 词汇和知识面广，接近真实网页和教科书；TinyStories 结构简单，小模型也能写出连贯故事。分来源统计 val loss，可以看出模型规模的收益在难/易文本上是否不同。

## 2. 下载的文件

只下载 parquet（共 3.5GB），没有下载仓库里的 txt / tar.gz / npy 等重复或无关文件。

| 来源 | 文件 | 大小（字节） | sha256 |
|---|---|---|---|
| smollm | train-00000-of-00009.parquet | 308,355,479 | `1717c4df0be91a0ad0140e86cd335a6258963f36052786df070767d572836141` |
| smollm | train-00001-of-00009.parquet | 308,497,991 | `f9ebd0d5d5bcf9e1bbe27f44fc691d98445568820566e5380c51701ef7d9a6a3` |
| smollm | train-00002-of-00009.parquet | 308,442,482 | `bee768a2a790dfd6eeca4ff0bb95bdbb5527bd9141fa87238d0a2dee0a83c13d` |
| smollm | train-00003-of-00009.parquet | 308,169,286 | `0242705280031163db43532255e42d7a2ca204ffb5bc999607301bbdc8e67b37` |
| smollm | train-00004-of-00009.parquet | 308,641,571 | `d93e016a79f02b584e0352eceb764f8a708357da2e13d2597a85fe4b2401bff7` |
| smollm | train-00005-of-00009.parquet | 308,682,733 | `0a882dc66c084fe4b0a21ec84655a7011fcb0a5f26b281f9bc91f30877ee63a2` |
| smollm | train-00006-of-00009.parquet | 308,592,919 | `5a7ab044b192864361d641a9e60d589550039acda3426a40453f82673456065b` |
| smollm | train-00007-of-00009.parquet | 308,340,660 | `24afc645177443fc8dee272b66a65a5d7461d36e5f5b9a1dc64caaf1d79697d0` |
| smollm | train-00008-of-00009.parquet | 206,020,195 | `0af763ed32eafa4c70668088bfcbf7fc9affdd23e00371cc0149a9ee3ed5d8b3` |
| tinystories | data/train-00000-of-00004-2d5a1467fff1081b.parquet | 248,731,111 | `77cf780cebe52b6e83e3a2ac84bc56d8059363113e41d17a023f1d8b2ed0fc0b` |
| tinystories | data/train-00001-of-00004-5852b56a2bd28fd9.parquet | 248,171,980 | `e2f173a6dcc2b5895183e8ad376e115bd5c06764eb0cfc01ce1add156b2d017f` |
| tinystories | data/train-00002-of-00004-a26307300439e943.parquet | 245,894,874 | `cac73cd1a0973a7fccde022a5a5c78f8ba58ea6849541835c58e7c5fc747f2ef` |
| tinystories | data/train-00003-of-00004-d243063613e5a057.parquet | 247,988,350 | `862514ae75723ddd339337f1ef7617b88f82290dcddd57445e1d38d257ca7891` |
| tinystories | data/validation-00000-of-00001-869c898b519ad725.parquet | 9,989,127 | `33406a6206554cfc279c29c11f4df51528af487aa1a602b075566fc83c49dcab` |

## 3. 清洗规则与丢弃统计

规则（`src/prepare.py` 的 `clean`）：

- **通用**：`\r\n` → `\n`；Unicode NFC 规范化；3 个以上连续换行合并成一个空行；去首尾空白；空文档丢弃；含乱码特征（`�`、`â€`、`Ã`）的文档丢弃。
- **SmolLM**：少于 200 字符丢弃；非 ASCII 字符比例 > 10% 丢弃（多为非英文或乱码）；保留常见 Unicode 标点，不截断长文档。
- **TinyStories**：
  1. **先修复乱码**：约 6% 的故事里弯引号被错误解码（UTF-8 被当成 Windows-1252 读），如 `’` → `â€™`、`“` → `â€œ`。这些模式是固定的，直接还原成 ASCII 引号/破折号/省略号。修复前 M1 单个分片就要丢 31,951 篇，修复后全量只丢 406 篇。
  2. 弯引号、破折号、省略号、不换行空格统一为 ASCII。
  3. 少于 20 字符丢弃；处理后仍含非 ASCII 字符的丢弃（剩下的基本是 `Ã©` 一类无法可靠还原的乱码）。

丢弃统计（全量）：

| 来源 / 集合 | 空文档 | 乱码 | 过短 | 非 ASCII | 保留篇数 |
|---|---|---|---|---|---|
| smollm（train+val 一起统计） | 0 | 3,155 | 135 | 841 | 1,007,335 |
| tinystories train | 230 | 406 | 0 | 147 | 2,118,936 |
| tinystories val | 0 | 4 | 0 | 3 | 21,983 |

SmolLM 共 1,011,466 篇，丢弃 4,131 篇（0.4%）。

## 4. 验证集切分

- **TinyStories**：用官方 validation 分片。
- **SmolLM**：没有官方验证集，**按文档**切出 0.5%。每篇文档用 `blake2b(seed=1337 + 文档内容)` 的哈希值决定去向（`is_val`），哈希值落在前 0.5% 的进验证集。
  - 按文档切而不是按 token 切：避免同一篇文章的前后半段分别进入训练集和验证集，导致 val loss 偏乐观。
  - 用哈希而不是随机数：结果只取决于文档内容，与读取顺序、分片数量无关，可复现；重复文档一定落在同一边，不会泄漏。
  - 实际切出 4,966 篇（0.49%）。

## 5. 分词器

| 项 | 值 |
|---|---|
| 类型 | ByteLevel BPE（HF `tokenizers` 0.23.3），预切分用 GPT-2 正则，无 normalizer |
| 词表大小 | **8192**（含 1 个特殊符号 `<|endoftext|>`，id = 0） |
| 训练语料 | 从两个来源的**训练集**抽约 4 亿字符，SmolLM : TinyStories = 2 : 1（每个分片开头各取一段；数据集已全局打乱）。不含验证集文档 |
| 训练用时 | 约 30 秒（CPU） |
| 文件 | `data/processed/tokenizer.json`（550KB，进 git） |
| 存储 | 8192 < 65536，token id 用 `np.uint16` 存，每 token 2 字节 |

压缩率（训练集，字符数 / token 数，不计 eot）：

| 来源 | chars/token |
|---|---|
| SmolLM | 3.90 |
| TinyStories | 3.94 |

两个来源压缩率接近，说明分词器没有偏向某一个来源。ByteLevel 意味着任何字符串都能编码，`decode(encode(s)) == s` 严格成立（每来源随机 5 篇检查全部为 True）。

为什么是 8192：FineWeb-Edu 词汇比 TinyStories 丰富得多，词表太小时一个词会被切成很多碎片，512 的上下文能装下的内容变少；8192 仍在 4k–8k 范围内，embedding 只有 `8192 × C`（s10m 315 万、s100m 629 万参数），远小于 GPT-2 的 5 万词表。

## 6. Token 数与混合权重

| 来源 | 训练篇数 | 训练 token | 验证篇数 | 验证 token | 混合权重 |
|---|---|---|---|---|---|
| SmolLM | 1,002,369 | 1,145,639,931 | 4,966 | 5,663,457 | 0.7028 |
| TinyStories | 2,118,936 | 484,538,699 | 21,983 | 4,871,561 | 0.2972 |
| **合计** | 3,121,305 | **1,630,178,630** | 26,949 | 10,535,018 | 1.0 |

- token 数包含每篇文档末尾的 `<|endoftext|>`。
- 混合权重 = 各来源训练 token 占比（自然比例），训满一轮时每个 token 恰好看一遍。实测约 70:30，比预计的 2:1 更偏 SmolLM（SmolLM 文档平均更长）。
- 训练 token 预算占全量比例：s10m 3 亿 ≈ 18%，s50m 10 亿 ≈ 61%，s100m 15 亿 ≈ 0.92 轮。
- 输出文件：`smollm_train.bin` 2.29GB、`tinystories_train.bin` 0.97GB、两个 val.bin 共 21MB（不进 git）。

## 7. 样例与切分示例

SmolLM（清洗后，前 300 字符）：

> Welcome to our latest unit all about exploring the wonderful world of books and how they can shape our lives! We will dive into different stories, characters, and themes while learning about what makes each book unique and memorable. By the end of this unit, you'll have a better understanding of why…

> As technology becomes more and more integrated in every aspect of our lives, it is inevitable that large amounts of electronic devices are bought and equally large amounts of electronic devices are rendered obsolete. There is an obvious upsurge in e-waste such as semi-conductors, sensors, monitors, …

TinyStories：

> Once upon a time, there was a big black cat. The black cat lived in a pretty garden with many flowers. The black cat liked to play in the garden all day. One day, a small brown dog came to the garden. The black cat saw the dog and paused. The dog saw the cat and said, "This is my garden now!" …

> Once upon a time, there lived a nice man called Joe. One day Joe wanted to do something special. He went to the pet shop and bought a white kitten. Joe was very happy, he loved the little white kitten. …

切分示例（`Ġ` 表示词前的空格）：

```
'Once upon a time'
  -> ['Once', 'Ġupon', 'Ġa', 'Ġtime']                                         4 tokens
'Photosynthesis converts light energy into chemical energy stored in glucose.'
  -> ['Ph', 'ot', 'osy', 'nt', 'hesis', 'Ġconver', 'ts', 'Ġlight', 'Ġenergy',
      'Ġinto', 'Ġchemical', 'Ġenergy', 'Ġstored', 'Ġin', 'Ġgluc', 'ose', '.']  17 tokens
```

常见词是一个 token；罕见的专业词（photosynthesis、glucose）被拆成子词——这正是 BPE 用有限词表覆盖任意文本的方式。

手写 BPE 演示（`python src/tokenizer.py`，在一小段故事上合并 20 次）：前几次合并是 `h+e → he`、`' '+t → ' t'`、`' t'+he → ' the'`，即最高频的组合最先被合并。

## 8. 许可与使用注意

| 来源 | 许可 | 对我们的要求 |
|---|---|---|
| SmolLM-Corpus 1B 子集 | ODC-By 1.0（Open Data Commons Attribution） | 使用、修改、再分发都允许，但必须**署名**数据来源。README 和报告里注明 HuggingFaceTB/smollm-corpus 及 ecreeth/1b-smollm-corpus 子集。子集中的 FineWeb-Edu 部分来自 CommonCrawl 网页，也需遵守 CommonCrawl 的使用条款。 |
| TinyStories | CDLA-Sharing-1.0（Community Data License Agreement – Sharing） | 可以使用和修改；如果**再分发数据本身**（包括清洗后的版本），必须用同样的许可。用数据训练出的模型和结果不受此限制。 |

- 本仓库**不再分发原始数据**：parquet 和 .bin 都不进 git，组员按 README 从 HuggingFace 下载。git 里只有分词器（`tokenizer.json`）和统计信息（`meta.json`）。
- 模型生成的内容（尤其是说明文）经常有事实错误，只用于展示训练流程，不应当作事实引用。
- 词表大小：BPE 8192（实验 C 的字符级词表 336，见 `data/processed_char/meta.json`）。

## 9. 复现

```bash
python src/prepare.py --datasets smollm,tinystories --tokenizer bpe --vocab_size 8192 --shards all
pytest tests/test_data.py
```

全量用时约 25 分钟（下载视网速而定；编码约 13 分钟，6 核 CPU）。M1（`--shards 1`）约 3 分钟，每个来源各 1 个分片，用于快速验证流程。
