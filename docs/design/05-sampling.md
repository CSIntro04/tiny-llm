# 阶段 5 · 生成与采样

## 目标

`src/sample.py` 从任意 run 加载 checkpoint，按给定 prompt 自回归生成文本，支持最大长度、temperature、top-k、top-p、随机种子和多条采样。`generate()` 函数同时供网页界面（`app/server.py`，流式版 `generate_stream()`）和 Gradio 备用界面（`app/app.py`）复用。

## 前置条件

- 阶段 4 至少有一个 `runs/<name>/ckpt_best.pt`（debug 模型即可）。

## 要写的文件与函数签名

```python
# src/sample.py
def load_model(run_dir: str, ckpt: str = "ckpt_best.pt",
               device: str = "cuda") -> tuple[GPT, Tokenizer, dict]: ...   # 模型、分词器、配置

@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, eot_id: int | None = None) -> torch.Tensor:
    """idx: (B, T0) → (B, T0 + 新生成长度)"""

def sample_next(logits: torch.Tensor, temperature: float,
                top_k: int | None, top_p: float | None) -> torch.Tensor: ...  # (B, V) → (B, 1)

def generate_text(model, tok, prompt: str, **kwargs) -> list[str]: ...         # 界面直接调用
def main() -> None: ...
```

## 实现要点

### 自回归生成循环

```text
重复 max_new_tokens 次：
  idx_cond = idx[:, -block_size:]          # 超过上下文长度就只保留最后 block_size 个
  logits, _ = model(idx_cond)              # 推理模式只返回最后位置：(B, 1, V)
  logits = logits[:, -1, :]                # (B, V)
  next_id = sample_next(logits, ...)       # (B, 1)
  idx = cat([idx, next_id], dim=1)
  若 next_id == eot_id 则停止（可选）
```

每一步把刚生成的 token 接到输入后面再预测下一个——这就是"自回归"。

### 采样参数

| 参数 | 做法 | 效果 |
|---|---|---|
| `temperature` | `logits / T` 再 softmax | T<1 分布更尖、更保守、更重复；T>1 更平、更有创意也更容易胡言乱语 |
| `temperature = 0` | 直接 `argmax`（greedy） | 结果确定；小模型容易陷入循环重复 |
| `top_k` | 只保留概率最高的 k 个，其余设为 `-inf` | 截掉长尾的离谱 token |
| `top_p`（nucleus） | 按概率从高到低累加，保留累计概率刚超过 p 的最小集合 | 候选集大小随"模型有多确定"自适应变化 |

最后 `torch.multinomial(softmax(logits), 1)` 按概率抽样。顺序：temperature → top-k → top-p → softmax → 抽样。

### KV cache（可选扩展，不必做）

朴素实现每生成一个 token 都要把整段序列重新算一遍注意力。KV cache 把之前每层算过的 K、V 缓存起来，新 token 只需算自己的 q/k/v 再与缓存拼接，复杂度从每步 O(T²) 降到 O(T)。本项目生成长度只有几百个 token，模型又小，不做也够快；做了可以作为加分项，在实验里对比生成速度。

### 命令行

```bash
python src/sample.py --run runs/s10m --prompt "Once upon a time" \
    --max_new_tokens 200 --temperature 0.8 --top_k 50 --top_p 0.95 \
    --num_samples 3 --seed 42 [--ckpt ckpt_tok300M.pt]
```

输出中把 prompt 和续写用分隔符区分，结尾打印生成速度（tokens/s）。

## 验收标准

- 同一 `--seed` 两次运行结果完全相同。
- `--temperature 0` 无论 seed 如何结果都相同。
- `--top_k 1` 与 `--temperature 0` 结果相同。
- prompt 长度超过 `block_size` 不报错（被截断）。
- debug 模型（训练完成后）对 "Once upon a time" 生成大体像英文的故事片段。
- `pytest tests/test_sample.py`：`sample_next` 在 top-k / top-p 下只会抽到允许的 token。

## 常见坑

- temperature 为 0 时做除法 → 除零，需要单独分支。
- top-p 实现时，排序后要把"累计概率超过 p 的第一个 token"也保留，否则可能一个都不剩。
- 推理前忘了 `model.eval()`。
- 推理可以用 `torch.autocast(bfloat16)` 提速，但 checkpoint 加载时注意 `map_location`。
- decode 时去掉 `<|endoftext|>`，或在界面上显示成换行。

## 答辩可能被问的问题

- 模型输出的是什么？怎么从 logits 变成一个具体的词？
- temperature 的数学含义？T→0 和 T→∞ 分别会怎样？
- top-k 和 top-p 区别？各自的优缺点？
- 为什么 greedy 解码容易重复？
- 上下文超过 512 个 token 后怎么办？

## 实测结果（2026-11-15，阶段 5 验收）

实现：`src/sample.py`（`load_model` / `sample_next` / `generate` / `generate_text` / `main`），测试 `tests/test_sample.py`（6 项全部通过）。没有做 KV cache。

`runs/<name>/` 里只有 `meta.json`，分词器文件 `tokenizer.json` 在 `data/processed/`。所以 `load_model` 先在 run 目录里找分词器，找不到就去配置里的 `data.dir` 找，不往 `runs/` 里写任何东西。

### 验收清单

- [x] 同一 `--seed` 两次运行结果完全相同（debug，seed 42，逐字一致）。
- [x] `--temperature 0`（seed 1、seed 2）和 `--top_k 1`（seed 3）三次输出完全相同。
- [x] 3900 个 token 的长 prompt 不报错，模型只看最后 512 个。
- [x] debug 模型对 "Once upon a time" 生成大体像英文的故事片段。
- [x] `pytest tests/test_sample.py`：top-k / top-p 只会抽到允许的 token，p 很小时至少保留 1 个。

### 样例（prompt "Once upon a time"，T=0.8，top-k 50，top-p 0.95，seed 42，每条 120 token）

**debug（5.39M）**：
> , there was a little boy named Timmy. Timmy loved to play with his toy for help. They all saw a big hay and flowers. "Sure, Lily!" Lily said. "Let's go home," Lily said him. …

句子单独看基本合乎语法，但上下文接不上：人物从 Timmy 突然换成 Lily，还有"said him"这样的错误。

**s10m（13.99M）**：
> , there was a little boy named Timmy. Timmy loved to play with his toy car all day long. One day, Timmy's mommy gave him a big package of a new toy car. … He played with it all day long, but he was not happy with his new toy car. …

**s100m（105.92M）**：
> , there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks all day long. One day, Timmy's mom told him it was time to take a bath. Timmy didn't want to stop playing, but he knew he had to listen to his mom. As Timmy was taking his bath, he noticed that his favorite toy truck was missing. …

s10m 已经能写出连贯的段落，但会自相矛盾（"he realized that it was very popular" 和后面的情节对不上）。s100m 的情节有因果关系（洗澡 → 发现玩具不见了 → 哭 → 妈妈来问），人物和前后文也保持一致。规模对比的系统分析放在阶段 6。

### 速度

RTX 5060 Ti，bf16，3 条样本 × 120 token 一起生成：s10m 约 80 tokens/s，s100m 约 170 tokens/s。计时包含 CUDA 第一次启动内核的预热时间，所以先跑的、模型更小的反而显得更慢，这里的数字只能当数量级看。没有 KV cache 时，每一步都要把整段序列重新算一遍。
