# 阶段 6 · 对比实验

## 目标

完成课程要求的"至少比较两种训练设置或采样参数"，本组做两组必做实验 + 一组可选：

- **实验 A（主实验）：模型规模 0.01B vs 0.05B vs 0.1B**
- **实验 B：采样参数**（temperature × top-k）
- 实验 C（可选）：字符级 vs BPE 分词

产出 `docs/report/experiments.md`：叠图、表格、生成样例、结论。

## 前置条件

- 阶段 2 用 `--shards all` 完成全量数据处理（两个来源）。
- 阶段 4、5 通过验收；已用 s100m 冒烟测试校准过训练时长。

## 要写的文件与函数签名

```python
# src/plot_compare.py
def load_logs(run_dirs: list[str]) -> dict[str, pd.DataFrame | list[dict]]: ...
def plot_overlay(logs, x: str = "tokens", y: str = "val_loss", out: str = ...) -> None: ...  # y 可为 val_smollm / val_tinystories
def plot_scaling(final: dict[str, dict[str, float]], out: str) -> None: ...   # 参数量 vs 最终 val loss（分来源）
def main() -> None: ...   # --runs runs/s10m runs/s50m runs/s100m --out docs/figures/

# src/eval_prompts.py
PROMPTS: dict[str, list[str]] = {"story": [...], "expository": [...]}   # 固定 10 个 prompt
def run_generation_grid(run_dirs, prompts, settings, seed) -> list[dict]: ...  # 生成并存 jsonl
def export_markdown(rows, out_md: str) -> None: ...
```

（不额外引入 pandas 的话，用 `csv` 模块读日志即可。）

## 实验 A：模型规模

### 控制变量

| 固定 | 值 |
|---|---|
| 数据 | SmolLM 1B 子集 + TinyStories，同一组 bin，同一混合比例（自然比例，约 2:1） |
| 分词器 | 同一个 BPE-8192 |
| 有效 batch、warmup 步数、调度形状、weight decay、grad clip、seed | 完全相同 |

| 变化 | s10m | s50m | s100m |
|---|---|---|---|
| 层/维/头 | 6/384/6 | 10/640/10 | 14/768/12 |
| 总参数 | 13.99M | 54.81M | 105.92M |
| lr | 1e-3 | 6e-4 | 4e-4 |
| 训练 token（按规模分配） | 3 亿 | 10 亿 | 15 亿 |
| 预计用时 | ~1h | ~7h | ~20h |

训练顺序：s10m → s50m → s100m（小的先跑，发现问题成本低）。每次开跑前报告预计耗时并确认。

**token 数不同带来的比较问题**：最终 loss 的差距同时来自"模型更大"和"看的数据更多"。为分开这两个因素，做两种比较：

- **相同 token 数**：s50m、s100m 在 3 亿 token 时保存的 `ckpt_tok300M.pt`，与 s10m 的最终模型比较；s100m 在 10 亿 token 时的 `ckpt_tok1000M.pt`，与 s50m 的最终模型比较。要注意此时大模型的学习率还没衰减完，loss 会比"专门训练到 3 亿就停"的情况偏高，结论要写明这一点。
- **各自训练完**：三个模型的最终结果，代表"各自接近计算最优时"的水平。

### 要记录的指标

1. **Loss 曲线**：
   - 三条加权 val loss 叠图，横轴为**已训练 token 数**（主图）和训练时间（辅图）。
   - **分来源**：`val_smollm` 和 `val_tinystories` 各一张叠图。
   - train loss 与 val loss 的差（看过拟合；s100m 训满一轮，预计差距很小）。
2. **最终结果表**（每个模型一行；"相同 token 数"另起一张表）：

| 模型 | 参数量 | 训练 token | val（加权） | val SmolLM | val TinyStories | 困惑度 | 训练时长 | tok/s | 显存峰值 | ckpt 大小 | 生成速度 |
|---|---|---|---|---|---|---|---|---|---|---|---|

3. **Scaling 图**：横轴参数量（对数），纵轴最终 val loss，两个来源各画一条线，观察规模收益在难（SmolLM）、易（TinyStories）文本上是否不同。
4. **生成质量**：固定 10 个 prompt，**故事类与说明文类各 5 个**：
   - 故事：`Once upon a time`、`Lily wanted to`、`The little dog was sad because`、`One day, Tom and his mom went to`、`There was a big tree in the park`
   - 说明文：`Photosynthesis is the process`、`The water cycle describes`、`In mathematics, a prime number`、`The French Revolution began`、`To stay healthy, people should`
   
   每个模型 `temperature=0.8, top_k=50, seed=42` 各生成 1 条 200 token。三位组员**独立盲评**（隐藏模型名），每条按 1–5 分打：语法通顺度、内容连贯性（前后一致、不跑题）、事实合理性（说明文类）、重复程度（越少越高分）。取三人平均，分故事/说明文两类列表比较。
5. **学习过程**：s100m 的 `ckpt_tok100M` / `ckpt_tok300M` / `ckpt_tok1000M` / 最终 四个 checkpoint 用同一 prompt 生成，展示"从乱码 → 有词 → 有句子 → 有内容"的演变。

### 预期讨论点（写结论时回答）

- 参数越多 val loss 越低吗？从 10M 到 50M、从 50M 到 100M 的收益是否递减？
- 规模收益在 SmolLM（难）和 TinyStories（易）上一样大吗？（预期：TinyStories 上小模型就接近饱和，SmolLM 上大模型优势更明显）
- 相同 token 数下，大模型是否"学得更快"（曲线更低）？
- 按 Chinchilla 比例分配 token 后，三个模型的"计算量 vs loss"是否落在一条线上？
- 生成质量的差距能否被人感知？0.01B 能写故事但写不了说明文吗？
- 成本：s100m 多花约 20 倍的时间，换来多少 loss 下降？值不值？

## 实验 B：采样参数

- 模型：s100m（最好的模型），训练成本为零。
- 网格：temperature ∈ {0.3, 0.7, 1.0, 1.5} × top-k ∈ {不限, 50}，外加 greedy（T=0）和 top-p=0.9 各一组。
- 每种设置、同 4 个 prompt（故事、说明文各 2 个）、各 3 个 seed。
- 记录：样例、平均重复率（如 distinct-2：不同 bigram 数 / 总 bigram 数）、人工感受。
- 预期：低温度重复、保守；高温度出现拼写错误和不连贯；top-k 能在高温下"兜底"。

## 实验 C（可选）：分词方式

- s10m 结构、相同 3 亿 token 预算，分别用字符级（过滤低频字符后词表约 200）和 BPE-8192 训练。
- **不能直接比 loss**：两者每个 token 代表的信息量不同。换算成 **bits-per-byte**：`bpb = val_loss / ln(2) × (token 数 / 字节数)`。
- 只在时间充裕时做（需要额外约 1 小时训练）。注意字符级下 3 亿 token 只对应约 3 亿字节文本，远少于 BPE 下 3 亿 token 覆盖的文本量，结论要写明这一点。

## 实验 D（扩展）：KV cache 生成加速

- 变量：生成时 `use_cache=False`（每步重算完整前缀）vs `True`（复用每层 K/V）。
- 固定：同一个 checkpoint、同一 prompt、同一采样参数（greedy）、同样多的新 token；先预热，再重复多次取最快一次。
- 指标：tok/s、加速比、两种模式生成的 token 是否一致。
- 预期：上下文越长、模型越大，加速越明显；batch=1 的小模型主要受每步固定开销限制；超过 `block_size` 后窗口重建，加速消失。
- 结果写到 [report/experiments.md](../report/experiments.md) 实验 D；测速命令见 `src/bench_kv_cache.py`。

## 实验 E：训练设置对比

- 固定：s10m（6 层 / 384 维 / 6 头）、BPE-8192、3 亿 token、warmup 200 + cosine、seed 1337，有效 batch 131072 token/步。
- 变量：学习率（1e-3 对照、5e-4、2e-3）、数据配比（混合语料 vs 只用 TinyStories）。
- 本机（RTX 5060 Laptop）加一组 `s10m_ctl` 作为硬件对照，和云端 5090 的 `s10m` 比较。
- 指标：加权 val、分来源 val、最大 grad norm、显存、有效训练时间、tok/s。
- 结果写到 [report/experiments.md](../report/experiments.md) 实验 E；配置见 `configs/s10m_{ctl,lr5e-4,lr2e-3,tts}.yaml`，绘图用 `src/plot_ablation.py`。

## 验收标准

```bash
python src/plot_compare.py --runs runs/s10m runs/s50m runs/s100m --out docs/figures/
python src/eval_prompts.py --runs runs/s10m runs/s50m runs/s100m --out docs/results/samples_A.jsonl
python src/eval_prompts.py --runs runs/s100m --grid sampling --out docs/results/samples_B.jsonl
```

- `docs/figures/` 下有 val loss 叠图（加权 + 两个来源各一张，横轴 tokens）、scaling 图（分来源）、train-val gap 图。
- `docs/report/experiments.md` 包含：实验设置表、最终结果表、样例表、人工打分表、结论（每个实验至少 3 条有数据支撑的结论）。

## 实验记录模板（每个 run 一条，写进 `docs/report/experiments.md`）

```markdown
### <run 名>
- 变量：...
- 配置：configs/<name>.yaml（commit: xxxx）
- 开始/结束时间、总时长：
- 最终 train / val loss、最佳 val loss（step）：
- tok/s、显存峰值：
- 样例（prompt → 续写）：
- 观察与结论：
```

## 常见坑

- 三个模型训练 token 数不同，叠图横轴必须用**已训练 token 数**而不是 step（有效 batch 相同时两者成正比，但要明确标注）。
- 只看加权 val loss 会掩盖两个来源的不同趋势，必须分来源报告。
- 比较时用 `ckpt_best.pt` 还是最终 checkpoint 要统一并写明。
- 盲评时别让打分人知道是哪个模型。
- 生成样例要固定 seed，否则无法复现。

## 答辩可能被问的问题

- 你们的对比公平吗？控制了哪些变量？
- 为什么大模型学习率要小？
- 什么是 scaling law？你们的结果符合吗？
- 三个模型训练的 token 数不同，怎么保证比较公平？
- 0.1B 模型数据够吗？怎么判断欠拟合还是过拟合？
- 为什么小模型在 TinyStories 上还行，在 SmolLM 上差很多？
- temperature 设多少最好？为什么？
