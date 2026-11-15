# 阶段 4 · 训练循环

## 目标

`src/train.py` 读取 yaml 配置，完成：AdamW、warmup + cosine 学习率、bf16 混合精度、梯度累积、梯度裁剪、定期验证、checkpoint（含 optimizer，可断点续训）、固定随机种子、CSV 日志、自动画 loss 曲线。debug 模型完整训练一次，s10m 冒烟测试通过。

## 前置条件

- 阶段 2 的 `train.bin` / `val.bin` / `meta.json`。
- 阶段 3 的 `GPT` 通过全部测试（尤其是过拟合测试）。

## 要写的文件与函数签名

```
configs/debug.yaml  configs/s10m.yaml  configs/s50m.yaml  configs/s100m.yaml
src/train.py
src/utils.py        （配置读取、种子、日志、画图、get_batch）
```

```python
# src/utils.py
def load_config(path: str, overrides: list[str]) -> dict: ...   # 支持 --max_iters 50 这类覆盖
def set_seed(seed: int) -> None: ...                              # random / numpy / torch / cuda
def get_lr(it: int, cfg: dict) -> float: ...                      # warmup + cosine
class CSVLogger:
    def __init__(self, path: str, fields: list[str]) -> None: ...
    def log(self, row: dict) -> None: ...
def plot_loss(log_path: str, out_png: str) -> None: ...

# src/train.py
@torch.no_grad()
def estimate_loss(model: GPT, cfg: dict) -> dict[str, float]: ...
    # {"train": ..., "val_smollm": ..., "val_tinystories": ..., "val": 按 mix 加权}
def save_checkpoint(path: str, model: GPT, optimizer, it: int, best_val: float, cfg: dict) -> None: ...
def load_checkpoint(path: str, model: GPT, optimizer) -> tuple[int, float]: ...
def train(cfg: dict) -> None: ...
def main() -> None: ...   # --config configs/s10m.yaml [--resume] [--max_iters 50] [--name xxx]
```

## 配置

三个正式配置**除模型结构、micro-batch、学习率和 token 预算（`max_iters`）外完全相同**：同数据、同混合比例、同 seed、同有效 batch、同 warmup。

```yaml
# configs/s10m.yaml
name: s10m
model: { preset: s10m, vocab_size: 8192, block_size: 512, dropout: 0.0, bias: true, grad_ckpt: false }
data:
  dir: data/processed
  sources: [smollm, tinystories]
  mix: natural            # 按各来源训练 token 数的比例混合；也可写成 {smollm: 0.67, tinystories: 0.33}
train:
  batch_size: 128         # micro-batch（云端 RTX 5090；本地 8GB 用 64 × 4）
  grad_accum: 2           # 有效 batch = 128 × 2 × 512 = 131072 token/步
  max_iters: 2290         # 2290 × 131072 ≈ 3.0 亿 token
  lr: 1.0e-3
  min_lr: 1.0e-4          # = lr / 10
  warmup_iters: 200
  weight_decay: 0.1
  betas: [0.9, 0.95]
  grad_clip: 1.0
  eval_interval: 200
  eval_iters: 50          # 每个来源各 50 个 batch
  ckpt_tokens: [1.0e8, 3.0e8, 1.0e9]   # 到达这些已训练 token 数时额外保存 ckpt_tok{N}.pt
  log_interval: 10
seed: 1337
dtype: bfloat16
compile: true             # WSL2/Linux 可开；原生 Windows 设 false
```

| 配置 | micro-batch × accum（云端 5090） | 本地 8GB 显卡 | lr | `max_iters` | 训练 token | 实际用时（5090） |
|---|---|---|---|---|---|---|
| debug | 32 × 1 | 32 × 1 | 1e-3 | 1200 | ~2000 万 | 0.6 分钟（本地 5060 Ti） |
| s10m | 128 × 2 | 64 × 4 | 1e-3 | 2290 | 3.0 亿 | **4.6 分钟** |
| s50m | 64 × 4 | 32 × 8 | 6e-4 | 7630 | 10.0 亿 | **45 分钟** |
| s100m | 64 × 4 | 16 × 16 | 4e-4 | 11450 | 15.0 亿 | **1 小时 56 分** |

- 模型越大学习率越小：参数越多，同样的步长造成的输出变化越大，容易不稳定。
- 每个模型的 cosine 调度按自己的 `max_iters` 走完，所以 s50m、s100m 在 3 亿 token 时学习率还很高——这是"相同 token 数对比"时要说明的混杂因素。
- 有效 batch 统一 131072 token/步（= 256 × 512）。换显卡时只调 micro-batch 和 `grad_accum`，保持两者乘积 256 不变；三个模型的对比因此不受硬件影响。

## 实现要点

### 训练主循环（伪代码）

```python
set_seed(cfg.seed)
model = GPT(cfg).to(device); optimizer = model.configure_optimizers(...)
if resume: it, best_val = load_checkpoint(...)
if cfg.compile: model = torch.compile(model)
while it < max_iters:
    lr = get_lr(it); 设置 optimizer 每个 param_group 的 lr
    if it % eval_interval == 0: estimate_loss → 记日志 → 若 val 最优则存 ckpt_best.pt；存 ckpt_last.pt
    for micro in range(grad_accum):
        x, y = get_batch("train")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        (loss / grad_accum).backward()      # 梯度累积：多次小 batch 的梯度相加 ≈ 一个大 batch
    clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    it += 1
plot_loss(...)
```

- **warmup + cosine**：前 `warmup_iters` 步学习率从 0 线性升到 `lr`（刚开始参数随机、梯度方向不可靠，Adam 的二阶矩估计也不准，大步长容易发散）；之后按余弦曲线降到 `min_lr`（后期小步长精细收敛）。
- **bf16 autocast**：矩阵乘法用 bf16 算，参数和优化器状态保持 fp32。bf16 的指数位和 fp32 一样多，表示范围相同，梯度不会下溢，所以**不需要 GradScaler**（fp16 才需要）。
- **梯度累积**：显存放不下大 batch 时，把一个大 batch 拆成 `grad_accum` 个 micro-batch，反向时梯度会自动累加，loss 要除以 `grad_accum` 才是平均。
- **梯度裁剪**：梯度范数超过 1.0 就整体缩放到 1.0，防止偶发的大梯度把模型"打飞"。记录裁剪前的梯度范数到日志，便于排查。
- **estimate_loss**：`model.eval()` + `torch.no_grad()`，在 train 和 val 上各取 `eval_iters` 个 batch 求平均，结束后 `model.train()`。
- **固定种子**：`random`、`numpy`、`torch`、`torch.cuda` 全部设置。注意 GPU 上部分算子不保证逐位复现，loss 曲线大致一致即可。
- `torch.backends.cuda.matmul.allow_tf32 = True`：fp32 部分也用 TF32 加速。

### checkpoint 与断点续训

- 内容：`model.state_dict()`（compile 后要取 `model._orig_mod`）、`optimizer.state_dict()`、`iter`、`best_val`、完整 `config`、`meta.json` 内容。
- 文件：`ckpt_last.pt`（每次 eval 覆盖）、`ckpt_best.pt`（val 最优）、`ckpt_tok{N}.pt`（已训练 token 数到达 `ckpt_tokens` 中的 1 亿 / 3 亿 / 10 亿时各存一次，供实验 A 的"相同 token 数对比"和"学习过程"）。
- `--resume`：从 `runs/<name>/ckpt_last.pt` 恢复模型、优化器和步数，学习率按步数自动接上。s50m 约 7 小时、s100m 约 20 小时，必须分多次跑；每次续训后检查 loss 曲线是否连续。
- 续训时还要恢复随机数状态（`torch.get_rng_state()`、`torch.cuda.get_rng_state()`），保证数据采样序列可复现。

### 日志与输出目录

```
runs/<name>/
  config.yaml          完整配置副本（含命令行覆盖后的值）
  meta.json            数据与分词器信息副本
  log.csv              step, tokens, train_loss, val_loss, val_smollm, val_tinystories, lr, grad_norm, tokens_per_sec, mem_gb, elapsed_s
  loss_curve.png       训练结束自动生成（train/val 两条线，横轴 step，另一张图横轴 tokens）
  ckpt_last.pt  ckpt_best.pt  ckpt_tok100M.pt  ckpt_tok300M.pt  ckpt_tok1000M.pt
```

- `runs/<name>` 已存在且不是 `--resume` 时**直接报错退出**，不覆盖已有结果（CLAUDE.md 规则）。冒烟测试用 `--name smoke_xxx`。
- 每次 eval 打印：step、loss、lr、tok/s、显存峰值（`torch.cuda.max_memory_allocated()`）、预计剩余时间。

### 资源估算方法

- 显存 ≈ 16 字节/参数（fp32 参数 + 梯度 + AdamW 的 m、v）+ 激活。激活随 `B × T × C × L` 增长，SDPA 避免了 `T×T` 的注意力矩阵。
- 每 token 计算量 ≈ `6N + 12·L·T·C` FLOPs。用冒烟测试的实测 tok/s 估算：总时长 = `max_iters × 有效 batch token 数 / tok/s`。
- 初估（5060 Ti）：s10m ≈ 1h；s50m ≈ 7h；s100m ≈ 20h。本地冒烟实测：18min / 3.6h / 8.9h（合计约 13h）。**正式训练改在云端 RTX 5090 上进行，实际合计不到 3 小时**，见下方"正式训练结果"。**每个正式训练开跑前先报告配置、预计时长、显存并确认。**
- 长时间训练用 `tmux` 或 `nohup python src/train.py ... > runs/<name>.out 2>&1 &` 后台跑。

### OOM 预案（按顺序）

1. 减 micro-batch（16 → 8 → 4），同时把 `grad_accum` 加倍，保持有效 batch 不变。
2. 缩短上下文 `block_size`（512 → 256；会影响对比公平性，三个模型要一起改）。
3. 开 `grad_ckpt: true`。
4. 缩小模型（最后手段）。

## 验收标准

```bash
# 冒烟：50 步，确认不报错、loss 在下降、日志和图都生成
python src/train.py --config configs/debug.yaml --max_iters 50 --name smoke_debug
python src/train.py --config configs/s100m.yaml --max_iters 50 --name smoke_s100m   # 记录 tok/s 和显存峰值
# 断点续训：跑 30 步中断，--resume 后 step 从 30 继续，loss 曲线连续
# debug 完整训练（几分钟）
python src/train.py --config configs/debug.yaml
```

- debug：val loss 从 ~9.0 明显下降，TinyStories 上的 val loss 应明显低于 SmolLM 上的（具体值以实测为准），`loss_curve.png` 生成。
- s100m 冒烟：显存峰值 < 7.5GB；把实测 tok/s 换算成完整训练时长，更新 00-roadmap.md 的估算表。
- 重新运行同名实验会被拒绝。

## 常见坑

- `torch.compile` 第一次编译要 1–2 分钟，tok/s 要看编译之后的。
- compile 后的 `state_dict` 键名带 `_orig_mod.` 前缀，保存前取原模型。
- 忘了 `loss / grad_accum` → 等效学习率被放大 `grad_accum` 倍。
- eval 后忘了 `model.train()` → 之后 dropout 一直关着（本项目 dropout=0 影响不大，但要写对）。
- 日志里记录的 train loss 要乘回 `grad_accum`（或单独记录未除的值）。
- 学习率按 step 计算而不是按 epoch；续训时 `it` 要正确恢复。
- 20 小时的训练中途可能遇到 Windows 更新重启、WSL 被关闭：`ckpt_last.pt` 每次 eval（约 200 步）都保存，最多损失几十分钟；写 checkpoint 时先写临时文件再 `os.replace`，防止写到一半断电导致文件损坏。
- 长训练期间每个来源的 val loss 都要看：如果只有 SmolLM 在降而 TinyStories 不动（或反过来），检查混合权重。
- **必须在激活的 conda 环境里运行**（`conda activate tiny-llm` 或 `conda run -n tiny-llm python ...`）。直接调用 `envs/tiny-llm/bin/python` 时 `PATH` 里没有环境的 `bin/`，`torch.compile` 会报 `Cannot find ptxas-blackwell`（Triton 找不到 conda 装的 `ptxas`）。
- **PyYAML 按 YAML 1.1 解析**：`1.0e8`、`3e-4` 这类写法会被读成**字符串**，浮点数的科学计数法要写成 `1.0e+8`、`3.0e-4`（带小数点和指数符号）。配置里已按此写法；命令行覆盖时 `load_config` 会再尝试转成 float。

## 实测结果（2026-11-14，阶段 4 验收）

| 检查 | 命令 / 名称 | 结果 |
|---|---|---|
| 单元测试 | `pytest tests/ -v` | 24 项全过（含 `test_train.py`：学习率、配置覆盖、同名拒绝、续训与不中断逐位一致） |
| 过拟合 | `--overfit --name overfit_debug --max_iters 300` | 固定一个 batch，loss 9.06 → 0.0003 |
| 冒烟 | `smoke_debug`（50 步） | 初始 val 9.06 ≈ ln 8192 = 9.01；50 步后 7.44 |
| 同名拒绝 | 再跑 `smoke_debug` | `FileExistsError`，未覆盖 |
| 断点续训 | `smoke_resume`：`--max_iters 60 --stop_iter 30`，再 `--resume` | 从 step 30 接上；step 40 loss 7.8305 与不中断运行完全相同 |
| 吞吐/显存 | `smoke_s10m` / `smoke_s50m` / `smoke_s100m` | 280k / 78k / 47k tok/s；峰值 4.7 / 5.7 / 5.6 GB（均 < 7.5GB） |
| debug 完整训练 | `runs/debug`（1200 步，1966 万 token，0.6 分钟） | val 9.06 → **4.40**；SmolLM 4.99，TinyStories **2.99**（明显更低，符合预期） |

`--stop_iter N`：跑到第 N 步存档退出，cosine 调度仍按 `max_iters` 计算，下次 `--resume` 无缝接上。适合把 s100m（~9h）拆成两晚。

## 正式训练结果（云端 RTX 5090，2026-11-15）

本地 5060 Ti（8GB）只用来开发和做冒烟测试，三个正式训练在云端 **RTX 5090（32GB，同为 Blackwell sm_120）** 上完成。软件环境按 `environment.yml` 创建，代码与配置和本地完全相同。显存更大，所以把 micro-batch 调大、`grad_accum` 相应调小，**有效 batch 保持 131072 token/步不变**。三次训练都一次跑完，中间没有续训。

| 模型 | 参数（非 emb / 总） | batch × accum | 步数 | 训练 token | 训练用时 | 平均吞吐 | 显存峰值 |
|---|---|---|---|---|---|---|---|
| s10m | 10.65M / 13.99M | 128 × 2 | 2290 | 3.0 亿 | 4.6 分钟 | 1113k tok/s | 9.2 GB |
| s50m | 49.24M / 54.81M | 64 × 4 | 7630 | 10.0 亿 | 45.2 分钟 | 370k tok/s | 10.4 GB |
| s100m | 99.23M / 105.92M | 64 × 4 | 11450 | 15.0 亿 | 115.9 分钟 | 216k tok/s | 16.5 GB |

- 训练用时是 `log.csv` 的 `elapsed_s`，只算训练步，不含评估和 `torch.compile` 编译，实际墙钟时间稍长一些。
- 显存峰值来自 `torch.cuda.max_memory_allocated()`。
- s100m 的吞吐约 216k tok/s，相当于约 150 TFLOPS 的有效算力（按 `6N + 12·L·T·C` 估算）；同样的模型在本地 5060 Ti 上是 47k tok/s。

**最终 loss**（最后一步评估，`ckpt_last.pt`；val 全程单调下降，所以 `ckpt_best.pt` 也是这一步）

| 模型 | train | val（按 mix 加权） | val SmolLM | val TinyStories | token/参数 |
|---|---|---|---|---|---|
| s10m | 3.096 | **3.098** | 3.629 | 1.845 | ~21 |
| s50m | 2.535 | **2.544** | 3.016 | 1.426 | ~18 |
| s100m | 2.389 | **2.409** | 2.865 | 1.330 | ~14 |

**观察**（详细分析放在阶段 6）

- **规模越大 loss 越低，但边际收益递减**：s10m→s50m 降了 0.55，s50m→s100m 只降了 0.14。
- **两个语料差距很大**：TinyStories 的词汇和句式简单、可预测性强，loss 比 SmolLM 低 1.5 左右。
- **几乎没有过拟合**：train 和 val 基本重合。s100m 也只把训练数据过了约 0.92 遍，其余两个更少。
- **训练稳定**：warmup 之后梯度范数最大约 1.08（s100m），没有出现 loss 尖峰或 NaN。
- **已经进入收益递减阶段**：最后 20% 的 token 只让 val 再降了 0.025–0.031（约 1%），说明按这个 token 预算已经训练充分。
- **相同 token 数下大模型更好**：在约 3 亿 token 处，s10m 的 val 是 3.098，s50m 约 2.86，s100m 约 2.82（取 step 2200 和 2400 两次评估之间的值）。但 s10m 那时学习率已经退火到最低，另外两个还没有，所以严格比较要在阶段 6 用 `ckpt_tok300M.pt` 来做。

结果文件：`runs/{s10m,s50m,s100m}/` 下的 `config.yaml`、`log.csv`、`loss_curve.png`，以及 `runs/*.out`。checkpoint 不进 git。

## 答辩可能被问的问题

- 为什么用 AdamW 而不是 SGD？AdamW 与 Adam 的区别？
- warmup 解决什么问题？为什么后期要降学习率？
- 梯度累积和直接用大 batch 等价吗？（数学上近似等价，差别只在 BatchNorm 之类依赖 batch 的层，本模型没有）
- 为什么 bf16 不需要 GradScaler？
- 训练 loss 和验证 loss 的差距说明了什么？
- 一个 epoch 是多少步？你们的模型看了几遍数据？
