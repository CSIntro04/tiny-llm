# 干净环境复现记录（2026-11-20/21）

> 目的：验证一位没参与开发的组员，能否只按根目录 [README.md](../../README.md) 从零跑到 debug 模型生成。
> 结论：**可以**。在一台没有 NVIDIA GPU 的 Linux 机器上，环境创建 94 秒、数据 3 分 36 秒、debug 训练 50.8 分钟，最终能正常生成；过程中发现的问题已补进 README 和 [01-environment.md](../design/01-environment.md)。

## 1. 复现环境

| 项目 | 值 |
|---|---|
| 时间 | 2026-11-20 22:55 – 2026-11-21 01:14（约 2 小时 20 分） |
| 系统 | WSL2，Ubuntu 26.04.1 LTS，内核 6.18.58.1-microsoft-standard-WSL2 |
| CPU | Intel Core Ultra 5 125H，18 线程 |
| 内存 / 磁盘 | 15GB / 可用 946GB |
| 显卡 | **无 NVIDIA GPU**（`nvidia-smi` 不存在），走 CPU 路径 |
| 仓库版本 | `main` 的 `8160728`（复现时最新；之后上游又合入了前端改动） |
| conda | 26.9.4，只用 conda-forge（清华 TUNA），`channel_priority: strict` |

为了不污染主仓库的 `runs/` 和 `data/`：

- 用 `git worktree` 拉了一份干净副本到 `/tmp/opencode/tiny-llm-repro`（detached HEAD）；
- 新建独立环境 `tiny-llm-repro`，没有覆盖原有的 `tiny-llm`。

conda 原配置里同时有 `conda-forge` 和 `defaults`（`channel_priority: flexible`），Miniconda 自带的 `~/miniconda3/.condarc` 也会合并进来。复现前先备份到 `/tmp/opencode/condarc-backup/`，然后：

```bash
conda config --file ~/miniconda3/.condarc --remove channels defaults
conda config --file ~/miniconda3/.condarc --set channel_priority strict
conda clean -i
```

## 2. 环境创建与自检

```bash
conda env create -f environment-cpu.yml -n tiny-llm-repro   # 94 秒
conda run -n tiny-llm-repro python src/check_env.py
conda run -n tiny-llm-repro pytest tests/
```

| 检查 | 结果 |
|---|---|
| 实际版本 | Python 3.12.16，torch 2.14.1（CPU 构建），tokenizers 0.23.3，huggingface_hub 1.32.0 |
| `check_env.py` | bf16 relative error 2.88e-03；SDPA max diff 7.64e-03，causality OK；CPU 模式三项全部通过 |
| `pytest tests/` | **35 passed, 7 skipped，26.95 秒** |

7 个 skipped 全部来自 `tests/test_server.py`，原因是没有 `runs/s10m/ckpt_best.pt`。干净副本本来就没有 checkpoint，属于预期；下载权重或自己训练后重跑即可全部执行。这一点 README 第 8 节已经写明。

## 3. 数据准备

```bash
export HF_ENDPOINT=https://hf-mirror.com
python src/prepare.py --datasets smollm,tinystories --tokenizer bpe --vocab_size 8192 --shards 1
```

- 第一次运行被手动中断：`smollm_train/val`、`tinystories_train` 已写完，但 `tinystories_val.bin` 和 `meta.json` 还没生成。直接重跑同一条命令，已下载的 parquet 会跳过。
- 重跑总用时 **3 分 36 秒**（训练 BPE 58 秒 + 编码约 1 分半），下载 541MB。
- 输出：原始 parquet 541MB，`data/processed/` 下 bin 共 496MB。

| 来源 | 训练文档 / token | 验证文档 / token | chars/token |
|---|---|---|---|
| SmolLM | 115,533 / 132,233,403 | 581 / 617,293 | 3.902 |
| TinyStories | 529,732 / 121,754,597 | 21,983 / 4,873,029 | 3.932 |
| 合计 | **253,988,000** | 5,490,322 | — |

自然混合权重：SmolLM **52.06%**、TinyStories **47.94%**。`--shards 1` 只下每个来源的第一个训练分片，所以比例和全量的约 70:30 不同，这是正常的。

> 注意：`meta.json`、`tokenizer.json` 在 git 里，重跑 `prepare.py` 会覆盖它们。复现时应在副本里跑，不要在主仓库直接跑。

## 4. 关键问题：CPU 上的 bf16 不可用

`configs/*.yaml` 默认 `dtype: bfloat16`。第一次直接按配置启动 debug 冒烟测试，日志停在 `[eval] step 0` 之后：

- 30 分钟走不完第 10 步（日志每 10 步才输出一次）；
- 进程只有一个线程 100% 满载，其余线程闲置；
- `log.csv` 里只有 step 0。

原因：这台机器的 Intel Core Ultra 5 125H（Meteor Lake）**没有 AVX512-BF16，也没有 AMX**。PyTorch 在 CPU 上遇到 bf16 矩阵乘法时退回到极慢的软件实现。支持 AVX512-BF16 的 Zen 4 等 CPU 不会遇到这个问题，所以之前 6 核 Ryzen 上记录的 25–30 分钟仍然成立。

改用 `--dtype float32` 后：

| 设置 | 结果 |
|---|---|
| bf16（默认） | 30 分钟走不完 1 个日志间隔，不可用 |
| fp32 | 5 步训练 12 秒，**2.4 秒/步，约 6.3k tok/s**，正常 |

生成、评估和测试不受影响，只有训练需要显式覆盖精度。

## 5. 完整 debug 训练

```bash
python src/train.py --config configs/debug.yaml --name repro_debug --dtype float32
```

| 项目 | 结果 |
|---|---|
| 模型 | debug 预设：4 层 / 256 维 / 4 头，5.39M 参数（非 embedding 3.16M） |
| 步数 / token | 1200 步 × 16,384 token = 19,660,800 token，约 7.7% 的一轮 |
| 吞吐 | 约 **6.5k tok/s**，2.4 秒/步 |
| 训练用时 | **50.8 分钟**（脚本统计）；整条命令 65.9 分钟（含 7 次评估和启动） |
| 最终 loss | train 3.9466，val **3.9930**（SmolLM 5.0968 / TinyStories 2.7944） |
| 初始 loss | val 9.0566，接近 ln(8192) ≈ 9.01，符合随机初始化预期 |

验证集 loss 变化：

| step | val（加权） | SmolLM | TinyStories |
|---|---|---|---|
| 0 | 9.0566 | 9.0682 | 9.0441 |
| 200 | 5.2478 | 6.2682 | 4.1396 |
| 400 | 4.7316 | 5.7468 | 3.6292 |
| 600 | 4.3910 | 5.4488 | 3.2422 |
| 800 | 4.1803 | 5.2688 | 2.9984 |
| 1000 | 4.0529 | 5.1520 | 2.8594 |
| 1200 | **3.9930** | 5.0968 | 2.7944 |

输出目录 `runs/repro_debug/`：`config.yaml`、`log.csv`、`loss_curve.png`、`meta.json`、`ckpt_best.pt`、`ckpt_last.pt`（含 AdamW 状态，各约 64.7MB）。曲线正常下降，没有出现 NaN 或发散。

## 6. 生成验证

```bash
python src/sample.py --run runs/repro_debug --prompt "Once upon a time" \
  --max_new_tokens 80 --temperature 0.8 --top_k 50 --seed 42
```

- 自动回退 CPU（默认 `--device cuda` 检测不到显卡）。
- 80 个 token 用时 **0.35 秒，约 230 tok/s**。
- 输出是连贯的英文故事片段，例如：

  > , there were two people in a house. One day, two children went for a little girl named Lily. They were playing outside and Lily loved playing in the big park…

  能看出语法基本正确、有简单的情节，但仍有小模型常见的指代混乱（"two children" 后面变成 "a little girl"）。这和 debug 模型只训练了 2000 万 token 的预期一致。

## 7. 本次改动

根据实测结果更新了三个文档（不改实验数字，不动 `runs/`）：

| 文件 | 改动 |
|---|---|
| `README.md` | 安装前三项检查（环境名 / condarc / 磁盘）；94 秒建环境实测；HF 镜像；CPU bf16 坑与 `--dtype float32`；`--shards 1` 实测统计；CPU 训练和生成速度；pytest 预期 35 passed / 7 skipped |
| `docs/design/01-environment.md` | 6.1 节补 CPU 精度说明；常见坑新增 HF DNS、环境重名、bf16 极慢、prepare 中断；新增 2026-11-20 复现记录 |
| `docs/defense/qa.md` | 新增「八、复现与环境」5 个问答；勾掉「干净环境复现」待办 |

提交方式（不直接改 `main`）：

- fork：`youshiyiniandong/tiny-llm`
- 分支：`docs/clean-env-repro`
- PR：<https://github.com/CSIntro04/tiny-llm/pull/3>

## 8. 遗留事项

- PR 等待人工审核，未合并。
- 干净副本里的 7 个服务端测试仍会跳过；要全部通过，需要先有 `runs/s10m/ckpt_best.pt`。
- 本次数据只下了 `--shards 1`，足够 debug 模型，但不能用来复现正式规模的训练。
- 复现用的 worktree、conda 环境和日志都在 `/tmp/opencode/`，确认不再需要后可以清理：

  ```bash
  git worktree remove /tmp/opencode/tiny-llm-repro
  conda env remove -n tiny-llm-repro
  ```

## 9. 完整命令清单

```bash
# 1. 干净副本与环境
git worktree add --detach /tmp/opencode/tiny-llm-repro HEAD
cd /tmp/opencode/tiny-llm-repro
conda env create -f environment-cpu.yml -n tiny-llm-repro
conda run -n tiny-llm-repro python src/check_env.py
conda run -n tiny-llm-repro pytest tests/

# 2. 数据
HF_ENDPOINT=https://hf-mirror.com conda run --no-capture-output -n tiny-llm-repro \
  python src/prepare.py --datasets smollm,tinystories --tokenizer bpe --vocab_size 8192 --shards 1

# 3. 冒烟（注意 --dtype float32）
conda run --no-capture-output -n tiny-llm-repro \
  python src/train.py --config configs/debug.yaml --name repro_smoke50 --max_iters 50 --dtype float32

# 4. 完整 debug 训练
conda run --no-capture-output -n tiny-llm-repro \
  python src/train.py --config configs/debug.yaml --name repro_debug --dtype float32

# 5. 生成
conda run --no-capture-output -n tiny-llm-repro \
  python src/sample.py --run runs/repro_debug --prompt "Once upon a time" \
  --max_new_tokens 80 --temperature 0.8 --top_k 50 --seed 42
```
