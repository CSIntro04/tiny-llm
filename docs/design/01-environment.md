# 阶段 1 · 环境

## 目标

建好一个可复现的 conda 环境 `tiny-llm`（Python 3.12），PyTorch 能在 RTX 5060 Ti（sm_120）上用 bf16 和 SDPA 正常运行。所有下载走清华 TUNA 镜像。

## 前置条件

- 已装 NVIDIA 驱动（WSL2 下装 Windows 端驱动即可，WSL 内**不要**再装 Linux 显卡驱动），`nvidia-smi` 能看到显卡。
- 已装 Miniconda（安装包可从 `https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/` 下载）。

## 要写的文件

```
environment.yml        conda 环境定义（主）
environment-cpu.yml    没有 NVIDIA 显卡时用的 CPU 版（Mac、无独显电脑）
requirements.txt       pip 参考清单（辅，版本与 environment.yml 一致）
.gitignore
src/check_env.py       环境自检脚本
```

```python
# src/check_env.py
def check_torch() -> None: ...       # 打印 torch 版本、CUDA 版本、arch 列表
def check_gpu() -> None: ...         # 设备名、显存、compute capability == (12, 0)
def check_bf16() -> None: ...        # bf16 矩阵乘法能跑，结果与 fp32 接近
def check_sdpa() -> None: ...        # F.scaled_dot_product_attention(is_causal=True) 能跑
def check_compile() -> None: ...     # 可选：torch.compile 一个小函数
if __name__ == "__main__": ...       # 依次运行，每项打印 OK / FAIL
```

## 实现要点

### 1. 配置 conda 镜像（`~/.condarc`）

```yaml
channels:
  - conda-forge
show_channel_urls: true
channel_priority: strict
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

然后 `conda clean -i` 清索引缓存。`channels` 里**只写 conda-forge，不写 defaults**。

> 为什么不混用 `defaults` 和 `conda-forge`：两个频道各自编译同名库（如 libstdc++、openblas），混装时容易出现二进制不兼容（ABI 冲突），求解依赖也会变慢。`channel_priority: strict` 保证同名包只从优先级最高的频道取。

### 2. pip 镜像（仅兜底用）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. `environment.yml`

```yaml
name: tiny-llm
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pytorch>=2.7          # conda-forge 的 CUDA 构建
  - cuda-version>=12.8    # 约束 CUDA 版本，50 系显卡需要 12.8+
  - numpy
  - tokenizers            # 训练 BPE
  - pyarrow               # 读 parquet
  - datasets              # 可选；下载用 huggingface_hub（datasets 的依赖）
  - pyyaml
  - matplotlib
  - gradio
  - tqdm
  - pytest
  - jupyterlab
```

```bash
conda env create -f environment.yml
conda activate tiny-llm
python src/check_env.py
```

**原则：能 `conda install` 就不用 pip。** 以后新加依赖，先 `conda search -c conda-forge <包名>`，有就用 conda 装并写进 `environment.yml`。

> 为什么尽量不 conda/pip 混装：conda 和 pip 是两套互不知情的包管理器。pip 装的包可能覆盖 conda 装的依赖，之后 `conda update` 又可能把它改回去，环境就"坏得莫名其妙"。

### 4. PyTorch 关键检查与兜底

装完必须满足：

```python
import torch
torch.__version__            # >= 2.7
torch.version.cuda           # >= 12.8
'sm_120' in torch.cuda.get_arch_list()   # True，否则会报 no kernel image
torch.cuda.get_device_capability()       # (12, 0)
```

**唯一允许 pip 的例外**：如果 conda-forge 上解析出来的 PyTorch 不包含 sm_120（`get_arch_list()` 里没有），在 `tiny-llm` 环境内：

```bash
conda remove pytorch --force
pip install "torch>=2.8"     # 走 TUNA 源；Linux 上 torch 2.8+ 的默认 wheel 即为 CUDA 12.8 构建
```

再跑一遍自检，并在本文件末尾的"环境记录"里写明用了 pip 兜底及原因。

### 5. `requirements.txt`

列出与 `environment.yml` 相同的包和版本（`conda list --export` 后手动整理主要包），供不用 conda 的人参考。

### 6. WSL2 / Windows 说明

- 当前开发环境是 **WSL2**：`torch.compile` 可以开（配置项 `compile: true`），Triton 可用。
- 原生 Windows：`compile` 默认关闭；DataLoader（如果用）`num_workers=0` 或加 `if __name__ == "__main__"`。
- WSL2 下数据放在 Linux 文件系统（如 `~/Codes/tiny-llm/data`），**不要**放 `/mnt/c/...`，跨文件系统读写很慢。

### 6.1 没有 NVIDIA 显卡（Mac / 无独显电脑）

`environment.yml` 用 `pytorch=...=cuda*` 强制装 GPU 版，Mac 和没有 NVIDIA 显卡的电脑装不上。这时用 `environment-cpu.yml`：

```bash
conda env create -f environment-cpu.yml   # 环境名同样是 tiny-llm，后面的命令都不用改
conda activate tiny-llm
python src/check_env.py                   # 自动进入 CPU 模式：检查版本、CPU 上的 bf16 和 SDPA
```

- 和 GPU 版的区别只有两点：PyTorch 是 CPU 构建（`=cpu*`），去掉了 `cuda-version`、`cuda-cudart-dev`、`gcc`。其余包版本相同。
- 已在 linux-64、win-64、osx-arm64 三个平台上解析通过。Mac 需要 Apple 芯片（M1 及以后）和 macOS 12 或更新；conda-forge 的 Mac 版 PyTorch 只有 CPU 构建，代码也没有用 Apple 的 MPS 后端。
- 代码在没有 GPU 时自动用 CPU：`train.py` 会自动关闭 `torch.compile`（CPU 上它要现场调 C++ 编译器，Mac 上通常没有，而且几乎不加速）；`sample.py`、两套界面、`demo_check.py` 都会退回 CPU。
- 速度参考：Ryzen 5 9600X（6 核，支持 AVX512-BF16）训练 debug 模型约 12k tok/s，完整跑完 `configs/debug.yaml`（约 2000 万 token）25–30 分钟；生成 s10m 约 230 tok/s、s100m 约 40 tok/s，演示和开发界面都够用。笔记本 CPU 会慢一些。
- **CPU 上的精度坑**：配置默认 `dtype: bfloat16`。bf16 矩阵乘法在 CPU 上要靠指令集加速：支持 AVX512-BF16 的 Zen 4 等没问题；Intel 12 代以后的大小核（如 Core Ultra 5 125H）没有 AVX512/AMX，会退化成极慢的软件实现——实测 30 分钟走不完 1 步。这类机器训练时加 `--dtype float32`：实测 2.4 秒/步、约 6.3–6.5k tok/s（18 核），完整 debug 训练约 1 小时。生成、评估和测试不受影响，只有训练需要改。
- CPU 上浮点运算的顺序和 GPU 不同，同一个 seed 生成的文字会在若干 token 后和 GPU 的记录分叉，`demo_check.py` 对此只提示不判失败。

### 7. `.gitignore`

```
data/raw/
data/processed/*.bin
runs/*/ckpt*.pt
__pycache__/
*.pyc
.ipynb_checkpoints/
.gradio/
```

## 验收标准

```bash
conda activate tiny-llm
python --version                     # Python 3.12.x
conda list | grep -E "^pytorch|^torch"   # 渠道列为 conda-forge（或已记录的 pip 兜底）
python src/check_env.py
```

预期输出类似：

```
[OK] torch 2.x.x, CUDA 12.8, arch list contains sm_120
[OK] NVIDIA GeForce RTX 5060 Ti, 8.0 GB, capability (12, 0)
[OK] bf16 matmul, max abs diff vs fp32 < 1e-1
[OK] SDPA causal attention
[OK] torch.compile
```

## 常见坑

- `CUDA error: no kernel image is available for execution on the device`：PyTorch 没编译 sm_120，见上面兜底。
- `conda env create` 很慢或卡住：检查 `.condarc` 是否还残留 `defaults`；用 `conda config --show channels` 确认。新版 conda 默认用 libmamba 求解器，速度应可接受。
- WSL 里 `nvidia-smi` 找不到：Windows 端驱动太旧，升级 Windows 驱动。
- conda-forge 的 pytorch 可能装成 CPU 版：检查 `torch.cuda.is_available()`；可以显式写 `pytorch=*=cuda*` 约束构建字符串。
- Miniconda 安装目录自带 `~/miniconda3/.condarc`，里面写了 `channels: [defaults]`，会和 `~/.condarc` 合并。用 `conda config --file ~/miniconda3/.condarc --remove channels defaults` 去掉，再用 `conda config --show-sources` 确认。
- `torch.compile` 报 `Failed to find C compiler` 或 `cuda.h: No such file or directory`：Triton 编译 kernel 时要现场用 C 编译器编一个小的启动器（launcher），需要 `gcc` 和 CUDA 头文件。从 conda-forge 装 `gcc` 和 `cuda-cudart-dev`（注意 `cuda-driver-dev` 只有 `libcuda.so` 的 stub，没有 `cuda.h`）。
- `curl: (6) Could not resolve host: huggingface.co`：国内网络常见，数据下载前先 `export HF_ENDPOINT=https://hf-mirror.com`。`prepare.py` 会跳过已下载的文件，可以放心重跑。
- `conda env create` 报 `CondaValueError: prefix already exists`：环境名 `tiny-llm` 已被占用。不要直接删旧环境，可以 `conda env create -f environment-cpu.yml -n tiny-llm-repro` 建一个副本，命令里用 `conda run -n tiny-llm-repro ...` 或 `conda activate tiny-llm-repro`。
- CPU 训练极慢、日志长时间停在 `[eval] step 0`：多半是 bf16 在不支持 AVX512-BF16 的 CPU 上走了软件实现，加 `--dtype float32`，见 6.1 节。
- `prepare.py` 中途被 Ctrl-C：`data/processed/*.bin` 可能只写了一半，`meta.json` 还是 git 里的旧版本（`shards: all`）。直接重跑同一条命令即可，下载会跳过、编码会覆盖。

## 答辩可能被问的问题

- 为什么 50 系显卡需要 CUDA 12.8？（新架构 sm_120 的 kernel 只有 12.8+ 的编译器能生成）
- bf16 和 fp16 有什么区别？为什么 bf16 不需要 GradScaler？（见 glossary）
- 为什么用 conda 而不是系统 Python？

## 环境记录

记录日期：2026-11-14

| 项目 | 值 |
|---|---|
| 系统 | WSL2（内核 6.18.58.1-microsoft-standard-WSL2），Ubuntu 26.04.1 LTS |
| Miniconda | `Miniconda3-latest`（py314_26.9.4-1），从 TUNA 下载，装在 `~/miniconda3` |
| conda | 26.9.4，频道仅 conda-forge（TUNA 镜像），`channel_priority: strict` |
| Python | 3.12.16 |
| PyTorch | 2.14.1，构建 `cuda129_mkl_py312_h0509a7e_300`，**来源 conda-forge**（未用 pip 兜底） |
| CUDA（PyTorch 构建） | 12.9 |
| Triton | 3.7.1（conda-forge） |
| torch.compile 额外依赖 | `gcc` 14.4.0、`cuda-cudart-dev` 12.9.79（conda-forge） |
| NVIDIA 驱动 | Windows 端 621.14（`nvidia-smi` 显示 CUDA UMD 13.5） |
| pip 镜像 | `https://pypi.tuna.tsinghua.edu.cn/simple`（目前没有用 pip 装任何包） |

`python src/check_env.py` 输出：

```
[OK] torch 2.14.1, CUDA 12.9, arch list contains sm_120
[OK] NVIDIA GeForce RTX 5060 Ti, 7.9 GB, capability (12, 0)
[OK] bf16 matmul, relative error vs fp32 = 2.88e-03
[OK] SDPA causal attention, max diff vs naive = 7.53e-03, causality OK
[OK] torch.compile

全部通过
```

已用锁定版本的 `environment.yml` 新建一个临时环境重新验证过，五项全部通过。

2026-11-17：新增 `environment-cpu.yml`。在本机（linux-64）用它新建临时环境验证：`check_env.py` CPU 模式三项通过，`pytest tests/` 42 项通过，debug 冒烟训练、`sample.py`、`demo_check.py` 均正常；osx-arm64 和 win-64 用 `conda env create --dry-run` 解析通过（未在真机上装过）。

### 2026-11-20 干净环境复现记录（无 GPU）

目的：验证"另一位组员在干净环境里按 README 从头跑到 debug 模型生成"。做法：用 `git worktree` 从 `main`（`8160728`）拉一份干净副本到 `/tmp/opencode/tiny-llm-repro`，新建独立环境 `tiny-llm-repro`，不碰主仓库的 `runs/` 和 `data/`。

| 项目 | 实测 |
|---|---|
| 机器 | WSL2，Ubuntu 26.04.1，Intel Core Ultra 5 125H（18 线程），15GB 内存，**无 NVIDIA 显卡** |
| conda | 26.9.4；`~/miniconda3/.condarc` 去掉 `defaults` 后只剩 conda-forge（TUNA），`channel_priority: strict` |
| 创建环境 | `conda env create -f environment-cpu.yml -n tiny-llm-repro`，**94 秒** |
| 版本 | Python 3.12.16，torch 2.14.1（CPU 构建），tokenizers 0.23.3，huggingface_hub 1.32.0 |
| `check_env.py` | CPU 模式三项 OK：bf16 relative error 2.88e-03；SDPA max diff 7.64e-03，causality OK |
| `pytest tests/` | **35 passed, 7 skipped，26.95 秒**；跳过的是 `tests/test_server.py` 里需要 `runs/s10m/ckpt_best.pt` 的 7 个用例（干净副本还没有 checkpoint） |
| 数据 | `HF_ENDPOINT=https://hf-mirror.com ... --shards 1`：下载 541MB，**3 分 36 秒**；训练集 2.54 亿 token（SmolLM 1.32 亿 + TinyStories 1.22 亿，52.06 : 47.94），验证集 549 万 |
| debug 训练 | bf16 不可用（30 分钟走不完 1 步），改 `--dtype float32`：约 **6.5k tok/s、2.4 秒/步，1200 步训练 50.8 分钟**；整条命令 66 分钟（含 7 次评估和启动） |
| 训练结果 | val 9.0566 → **3.9930**（SmolLM 5.0968 / TinyStories 2.7944），train 3.9466；`log.csv` 和 `loss_curve.png` 正常生成 |
| 生成 | `sample.py --run runs/repro_debug --prompt "Once upon a time" --max_new_tokens 80 --seed 42`：0.35 秒、**230 tok/s**，输出是连贯的英文故事片段 |

结论：README 的 CPU 路径可以从零跑通。过程中发现、并已补进上文的问题：Miniconda 自带的 `.condarc` 残留 `defaults`；`huggingface.co` DNS 解析失败，要改用 `hf-mirror.com`；没有 AVX512-BF16 的 CPU 上 bf16 训练不可用；干净副本里 pytest 是 35 passed / 7 skipped，不是全过。复现用的 worktree 和 conda 环境是临时的，验证完可以删除。
