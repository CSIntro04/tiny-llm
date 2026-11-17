# CLAUDE.md — 迷你语言模型项目（Topic B03）

## 项目背景

- 这是上海交通大学 Fall 2026 计算机科学导论小组项目，题目是 **B03：从零训练一个迷你语言模型**。
- 课程要求：
  1. 准备文本数据集，完成预处理和词元化（tokenization）。
  2. **从零实现** Decoder-only Transformer，用 next-token prediction 训练，并记录训练损失曲线。
  3. 做交互界面：用户输入提示文本，查看续写结果，可以调生成长度、采样温度（temperature）等参数。
  4. 至少比较两种训练设置或采样参数下的生成结果。
  5. 说明数据来源、模型结构和训练设置。
- 重点是**理解流程**，不追求商业大模型的生成质量。答辩时会被追问原理，所以组员必须能看懂每一行核心代码。

## 硬件与环境

- 本地开发：CPU AMD Ryzen 5 9600X；GPU **RTX 5060 Ti，8GB 显存**（Blackwell 架构，sm_120）。
- 正式训练：云端 **RTX 5090，32GB 显存**（同为 sm_120，环境同 `environment.yml`）。`configs/` 里的 batch 按 5090 设置，本地 8GB 运行时按各配置注释里的值，用 `--batch_size` / `--grad_accum` 覆盖（乘积保持 256）。
- PyTorch 必须是 **2.7 及以上、CUDA 12.8 及以上**的版本，老版本不支持 50 系显卡。遇到 `no kernel image` 之类的报错，先查版本。
- 训练用 **bf16 混合精度**（`torch.autocast(dtype=torch.bfloat16)`），不需要 GradScaler。
- 注意力用 `F.scaled_dot_product_attention`，可以自动走 FlashAttention 来省显存。
- 如果在原生 Windows 上运行，`torch.compile` 默认关闭（Triton 支持不好），通过配置项 `compile` 来开关。DataLoader 的 `num_workers` 在 Windows 上设成 0 或者加 `if __name__ == "__main__"` 保护。
- 环境用 **Miniconda + Python 3.12**，环境名 `tiny-llm`。安装包一律走**清华 TUNA 镜像**（conda 和 pip 都是）；数据集从 HuggingFace 下载（见下方"工作方式"中的数据下载规则）。
- **能 `conda install`（conda-forge）就不用 pip**；pip 只作兜底（例如 conda-forge 没有支持 sm_120 的 PyTorch 构建时），用了要在文档里记录原因。不要混用 `defaults` 和 `conda-forge` 频道。
- 依赖以 `environment.yml` 为主，`requirements.txt` 为辅（版本保持一致）。没有 NVIDIA 显卡的组员（Mac、无独显电脑）用 `environment-cpu.yml`，三个文件的包版本要一起改。详见 `docs/design/01-environment.md`。

## 模型规模目标

本组目标：分别训练 **0.01B / 0.05B / 0.1B** 三个规模并对比（见 `TARGET.md` 末尾"本组目标调整"）。数据为 SmolLM-Corpus 1B 子集 + TinyStories 混合；词表 8192、上下文 512、head_size 64：

| 名称 | 用途 | 结构（层/维/头） | 非 embedding 参数 | 总参数 | 训练 token 预算 |
|---|---|---|---|---|---|
| debug | 流程验证、过拟合测试 | 4 / 256 / 4 | 3.16M | 5.39M | ~2000 万 |
| s10m | 0.01B | 6 / 384 / 6 | 10.65M | 13.99M | 3 亿 |
| s50m | 0.05B | 10 / 640 / 10 | 49.24M | 54.81M | 10 亿 |
| s100m | 0.1B | 14 / 768 / 12 | 99.23M | 105.92M | 15 亿 |

- token 预算按规模分配（约 14–21 token/参数，接近 Chinchilla 比例）。三个正式训练已于 2026-11-15 在云端 5090 上完成（合计约 2.8 小时，结果在 `runs/s10m|s50m|s100m`）；之后再开新的长时间训练仍要先确认。

- 上下文长度默认 512。8GB 显存吃紧时，优先减小 batch 并用梯度累积补回来，其次开梯度检查点。
- **不要超过 0.1B（约 1.1 亿参数）**，除非用户明确要求。
- **词表要小**：中文用字符级（过滤低频字，约 5000 字）；英文用自己训练的 BPE（4k–8k），或者字符级。不要直接用 GPT-2 的 5 万词表，否则嵌入层会占掉大部分参数。
- 新模型配置写好后，先打印参数量（区分是否包含 embedding），再估算显存和训练时间，告诉用户。

## 目录结构

```
data/            原始和处理后的数据（大文件不进 git）
  raw/
  processed/     {smollm,tinystories}_{train,val}.bin（uint16 token id）、meta.json、tokenizer.json
  processed_char/  字符级分词的同类文件（实验 C）
configs/         每个实验一个 yaml 配置
src/
  tokenizer.py   词元化（字符级 / BPE）
  prepare.py     数据清洗、切分训练集和验证集、编码保存
  model.py       Transformer 实现（手写）
  train.py       训练循环
  sample.py      命令行生成（generate / generate_stream 供界面复用）
  utils.py
  evaluate.py  eval_prompts.py  plot_compare.py  score_blind.py  export.py  demo_check.py  check_env.py
app/             交互界面：server.py（FastAPI）+ web/（React 前端，dist/ 已提交）；app.py 是 Gradio 备用版；common.py 共用
tests/           pytest 测试
runs/            每次训练的输出：checkpoint、日志、loss 曲线、配置副本
docs/            README.md 索引；DEVELOPING.md 继续开发指南
  design/        阶段设计文档 00–08（计划 + 实现记录）
  report/        交付文档：data.md  model.md  experiments.md
  defense/       答辩：qa.md  demo.md  glossary.md  slides/
  results/       生成样例 samples_*、盲评 blind_eval/（脚本生成）
  figures/       图和结果表（界面与 PPT 直接读这里，路径不要改）
```

## 实现约定

- **model.py 必须手写**：token embedding + 位置编码、多头因果自注意力、MLP、LayerNorm、残差连接、权重共享（输出层与 embedding 共享）。
  - 可以参考 nanoGPT 的结构，但**不能**直接用 HuggingFace 的 `GPT2LMHeadModel` 或 `Trainer` 之类的现成训练封装。
  - `torch.nn` 的基础层（Linear、Embedding、LayerNorm）可以用。
- 关键代码写**中文注释**，说清楚张量形状和这一步的原理。比如注意力里的 `(B, T, C) -> (B, nh, T, hs)`、为什么要因果掩码、为什么损失是 cross entropy。
- 训练循环要包含：AdamW、warmup + cosine 学习率、梯度裁剪、梯度累积、定期在验证集上评估、定期保存 checkpoint（包括 optimizer 状态，方便断点续训）、固定随机种子。
- 训练日志用 CSV 或 JSON 记录 step、train loss、val loss、学习率、每秒处理的 token 数，结束后自动画 loss 曲线保存成 png。每次运行把完整配置复制到 `runs/<实验名>/`。
- 采样支持 temperature、top-k（可选 top-p）、最大生成长度。
- 代码风格：函数短小，有类型标注，不搞过度抽象。这是学生项目，可读性优先于工程化。

## 常用命令

```bash
python src/prepare.py --datasets smollm,tinystories --tokenizer bpe --vocab_size 8192  # 处理数据
python src/train.py --config configs/s10m.yaml      # 训练
python src/sample.py --run runs/<实验名> --prompt "Once upon a time"  # 命令行生成
python app/app.py                                   # 启动交互界面
```

## 工作方式（给 Claude 的规则）

- **长时间训练先问**：预计超过 10 分钟的训练，先告诉用户配置、预计耗时和显存，用户确认后再跑。验证代码时用 `--max_iters 50` 之类的参数做冒烟测试。
- 改动模型结构或训练逻辑后，先跑一个**过拟合测试**：在一个很小的 batch 上，loss 应该能降到接近 0。
- 不要删除或覆盖 `runs/` 下已有的实验结果。新实验用新目录名。
- **数据下载**：下面两个数据集的指定文件**已经过用户确认**（总计约 3.7GB），按 `docs/design/02-data.md` 的两步计划下载即可，不用再问：
  - `ecreeth/1b-smollm-corpus`：`train-0000{0..8}-of-00009.parquet`，约 2.67GB。
  - `roneneldan/TinyStories`：`data/train-0000{0..3}-of-00004-*.parquet` + `data/validation-*.parquet`，约 1GB。
  - 不要下载这两个仓库里的其他大文件（如 `TinyStories-train.txt`、`TinyStoriesV2-GPT4-train.txt`、`TinyStories_all_data.tar.gz`、`MLP_input_output.npy`）。
  - 下载前检查磁盘剩余空间 ≥ 12GB；不够就停下来告诉用户。
  - 除此之外的任何数据集，或单次下载超过 1GB 的其他文件，仍要先和用户确认来源和大小。
- 报错时先解释原因，再改。显存不足（OOM）时按这个顺序处理：减 batch → 缩短上下文 → 开梯度检查点 → 缩小模型。
- 用户是在学习，涉及新概念（比如 RoPE、KV cache、梯度累积）时，用两三句话解释它是什么、为什么要用。
- 回复用中文；代码标识符用英文。

## 交付物清单

- [x] 数据集说明：来源、许可、大小、清洗步骤、词表大小
- [x] 至少一个训练好的模型和 loss 曲线
- [x] 至少两组对比实验（比如模型大小、数据量、学习率、temperature），保留生成样例
- [x] 交互界面（网页版 + Gradio 备用）：输入 prompt，调节长度和温度，选择不同的 checkpoint
- [x] 模型结构图和参数量统计
- [x] README：如何复现
