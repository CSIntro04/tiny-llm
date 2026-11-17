# 继续开发指南

写给要在这个项目上继续改代码、加实验的组员（以及 Claude Code）。项目规则的原文在根目录 [CLAUDE.md](../CLAUDE.md)，本文是它的操作版。

## 1. 当前状态（2026-11-15）

八个阶段全部完成：三个正式模型（s10m / s50m / s100m）已在云端 RTX 5090 上训练完，实验 A（规模）、B（采样参数）、C（字符级 vs BPE）都有结论，网页界面和答辩材料都已就绪。结果摘要见根 [README.md](../README.md) 第 7 节。

### 哪些在 git 里，哪些不在

| 东西 | 在哪 | 进 git？ | 说明 |
|---|---|---|---|
| 代码、配置、文档、图 | `src/` `app/` `configs/` `docs/` `tests/` | ✅ | |
| 前端构建产物 | `app/web/dist/` | ✅ | 演示机不用装 Node；改前端后要重新构建并提交 |
| 分词器和数据统计 | `data/processed/{meta,tokenizer}.json`、`data/processed_char/*.json` | ✅ | |
| 训练日志 | `runs/<run>/{config.yaml,log.csv,loss_curve.png,meta.json}` | ✅ | |
| 正式训练的终端输出 | `runs/{s10m,s50m,s100m,s10m_char}.out` | ✅ | 云端 `nohup` 的 stdout，记录了参数量、吞吐、每步 loss |
| 原始数据和 token 文件 | `data/raw/`、`data/**/*.bin` | ❌ | 约 8GB，用 `src/prepare.py` 重新生成 |
| checkpoint | `runs/*/ckpt*.pt` | ❌ | 含 AdamW 状态，可续训 |
| 冒烟/过拟合测试 run | `runs/smoke_*`、`runs/overfit_*` | ❌ | 随时可删、可重跑 |
| 仅权重的发布包 | `release/`（含 `tiny-llm-weights.tar`） | ❌ | `src/export.py` 生成，已上传交大云盘 |
| 云端训练结果的完整打包 | 根目录 `runs.tar.gz`（8.8GB） | ❌ | 从 5090 上拉回来的原始备份（含全部 checkpoint 和优化器状态），`runs/` 下的正式 run 就来自它。本机空间紧张时可以挪到移动硬盘；**删除之前**先确认 `runs/` 里每个正式 run 的 `ckpt_*.pt` 都在、能加载 |
| 答辩 PPT | `docs/defense/slides/B03-答辩.pptx` | ❌ | `build_slides.py` 生成；模板 `模板.pptx` 进 git |

### 新机器上恢复工作环境

1. 建环境：`conda env create -f environment.yml && conda activate tiny-llm && python src/check_env.py`（镜像配置见根 README 第 1 节）。没有 NVIDIA 显卡（Mac 等）把 `environment.yml` 换成 `environment-cpu.yml`。
2. 数据：`python src/prepare.py --datasets smollm,tinystories --tokenizer bpe --vocab_size 8192 --shards all`（约 25 分钟），或者直接从旧机器拷 `data/processed/*.bin`。
3. 模型：只需要生成 → 从交大云盘下载 `tiny-llm-weights.tar` 解压到 `runs/`；需要续训 → 用 `runs.tar.gz`（`tar xzf runs.tar.gz`，含优化器状态）。
4. 检查：`pytest tests/`，再 `python src/demo_check.py`。

## 2. 代码地图

```
parquet ──prepare.py──▶ data/processed/*.bin（uint16） + tokenizer.json + meta.json
                              │
                 utils.get_batch（按 mix 权重从各来源随机截取 513 个 token：x=前 512，y=后 512）
                              │
                 model.GPT ◀── train.py（AdamW、warmup+cosine、bf16、梯度累积/裁剪、eval、ckpt、log.csv）
                              │
                    runs/<name>/ckpt_*.pt
                              │
        sample.load_model ──▶ generate / generate_text / generate_stream
                              │
              app/common.py（找 checkpoint、缓存模型、GPU 锁）
                 ├── app/server.py（FastAPI + SSE）◀── app/web/（React 前端）
                 └── app/app.py（Gradio 备用）
```

| 文件 | 用途 |
|---|---|
| `src/tokenizer.py` | `BPETokenizer`、`CharTokenizer`、教学用的 `train_bpe_toy`、`load_tokenizer(meta_path)` |
| `src/prepare.py` | 下载、清洗、按文档哈希切验证集、训练分词器、编码成 `.bin` |
| `src/model.py` | `GPTConfig`、`MODEL_PRESETS`、`CausalSelfAttention`、`MLP`、`Block`、`GPT`；`python src/model.py` 打印参数量 |
| `src/utils.py` | `get_batch`、`load_config`（yaml + `--键 值` 覆盖）、`set_seed`、`get_lr`、`CSVLogger`、`plot_loss` |
| `src/train.py` | 训练入口，`--resume` 续训、`--overfit` 过拟合测试 |
| `src/sample.py` | 加载模型、采样（`sample_next`：temperature / top-k / top-p）、命令行生成 |
| `src/evaluate.py` | 在同一批验证数据上重新算 loss（`loss`）和 bits-per-byte（`bpb`） |
| `src/eval_prompts.py` | 固定 prompt 批量生成样例、distinct-2、盲评表 |
| `src/plot_compare.py` | 多 run 的 loss 叠图、scaling 图、最终结果表 |
| `src/score_blind.py` | 汇总人工盲评（R1–R5） |
| `src/export.py` | 导出仅权重 checkpoint 到 `release/` |
| `src/demo_check.py` | 答辩前自检：GPU、checkpoint、生成结果与记录是否一致 |
| `src/check_env.py` | 环境自检：版本、bf16、SDPA、torch.compile |

测试在 `tests/`，按模块分：`test_data`、`test_model`、`test_train`、`test_sample`、`test_eval`、`test_export`、`test_server`。

## 3. 开发规则速查

完整规则见 [CLAUDE.md](../CLAUDE.md)，最容易忘的几条：

- **`src/model.py` 必须手写**，不能用 HuggingFace 的模型类或 `Trainer`。关键代码写中文注释，标出张量形状。
- **参数量不超过 0.1B**（约 1.1 亿），词表保持小（BPE 8192 / 字符级）。
- **预计超过 10 分钟的训练先和大家确认**配置、耗时、显存；验证代码一律先跑 `--max_iters 50` 冒烟。
- **新实验用新的 run 名**（`--name`），不要覆盖 `runs/` 下已有的结果。`train.py` 发现目录已存在会拒绝运行（`tests/test_train.py::test_refuse_existing_run`），除非加 `--resume`。
- **改了模型结构或训练逻辑，先跑过拟合测试**：`python src/train.py --config configs/debug.yaml --name overfit_xxx --overfit`，loss 应接近 0。
- 依赖：能 `conda install`（conda-forge）就不用 pip，都走清华 TUNA 镜像；`environment.yml`、`environment-cpu.yml` 和 `requirements.txt` 版本保持一致。
- 本地 8GB 显卡跑正式配置：用 `--batch_size` / `--grad_accum` 覆盖，乘积保持 256。显存不够时依次：减 batch → 缩短上下文 → 开 `grad_ckpt` → 缩小模型。

## 4. 常见改动的操作步骤

### 新增一个实验配置

1. 复制最接近的 `configs/*.yaml`，改 `name` 和要比较的那一项（其他保持不变，对比才公平）。如果是新规模，在 `src/model.py` 的 `MODEL_PRESETS` 里加一项。
2. `python src/model.py` 看参数量（区分是否含 embedding），确认 ≤ 0.1B。
3. 冒烟：`python src/train.py --config configs/xxx.yaml --name smoke_xxx --max_iters 50`，从输出的 tok/s 和显存峰值估算完整训练的时间。
4. 把配置、预计耗时、显存发到群里确认，再正式训练。
5. 训练完：`python src/plot_compare.py --runs ... --out docs/figures/` 更新图表；需要的话用 `src/evaluate.py`、`src/eval_prompts.py` 补评估和样例。
6. 结论写进 [report/experiments.md](report/experiments.md)。注意 `tests/test_train.py::test_configs_consistent` 只检查三个正式配置（只允许 preset、batch、lr、max_iters 不同），不要为了新实验去改这三个文件，新建一个配置就好。

### 改模型结构（例如换成 RoPE）

1. 改 `src/model.py`，新功能用 `GPTConfig` 里的开关控制，默认值保持旧行为，这样已有 checkpoint 还能加载。
2. 参数量变了就更新 `tests/test_model.py` 顶部的期望值，以及 [report/model.md](report/model.md) 的参数表。
3. `pytest tests/test_model.py`（形状、因果性、手写注意力与 SDPA 等价、权重共享、单 batch 过拟合）。
4. 用 `--overfit` 跑一次过拟合测试，再在 debug 配置上完整训练一次（本地约 1 分钟），和 `runs/debug/log.csv` 对比曲线。

### 改采样或生成

- `src/sample.py` 里的 `generate()` 和 `generate_stream()` 是两份循环，**随机数的调用顺序必须一致**，否则同一个 seed 下网页和命令行结果不同。`tests/test_server.py::test_stream_matches_generate_text` 会检查。
- 采样参数的范围同时出现在三处：`src/sample.py` 命令行参数、`app/server.py` 的 `GenerateRequest`、前端 `app/web/src/components/SamplingPanel.tsx`，改了要一起改。

### 改网页界面

- 前端：按 [app/web/README.md](../app/web/README.md) 装 Node + pnpm，`pnpm dev` 开发，改完 `pnpm build` 并**把 `app/web/dist/` 一起提交**。
- 新接口：`app/server.py` 加路由 → `app/web/src/lib/api.ts` 加类型和调用 → `tests/test_server.py` 加测试。两套界面都要用的逻辑放 `app/common.py`。
- 安全约定：只接受 `list_checkpoints()` 列出的路径，不把 `runs/` 整个挂成静态目录。
- Gradio 备用界面 `app/app.py` 不必跟着加新功能，但要保证还能启动（答辩兜底用）。

### 重新生成样例、盲评、PPT

```bash
python src/eval_prompts.py --runs runs/s10m runs/s50m runs/s100m --grid scale --out docs/results/samples_A.jsonl
python src/eval_prompts.py --runs runs/s100m --grid sampling --out docs/results/samples_B.jsonl
python src/eval_prompts.py --runs runs/s100m --grid progress --out docs/results/samples_progress.jsonl
python src/score_blind.py --sheets docs/results/blind_eval/R*.md --flip_repeat R3
conda run -n slides python docs/defense/slides/build_slides.py   # 第一次先 conda env create -f docs/defense/slides/environment.yml
```

注意：`--grid scale` 会重写 `docs/results/blind_eval/sheet.md` 和 `key.json`；如果生成结果变了，已有的 R1–R5 打分就和新样本对不上了。`demo_check.py` 也以 `samples_A.jsonl` 为参考答案。

## 5. 提交前检查

- [ ] `pytest tests/` 全部通过。
- [ ] 改了模型、采样或界面：有 checkpoint 时跑 `python src/demo_check.py`。
- [ ] 文档里的数字（README、report/、PPT 脚本）与 `docs/figures/results.md` 一致。
- [ ] 没有把大文件加进 git（`git status` 里不应出现 `.pt`、`.bin`、`.parquet`、`.tar`）。
- [ ] 新文档放进对应的 `docs/` 子目录，并在 [docs/README.md](README.md) 的索引里加一行。

## 6. 可以继续做的方向

按从易到难排。每一项都要遵守"≤ 0.1B"和"长时间训练先确认"。

1. **在 s10m 上做小消融**（几分钟一次）：学习率、warmup 长度、数据配比（`data.mix` 可以写成 `{smollm: 0.5, tinystories: 0.5}` 这样的字典，或者把 `data.sources` 改成只有 `[tinystories]`）、dropout。s10m 在 5090 上 4.6 分钟、本地约 18 分钟就能训完，适合多试几组，补充实验 B 以外的"训练设置对比"。
2. **KV cache**：生成时把每层已经算过的 k、v 存下来，新 token 只算自己的 q、k、v，不用每步重算整个前缀。现在每步都做完整前向（见 report/experiments.md 里关于生成速度的说明），加上后 200 token 的生成应该明显变快。改 `CausalSelfAttention.forward` 和 `sample.generate*`，并加测试：有无 cache 的输出 logits 应一致。
3. **RoPE（旋转位置编码）**：不再加可学习的位置向量 `wpe`，而是把位置信息以"旋转"的方式乘进 q、k，注意力分数只依赖相对距离，长度外推更好，还省掉 `512 × C` 个参数。按上面"改模型结构"的步骤做，再用 s10m 和现有 run 对比 val loss。
4. **中文字符级小模型**：课程建议里提到的中文路线。字符级、过滤低频字后约 5000 字的词表，数据集需要另找并先确认来源和大小。可以复用 `CharTokenizer` 和 `configs/s10m_char.yaml` 的思路。
5. **更多评估**：例如在 TinyStories 上用固定 prompt 测重复率随 temperature 的变化曲线，或给说明文样例加事实性的人工检查表。

完成的方向请在 [report/experiments.md](report/experiments.md) 追加一节，并更新根 README 的结果摘要。
