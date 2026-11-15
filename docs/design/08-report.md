# 阶段 8 · 报告、README 与答辩准备

## 目标

整理所有交付物：根目录 `README.md`（可复现）、数据说明、模型结构图与参数量统计、训练设置、实验对比与结论、答辩问题清单。

## 前置条件

- 阶段 6 的实验结果和图表；阶段 7 的界面可运行。

## 要写的文件

```
README.md            项目简介 + 完整复现步骤
docs/report/data.md         数据集说明（阶段 2 已有草稿，补全）
docs/report/model.md        模型结构图、参数量统计、训练设置表
docs/report/experiments.md  实验对比（阶段 6 产出，补全结论）
docs/defense/qa.md           答辩问题清单与参考回答
```

## 实现要点

### README.md 结构

1. 一句话介绍 + 界面截图 + 三模型生成样例
2. 硬件与环境：Miniconda、TUNA 镜像配置、`conda env create -f environment.yml`、`python src/check_env.py`
3. 数据：`python src/prepare.py --datasets smollm,tinystories --tokenizer bpe --vocab_size 8192 --shards all`（说明下载约 3.7GB、编码约 20–40 分钟）
4. 训练：三个配置的命令、预计耗时、显存
5. 生成：`python src/sample.py --run runs/s100m --prompt "Once upon a time"`
6. 界面：`python app/server.py`（备用：`python app/app.py`）
7. 结果摘要：最终结果表 + val loss 叠图
8. 目录结构、参考文献（TARGET.md 中的 4 条 + TinyStories、SmolLM-Corpus、FineWeb-Edu）；数据许可说明（ODC-By 需署名、CDLA-Sharing-1.0）

验证方式：另一位组员在**干净环境**里按 README 从头跑到 debug 模型生成，全程不需要问人。

### docs/report/model.md

- 结构图：复用 [03-model.md](03-model.md) 的 Mermaid 图，再画一张单个 Block 内部张量形状的流程图（标注 `(B,T,C)`、`(B,nh,T,hs)`、`(B,nh,T,T)`）。
- 参数量统计表：按组件（wte、wpe、每层 attn、每层 mlp、LayerNorm）分别列出四档配置的参数量，与代码打印结果一致。
- 训练设置表：三个正式配置的全部超参数并排列出，突出"只有结构、micro-batch 和学习率不同"。

### docs/defense/qa.md：答辩问题清单

从各阶段文件的"答辩可能被问的问题"汇总，每题写 2–4 句参考回答，并指向代码行（`src/model.py:行号`）。重点：

| 主题 | 问题 |
|---|---|
| 整体 | 什么是 next-token prediction？为什么它能学到语言？Decoder-only 是什么？ |
| 数据 | BPE 怎么训练？为什么词表要小？x 和 y 什么关系？ |
| 模型 | 注意力的计算过程和张量形状；因果掩码的作用；为什么除以 √d；多头的意义；残差与 LayerNorm；权重共享；参数量怎么算 |
| 损失 | 为什么用 cross entropy？初始 loss 为什么 ≈ ln(V)？困惑度是什么？ |
| 训练 | AdamW；warmup + cosine；梯度裁剪；梯度累积；bf16 与 GradScaler |
| 生成 | temperature 数学含义；top-k vs top-p；greedy 为何重复 |
| 实验 | 对比是否公平？scaling 结论？0.1B 数据是否够？ |

每位组员都要能回答全部问题，不只是自己负责的部分。

### 答辩演示脚本（5 分钟）

1. 打开界面，三模型对比页，输入 "Once upon a time"，展示差别（1 分钟）
2. 单模型页调 temperature 0 → 0.8 → 1.5，解释现象（1 分钟）
3. 展示 val loss 叠图和 scaling 图，讲结论（1.5 分钟）
4. 打开 `src/model.py` 讲注意力 forward（1.5 分钟）

## 验收标准

- 对照 [00-roadmap.md](00-roadmap.md) 的交付物对照表和 CLAUDE.md 的交付物清单，全部打勾。
- 干净环境按 README 复现成功。
- `docs/report/model.md` 中的参数量与 `python src/model.py` 输出一致；各文档中三档规模的数字一致。
- 每位组员完成一次模拟答辩。

## 常见坑

- README 里的命令与实际脚本参数不一致——最后一天统一检查一遍。
- 图片用相对路径引用，确保在 GitHub 上能显示。
- 大文件（checkpoint、bin）不要进 git；如需提交最终模型，单独说明获取方式。

## 答辩可能被问的问题

- 如果给你们更多时间/算力，下一步做什么？（更多数据、RoPE、KV cache、更长上下文……）
- 这个项目里你个人负责什么？遇到的最大问题是什么？
