# 项目文档索引

本目录是 B03「从零训练迷你语言模型」的分阶段实现文档。目标：在 SmolLM-Corpus 1B 子集 + TinyStories 的混合语料（约 15 亿 token）上分别训练 **0.01B / 0.05B / 0.1B** 三个 Decoder-only Transformer 并对比。

## 阅读顺序

| 文件 | 内容 | 什么时候读 |
|---|---|---|
| [00-roadmap.md](00-roadmap.md) | 总路线、里程碑、模型规模、资源估算、交付物对照 | 最先读 |
| [01-environment.md](01-environment.md) | 阶段 1：Miniconda + TUNA 镜像环境、GPU 自检 | 开工第一天 |
| [02-data.md](02-data.md) | 阶段 2：两个语料的下载、清洗、验证集切分、BPE-8192、按来源混合采样 | 环境就绪后 |
| [03-model.md](03-model.md) | 阶段 3：手写 GPT（注意力、MLP、权重共享） | 数据就绪后 |
| [04-training.md](04-training.md) | 阶段 4：训练循环、日志、checkpoint、资源估算 | 模型通过测试后 |
| [05-sampling.md](05-sampling.md) | 阶段 5：temperature / top-k / top-p 采样、命令行生成 | 有第一个 checkpoint 后 |
| [06-experiments.md](06-experiments.md) | 阶段 6：三模型规模对比 + 采样参数对比 | 训练正式模型时 |
| [07-app.md](07-app.md) | 阶段 7：Gradio 交互界面 | 可与阶段 6 并行 |
| [08-report.md](08-report.md) | 阶段 8：README、报告素材、答辩准备 | 收尾 |
| [glossary.md](glossary.md) | 概念速查（每条 2–3 句） | 随时查 |

## 每个阶段文件的统一结构

1. **目标**：这一阶段做完能得到什么
2. **前置条件**：依赖哪些前面阶段的产物
3. **要写的文件与函数签名**：只给签名和注释，实现留给编码阶段
4. **实现要点**：张量形状、原理、关键细节
5. **验收标准**：可以直接运行的命令 + 预期输出
6. **常见坑**
7. **答辩可能被问的问题**

## 实现阶段才会生成的文档

- `docs/data.md`：数据集说明（阶段 2 产出）
- `docs/experiments.md`：实验记录与生成样例（阶段 6 产出）
- `docs/model.md`：模型结构图与参数量表（阶段 8 整理）
