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

## 实现阶段产出的文档

| 文件 | 内容 |
|---|---|
| [data.md](data.md) | 数据集说明：来源、许可、下载文件与 sha256、清洗与丢弃统计、分词器、token 数 |
| [model.md](model.md) | 模型结构图（含 Block 内张量形状）、按组件的参数量统计、训练设置表 |
| [experiments.md](experiments.md) | 实验 A/B/C 的结果、图表、生成样例、盲评和结论 |
| [qa.md](qa.md) | 答辩问题清单与参考回答、5 分钟演示脚本 |
| [demo.md](demo.md) | 答辩现场运行方案：启动步骤、演示脚本与预期输出、兜底方案 |
| `slides/` | 答辩 PPT |
| `samples_*.md`、`blind_eval*` | 全部生成样例、盲评打分表与汇总 |
| `figures/` | loss 叠图、scaling 图、结果表 |
