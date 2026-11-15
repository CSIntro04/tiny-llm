Description: 大语言模型通过根据已有文本预测下一个词元来学习语言规律。本项目要求从零训练一个小型自回归语言模型，并制作交互界面展示训练和生成过程。项目应准备一个规模适中的文本数据集，完成文本预处理和词元化，使用 Decoder-only Transformer 进行训练，并记录训练损失的变化。建议尝试训练约0.01B（1000 万）参数的模型；也可以先从更小的字符级模型开始验证流程，再根据设备和训练时间决定是否扩大模型规模。用户应能够输入提示文本、查看模型生成的续写结果，并调整生成长度或采样温度等参数。项目还应说明数据来源、模型结构和训练设置，并比较至少两种训练设置或采样参数下的生成结果。项目重点是理解数据准备、Next-token Prediction、模型训练和文本生成的基本流程，不要求达到商业大语言模型的生成质量。
Reference: [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., &Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30. https://arxiv.org/abs/1706.03762
[2] Karpathy, A. nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs. https://github.com/karpathy/nanoGPT
[3] https://huggingface.co/learn/llm-course/en/chapter7/6
[4] https://dujh22.github.io/Dujinhua_wiki/aml2024/

## 本组目标调整

在课程要求（约 0.01B 参数）的基础上，本组**分别训练 0.01B、0.05B、0.1B 三个规模**的 Decoder-only Transformer，作为核心对比实验：

| 名称 | 结构（层/维/头） | 总参数 | 训练 token 数 |
|---|---|---|---|
| s10m | 6 / 384 / 6 | ≈ 14.0M（0.01B） | 3 亿 |
| s50m | 10 / 640 / 10 | ≈ 54.8M（0.05B） | 10 亿 |
| s100m | 14 / 768 / 12 | ≈ 105.9M（0.1B） | 15 亿 |

- 数据：混合两个英文语料，共约 15 亿 token：
  - [ecreeth/1b-smollm-corpus](https://huggingface.co/datasets/ecreeth/1b-smollm-corpus)：SmolLM-Corpus 的 10 亿 token 子集（FineWeb-Edu 87% + Cosmopedia v2 13%），ODC-By 许可。
  - [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)：约 5 亿 token 的儿童短故事，CDLA-Sharing-1.0 许可。
- 词元化：在混合语料上自训练 BPE（词表 8192）。
- 训练 token 数按模型规模分配（接近 Chinchilla 的约 20 token/参数）。三个模型使用相同数据、相同随机种子、相同有效 batch，比较：两个语料上各自的验证集 loss、**相同已训练 token 数**下的 loss 曲线、生成质量（人工打分）、训练成本（时间、显存、吞吐）。
- 另外比较不同采样参数（temperature、top-k）下的生成结果。
- 分阶段实现方案见 `docs/design/00-roadmap.md`。
