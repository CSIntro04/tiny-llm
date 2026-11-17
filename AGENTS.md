# AGENTS.md

给 Codex 等 AI 编程助手看的说明。项目规则只维护一份，写在 `CLAUDE.md` 里，本文件不重复。

**开始工作前先完整阅读：**

1. `CLAUDE.md`：项目背景、硬件与环境、模型规模、实现约定、工作方式（必须遵守）。
2. `docs/DEVELOPING.md`：怎么恢复环境、改代码、跑测试、加实验。

下面几条最容易出错，单独列出来：

- 回复用中文，代码标识符用英文；关键代码写中文注释（张量形状和原理）。
- 不要删除或覆盖 `runs/` 下已有的结果；新实验用新目录名。
- 预计超过 10 分钟的训练先问用户；验证代码用 `--max_iters 50` 之类的冒烟测试。
- `src/model.py` 必须手写，不能用 HuggingFace 的现成模型或 Trainer。模型不超过 0.1B 参数。
- 依赖能用 conda-forge 就不用 pip。没有 NVIDIA 显卡的电脑用 `environment-cpu.yml`；改包版本时 `environment.yml`、`environment-cpu.yml`、`requirements.txt` 一起改。
- 改完代码跑 `pytest tests/`；只在用户要求时提交。
