# 阶段 7 · Gradio 交互界面

## 目标

`python app/app.py` 启动本地网页：用户输入 prompt，选择 checkpoint，调节生成长度、temperature、top-k、top-p、seed，查看续写结果；并提供三模型并排对比页，直观展示 0.01B / 0.05B / 0.1B 的差别。

## 前置条件

- 阶段 5 的 `load_model`、`generate_text` 可用。至少一个训练好的 run（开发时用 debug 模型即可，可与阶段 6 并行）。

## 要写的文件与函数签名

```python
# app/app.py
def list_checkpoints(runs_dir: str = "runs") -> list[str]: ...     # 扫描 runs/*/ckpt_*.pt
@functools.lru_cache(maxsize=4)
def get_model(ckpt_path: str) -> tuple[GPT, Tokenizer, dict]: ...  # 缓存，避免重复加载
def model_info(ckpt_path: str) -> str: ...                          # 参数量、结构、训练步数、val loss（Markdown）
def generate_single(ckpt, prompt, max_new_tokens, temperature, top_k, top_p, seed) -> str: ...
def generate_compare(prompt, max_new_tokens, temperature, top_k, top_p, seed) -> tuple[str, str, str]: ...
def build_ui() -> gr.Blocks: ...
if __name__ == "__main__":
    build_ui().launch()
```

## 界面布局

**标签页 1：单模型生成**

| 组件 | 说明 |
|---|---|
| checkpoint 下拉框 | `list_checkpoints()` 结果，默认 `runs/s100m/ckpt_best.pt`；附"刷新"按钮 |
| 模型信息 | Markdown：结构、参数量、训练 token 数、最佳 val loss |
| prompt 文本框 | 默认 "Once upon a time" |
| max_new_tokens 滑块 | 10–500，默认 200 |
| temperature 滑块 | 0–2，步长 0.05，默认 0.8（0 = greedy） |
| top_k 滑块 | 0–200，默认 50（0 = 不限） |
| top_p 滑块 | 0.1–1.0，默认 1.0（1.0 = 不限） |
| seed 数字框 | 默认 42；-1 表示随机 |
| 生成按钮 / 输出框 | prompt 部分加粗或另色显示，续写部分正常显示 |
| 示例 | `gr.Examples` 给 6 个预置 prompt：故事类、说明文类各 3 个（与实验 A 的 prompt 一致） |

**标签页 2：三模型对比**：同一 prompt、同一组采样参数、同一 seed，并排三列输出 s10m / s50m / s100m 结果，每列标题显示参数量。

**标签页 3：训练过程**：选择 run，显示 `loss_curve.png`、`docs/figures/` 中的叠图，以及最终结果表。

可选：流式输出（`generate` 改成生成器，每生成一个 token `yield` 一次，界面逐字出现）。

## 实现要点

- 模型缓存：三个模型 bf16 推理权重合计约 0.35GB，加上运行时开销也不到 1GB，可以同时常驻显存。
- 推理用 `torch.no_grad()` + `torch.autocast(bfloat16)`；`seed` 在每次生成前 `torch.manual_seed`。
- 界面只做"调用"，所有生成逻辑复用 `src/sample.py`，不要复制一份。
- 从 `app/` 导入 `src`：在 `app.py` 顶部把项目根目录加入 `sys.path`，或把项目装成可编辑包。
- 只监听本机（默认 `127.0.0.1`）；需要给组员演示时再考虑 `share=True`。
- 训练进行中别同时开界面跑大模型，以免抢显存导致训练 OOM。

## 验收标准

```bash
python app/app.py      # 浏览器打开 http://127.0.0.1:7860
```

- 下拉框能列出所有 run 的 checkpoint；切换后模型信息随之变化。
- 同参数同 seed 重复点击结果一致；seed=-1 每次不同。
- temperature=0 时输出确定。
- 对比页一次点击得到三列结果，每次生成 200 token 总耗时在几秒以内。
- 训练过程页能显示 loss 曲线图。

## 常见坑

- Gradio 版本差异：组件参数名（如 `gr.Slider(value=...)`）以 environment.yml 锁定的版本为准。
- 第一次生成慢：模型加载 + CUDA 初始化，可在启动时预加载默认模型。
- `lru_cache` 的参数必须可哈希（路径字符串可以）。
- 输出里的 `<|endoftext|>` 要处理掉。

## 答辩可能被问的问题

- 现场演示：调高/调低 temperature 会怎样？为什么？
- 三个模型的输出差异在哪？与 loss 的差距是否一致？
- 生成一段文本时，模型内部前向了多少次？
