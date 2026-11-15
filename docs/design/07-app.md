# 阶段 7 · 交互界面（网页版 + Gradio 备用）

> 本文前半部分是最初的 Gradio 方案（现为备用界面 `app/app.py`）；答辩用的正式界面是后半部分「网页界面：React + shadcn/ui」一节（`app/server.py` + `app/web/`）。

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

**外观**：`gr.themes.Soft`（indigo / cyan）+ `app.py` 里的 `CSS` 常量，顶部渐变横幅；颜色全用主题 CSS 变量，深色模式自动适配；只用系统字体（演示现场离线）。Gradio 6 起 `theme` / `css` 传给 `launch()`。

**标签页 1：单模型生成**（左栏参数、右栏输入与输出的两栏布局）

| 组件 | 说明 |
|---|---|
| checkpoint 下拉框 | 左侧参数栏顶部；`list_checkpoints()` 结果，默认 `runs/s100m/ckpt_best.pt`；附 🔄 刷新按钮 |
| 采样预设 | Greedy / 保守（T=0.5, k=20）/ 默认（T=0.8, k=50）/ 创意（T=1.2, p=0.95），一键改写下面的滑块 |
| 模型信息 | 右侧顶部一排指标卡：结构、参数量、分词器、训练 token 数、最佳 val loss |
| prompt 文本框 | 默认 "Once upon a time" |
| max_new_tokens 滑块 | 10–500，默认 200 |
| temperature 滑块 | 0–2，步长 0.05，默认 0.8（0 = greedy） |
| top_k 滑块 | 0–200，默认 50（0 = 不限） |
| top_p 滑块 | 0.1–1.0，默认 1.0（1.0 = 不限） |
| seed 数字框 | 默认 42；-1 表示随机 |
| 生成按钮 / 输出框 | 输出卡片：prompt 部分高亮底色，续写用衬线字体；底部胶囊显示 seed、token 数、用时、tok/s；回车也能生成，"清空"恢复空白 |
| 示例 | `gr.Examples` 给 6 个预置 prompt：故事类、说明文类各 3 个（与实验 A 的 prompt 一致） |

**标签页 2：三模型对比**：同一 prompt、同一组采样参数、同一 seed，并排三列输出 s10m / s50m / s100m 结果，每列列头是模型名 + 参数量徽章（颜色由浅到深）；采样参数收在默认折叠的面板里。

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

## 实现记录（2026-11-15）

- 实现：`app/app.py`，三个标签页（单模型生成 / 三模型对比 / 训练过程）按上面的布局完成。生成全部复用 `src/sample.py` 的 `load_model`、`generate_text`；示例 prompt 复用 `src/eval_prompts.py` 的 `PROMPTS`。
- `model_info` 需要训练步数和 best val，这两个值用 `torch.load(..., mmap=True)` 读取，不会把含优化器状态的整个 checkpoint 读进内存。
- `src/sample.py:load_model` 改成先加载到 CPU，再只把模型权重搬上 GPU。原来直接 `map_location="cuda"` 时，AdamW 的 m、v 也会占用显存（s100m 的 checkpoint 有 1.27GB）。
- seed=-1 时随机生成一个 seed，并显示在输出下方，方便复现那次结果。
- 流式输出（可选项）这次没做。
- 自动验收（用 `gradio_client` 调接口，本地 RTX 5060 Ti）：同参数同 seed 结果一致 ✅；seed=-1 每次不同 ✅；T=0 时换 seed 结果也不变 ✅；三模型对比 200 token 共 3.1 秒（包含首次加载 s10m、s50m 的时间）✅；切换 checkpoint 时模型信息随之更新 ✅。

## 网页界面：React + shadcn/ui（2026-11-15 新增）

Gradio 版保留作备用；正式演示改用自己写的网页前端，界面更清爽，并实现了流式输出。

```
浏览器（app/web/dist 静态页面，React + shadcn/ui）
   │  fetch /api/*            ▲ SSE：每个 token 一条 "data: {...}"
   ▼                          │
app/server.py（FastAPI） ── app/common.py（找 checkpoint、缓存模型） ── src/sample.py（generate_stream）
```

- **shadcn/ui** 不是装进来的黑盒组件库，而是把按钮、滑块、标签页等组件的源码复制进项目（`app/web/src/components/ui/`），样式用 Tailwind 写，可以直接改。
- **流式输出（SSE，Server-Sent Events）**：生成接口的 HTTP 响应不立刻结束，后端每生成一个 token 就往响应里写一行 `data: {"delta": "..."}`，前端边读边追加到页面上。`src/sample.py:generate_stream` 的循环和 `generate()` 一样、随机数调用顺序也一样，所以同一个 seed 下结果与 `generate_text()` 完全相同（`tests/test_server.py` 验证）。
- **增量解码**：每步把全部新 token 重新 decode 再减去上次的文本。BPE 按字节合并，一个汉字可能跨两个 token，单独 decode 一个 token 会出现乱码（`\ufffd`），这时先不发，等下一个 token 补齐。
- **GPU 锁**：`common.GPU_LOCK` 让生成请求排队。`set_seed` 是全局的，两个请求交错执行会互相打乱随机数，结果就不可复现了。三模型对比也是前端依次请求三个模型。
- **安全**：只接受 `list_checkpoints()` 列出的 checkpoint 路径，`runs/` 目录不整个挂出去（里面有 checkpoint 和优化器状态），只单独提供 `loss_curve.png`。
- **外观与动效**：参考 beautifului.dev 首页的版式——居中的细线"纸面"，左侧吸顶侧栏（标题、深浅色切换、目录、项目信息），右侧是编号区块（01 单模型生成 … 05 结果表），区块之间用虚线分隔，展示区是浅灰底上放白色卡片。只用中性灰，不用亮色。动效：区块进场依次淡入上移（错开 120ms）；目录的悬停底色会"滑"到鼠标所在条目，并随滚动高亮当前区块；分段选择器的滑块平滑移动；等待第一个 token 时显示 3×3 像素点阵 + 扫光文字 + 计时；新生成的文本片段淡入去模糊，末尾是细竖线光标。系统开启"减少动态效果"时全部关闭。
- **字体**：英文和数字用 Inter，中文用思源黑体（Noto Sans SC），seed、路径、tok/s 这类等宽内容用 JetBrains Mono。三者都是 OFL 开源字体，用 `@fontsource-variable/*` 的可变字重版打包进 `dist/`（共约 5MB）。思源黑体按 unicode-range 切成约 100 个小文件，浏览器只下载页面上用到的那几片。选型过程：在一个字体对比页里比较过 Inter / Geist / IBM Plex / 霞鹜文楷 / Maple Mono / 系统字体。
- **离线**：不引用任何 CDN 或在线字体（字体文件都在 `dist/assets/` 里）；`app/web/dist/` 提交进 git，运行时不需要 Node。

| 文件 | 内容 |
|---|---|
| `app/server.py` | 接口：`/api/checkpoints`、`/api/model`、`/api/generate`（SSE）、`/api/compare-models`、`/api/runs`、`/api/figures`、`/api/results`；托管 `dist/` |
| `app/web/src/lib/api.ts` | 类型定义、fetch 封装、`streamGenerate()` 读 SSE |
| `app/web/src/components/` | `Section`（编号区块）、`GlideNav`（侧栏目录）、`Segmented`（分段选择器）、`Composer`（输入框）、`SamplingPanel`、`ModelStats`、`OutputCard`、`PixelLoader` |
| `app/web/src/sections/` | 五个区块：`GenerateSection`、`CompareSection`、`TrainingSections`（loss 曲线 / 对比图 / 结果表） |

改前端：`conda env create -f app/web/environment.yml`（Node.js + pnpm），然后见 `app/web/README.md`。
