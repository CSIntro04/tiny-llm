# 答辩现场运行方案

演示机：自带的 RTX 5060 Ti 笔记本（WSL2 + `tiny-llm` 环境）。界面跑在本机，**全程不依赖网络**。PPT 第 11 页是演示环节，约 2 分钟。

## 一、提前一天：彩排

- [ ] 按本文第三节完整走一遍，计时（目标 2 分钟以内）。
- [ ] 接投影仪（或外接显示器，1920×1080）检查：浏览器缩放到 125%，Gradio 的输出文字后排能看清。
- [ ] 录一段完整演示的屏幕录像（Windows：Win+Alt+R 或 OBS），存到 U 盘和笔记本桌面，作为最后一级兜底。
- [ ] 确认 `runs/{s10m,s50m,s100m}/ckpt_best.pt` 都在（`python src/demo_check.py` 会检查）。
- [ ] PPT 拷到 U 盘一份，以防演示机不能投影时用教室电脑放 PPT。

## 二、开场前 30 分钟：启动

1. 插上电源（电池模式下显卡会降频），关闭睡眠/屏保，关掉游戏、浏览器视频等占 GPU 的程序。
2. 打开 WSL 终端：

   ```bash
   cd ~/Codes/tiny-llm
   conda activate tiny-llm
   python src/demo_check.py           # 自检：GPU、三个 checkpoint、生成结果是否与记录一致
   ```

   预期输出（全部"通过"、"与样例 一致"）：

   ```
   GPU：NVIDIA GeForce RTX 5060 Ti，torch 2.14.1
   [通过] s10m   加载  0.5s | 50 token 用时  0.2s（  209 tok/s）| 与样例 一致
   [通过] s50m   加载  1.4s | 50 token 用时  0.2s（  304 tok/s）| 与样例 一致
   [通过] s100m  加载  1.9s | 50 token 用时  0.4s（  141 tok/s）| 与样例 一致
   显存占用（三个模型都在显存里）：0.74 GB
   全部通过，可以启动：python app/app.py --preload all
   ```

3. 启动界面（这个终端不要关）：

   ```bash
   python app/app.py --preload all    # 三个模型启动时就加载好，约 5 秒
   ```

4. 在 Windows 的浏览器打开 `http://127.0.0.1:7860`（WSL2 会自动转发端口）。浏览器缩放 125%。
5. **热身**：在"三模型对比"页点一次"生成"（第一次调用有 CUDA 初始化开销），确认三栏都有输出。
6. 再开一个终端标签页，`cd ~/Codes/tiny-llm && conda activate tiny-llm`，留着给兜底命令用；VS Code 打开 `src/model.py` 并跳到第 66 行，留着讲代码。

## 三、演示脚本（2 分钟）

参数都固定了 seed，**现场输出和下面的预期完全一致**（GPU 上已验证）。讲解词是要点，不用逐字背。

### 第 1 步：三模型对比（约 50 秒）

"三模型对比"页，默认参数（max_new_tokens 200，T=0.8，top-k 50，top-p 1.0，seed 42）。

1. prompt：`Once upon a time` → 生成（三个模型共约 4 秒）。

   | 模型 | 预期开头 |
   |---|---|
   | s10m | , there was a little boy named Timmy. Timmy loved to play with his toy car all day long. One day, Timmy's mommy gave him a big package of a new toy car… |
   | s50m | , there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks. One day, Timmy's mommy gave him a big box of blocks… |
   | s100m | , there was a little boy named Timmy… One day, Timmy's mom told him it was time to take a bath. Timmy didn't want to stop playing, but he knew he had to listen to his mom. |

   讲："写儿童故事，0.01B 的模型就已经很通顺了，三个模型差别不大——这和 TinyStories 上 val loss 只差 0.5 一致。"

2. prompt：`The French Revolution began` → 生成（约 2 秒）。

   | 模型 | 预期开头 |
   |---|---|
   | s10m | in the aftermath of the Revolution, and the Roman Empire was the first settlement of the Roman Empire… |
   | s50m | in the afternoon of 1864, and the French Revolution resulted in the establishment of its own government. Under Louis XIV's rule… |
   | s100m | in the aftermath of the Revolutionary War, and on 17 September 1789, the French Revolution was declared an international war of war… |

   讲："说明文的差距就明显了：小模型反复说 Roman Empire，大模型开始出现 1789 这样的正确年份。但即使 0.1B，内容也是编的——它学到的是'写得像'，不是'写得对'。"

### 第 2 步：调 temperature（约 50 秒）

切到"单模型生成"页，checkpoint 选 `runs/s100m/ckpt_best.pt`，prompt `Photosynthesis is the process`，max_new_tokens 150，seed 42。依次改 temperature 和 top-k，每次点"生成"（每次不到 1 秒）：

| 设置 | 预期 | 讲 |
|---|---|---|
| T=0，top-k 50 | of converting organic matter into chemical energy. What is the difference between photosynthesis and photosynthesis? Photosynthesis is the process of converting… （整句复读） | "greedy 每步都选最可能的词，一旦进入循环就出不来。" |
| T=0.8，top-k 50 | of the photosynthesis that converts light energy (solar) into chemical energy… with carbon dioxide (CO2) and water vapor… | "正常的设置，也是界面的默认值。" |
| T=1.5，top-k **0** | by which bacteria perform itself ASE Washington IT Dangerers Just bood for these sorts… （乱码） | "温度高把分布拉平，几乎不可能的词也被抽到了。" |
| T=1.5，top-k 50 | by which bacteria perform this type of chemiluminescence. The food for these two types of bacteria can contain an amino acids… | "加上 top-k 只在前 50 个词里抽，拼写又正常了，虽然内容还是很荒诞。" |

### 第 3 步：训练过程（约 20 秒）

切到"训练过程"页，选 s100m，展示 loss 曲线和结果表。讲："从 9.2（≈ ln 8192，随机猜）降到 2.4；train 和 val 两条线几乎重合，没有过拟合。"然后切回 PPT 讲实验结论。

## 四、兜底方案（逐级降级）

| 现象 | 处理 | 用时 |
|---|---|---|
| 浏览器页面卡住、按钮没反应 | 刷新页面（模型已经在显存里，不用重新加载）。还不行就在终端 Ctrl+C，重新 `python app/app.py --preload all` | 10 秒 |
| 界面打不开 / Gradio 报错 | 在备用终端直接用命令行生成（见下面的命令），终端字体调大 | 立即 |
| CUDA 报错、显卡驱动异常 | 改用 CPU：`CUDA_VISIBLE_DEVICES= python app/app.py --preload all`（`load_model` 没有 GPU 时自动用 CPU）。CPU 实测：s10m 142、s50m 46、s100m 25 tok/s，把 max_new_tokens 调到 100，三模型对比约 7 秒 | 30 秒 |
| 演示机无法使用 | 用教室电脑放 PPT，播放提前录好的演示视频；PPT 第 7–10 页已经有全部样例 | 立即 |

命令行兜底（复制粘贴即可，结果与界面相同）：

```bash
# 第 1 步：三模型对比
for r in s10m s50m s100m; do
  python src/sample.py --run runs/$r --prompt "The French Revolution began" --max_new_tokens 80 --seed 42
done
# 第 2 步：调温度
python src/sample.py --run runs/s100m --prompt "Photosynthesis is the process" --max_new_tokens 80 --temperature 0
python src/sample.py --run runs/s100m --prompt "Photosynthesis is the process" --max_new_tokens 80 --temperature 1.5 --top_k 0
python src/sample.py --run runs/s100m --prompt "Photosynthesis is the process" --max_new_tokens 80 --temperature 1.5 --top_k 50
```

## 五、现场可能被要求的临时操作

- **"换个 prompt 试试"**：直接输入，seed 设 -1（每次随机，界面会显示用了哪个 seed）。英文 prompt 效果好；中文 prompt 模型没学过，会输出乱码（数据里没有中文，可以顺便解释词表和训练数据的关系）。
- **"看训练中途的模型"**：单模型页的 checkpoint 下拉框选 `runs/s100m/ckpt_tok100M.pt`，对比最终模型（见 [samples_progress.md](samples_progress.md)）。
- **"讲一下代码"**：VS Code 里的 `src/model.py:66`（注意力 forward），讲稿见 [qa.md](qa.md) 第三节。
