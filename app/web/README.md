# 网页前端（React + TypeScript + Tailwind + shadcn/ui）

只有修改前端时才需要这里的步骤；运行界面只需要 `python app/server.py`（使用已提交的 `dist/`）。

包管理用 **pnpm**（依赖锁定在 `pnpm-lock.yaml`，`package.json` 的 `packageManager` 字段固定了版本）。`node_modules/` 不进 git。

```bash
conda env create -f app/web/environment.yml   # 一次性：装 Node.js 22 + pnpm 10（conda-forge）
conda activate web
cd app/web
pnpm install                                  # 按 pnpm-lock.yaml 安装依赖（.npmrc 配置了 npmmirror 镜像，pnpm 同样读取）

# 开发：两个终端
python app/server.py --preload default        # 终端 1（tiny-llm 环境，在项目根目录）：后端，端口 7860
pnpm dev                                      # 终端 2（web 环境）：打开 http://localhost:5173，改代码页面自动刷新

pnpm build                                    # 改完后重新构建 dist/，记得把 dist/ 一起提交
```

- `src/components/ui/`：用 `pnpm dlx shadcn@latest add <组件名>` 生成的 shadcn 组件源码，可以直接改。
- `src/lib/api.ts`：和后端通信；`src/sections/`：页面上的五个编号区块；`src/components/`：复用的组件；动效的 keyframes 都在 `src/index.css`。
- 字体（Inter、思源黑体、JetBrains Mono）通过 `@fontsource-variable/*` 打包进 `dist/`，在 `src/main.tsx` 引入。
