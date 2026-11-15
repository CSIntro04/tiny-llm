// Vite 配置：开发时把 /api 和 /figures 转发给 FastAPI 后端（python app/server.py）
import path from "node:path"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"

const backend = "http://127.0.0.1:7860"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  build: { chunkSizeWarningLimit: 800 }, // 本地界面，一个 ~580KB 的包没关系
  server: { proxy: { "/api": backend, "/figures": backend } },
})
