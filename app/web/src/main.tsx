import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

// 字体打包进 dist/（@fontsource 的可变字重版），演示现场断网也能显示：
// Inter（英文 / 数字）、思源黑体 Noto Sans SC（中文，按 unicode-range 切片，浏览器只下载用到的片段）、JetBrains Mono（等宽）
import "@fontsource-variable/inter"
import "@fontsource-variable/noto-sans-sc"
import "@fontsource-variable/jetbrains-mono"
import "./index.css"
import App from "./App"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
