// 页面骨架：居中的细线纸面；左边吸顶侧栏（标题、深浅色切换、目录、项目信息），右边编号区块依次排列
import { useCallback, useEffect, useState } from "react"
import { Moon, Sun } from "lucide-react"

import { api, type Checkpoints } from "@/lib/api"
import { GlideNav, type NavItem } from "@/components/GlideNav"
import { Segmented } from "@/components/Segmented"
import { TooltipProvider } from "@/components/ui/tooltip"
import { CompareSection } from "@/sections/CompareSection"
import { GenerateSection } from "@/sections/GenerateSection"
import { CurveSection, FiguresSection, ResultsSection } from "@/sections/TrainingSections"

const NAV: NavItem[] = [
  { id: "generate", label: "单模型生成" },
  { id: "compare", label: "三模型对比" },
  { id: "curve", label: "Loss 曲线" },
  { id: "figures", label: "对比图" },
  { id: "results", label: "结果表" },
]

const FACTS: [string, string][] = [
  ["结构", "Decoder-only Transformer"],
  ["数据", "SmolLM-Corpus + TinyStories"],
  ["词表", "BPE 8192"],
  ["上下文", "512"],
  ["规模", "0.01B / 0.05B / 0.1B"],
]

// 深浅色：在 <html> 上加 / 去 .dark 类；偏好存 localStorage（隐私模式下可能不可用，所以 try/catch）
type Theme = "light" | "dark"
function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const saved = localStorage.getItem("theme")
      if (saved === "light" || saved === "dark") return saved
    } catch { /* 忽略 */ }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
  })
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark")
    try { localStorage.setItem("theme", theme) } catch { /* 忽略 */ }
  }, [theme])
  return [theme, setTheme]
}

// 侧栏左上角的标志：3×3 小方块，与生成时的点阵加载器呼应
function Mark() {
  return (
    <span aria-hidden className="grid grid-cols-[repeat(3,6px)] gap-[2px]">
      {[1, 0.35, 1, 0.35, 1, 0.35, 1, 0.35, 1].map((o, i) => <span key={i} className="bg-ink size-1.5 rounded-[1.5px]" style={{ opacity: o }} />)}
    </span>
  )
}

export default function App() {
  const [theme, setTheme] = useTheme()
  const [ckpts, setCkpts] = useState<Checkpoints | null>(null)
  const [error, setError] = useState("")

  const loadCkpts = useCallback(() => {
    api.checkpoints().then(setCkpts).catch((e) => setError(String(e)))
  }, [])
  useEffect(loadCkpts, [loadCkpts])

  return (
    <TooltipProvider delayDuration={200}>
      <main className="bg-page relative mx-auto min-h-screen max-w-[1120px] shadow-[0_0_0_1px_var(--line)]">
        <div className="lg:grid lg:grid-cols-[264px_minmax(0,1fr)]">
          <aside className="border-line flex flex-col border-b border-dashed px-5 pt-12 pb-6 sm:px-7 lg:sticky lg:top-0 lg:h-screen lg:border-r lg:border-b-0 lg:pt-[clamp(2.5rem,8vh,5rem)]">
            <div className="reveal flex items-center justify-between">
              <Mark />
              <Segmented size="sm" value={theme} onChange={setTheme} options={[
                { value: "light", label: <Sun className="size-3.5" />, title: "浅色" },
                { value: "dark", label: <Moon className="size-3.5" />, title: "深色" },
              ]} />
            </div>
            <div className="reveal" style={{ animationDelay: "60ms" }}>
              <h1 className="text-ink mt-10 text-[21px] leading-snug font-semibold tracking-[-0.02em] text-balance">
                从零训练的迷你语言模型。
              </h1>
              <p className="text-ink-3 mt-2 text-[12.5px] leading-relaxed text-pretty">
                手写的 Decoder-only Transformer，用 next-token prediction 训练。输入一段开头，看它逐个 token 续写。
              </p>
            </div>
            <div className="reveal border-line mt-7 hidden border-t border-dashed pt-6 lg:block" style={{ animationDelay: "120ms" }}>
              <GlideNav items={NAV} />
            </div>
            <dl className="reveal border-line mt-auto hidden space-y-1.5 border-t border-dashed pt-5 text-[11.5px] lg:block" style={{ animationDelay: "180ms" }}>
              {FACTS.map(([k, v]) => (
                <div key={k} className="flex justify-between gap-3">
                  <dt className="text-ink-3">{k}</dt>
                  <dd className="text-ink-2 truncate text-right">{v}</dd>
                </div>
              ))}
            </dl>
          </aside>

          <div className="min-w-0 pt-4 pb-12 lg:pt-6">
            {error && <div className="text-ink-2 mx-5 mt-4 rounded-[10px] p-3 text-[12.5px] shadow-[var(--shadow-card)] sm:mx-8">连不上后端：{error}</div>}
            {ckpts && <GenerateSection index={0} ckpts={ckpts} reload={loadCkpts} />}
            <CompareSection index={1} examples={ckpts?.examples ?? []} />
            <CurveSection index={2} />
            <FiguresSection index={3} />
            <ResultsSection index={4} />
            <footer className="text-ink-3 px-5 pt-8 text-[11.5px] sm:px-8">
              上海交通大学 计算机科学导论 · B03 从零训练一个迷你语言模型
            </footer>
          </div>
        </div>
      </main>
    </TooltipProvider>
  )
}
