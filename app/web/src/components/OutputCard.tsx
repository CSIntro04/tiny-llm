// 生成结果卡片：prompt 灰底标记 + 续写正文（每个新片段淡入）+ 细竖线光标 + 底部统计 + 复制
import { useState } from "react"
import { Check, Copy } from "lucide-react"

import type { GenDone } from "@/lib/api"
import { PixelLoader } from "@/components/PixelLoader"
import { cn } from "@/lib/utils"

// chunks：后端每发来一段文本就追加一项，渲染成一个个 <span class="token">，新片段各自播放淡入动画
export type Output = { prompt: string; chunks: string[]; running: boolean; done: GenDone | null; error?: string }

export const EMPTY_OUTPUT: Output = { prompt: "", chunks: [], running: false, done: null }

export function OutputCard({ out, placeholder, className }: { out: Output; placeholder?: string; className?: string }) {
  const [copied, setCopied] = useState(false)
  const started = out.running || out.chunks.length > 0 || out.done || out.error
  const waiting = out.running && out.chunks.length === 0

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(out.prompt + out.chunks.join(""))
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* 无剪贴板权限时忽略 */ }
  }

  return (
    <div className={cn("bg-surface shadow-card group/out relative flex min-h-44 flex-col rounded-[10px] p-5", className)}>
      {!started ? (
        <div className="text-ink-3 m-auto text-[12.5px]">{placeholder ?? "输入 prompt 后点击「生成」"}</div>
      ) : waiting ? (
        <div className="m-auto"><PixelLoader /></div>
      ) : (
        <>
          {out.chunks.length > 0 && !out.running && (
            <button onClick={copy} title="复制"
              className="text-ink-3 hover:bg-hover hover:text-ink absolute top-2.5 right-2.5 flex size-7 items-center justify-center rounded-md opacity-0 transition-[opacity,background-color] duration-150 group-hover/out:opacity-100">
              {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
            </button>
          )}
          <p className="text-ink text-[14.5px] leading-[1.75] whitespace-pre-wrap">
            <span className="bg-mark rounded-[4px] px-1 py-px font-medium">{out.prompt}</span>
            {out.chunks.map((c, i) => <span key={i} className="token">{c}</span>)}
            {(out.running || out.done) && <span className={cn("caret", out.running && "is-streaming")} />}
          </p>
          {out.error && <p className="text-ink-3 mt-3 text-[12px]">{out.error}</p>}
          {out.done && (
            <div className="text-ink-3 mt-auto flex flex-wrap gap-x-3 gap-y-1 pt-4 font-mono text-[11px] tabular-nums"
              style={{ animation: "token-in 300ms ease-out both" }}>
              <span>seed {out.done.seed}</span>
              <span>{out.done.n_tokens} tok</span>
              <span>{out.done.seconds.toFixed(2)}s</span>
              <span>{out.done.tok_s.toFixed(0)} tok/s</span>
            </div>
          )}
        </>
      )}
    </div>
  )
}
