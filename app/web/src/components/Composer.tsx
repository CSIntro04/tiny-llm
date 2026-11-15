// 输入框：prompt 文本 + 右下角生成 / 停止按钮；下面一排示例 prompt
import { ArrowUp, Square } from "lucide-react"

import { cn } from "@/lib/utils"

type Props = {
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  onStop: () => void
  running: boolean
  examples: string[]
  submitLabel?: string
  rows?: number
}

export function Composer({ value, onChange, onSubmit, onStop, running, examples, submitLabel = "生成", rows = 2 }: Props) {
  return (
    <div className="space-y-2">
      <div className="bg-surface shadow-card focus-within:shadow-[0_0_0_1px_var(--line-strong),0_6px_18px_oklch(0_0_0/0.05)] rounded-[10px] transition-shadow duration-200">
        <textarea value={value} rows={rows} onChange={(e) => onChange(e.target.value)}
          placeholder="输入英文开头，例如 Once upon a time"
          onKeyDown={(e) => {
            // Enter 生成，Shift+Enter 换行
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!running) onSubmit() }
          }}
          className="text-ink placeholder:text-ink-3 block w-full resize-none bg-transparent px-3.5 pt-3 text-[14px] leading-relaxed outline-none" />
        <div className="flex items-center justify-between gap-2 px-2.5 pb-2.5">
          <span className="text-ink-3 pl-1 text-[11px]">Enter 生成 · Shift+Enter 换行</span>
          <button type="button" onClick={running ? onStop : onSubmit} disabled={!running && !value.trim()}
            className={cn(
              "flex h-7 items-center gap-1.5 rounded-full pr-3 pl-2.5 text-[12px] font-medium transition-[background-color,transform,opacity] duration-150 active:scale-[0.96] disabled:opacity-40",
              running ? "bg-field text-ink shadow-btn hover:bg-hover" : "bg-ink text-page hover:opacity-90",
            )}>
            {running ? <Square className="size-3 fill-current" /> : <ArrowUp className="size-3.5" />}
            {running ? "停止" : submitLabel}
          </button>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {examples.map((ex) => (
          <button key={ex} type="button" onClick={() => onChange(ex)}
            className="bg-surface text-ink-2 shadow-btn hover:bg-hover hover:text-ink h-6 rounded-full px-2.5 text-[11.5px] transition-[background-color,color,transform] duration-150 active:scale-[0.96]">
            {ex}
          </button>
        ))}
      </div>
    </div>
  )
}
