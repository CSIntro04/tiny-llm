// 分段选择器：一排选项，选中项下面有一块"滑块"平滑移动过去（采样预设、深浅色、run 选择共用）
import { useLayoutEffect, useRef, useState, type ReactNode } from "react"

import { cn } from "@/lib/utils"

type Option<T extends string> = { value: T; label: ReactNode; title?: string }

type Props<T extends string> = {
  options: Option<T>[]
  value: T | null // null：没有选中项（比如手动调了滑块，不再等于任何预设）
  onChange: (v: T) => void
  className?: string
  size?: "sm" | "md"
}

export function Segmented<T extends string>({ options, value, onChange, className, size = "md" }: Props<T>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const [thumb, setThumb] = useState({ left: 0, width: 0, visible: false })

  // 选中项变化时，量出它的位置和宽度，把滑块移过去（CSS transition 负责动画）
  useLayoutEffect(() => {
    const el = refs.current[options.findIndex((o) => o.value === value)]
    setThumb(el ? { left: el.offsetLeft, width: el.offsetWidth, visible: true } : (t) => ({ ...t, visible: false }))
  }, [value, options])

  return (
    <div className={cn("bg-field relative inline-flex rounded-full p-0.5", className)}>
      <span aria-hidden className="bg-surface shadow-btn absolute inset-y-0.5 rounded-full"
        style={{
          left: thumb.left, width: thumb.width, opacity: thumb.visible ? 1 : 0,
          transition: "left 260ms var(--ease-out-strong), width 260ms var(--ease-out-strong), opacity 150ms",
        }} />
      {options.map((o, i) => (
        <button key={o.value} ref={(el) => { refs.current[i] = el }} type="button" title={o.title}
          onClick={() => onChange(o.value)}
          className={cn(
            "relative z-10 flex flex-1 items-center justify-center gap-1 rounded-full font-medium whitespace-nowrap transition-colors duration-150",
            size === "sm" ? "h-6 px-2.5 text-[11.5px]" : "h-7 px-3 text-[12px]",
            o.value === value ? "text-ink" : "text-ink-3 hover:text-ink-2",
          )}>
          {o.label}
        </button>
      ))}
    </div>
  )
}
