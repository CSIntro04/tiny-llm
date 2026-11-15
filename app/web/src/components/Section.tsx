// 一个编号区块："01  标题  一句说明"，下面是灰底展示区；进场时按 index 依次错开淡入上移
import type { ReactNode } from "react"

import { cn } from "@/lib/utils"

type Props = {
  id: string
  index: number
  title: string
  desc: string
  aside?: ReactNode // 标题行右侧的控件（比如 run 选择）
  children: ReactNode
  surfaceClassName?: string
}

export function Section({ id, index, title, desc, aside, children, surfaceClassName }: Props) {
  return (
    <section id={id} className="reveal border-line flex scroll-mt-6 flex-col border-b border-dashed px-5 py-8 sm:px-8 sm:py-10"
      style={{ animationDelay: `${index * 120}ms` }}>
      <div className="mb-3 flex items-center gap-3">
        <div className="flex min-w-0 flex-1 items-baseline gap-2">
          <span className="text-ink-3 font-mono text-[11px] tabular-nums">{String(index + 1).padStart(2, "0")}</span>
          <h2 className="text-ink text-[13px] font-semibold whitespace-nowrap">{title}</h2>
          <p className="text-ink-3 truncate text-[12.5px]">{desc}</p>
        </div>
        {aside}
      </div>
      <div className={cn("bg-canvas shadow-hairline relative rounded-[14px] p-3", surfaceClassName)}>{children}</div>
    </section>
  )
}
