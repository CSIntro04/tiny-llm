// 侧栏目录：鼠标移过时，一块浅灰底色"滑"到对应条目；当前滚动到的区块加粗（scroll-spy）
import { useEffect, useRef, useState } from "react"

import { cn } from "@/lib/utils"

export type NavItem = { id: string; label: string }

// ready：区块渲染完成后再开始观察（单模型生成区块要等 checkpoint 列表加载完才出现）
export function GlideNav({ items, ready = true }: { items: NavItem[]; ready?: boolean }) {
  const listRef = useRef<HTMLUListElement>(null)
  const [hover, setHover] = useState({ top: 0, height: 0, visible: false })
  const [active, setActive] = useState(items[0]?.id)

  // scroll-spy：IntersectionObserver 观察每个区块，取最靠上、正在视口中部的那个
  useEffect(() => {
    const seen = new Map<string, boolean>()
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => seen.set(e.target.id, e.isIntersecting))
        const first = items.find((it) => seen.get(it.id))
        if (first) setActive(first.id)
      },
      { rootMargin: "-20% 0px -55% 0px" },
    )
    items.forEach((it) => { const el = document.getElementById(it.id); if (el) io.observe(el) })
    return () => io.disconnect()
  }, [items, ready])

  const glideTo = (el: HTMLElement) => setHover({ top: el.offsetTop, height: el.offsetHeight, visible: true })

  return (
    <nav aria-label="目录">
      <p className="text-ink-3 mb-2 text-[11.5px]">目录</p>
      <ul ref={listRef} className="relative flex flex-col" onMouseLeave={() => setHover((h) => ({ ...h, visible: false }))}>
        <span aria-hidden className="bg-hover pointer-events-none absolute inset-x-0 rounded-[7px]"
          style={{
            top: hover.top, height: hover.height, opacity: hover.visible ? 1 : 0,
            transition: "top 220ms var(--ease-out-strong), height 220ms var(--ease-out-strong), opacity 150ms ease",
          }} />
        {items.map((it, i) => (
          <li key={it.id} onMouseEnter={(e) => glideTo(e.currentTarget)}>
            <a href={`#${it.id}`}
              className={cn(
                "relative z-10 flex items-center gap-2.5 rounded-[7px] px-2 py-[5px] text-[12.5px] transition-colors duration-150",
                active === it.id ? "text-ink font-medium" : "text-ink-2 hover:text-ink",
              )}>
              <span className="text-ink-3 font-mono text-[10.5px] tabular-nums">{String(i + 1).padStart(2, "0")}</span>
              {it.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  )
}
