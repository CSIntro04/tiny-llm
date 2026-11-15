// 区块 03–05：loss 曲线、对比图（点击放大）、结果表
import { useEffect, useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

import { api, type ResultTable } from "@/lib/api"
import { Section } from "@/components/Section"
import { Segmented } from "@/components/Segmented"
import { Dialog, DialogContent, DialogTitle, DialogTrigger } from "@/components/ui/dialog"

export function CurveSection({ index }: { index: number }) {
  const [runs, setRuns] = useState<string[]>([])
  const [run, setRun] = useState<string | null>(null)
  useEffect(() => {
    api.runs().then((r) => { setRuns(r); setRun(r.includes("s100m") ? "s100m" : (r[0] ?? null)) }).catch(console.error)
  }, [])

  return (
    <Section id="curve" index={index} title="Loss 曲线" desc="左：按 step；右：按已训练 token 数。"
      aside={runs.length > 0 && <Segmented size="sm" value={run} onChange={setRun} options={runs.map((r) => ({ value: r, label: r }))} />}>
      <div className="bg-surface shadow-card overflow-hidden rounded-[10px] p-2">
        {/* key=run：换 run 时重新挂载 img，重播淡入动画 */}
        {run && <img key={run} src={`/api/runs/${encodeURIComponent(run)}/curve`} alt={`${run} loss 曲线`}
          className="token w-full rounded-md bg-white" />}
      </div>
    </Section>
  )
}

export function FiguresSection({ index }: { index: number }) {
  const [figures, setFigures] = useState<string[]>([])
  useEffect(() => { api.figures().then(setFigures).catch(console.error) }, [])

  return (
    <Section id="figures" index={index} title="对比图" desc="docs/figures/ 下的实验图，点击放大。">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {figures.map((f) => (
          <Dialog key={f}>
            <DialogTrigger asChild>
              <button className="bg-surface shadow-card group overflow-hidden rounded-[10px] text-left transition-transform duration-200 ease-[var(--ease-out-strong)] hover:-translate-y-0.5 active:scale-[0.98]">
                <div className="bg-white p-1.5">
                  <img src={`/figures/${f}`} alt={f} loading="lazy" className="aspect-[4/3] w-full object-contain" />
                </div>
                <div className="border-line text-ink-3 group-hover:text-ink-2 border-t px-3 py-2 font-mono text-[11px] transition-colors">{f}</div>
              </button>
            </DialogTrigger>
            <DialogContent className="shadow-overlay border-0 sm:max-w-4xl">
              <DialogTitle className="text-ink-2 font-mono text-[12px] font-normal">{f}</DialogTitle>
              <img src={`/figures/${f}`} alt={f} className="w-full rounded-md bg-white" />
            </DialogContent>
          </Dialog>
        ))}
      </div>
    </Section>
  )
}

export function ResultsSection({ index }: { index: number }) {
  const [tables, setTables] = useState<ResultTable[]>([])
  useEffect(() => { api.results().then(setTables).catch(console.error) }, [])

  return (
    <Section id="results" index={index} title="结果表" desc="实验 A（模型规模）与实验 C（字符级 vs BPE）的数字。">
      <div className="space-y-3">
        {tables.map((t) => (
          <div key={t.title} className="bg-surface shadow-card overflow-hidden rounded-[10px]">
            <div className="border-line text-ink border-b px-3.5 py-2.5 text-[12.5px] font-semibold">{t.title}</div>
            <div className="md-table overflow-x-auto px-1 pb-1">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{t.markdown}</ReactMarkdown>
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}
