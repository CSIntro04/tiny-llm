// 区块 02：三模型对比。同一 prompt、同一组参数、同一 seed，依次让 0.01B / 0.05B / 0.1B 流式生成
import { useEffect, useRef, useState } from "react"
import { ChevronDown, SlidersHorizontal } from "lucide-react"

import { api, fmtM, type CompareModel, type SamplingParams } from "@/lib/api"
import { runInto } from "@/lib/run"
import { resolveSeed } from "@/lib/seed"
import { Composer } from "@/components/Composer"
import { EMPTY_OUTPUT, OutputCard, type Output } from "@/components/OutputCard"
import { DEFAULT_PARAMS, SamplingPanel } from "@/components/SamplingPanel"
import { Section } from "@/components/Section"
import { cn } from "@/lib/utils"

export function CompareSection({ index, examples }: { index: number; examples: string[] }) {
  const [models, setModels] = useState<CompareModel[]>([])
  const [prompt, setPrompt] = useState("Once upon a time")
  const [params, setParams] = useState<SamplingParams>(DEFAULT_PARAMS)
  const [outs, setOuts] = useState<Output[]>([EMPTY_OUTPUT, EMPTY_OUTPUT, EMPTY_OUTPUT])
  const [running, setRunning] = useState(false)
  const [showParams, setShowParams] = useState(false)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => { api.compareModels().then(setModels).catch(console.error) }, [])

  const generate = async () => {
    setRunning(true)
    setOuts([EMPTY_OUTPUT, EMPTY_OUTPUT, EMPTY_OUTPUT])
    abort.current = new AbortController()
    // seed = -1 时在这里先定下来，保证三个模型用同一个随机种子
    const p = { ...params, seed: resolveSeed(params.seed) }
    for (const [i, m] of models.entries()) {
      if (m.missing || abort.current.signal.aborted) continue
      const setOne = (f: (o: Output) => Output) => setOuts((all) => all.map((o, j) => (j === i ? f(o) : o)))
      await runInto(m.ckpt, prompt, p, setOne, abort.current.signal)
    }
    setRunning(false)
  }

  const summary = `T ${params.temperature} · k ${params.top_k || "∞"} · p ${params.top_p} · ${params.max_new_tokens} tok · seed ${params.seed}`

  return (
    <Section id="compare" index={index} title="三模型对比" desc="同一 prompt、同一组参数、同一 seed，比较三个规模。">
      <div className="space-y-3">
        <Composer value={prompt} onChange={setPrompt} onSubmit={generate} onStop={() => abort.current?.abort()}
          running={running} examples={examples} submitLabel="三个一起生成" rows={1} />

        {/* 参数面板：grid-rows 从 0fr 过渡到 1fr，实现高度自适应的展开动画 */}
        <div className="bg-surface shadow-card rounded-[10px]">
          <button onClick={() => setShowParams((s) => !s)}
            className="text-ink-2 hover:text-ink flex w-full items-center gap-2 px-3.5 py-2.5 text-[12px] transition-colors">
            <SlidersHorizontal className="size-3.5" />
            采样参数
            <span className="text-ink-3 truncate font-mono text-[11px]">{summary}</span>
            <ChevronDown className={cn("ml-auto size-3.5 transition-transform duration-300", showParams && "rotate-180")} />
          </button>
          <div className="grid transition-[grid-template-rows] duration-300 ease-[var(--ease-out-strong)]"
            style={{ gridTemplateRows: showParams ? "1fr" : "0fr" }}>
            <div className="overflow-hidden">
              <div className="border-line max-w-sm border-t border-dashed px-3.5 pt-4 pb-4"><SamplingPanel value={params} onChange={setParams} /></div>
            </div>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          {models.map((m, i) => (
            <div key={m.run} className="flex min-w-0 flex-col gap-2">
              <div className="flex items-baseline gap-2 px-1">
                <span className="text-ink text-[12.5px] font-semibold">{m.run}</span>
                <span className="text-ink-3 font-mono text-[11px] tabular-nums">{m.params ? fmtM(m.params) : "缺失"}</span>
                {outs[i].running && <span className="bg-ink ml-auto size-1.5 animate-pulse rounded-full" />}
              </div>
              <OutputCard out={outs[i]} className="flex-1" placeholder={m.missing ? `找不到 ${m.ckpt}` : "等待生成"} />
            </div>
          ))}
        </div>
      </div>
    </Section>
  )
}
