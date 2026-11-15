// 区块 01：单模型生成。左边选 checkpoint + 采样参数，右边模型信息、输入框、输出
import { useEffect, useRef, useState } from "react"
import { RotateCw } from "lucide-react"

import { api, type Checkpoints, type ModelStats as Stats, type SamplingParams } from "@/lib/api"
import { runInto } from "@/lib/run"
import { Composer } from "@/components/Composer"
import { ModelStats } from "@/components/ModelStats"
import { EMPTY_OUTPUT, OutputCard, type Output } from "@/components/OutputCard"
import { DEFAULT_PARAMS, SamplingPanel } from "@/components/SamplingPanel"
import { Section } from "@/components/Section"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

export function GenerateSection({ index, ckpts, reload }: { index: number; ckpts: Checkpoints; reload: () => void }) {
  const [ckpt, setCkpt] = useState(ckpts.default ?? "")
  const [stats, setStats] = useState<Stats | null>(null)
  const [prompt, setPrompt] = useState("Once upon a time")
  const [params, setParams] = useState<SamplingParams>(DEFAULT_PARAMS)
  const [out, setOut] = useState<Output>(EMPTY_OUTPUT)
  const abort = useRef<AbortController | null>(null)

  // 切换 checkpoint 时重新取模型信息（第一次会触发后端加载模型，稍等几秒）
  useEffect(() => {
    if (!ckpt) return
    setStats(null)
    api.model(ckpt).then(setStats).catch(console.error)
  }, [ckpt])

  const generate = () => {
    abort.current = new AbortController()
    runInto(ckpt, prompt, params, setOut, abort.current.signal)
  }

  return (
    <Section id="generate" index={index} title="单模型生成" desc="选一个 checkpoint，调采样参数，看它逐 token 续写。">
      <div className="grid gap-3 lg:grid-cols-[236px_minmax(0,1fr)]">
        <div className="bg-surface shadow-card h-fit space-y-5 rounded-[10px] p-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-ink-2 text-[12px]">Checkpoint</span>
              <button onClick={reload} title="刷新列表"
                className="text-ink-3 hover:text-ink transition-transform duration-300 active:rotate-180">
                <RotateCw className="size-3" />
              </button>
            </div>
            <Select value={ckpt} onValueChange={setCkpt}>
              <SelectTrigger size="sm" className="bg-field shadow-btn w-full border-0 font-mono text-[11px]"><SelectValue /></SelectTrigger>
              <SelectContent>
                {ckpts.items.map((c) => <SelectItem key={c} value={c} className="font-mono text-[11px]">{c}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div className="border-line border-t border-dashed" />
          <SamplingPanel value={params} onChange={setParams} />
        </div>

        <div className="min-w-0 space-y-3">
          <ModelStats stats={stats} />
          <Composer value={prompt} onChange={setPrompt} onSubmit={generate} onStop={() => abort.current?.abort()}
            running={out.running} examples={ckpts.examples} />
          <OutputCard out={out} />
        </div>
      </div>
    </Section>
  )
}
