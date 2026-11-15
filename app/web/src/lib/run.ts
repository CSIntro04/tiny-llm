// 跑一次流式生成，并把进度写进一个 Output 状态（单模型和三模型对比共用）
import { streamGenerate, type SamplingParams } from "@/lib/api"
import type { Output } from "@/components/OutputCard"

export async function runInto(
  ckpt: string,
  prompt: string,
  params: SamplingParams,
  setOut: (f: (o: Output) => Output) => void,
  signal: AbortSignal,
): Promise<void> {
  setOut(() => ({ prompt, chunks: [], running: true, done: null }))
  try {
    const done = await streamGenerate({ ckpt, prompt, ...params },
      (d) => setOut((o) => ({ ...o, chunks: [...o.chunks, d] })), signal)
    setOut((o) => ({ ...o, running: false, done }))
  } catch (e) {
    const aborted = (e as Error).name === "AbortError"
    setOut((o) => ({ ...o, running: false, error: aborted ? "已停止" : String((e as Error).message) }))
  }
}
