// 和后端 app/server.py 通信：类型定义 + fetch 封装 + 流式生成（SSE）

export type Checkpoints = { default: string | null; items: string[]; examples: string[] }

export type ModelStats = {
  ckpt: string
  n_layer: number
  n_embd: number
  n_head: number
  block_size: number
  params: number
  params_non_emb: number
  tokenizer: string
  vocab_size: number
  iters: number
  tokens: number
  best_val: number
}

export type CompareModel = { run: string; ckpt: string; missing: boolean; params: number | null }
export type ResultTable = { title: string; markdown: string }

export type SamplingParams = {
  max_new_tokens: number
  temperature: number
  top_k: number // 0 = 不限
  top_p: number // 1.0 = 不限
  seed: number // -1 = 随机
}

export type GenDone = { seed: number; n_tokens: number; seconds: number; tok_s: number }

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`)
  return r.json() as Promise<T>
}

export const api = {
  checkpoints: () => getJSON<Checkpoints>("/api/checkpoints"),
  model: (ckpt: string) => getJSON<ModelStats>(`/api/model?ckpt=${encodeURIComponent(ckpt)}`),
  compareModels: () => getJSON<CompareModel[]>("/api/compare-models"),
  runs: () => getJSON<string[]>("/api/runs"),
  figures: () => getJSON<string[]>("/api/figures"),
  results: () => getJSON<ResultTable[]>("/api/results"),
}

/**
 * 流式生成：POST /api/generate，后端每生成一个 token 发一条 "data: {...}\n\n"。
 * 用 fetch + ReadableStream 逐块读取（EventSource 只支持 GET，所以不用它）。
 * signal 来自 AbortController：点"停止"时中断连接，后端随之停止生成。
 */
export async function streamGenerate(
  body: SamplingParams & { ckpt: string; prompt: string },
  onDelta: (text: string) => void,
  signal?: AbortSignal,
): Promise<GenDone | null> {
  const r = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  })
  if (!r.ok || !r.body) throw new Error(`生成失败：${r.status} ${await r.text()}`)

  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buf = ""
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return null
    buf += decoder.decode(value, { stream: true })
    // 一条消息以空行结束；最后一段可能不完整，留到下次拼接
    const parts = buf.split("\n\n")
    buf = parts.pop() ?? ""
    for (const part of parts) {
      if (!part.startsWith("data: ")) continue
      const msg = JSON.parse(part.slice(6))
      if (msg.done) return msg as GenDone
      onDelta(msg.delta)
    }
  }
}

// 数字格式：105920000 -> "105.9M"
export const fmtM = (n: number, digits = 1) => `${(n / 1e6).toFixed(digits)}M`
