// 模型信息：一条细线分隔的指标行（结构、参数量、词表、已训练 token、best val loss）
import { fmtM, type ModelStats as Stats } from "@/lib/api"
import { Skeleton } from "@/components/ui/skeleton"

export function ModelStats({ stats }: { stats: Stats | null }) {
  const items: [string, string, string][] = stats
    ? [
        ["结构", `${stats.n_layer}/${stats.n_embd}/${stats.n_head}`, "层 / 维 / 头"],
        ["参数", fmtM(stats.params), `非 emb ${fmtM(stats.params_non_emb)}`],
        ["词表", `${stats.vocab_size}`, stats.tokenizer.replace("Tokenizer", "")],
        ["已训练 token", fmtM(stats.tokens, 0), `${stats.iters} 步`],
        ["val loss", stats.best_val.toFixed(3), `上下文 ${stats.block_size}`],
      ]
    : []
  return (
    <div className="bg-surface shadow-card divide-line grid grid-cols-2 divide-x divide-y rounded-[10px] sm:grid-cols-5 sm:divide-y-0">
      {stats
        ? items.map(([k, v, sub]) => (
            <div key={k} className="px-3.5 py-2.5" style={{ animation: "token-in 300ms ease-out both" }}>
              <div className="text-ink-3 text-[11px]">{k}</div>
              <div className="text-ink mt-0.5 text-[15px] font-semibold tracking-tight tabular-nums">{v}</div>
              <div className="text-ink-3 truncate text-[10.5px]">{sub}</div>
            </div>
          ))
        : Array.from({ length: 5 }, (_, i) => <div key={i} className="px-3.5 py-3"><Skeleton className="h-10" /></div>)}
    </div>
  )
}
