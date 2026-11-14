| run | checkpoint | 已训练 token | val（加权） | val SmolLM | val TinyStories |
|---|---|---|---|---|---|
| s10m (14M) | ckpt_tok100M.pt | 100M | 3.592 | 4.158 | 2.253 |
| s50m (55M) | ckpt_tok100M.pt | 100M | 3.357 | 3.900 | 2.074 |
| s100m (106M) | ckpt_tok100M.pt | 100M | 3.357 | 3.903 | 2.065 |
| s10m (14M) | ckpt_best.pt | 300M | 3.082 | 3.608 | 1.838 |
| s50m (55M) | ckpt_tok300M.pt | 300M | 2.843 | 3.344 | 1.660 |
| s100m (106M) | ckpt_tok300M.pt | 300M | 2.810 | 3.309 | 1.630 |
| s50m (55M) | ckpt_best.pt | 1000M | 2.529 | 3.000 | 1.413 |
| s100m (106M) | ckpt_tok1000M.pt | 1000M | 2.467 | 2.930 | 1.373 |
| s100m (106M) | ckpt_best.pt | 1501M | 2.394 | 2.849 | 1.317 |

每个来源在同一批 0.82M 个验证 token 上评估（固定随机窗口）。
