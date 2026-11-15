| run | 分词器（词表） | SmolLM loss | SmolLM 字节/token | **SmolLM bpb** | TinyStories loss | TinyStories 字节/token | **TinyStories bpb** |
|---|---|---|---|---|---|---|---|
| s10m | BPETokenizer (8192) | 3.665 | 3.87 | **1.367** | 1.897 | 4.01 | **0.683** |
| s10m_char | CharTokenizer (336) | 1.114 | 1.00 | **1.602** | 0.643 | 1.00 | **0.929** |
| s50m | BPETokenizer (8192) | 3.054 | 3.87 | **1.139** | 1.486 | 4.01 | **0.535** |
| s100m | BPETokenizer (8192) | 2.903 | 3.87 | **1.082** | 1.396 | 4.01 | **0.502** |

bpb = 总 NLL / ln2 / UTF-8 字节数，所有 run 用同一段验证原文。
