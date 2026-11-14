"""手写 GPT 风格的 Decoder-only Transformer。

结构：token embedding + 可学习位置编码 → L 个 Pre-LN Block（因果自注意力 + MLP）
      → 最后的 LayerNorm → lm_head（与 token embedding 共享权重）→ logits。

记号：B = batch 大小，T = 序列长度，C = n_embd，nh = n_head，hs = C // nh，V = vocab_size。

运行 `python src/model.py` 打印四档预设的参数量和资源粗估。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 512  # 最大上下文长度 T
    n_layer: int = 6  # L：Block 层数
    n_head: int = 6  # nh：注意力头数
    n_embd: int = 384  # C：隐藏维度；每个头的维度 hs = C // nh
    dropout: float = 0.0
    bias: bool = True  # Linear 和 LayerNorm 是否带 bias
    grad_ckpt: bool = False  # 梯度检查点（显存不够时开）
    manual_attn: bool = False  # True：手写注意力（教学/对比）；False：用 SDPA


# 四档模型规模（词表、上下文长度用 GPTConfig 的默认值）
MODEL_PRESETS: dict[str, dict] = {
    "debug": dict(n_layer=4, n_embd=256, n_head=4),
    "s10m": dict(n_layer=6, n_embd=384, n_head=6),
    "s50m": dict(n_layer=10, n_embd=640, n_head=10),
    "s100m": dict(n_layer=14, n_embd=768, n_head=12),
}


class CausalSelfAttention(nn.Module):
    """多头因果自注意力：(B, T, C) -> (B, T, C)。"""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd 必须能被 n_head 整除"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.manual_attn = cfg.manual_attn
        # 一个 Linear 同时算出 q、k、v，比三个 Linear 更快（一次大矩阵乘法）
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # 多头结果拼接后再做一次线性变换，混合各个头的信息
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # 因果掩码：下三角为 1。register_buffer 让它跟着 .to(device) 走，但不是可训练参数；
        # persistent=False：不存进 state_dict（它可以随时重建，没必要占 checkpoint）
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        nh, hs = self.n_head, C // self.n_head

        q, k, v = self.c_attn(x).split(C, dim=2)  # (B, T, 3C) -> 3 个 (B, T, C)
        # 拆成多头：(B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        # 把 nh 移到前面，后面的矩阵乘法就对每个头独立进行
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        if self.manual_attn:
            # 打分：(B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)，第 i 行 = 位置 i 对各位置的打分
            # 除以 sqrt(hs)：点积的方差随 hs 增大，不缩放会让 softmax 过尖、梯度消失
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
            # 因果掩码：位置 i 只能看 j <= i，未来位置设为 -inf，softmax 后权重为 0
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)  # 每行变成和为 1 的权重
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        else:
            # 数学上与上面等价；自动选 FlashAttention 等内核，不显式构造 T×T 矩阵，省显存
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )

        # 合并多头：(B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        # transpose 后内存不连续，view 之前必须 contiguous()
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """逐位置的前馈网络：C -> 4C -> GELU -> C。"""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """一个 Transformer 层（Pre-LN）：注意力负责位置间交换信息，MLP 负责每个位置上的非线性变换。"""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接：子层只学"增量"，梯度可沿 x 这条恒等路径直接传回浅层
        # Pre-LN：LayerNorm 放在子层之前，深层训练更稳定
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token id -> 向量
        # 可学习的绝对位置编码：注意力本身不区分顺序，必须显式告诉模型"第几个位置"
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 权重共享：输出层"向量 -> 每个 token 的打分"与输入"id -> 向量"用同一张表，省 V×C 个参数
        self.lm_head.weight = self.wte.weight

        # 初始化：先 tie 再初始化，两者指向同一个 Parameter，不会被拆开
        self.apply(self._init_weights)
        # 残差分支的输出投影缩小初始幅度：残差流上叠加了 2L 个分支（每层 attn + mlp），
        # 不缩小的话深层模型初始输出方差会随层数增大
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) token id。有 targets 时返回 (logits (B,T,V), loss)；否则 (logits (B,1,V), None)。"""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长度 {T} 超过 block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)  # (T,)；按实际 T 取，不写死 block_size

        # (B, T, C) + (T, C) 广播 -> (B, T, C)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            if self.cfg.grad_ckpt and self.training:
                # 梯度检查点：前向不保存块内激活，反向时重算一遍，用计算换显存
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)  # (B, T, V)
            # cross entropy = 正确 token 的 -log p 的平均；最小化它 = 最大化训练数据似然
            # 展平成 (B*T, V) 和 (B*T,)，每个位置都是一道"预测下一个 token"的分类题
            # targets 可能是切片得到的非连续张量（如 data[:, 1:]），用 reshape 而不是 view
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        # 推理时只需要最后一个位置的预测，省掉 T-1 个位置的 lm_head 计算
        logits = self.lm_head(x[:, [-1], :])  # (B, 1, V)；用 [-1] 保留 T 维
        return logits, None

    def num_params(self, non_embedding: bool = True) -> int:
        """参数量。parameters() 会对共享的 lm_head/wte 去重，只算一次。
        non_embedding=True 时减去 wte 和 wpe。"""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n

    def configure_optimizers(
        self, weight_decay: float, lr: float, betas: tuple[float, float]
    ) -> torch.optim.AdamW:
        """二维及以上的权重（Linear、Embedding）做 weight decay；bias 和 LayerNorm 参数不做。
        衰减 bias/LayerNorm 没有正则化意义，反而会限制它们的缩放/平移能力。"""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        print(f"decay 参数 {sum(p.numel() for p in decay):,}，"
              f"不 decay 参数 {sum(p.numel() for p in no_decay):,}")
        use_fused = next(self.parameters()).is_cuda  # fused 内核把多个张量的更新合并，CUDA 上更快
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=use_fused)


def _report() -> None:
    """打印四档预设的参数量，以及按 docs/00-roadmap.md 的方法粗估的显存和计算量。"""
    print(f"{'名称':<6} {'层/维/头':<10} {'非emb(M)':>9} {'总参数(M)':>10} "
          f"{'权重+优化器(GB)':>15} {'FLOPs/token(M)':>15}")
    for name, preset in MODEL_PRESETS.items():
        cfg = GPTConfig(**preset)
        with torch.device("meta"):  # meta 设备只记录形状不分配内存，构建 s100m 也是瞬间完成
            model = GPT(cfg)
        n_total = model.num_params(non_embedding=False)
        n_non_emb = model.num_params(non_embedding=True)
        # fp32 参数 4 + 梯度 4 + AdamW 两个状态 8 = 16 字节/参数（不含激活）
        state_gb = 16 * n_total / 1e9
        # 每 token 训练 FLOPs ≈ 6N（前向 2N + 反向 4N）+ 注意力的 12·L·T·C
        flops = 6 * n_total + 12 * cfg.n_layer * cfg.block_size * cfg.n_embd
        arch = f"{cfg.n_layer}/{cfg.n_embd}/{cfg.n_head}"
        print(f"{name:<8} {arch:<12} {n_non_emb / 1e6:>9.2f} {n_total / 1e6:>10.2f} "
              f"{state_gb:>15.2f} {flops / 1e6:>15.0f}")


if __name__ == "__main__":
    _report()
