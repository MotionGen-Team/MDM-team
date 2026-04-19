"""全局分支里的时间主线 GQA 模块。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGQA(nn.Module):
    """
    时间主线上的轻量 GQA 模块。

    当前是一个“GQA 风格”的实现：
    - Query 用更多头
    - Key / Value 用较少的 KV 头
    - KV 头通过 repeat 的方式共享给多组 query 头

    这样保留了 GQA 的结构意图，同时实现复杂度仍然可控。
    """

    def __init__(self, d_model: int, num_query_heads: int = 8, num_kv_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        if d_model % num_query_heads != 0:
            raise ValueError(f'd_model={d_model} must be divisible by num_query_heads={num_query_heads}')
        if num_query_heads % num_kv_heads != 0:
            raise ValueError('num_query_heads must be divisible by num_kv_heads')

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.q_per_kv = num_query_heads // num_kv_heads

        # Query 保持完整头数，因为时间主线是 global branch 的主要表达载体。
        self.q_proj = nn.Linear(d_model, d_model)
        # Key / Value 只投影到较少的 KV 头上，这正是 GQA 省算力的关键。
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        # 最后统一投回 D，方便继续走 residual / FFN。
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: [T, B, D]

        输出:
            y: [T, B, D]
        """
        if x.ndim != 3:
            raise ValueError(f'Expected 3D input, got shape {tuple(x.shape)}')

        timesteps, batch_size, _ = x.shape

        # [T, B, D] -> [B, T, D]
        # 后面实现 attention 时，按 batch-first 组织会更直观。
        x_bt = x.permute(1, 0, 2).contiguous()

        # 把输入分别投影成 Q / K / V。
        # 其中 Q 头数更多，KV 头数更少，这是 GQA 的核心设定。
        q = self.q_proj(x_bt).view(batch_size, timesteps, self.num_query_heads, self.head_dim)
        k = self.k_proj(x_bt).view(batch_size, timesteps, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x_bt).view(batch_size, timesteps, self.num_kv_heads, self.head_dim)

        # 把较少的 KV 头共享给更多的 query 头。
        # 这里用 repeat_interleave 做显式共享，逻辑清楚，第一版也更好排错。
        # 后面如果要进一步优化算力，再考虑写更底层的高效实现。
        k = k.repeat_interleave(self.q_per_kv, dim=2)
        v = v.repeat_interleave(self.q_per_kv, dim=2)

        # [B, T, H, D_h] -> [B, H, T, D_h]
        # 这一步只是把 head 维提前，方便后面做 batched attention 计算。
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 标准 scaled dot-product attention。
        # 这里只在时间主线上建模“token 与 token 的时间关系”，
        # 不再混入 summary，因为 summary 已经在前一层完成注入了。
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        # [B, H, T, D_h] -> [B, T, D]
        # 把所有 query heads 合并回统一特征维 D。
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, timesteps, self.d_model)
        attn_output = self.out_proj(attn_output)

        # 还原回 [T, B, D]，保持和整个方案里统一的 token-first 格式一致。
        attn_output = attn_output.permute(1, 0, 2).contiguous()

        # 后续可尝试的替代结构：
        # 1. chunked temporal attention:
        #    如果后面 T 变长，可以按时间块做注意力，进一步降算力。
        # 2. window attention:
        #    让每个 token 只看局部时间窗，适合更强调局部连续性时尝试。
        # 3. 真正的高效 GQA 实现:
        #    当前版本为了清晰用了 repeat 共享 KV，后面可以换成更省显存的实现。
        # 4. rotary / relative position bias:
        #    如果后面发现纯内容注意力不够稳定，可以补更强的位置建模。
        return attn_output
