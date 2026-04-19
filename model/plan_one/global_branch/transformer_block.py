"""全局分支里的 transformer block 组装模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .temporal_gqa import TemporalGQA


class GlobalTransformerBlock(nn.Module):
    """
    全局分支里的单层 transformer block。

    当前结构：
    - Temporal GQA
    - residual + norm
    - FFN
    - residual + norm

    设计原因：
    - 保留 transformer block 的基本范式，尽量和官方主干风格一致。
    - 但把 attention 实现替换成更轻的 GQA。
    - 这样既保留了“官方 transformer 的骨架”，又满足当前方案对轻量 attention 的要求。
    """

    def __init__(self, d_model: int, ff_mult: int = 4, num_query_heads: int = 8, num_kv_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        # 注意力子层：这里不再用标准 MHA，而是换成时间主线上的 GQA。
        self.temporal_gqa = TemporalGQA(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
        )
        # attention 残差后的归一化。
        self.attn_norm = nn.LayerNorm(d_model)
        # FFN 仍沿用标准 transformer 的两层 MLP 结构。
        # 这一层主要负责在 attention 后做逐 token 的非线性重整。
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        # FFN 残差后的归一化。
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        输入:
            x: [T, B, D]

        输出:
            - `attn_out`: [T, B, D]
            - `x_after_attn`: [T, B, D]
            - `x_out`: [T, B, D]
        """
        # 第一步：时间主线 GQA。
        # 它负责建模“结构增强后的 temporal tokens”之间的全局时间关系。
        attn_out = self.temporal_gqa(x)
        # 第二步：attention residual + norm。
        # 让注意力更像对原 token 的更新，而不是直接替换。
        x_after_attn = self.attn_norm(x + attn_out)
        # 第三步：标准 FFN，对每个时间 token 做逐点非线性变换。
        ffn_out = self.ffn(x_after_attn)
        # 第四步：FFN residual + norm。
        x_out = self.ffn_norm(x_after_attn + ffn_out)

        # 后续可尝试的替代结构：
        # 1. pre-norm 版本：
        #    如果后面层数加深，pre-norm 有时会更稳定。
        # 2. attention 后再加一个 summary-aware FFN：
        #    如果后面觉得 summary 注入还不够，可以在 block 内再补轻量结构调制。
        # 3. 更轻的 FFN：
        #    比如 gated MLP / SwiGLU，适合后面继续压参数量时尝试。
        return {
            'attn_out': attn_out,
            'x_after_attn': x_after_attn,
            'x_out': x_out,
        }
