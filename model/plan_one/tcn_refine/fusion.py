"""方案一 refine 层里的融合模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    """
    按定稿版执行 `concat + projection` 融合。

    输入:
        L_t [T, B, D]
        G_t [T, B, D]

    输出:
        F_t [T, B, D]

    说明：
    - 第一版先用最直接的拼接融合，优先保证稳定、简单、易排查。
    - 后续如果发现 local / global 之间需要更强的双向对齐，
      可以把这里替换成 cross-attention 融合。
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 当前先把 local/global 在特征维直接拼接，再投回统一宽度 D。
        self.proj = nn.Linear(d_model * 2, d_model)
        # 投影后做一层归一化，避免 local/global 两路特征数值尺度差异过大。
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, l_t: torch.Tensor, g_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        if l_t.shape != g_t.shape:
            raise ValueError(f'L_t and G_t must share shape, got {tuple(l_t.shape)} vs {tuple(g_t.shape)}')
        if l_t.ndim != 3:
            raise ValueError(f'Expected 3D tensors, got shape {tuple(l_t.shape)}')

        fused = torch.cat([l_t, g_t], dim=-1)
        f_t = self.out_norm(self.proj(fused))
        return {
            'fused': fused,
            'f_t': f_t,
        }
