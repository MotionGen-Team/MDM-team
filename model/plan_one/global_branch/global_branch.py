"""全局分支总装模块。"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .group_summary_builder import GroupSummaryBuilder
from .summary_fusion import SummaryFusion
from .transformer_block import GlobalTransformerBlock


class GlobalBranch(nn.Module):
    """
    方案一的全局分支。

    当前实现严格对齐实现文档中的顺序：
    1. `GroupSummaryBuilder` 从 `H_struct` 生成 `S_t`
    2. `SummaryFusion` 先把 `S_t` 注入 `H_global`
    3. 再让增强后的 `H_global_enh` 进入 GQA transformer 主干
    4. 输出 `G_t [T, B, D]`
    """

    def __init__(
        self,
        d_struct: int,
        d_model: int,
        num_layers: int = 2,
        ff_mult: int = 4,
        num_query_heads: int = 8,
        num_kv_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.group_summary_builder = GroupSummaryBuilder(d_struct=d_struct, d_model=d_model)
        self.summary_fusion = SummaryFusion(d_model=d_model, num_heads=1, dropout=dropout)
        self.blocks = nn.ModuleList([
            GlobalTransformerBlock(
                d_model=d_model,
                ff_mult=ff_mult,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, h_struct: torch.Tensor, h_global: torch.Tensor, c: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """
        输入:
            h_struct: [T, B, 22, D_s]
            h_global: [T, B, D]
            c:        [1, B, D] 或 None

        输出:
            - `s_t`: [T, B, 3, D]
            - `h_global_enh`: [T, B, D]
            - `g_t`: [T, B, D]
            - `c`: 原样透传
        """
        summary_outputs = self.group_summary_builder(h_struct)
        s_t = summary_outputs['s_t']

        fusion_outputs = self.summary_fusion(h_global, s_t)
        x = fusion_outputs['h_global_enh']

        block_outputs: List[Dict[str, torch.Tensor]] = []
        for block in self.blocks:
            outputs = block(x)
            x = outputs['x_out']
            block_outputs.append(outputs)

        return {
            'group_summary': summary_outputs,
            's_t': s_t,
            'summary_fusion': fusion_outputs,
            'h_global_enh': fusion_outputs['h_global_enh'],
            'block_outputs': block_outputs,
            'g_t': x,
            'c': c,
        }
