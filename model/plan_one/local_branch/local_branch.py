"""局部分支总装模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .body_groups import BODY_GROUPS, split_body_groups
from .group_attention import GroupSelfAttention
from .group_pooling import MeanGroupPooling
from .multi_scale_temporal import get_multi_scale_tcn


class LocalBranch(nn.Module):
    """
    方案一的局部分支。

    当前实现严格对齐实现文档中的顺序：
    1. 按 body groups 切分 `H_struct`
    2. 每个 group 做组内 mean pooling
    3. 每个 group 做 multi-scale temporal conv
    4. 对 3 个 group token 做单层 group self-attention
    5. concat 3 个 group token，再映射到统一的 `L_t`
    """

    def __init__(self, d_struct: int, d_model: int, dropout: float = 0.1, multi_scale_variant: str = 'baseline'):
        super().__init__()
        self.group_names = ['left_leg', 'right_leg', 'torso_upper']
        self.group_pooling = MeanGroupPooling()
        self.group_temporal = nn.ModuleDict({
            group_name: get_multi_scale_tcn(multi_scale_variant, latent_dim=d_struct, dropout=dropout)
            for group_name in self.group_names
        })
        self.group_attention = GroupSelfAttention(d_struct=d_struct, num_heads=1, dropout=dropout)
        self.out_proj = nn.Linear(d_struct * len(self.group_names), d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, h_struct: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        输入:
            h_struct: [T, B, 22, D_s]

        输出:
            - `group_splits`: 切分后的各组张量
            - `group_tokens`: pooling + temporal conv 后的各组时序特征
            - `group_stack`: attention 前的 3 组堆叠结果
            - `group_stack_attn`: attention 后的 3 组堆叠结果
            - `l_t`: 局部分支最终输出 [T, B, D]
        """
        group_splits = split_body_groups(h_struct)

        group_tokens = {}
        for group_name in self.group_names:
            h_group = group_splits[group_name]
            z_group = self.group_pooling(h_group)
            f_group = self.group_temporal[group_name](z_group)
            group_tokens[group_name] = f_group

        group_stack = torch.stack([group_tokens[name] for name in self.group_names], dim=2)
        group_stack_attn = self.group_attention(group_stack)

        # [T, B, 3, D_s] -> [T, B, 3 * D_s]
        timesteps, batch_size, num_groups, channels = group_stack_attn.shape
        local_flat = group_stack_attn.reshape(timesteps, batch_size, num_groups * channels)
        l_t = self.out_proj(local_flat)
        l_t = self.out_norm(l_t)

        return {
            'group_splits': group_splits,
            'group_tokens': group_tokens,
            'group_stack': group_stack,
            'group_stack_attn': group_stack_attn,
            'l_t': l_t,
        }
