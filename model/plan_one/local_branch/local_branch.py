"""局部分支总装模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .body_groups import BODY_GROUPS, split_body_groups
from .group_attention import GroupSelfAttention
from .group_pooling import MeanGroupPooling
from .multi_scale_temporal import MultiScaleTemporalConv


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

    def __init__(self, d_struct: int, d_model: int, dropout: float = 0.1, variant: str = 'full'):
        super().__init__()
        if variant not in ('full', 'shared_gate'):
            raise ValueError(f'Unsupported LocalBranch variant: {variant}')

        self.variant = variant
        self.group_names = ['left_leg', 'right_leg', 'torso_upper']
        self.group_pooling = MeanGroupPooling()

        if self.variant == 'full':
            self.group_temporal = nn.ModuleDict({
                group_name: MultiScaleTemporalConv(d_struct=d_struct, dropout=dropout)
                for group_name in self.group_names
            })
            self.group_attention = GroupSelfAttention(d_struct=d_struct, num_heads=1, dropout=dropout)
        else:
            # v2 local branch:
            # - left/right legs share the same temporal filters
            # - group self-attention is disabled
            # - local output is capped by a small learnable scalar gate
            self.leg_temporal = MultiScaleTemporalConv(d_struct=d_struct, dropout=dropout)
            self.torso_temporal = MultiScaleTemporalConv(d_struct=d_struct, dropout=dropout)
            self.local_gate_max = 0.25
            self.local_gate_logit = nn.Parameter(torch.tensor(-1.3862944))

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
        if self.variant == 'full':
            return self._forward_full(h_struct)
        return self._forward_shared_gate(h_struct)

    def _forward_full(self, h_struct: torch.Tensor) -> Dict[str, torch.Tensor]:
        group_splits = split_body_groups(h_struct)
        group_tokens = {}
        for group_name in self.group_names:
            h_group = group_splits[group_name]
            z_group = self.group_pooling(h_group)
            f_group = self.group_temporal[group_name](z_group)
            group_tokens[group_name] = f_group

        group_stack = torch.stack([group_tokens[name] for name in self.group_names], dim=2)
        group_stack_attn = self.group_attention(group_stack)
        l_t = self._project_group_stack(group_stack_attn)

        return {
            'group_splits': group_splits,
            'group_tokens': group_tokens,
            'group_stack': group_stack,
            'group_stack_attn': group_stack_attn,
            'l_t': l_t,
        }

    def _forward_shared_gate(self, h_struct: torch.Tensor) -> Dict[str, torch.Tensor]:
        group_splits = split_body_groups(h_struct)
        group_tokens = {}
        for group_name in self.group_names:
            h_group = group_splits[group_name]
            z_group = self.group_pooling(h_group)
            if group_name in ('left_leg', 'right_leg'):
                f_group = self.leg_temporal(z_group)
            else:
                f_group = self.torso_temporal(z_group)
            group_tokens[group_name] = f_group

        group_stack = torch.stack([group_tokens[name] for name in self.group_names], dim=2)
        group_stack_attn = group_stack
        l_t_pre_gate = self._project_group_stack(group_stack_attn)
        local_gate = torch.sigmoid(self.local_gate_logit) * self.local_gate_max
        l_t = l_t_pre_gate * local_gate

        return {
            'group_splits': group_splits,
            'group_tokens': group_tokens,
            'group_stack': group_stack,
            'group_stack_attn': group_stack_attn,
            'l_t_pre_gate': l_t_pre_gate,
            'local_gate': local_gate,
            'l_t': l_t,
        }

    def _project_group_stack(self, group_stack: torch.Tensor) -> torch.Tensor:
        # [T, B, 3, D_s] -> [T, B, 3 * D_s]
        timesteps, batch_size, num_groups, channels = group_stack.shape
        local_flat = group_stack.reshape(timesteps, batch_size, num_groups * channels)
        l_t = self.out_proj(local_flat)
        return self.out_norm(l_t)
