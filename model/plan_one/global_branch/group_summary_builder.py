"""全局分支里的 group summary builder。"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


# 当前定义与实现方案文档保持一致。
GROUP_SUMMARY_INDICES: Dict[str, List[int]] = {
    'left_leg': [2, 5, 8, 11],
    'right_leg': [1, 4, 7, 10],
    'torso_upper': [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}


class GroupSummaryBuilder(nn.Module):
    """
    从 `H_struct` 生成 `S_t`。

    当前第一版：
    - 先按 3 个 body groups 切分
    - 每组做 mean pooling
    - 再做线性投影
    - 输出 `S_t [T, B, 3, D]`
    """

    def __init__(self, d_struct: int, d_model: int):
        super().__init__()
        self.group_names = ['left_leg', 'right_leg', 'torso_upper']
        self.proj = nn.Linear(d_struct, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, h_struct: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        输入:
            h_struct: [T, B, 22, D_s]

        输出:
            - `group_splits`: 每组切分后的原始张量
            - `group_tokens`: 每组 pooling 后的 token
            - `s_t`: [T, B, 3, D]
        """
        if h_struct.ndim != 4:
            raise ValueError(f'Expected 4D h_struct, got shape {tuple(h_struct.shape)}')
        if h_struct.shape[2] != 22:
            raise ValueError(f'Expected 22 structured tokens, got {h_struct.shape[2]}')

        group_splits: Dict[str, torch.Tensor] = {}
        group_tokens: Dict[str, torch.Tensor] = {}

        for group_name in self.group_names:
            group_indices = torch.tensor(
                GROUP_SUMMARY_INDICES[group_name], device=h_struct.device, dtype=torch.long
            )
            h_group = h_struct.index_select(dim=2, index=group_indices)
            group_splits[group_name] = h_group

            # 当前先用 mean pooling 生成 group summary token。
            z_group = h_group.mean(dim=2)
            z_group = self.proj(z_group)
            z_group = self.out_norm(z_group)
            group_tokens[group_name] = z_group

        s_t = torch.stack([group_tokens[name] for name in self.group_names], dim=2)
        return {
            'group_splits': group_splits,
            'group_tokens': group_tokens,
            's_t': s_t,
        }
