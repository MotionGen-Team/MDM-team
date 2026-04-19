"""局部分支里的组内特征汇聚模块。"""

from __future__ import annotations

import torch
import torch.nn as nn


class MeanGroupPooling(nn.Module):
    """
    对每个 body group 内的多个 joint token 做 mean pooling。

    按当前实现方案，第一版先固定使用 mean pooling，
    保证局部分支稳定、简单、便于调试。
    """

    def forward(self, h_group: torch.Tensor) -> torch.Tensor:
        """
        输入:
            h_group: [T, B, J_g, D_s]

        输出:
            z_group: [T, B, D_s]
        """
        if h_group.ndim != 4:
            raise ValueError(f'Expected 4D group tensor, got shape {tuple(h_group.shape)}')
        return h_group.mean(dim=2)
