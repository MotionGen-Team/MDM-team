"""局部分支里的 group-attention 模块。"""

from __future__ import annotations

import torch
import torch.nn as nn


class GroupSelfAttention(nn.Module):
    """
    对 3 个 group token 做单层 self-attention。

    注意：
    - attention 只在 group 维度上做，不在 joint 维上做
    - 每个时间步独立处理
    - 默认带 residual 和 layer norm

    设计原因：
    - 当前 local 分支只需要建模“左腿 / 右腿 / 躯干”之间的交互。
    - token 数固定为 3，attention 规模很小，不属于难调的大注意力模块。
    - 相比 gating，这里能更自然地表达 group 间的双向信息交换。
    """

    def __init__(self, d_struct: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        # MultiheadAttention 直接作用在 3 个 group token 上。
        # 第一版先固定为单头，优先保证稳定和简单。
        self.attn = nn.MultiheadAttention(
            embed_dim=d_struct,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # attention 后接 layer norm，配合 residual 稳定训练。
        self.out_norm = nn.LayerNorm(d_struct)

    def forward(self, f_groups: torch.Tensor) -> torch.Tensor:
        """
        输入:
            f_groups: [T, B, 3, D_s]

        输出:
            f_groups_attn: [T, B, 3, D_s]
        """
        if f_groups.ndim != 4:
            raise ValueError(f'Expected 4D group tensor, got shape {tuple(f_groups.shape)}')
        if f_groups.shape[2] != 3:
            raise ValueError(f'Group attention expects exactly 3 groups, got {f_groups.shape[2]}')

        timesteps, batch_size, num_groups, channels = f_groups.shape

        # [T, B, 3, D_s] -> [T*B, 3, D_s]
        # 这里把每个时间步、每个 batch 样本都视作一个独立的小 attention 问题。
        # 这样 attention 只在 group 维上做，时间维不会在这里被再次混合。
        attn_input = f_groups.reshape(timesteps * batch_size, num_groups, channels)

        # self-attention:
        # query / key / value 都是同一个 3-token group 序列。
        # 作用是让左腿、右腿、躯干在同一时刻互相看见对方的状态。
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        # residual + norm:
        # 让 attention 更像在原 group token 上做轻量修正，而不是完全重写。
        attn_output = self.out_norm(attn_input + attn_output)

        # 后续可尝试的替代结构：
        # 1. group gating:
        #    如果后面发现 attention 没带来明显收益，可以退回更简单的 gating 结构。
        # 2. 两层 group-attention:
        #    如果 1 层不够表达 group 交互，再尝试叠 2 层，但不建议一开始就堆深。
        # 3. 带 FFN 的 transformer-style block:
        #    在 attention 后面补一个小 FFN，形成更完整的 block。
        # 4. 动态 group token:
        #    后面如果手臂单独拆组，可以把 token 数从 3 扩到 4/5，再复用这套模块。
        return attn_output.view(timesteps, batch_size, num_groups, channels)
