"""局部分支里的多尺度时间卷积模块。"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleTemporalConv(nn.Module):
    """
    每个 group 的多尺度时间卷积模块。

    当前固定 3 个并联分支：
    - kernel_size = 3
    - kernel_size = 5
    - kernel_size = 3, dilation = 2

    设计原因：
    - `k=3` 负责抓短局部变化，最接近“相邻几帧”的动态模式。
    - `k=5` 负责看更宽一点的时间窗口，补充较慢一点的局部趋势。
    - `k=3, dilation=2` 在不明显增加参数量的情况下扩大感受野。
    - 3 路并联后再融合，比只用单一路径更容易保留不同时间尺度的信息。
    """

    def __init__(self, d_struct: int, dropout: float = 0.1):
        super().__init__()
        # 分支 1：标准 3x1 时间卷积，优先看最短时程的局部变化。
        self.branch_k3 = _TemporalConvBranch(d_struct, kernel_size=3, dilation=1, dropout=dropout)
        # 分支 2：更宽的 5x1 时间卷积，用来补充稍长一点的局部模式。
        self.branch_k5 = _TemporalConvBranch(d_struct, kernel_size=5, dilation=1, dropout=dropout)
        # 分支 3：带 dilation 的 3x1 卷积，用更低成本扩大时间感受野。
        self.branch_d2 = _TemporalConvBranch(d_struct, kernel_size=3, dilation=2, dropout=dropout)
        # 把 3 路输出在特征维拼接后，再压回原始宽度 D_s。
        # 这样后续模块不用感知这里有几条分支。
        self.merge = nn.Linear(d_struct * 3, d_struct)
        # 合并后做一次归一化，避免 3 路分支数值尺度差异过大。
        self.out_norm = nn.LayerNorm(d_struct)

    def forward(self, z_group: torch.Tensor) -> torch.Tensor:
        """
        输入:
            z_group: [T, B, D_s]

        输出:
            f_group: [T, B, D_s]
        """
        if z_group.ndim != 3:
            raise ValueError(f'Expected 3D group feature, got shape {tuple(z_group.shape)}')

        # 3 路并联，分别提取不同时间尺度下的 group 时序模式。
        b1 = self.branch_k3(z_group)
        b2 = self.branch_k5(z_group)
        b3 = self.branch_d2(z_group)

        # 在最后一维拼接，保留 3 条分支各自提取出来的时序响应。
        merged = torch.cat([b1, b2, b3], dim=-1)
        # 线性层负责把多尺度信息融合回统一宽度，方便后续接 attention。
        merged = self.merge(merged)
        return self.out_norm(merged)


class _TemporalConvBranch(nn.Module):
    """
    单个时间卷积分支。

    输入输出都保持 [T, B, D_s]，这样多分支并联后容易融合。
    """

    def __init__(self, d_struct: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        # 这里按 same padding 的思路设置 padding，保证时间长度 T 不变。
        # 我们不希望 local 分支在这一层压缩时间分辨率，否则后面很难和 global 分支逐帧对齐。
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=d_struct,
            out_channels=d_struct,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        # SiLU 比较平滑，作为局部分支里的轻量非线性更稳。
        self.activation = nn.SiLU()
        # 保留 dropout，后面如果局部分支过拟合可以直接调这个位置。
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        timesteps, batch_size, channels = x.shape

        # Conv1d 需要 [N, C, L]，这里把 batch 作为 N，时间 T 作为卷积长度。
        # 这样每个 body group 的时序特征都可以独立经过同一套卷积核。
        x = x.permute(1, 2, 0).contiguous()

        # 第一步：时间卷积。
        # 它只沿时间维聚合上下文，不改特征维宽度。
        x = self.conv(x)
        # 第二步：非线性激活，让卷积提取到的局部模式有更强表达能力。
        x = self.activation(x)
        # 第三步：dropout，控制这一分支的过拟合风险。
        x = self.dropout(x)
        # [B, D_s, T] -> [T, B, D_s]
        # 还原回局部分支统一使用的张量格式。
        x = x.permute(2, 0, 1).contiguous()

        if x.shape != (timesteps, batch_size, channels):
            raise RuntimeError(f'Unexpected temporal branch output shape: {tuple(x.shape)}')

        # 后续可尝试的替代结构：
        # 1. 再加一层 residual temporal block：
        #    适合后面发现单层卷积提取能力不够时使用。
        # 2. 把 3 路分支改成更多 dilation 组合：
        #    比如 dilation = 1 / 2 / 4 / 8。
        #    适合需要更长时间感受野时尝试。
        # 3. 深度可分离时间卷积：
        #    进一步减参数量，适合后面模型整体变大时尝试。
        # 4. 轻量 gated temporal conv：
        #    给不同时间模式加门控，适合后面发现卷积响应不够有选择性时尝试。
        return x
