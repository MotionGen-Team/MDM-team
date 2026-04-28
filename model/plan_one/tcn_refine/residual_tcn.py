"""方案一 refine 层里的残差 TCN 模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ResidualTCNBlock(nn.Module):
    """
    单个残差 TCN block。

    当前固定：
    - 通道宽度保持不变
    - kernel_size = 3
    - 通过 dilation 扩感受野
    - residual add + layer norm

    设计原因：
    - refine 层的职责是“轻量修正”，不是重新做一遍主干建模。
    - 所以 block 内保持宽度不变，结构尽量简单。
    - 通过 dilation 扩大时间感受野，但不做时间降采样，避免破坏逐帧对齐。
    """

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        # 这里用 kernel_size=3，配合 dilation 扩时间感受野。
        # padding 直接取 dilation，可以保持时间长度 T 不变。
        padding = dilation
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        # 第二层卷积继续保持通道不变，让整个 block 更像一个残差修正单元。
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        # SiLU 作为轻量非线性，通常比 ReLU 更平滑。
        self.activation = nn.SiLU()
        # dropout 先保留在 block 内，后面如果 refine 层过拟合可以直接调这里。
        self.dropout = nn.Dropout(dropout)
        # 残差相加后做 layer norm，稳定每个时间 token 的数值分布。
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: [T, B, C]

        输出:
            x_out: [T, B, C]
        """
        if x.ndim != 3:
            raise ValueError(f'Expected 3D input, got shape {tuple(x.shape)}')

        timesteps, batch_size, channels = x.shape
        residual = x

        # Conv1d 只沿时间维做建模，不改变时间长度。
        # [T, B, C] -> [B, C, T] 是为了对接 PyTorch 的 Conv1d 输入格式。
        x = x.permute(1, 2, 0).contiguous()
        # 第一步卷积：先提取一轮局部时间模式。
        x = self.conv1(x)
        # 非线性激活：增强 block 的表达能力。
        x = self.activation(x)
        # dropout：减轻 refine block 记忆训练集局部修正模式的风险。
        x = self.dropout(x)
        # 第二步卷积：在同一 dilation 下继续整合时间上下文。
        x = self.conv2(x)
        x = self.dropout(x)
        # [B, C, T] -> [T, B, C]
        # 还原回整个方案里统一的 token-first 表示格式。
        x = x.permute(2, 0, 1).contiguous()

        if x.shape != (timesteps, batch_size, channels):
            raise RuntimeError(f'Unexpected TCN block output shape: {tuple(x.shape)}')
        # 后续可尝试的替代结构：
        # 1. bottleneck TCN block:
        #    先降通道再升通道，适合后面想压参数量时尝试。
        # 2. gated TCN:
        #    在时间卷积后增加门控，适合后面想让修正更有选择性时尝试。
        # 3. 更大的 dilation 组合:
        #    比如 1 / 2 / 4 / 8，适合后面想让 refine 层看更长时间范围时尝试。
        # 4. depthwise temporal conv:
        #    适合后面整体模型变大、需要继续减轻 refine 算力时尝试。
        # 残差连接：
        # 让 block 更像对输入进行小幅修正，而不是彻底覆盖。
        # 这很符合 refine 层的定位。
        return self.out_norm(residual + x)


class CoordinationResidualTCN(nn.Module):
    """
    两层 residual TCN + 输出投影。

    输入:
        R_in [T, B, 2D]

    输出:
        delta_raw [T, B, C]

    设计原因：
    - 两个 block 已经能覆盖“短局部修正 + 稍长一点的修正”。
    - 第一版不把 refine 层做太深，避免它反过来压过 base prediction head。
    - 最后单独接一个线性层输出到 motion residual，更清楚地区分
      “时序建模部分”和“输出映射部分”。
    """

    def __init__(self, d_model: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        channels = d_model * 2
        # block1 先用 dilation=1，看最短时程的协调误差。
        self.block1 = ResidualTCNBlock(channels=channels, dilation=1, dropout=dropout)
        # block2 再用 dilation=2，把修正感受野再往外扩一点。
        self.block2 = ResidualTCNBlock(channels=channels, dilation=2, dropout=dropout)
        # 最后把 2D latent 修正结果映射回 motion residual 空间 C。
        self.out_proj = nn.Linear(channels, out_dim)
        self.gate_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, r_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 第一个残差 TCN block：优先修正短时间范围内的局部不协调。
        x1 = self.block1(r_in)
        # 第二个残差 TCN block：在第一轮修正基础上进一步看稍长一点的时间范围。
        x2 = self.block2(x1)
        # 后续可尝试的替代结构：
        # 1. 增加第三个 residual TCN block:
        #    如果后面发现两层修正能力不够，再考虑往上加深。
        # 2. 多尺度 TCN:
        #    在 refine 层里也做并联 dilation 分支，适合后面想增强修正弹性时尝试。
        # 3. 输出前增加一个小 MLP:
        #    如果线性 out_proj 不够表达，可以在输出前补一层非线性。
        # 4. 条件化 refine:
        #    后面如果想让 refine 对文本或时间步更敏感，可以把 `c` 以 FiLM/gating 方式注入。
        # 输出投影：把 refine latent 映射回 motion residual。
        gate = torch.sigmoid(self.gate_logit).to(dtype=x2.dtype, device=x2.device)
        delta_raw = gate * self.out_proj(x2)
        return {
            'x1': x1,
            'x2': x2,
            'gate': gate,
            'delta_raw': delta_raw,
        }
