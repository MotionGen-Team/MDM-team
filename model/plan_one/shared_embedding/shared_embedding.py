"""方案一的 Shared Embedding 模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .structure_adapter import StructureAdapter


class SharedEmbeddingBlock(nn.Module):
    """
    方案一的共享嵌入模块。

    当前流程：
    1. `StructureAdapter` 先把输入整理成逐关节结构化特征。
    2. 在投影前做轻量时序混合，给结构特征补上短程时间上下文。
    3. 再做共享线性投影，得到结构侧 latent 表示。
    4. `H_struct` 保留关节维度，给 local/group summary 使用。
    5. `H_global` 通过对结构 token 做 mean pooling 得到。
    """

    def __init__(
        self,
        data_rep: str = 'hml_vec',
        joints_num: int = 22,
        d_model: int = 512,
        d_struct: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.data_rep = data_rep
        self.joints_num = joints_num
        self.d_model = d_model
        self.d_struct = d_struct

        self.structure_adapter = StructureAdapter(data_rep=data_rep, joints_num=joints_num)
        self.temporal_mixer = _TemporalMixer(
            feature_dim=self.structure_adapter.output_feat_dim,
            dropout=dropout,
        )
        self.temporal_norm = nn.LayerNorm(self.structure_adapter.output_feat_dim)
        self.input_proj = nn.Linear(self.structure_adapter.output_feat_dim, d_struct)
        self.pre_norm = nn.LayerNorm(d_struct)
        self.post_norm = nn.LayerNorm(d_struct)
        self.global_proj = nn.Linear(d_struct, d_model)

    def forward(self, x_t: torch.Tensor, c: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """
        输入：
            x_t: 模型外部接口格式下的原始动作张量
            c: 可选条件嵌入 `[1, B, D]`，当前只透传，不在本层内注入

        输出：
            - `x_struct`: 投影前的结构化特征
            - `h_struct`: 结构侧 latent 表示 `[T, B, G_in, D_s]`
            - `h_global`: 全局时序表示 `[T, B, D]`
            - `c`: 原样透传的条件嵌入
        """
        x_struct = self.structure_adapter(x_t)

        # 轻量时序混合放在投影前，这样时间上下文是在“结构化原始特征”层面
        # 先被注入的，而不是等投影后再补。
        x_struct = self.temporal_mixer(x_struct)
        x_struct = self.temporal_norm(x_struct)

        # 共享投影把结构特征映射到后续分支统一使用的结构 latent 宽度。
        h_struct = self.input_proj(x_struct)
        h_struct = self.pre_norm(h_struct)
        h_struct = self.post_norm(h_struct)

        # 当前先用 mean pooling 做全局汇聚，优先保证稳定和易接入。
        pooled = h_struct.mean(dim=2)
        h_global = self.global_proj(pooled)

        return {
            'x_struct': x_struct,
            'h_struct': h_struct,
            'h_global': h_global,
            'c': c,
        }


class _TemporalMixer(nn.Module):
    """
    轻量时序卷积模块。

    这里只沿时间维做混合，关节维/分组维会先折叠进 batch，
    这样同一套时间卷积就能复用到所有结构 token 上。
    """

    def __init__(self, feature_dim: int, dropout: float):
        super().__init__()
        # depthwise temporal conv:
        # 只在每个通道内部沿时间维做卷积，不在这里混合通道信息。
        # 这样参数量小，适合先给结构特征补一个“轻量时间上下文”。
        self.dw_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1,
            groups=feature_dim,
        )
        # pointwise conv:
        # 在 depthwise 之后用 1x1 卷积做通道混合，相当于把“时间提取”和“通道融合”拆开。
        # 这种 depthwise + pointwise 的组合比直接上一个重卷积更轻，也更容易稳定训练。
        self.pw_conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=1,
        )
        # SiLU 比 ReLU 更平滑，作为共享层里的轻量非线性通常更稳。
        self.activation = nn.SiLU()
        # 先保留 dropout，方便后面在小数据量或分支堆深时抑制过拟合。
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [T, B, G_in, C]

        输出：
            与输入同形状的张量
        """
        # 这里的最后一维 C 就是每个结构 token 的特征维，在 Conv1d 里对应 channel 维。
        timesteps, batch_size, groups_in, channels = x.shape
        # 残差分支保留原始结构特征。
        # 这里不希望时序混合一上来就把局部结构“洗掉”，所以最后做 residual add。
        residual = x

        # [T, B, G, C] -> [B*G, C, T]
        # Conv1d 默认吃的是 [N, C, L]，所以这里把每个 joint/group token 视作一个独立样本，
        # 统一沿时间维 T 做卷积。这样同一套时序卷积核会共享到所有结构 token 上。
        x = x.permute(1, 2, 3, 0).reshape(batch_size * groups_in, channels, timesteps)

        # 第一步：depthwise temporal conv
        # 作用：在“每个通道内部”先聚合相邻时间步信息。
        # 设计原因：我们这里想先补短程时序上下文，而不是一上来做重型时序建模。
        x = self.dw_conv(x)

        # 第二步：非线性激活
        # 给局部时间模式一点表达能力，否则两层线性卷积叠起来还是偏线性。
        x = self.activation(x)

        # 第三步：pointwise conv
        # 作用：把不同通道的时间响应再做一次轻量融合。
        # 如果只做 depthwise，通道之间完全独立，表达力会偏弱。
        x = self.pw_conv(x)

        # 第四步：dropout
        # 这里先保留一个标准位置，后面如果发现共享层过拟合，可以直接调这个超参。
        x = self.dropout(x)

        # [B*G, C, T] -> [T, B, G, C]
        # 还原回后续模块统一使用的结构张量格式。
        x = x.view(batch_size, groups_in, channels, timesteps).permute(3, 0, 1, 2).contiguous()

        # 后续可以尝试的替代结构（先写在这里，方便队友直接看到）：
        # 1. 多尺度 temporal conv:
        #    并联多个 kernel_size / dilation，再 concat 或 sum。
        #    适合想让共享层更早看到不同时间尺度模式时使用。
        # 2. dilated temporal conv:
        #    用 dilation=2/4/8 扩大感受野，参数开销仍然可控。
        #    适合后面发现当前 kernel_size=3 看得太短时尝试。
        # 3. gated temporal conv:
        #    把一部分通道当作 gate，对时序响应做门控。
        #    适合想提高共享层选择性时尝试。
        # 4. 轻量 self-attention / linear attention:
        #    如果后面发现卷积对长程依赖不够，可以在这里替换成更轻的 attention。
        #    但这一步不建议太早做，否则会把共享层做重，影响后续排错。

        # 残差连接：
        # 让这层更像“轻量修正”，而不是彻底重写结构特征。
        # 这对共享嵌入阶段尤其重要，因为后面 local/global 分支都依赖这里的输出稳定性。
        return residual + x
