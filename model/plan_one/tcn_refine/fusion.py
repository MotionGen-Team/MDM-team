"""方案一 refine 层里的融合模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    """
    显式拆分 local/global contribution 后再融合。

    输入:
        L_t [T, B, D]
        G_t [T, B, D]

    输出:
        F_t [T, B, D]

    说明：
    - 保留原 `proj` 参数名，兼容已有 checkpoint 的主体权重。
    - forward 中把 `proj.weight` 拆成 local/global 两半，显式限制
      local contribution，避免小幅 `L_t` 被 fusion Linear 重新放大。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.local_gate_max = 0.25
        # Initial effective gate is 0.10: sigmoid(-0.4054651) * 0.25.
        self.local_gate_logit = nn.Parameter(torch.tensor(-0.4054651))

    def forward(self, l_t: torch.Tensor, g_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        if l_t.shape != g_t.shape:
            raise ValueError(f'L_t and G_t must share shape, got {tuple(l_t.shape)} vs {tuple(g_t.shape)}')
        if l_t.ndim != 3:
            raise ValueError(f'Expected 3D tensors, got shape {tuple(l_t.shape)}')

        d_model = l_t.shape[-1]
        if self.proj.in_features != d_model * 2:
            raise ValueError(
                f'Fusion proj expects {self.proj.in_features} input channels, '
                f'but got local/global width {d_model}.'
            )

        fused = torch.cat([l_t, g_t], dim=-1)
        local_pre_raw = torch.nn.functional.linear(l_t, self.proj.weight[:, :d_model], None)
        global_pre = torch.nn.functional.linear(g_t, self.proj.weight[:, d_model:], self.proj.bias)
        local_gate = self.get_local_gate(dtype=l_t.dtype, device=l_t.device)
        local_pre_gated = local_pre_raw * local_gate
        fusion_pre = global_pre + local_pre_gated
        f_t = self.out_norm(fusion_pre)
        return {
            'fused': fused,
            'local_pre_raw': local_pre_raw,
            'local_pre_gated': local_pre_gated,
            'global_pre': global_pre,
            'local_gate': local_gate,
            'fusion_pre': fusion_pre,
            'f_t': f_t,
        }

    def get_local_gate(self, dtype=None, device=None) -> torch.Tensor:
        gate = torch.sigmoid(self.local_gate_logit) * self.local_gate_max
        if dtype is not None or device is not None:
            gate = gate.to(dtype=dtype or gate.dtype, device=device or gate.device)
        return gate
