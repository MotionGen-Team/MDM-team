"""全局分支里的 summary fusion 模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SummaryFusion(nn.Module):
    """
    用不同策略把 `S_t` 注入到 `H_global` 中。

    mode 可选：
    - 'crossattn' : cross-attention (baseline)
    - 'gating'    : summary gating
    - 'film'      : mean(S_t) 后线性调制 (FiLM)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        mode: str = "crossattn",
    ):
        super().__init__()
        self.mode = mode

        if mode == "crossattn":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.gate_logit = nn.Parameter(torch.tensor(-3.0))

        elif mode == "gating":
            # 把 s_t 的平均向量投影到门控值
            self.gate_proj = nn.Linear(d_model, d_model)

        elif mode == "film":
            # FiLM 的两个调制分支
            self.scale_proj = nn.Linear(d_model, d_model)
            self.shift_proj = nn.Linear(d_model, d_model)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 增强后的 layer norm
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self, h_global: torch.Tensor, s_t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        输入:
            h_global: [T, B, D]
            s_t:      [T, B, 3, D]

        输出:
            - 'delta':         [T, B, D]
            - 'gate':          标量或平均门控值（用于日志）
            - 'h_global_enh':  [T, B, D]
        """
        # ========= 通用输入校验 =========
        if h_global.ndim != 3:
            raise ValueError(
                f"Expected 3D h_global, got shape {tuple(h_global.shape)}"
            )
        if s_t.ndim != 4:
            raise ValueError(f"Expected 4D s_t, got shape {tuple(s_t.shape)}")
        if h_global.shape[0] != s_t.shape[0] or h_global.shape[1] != s_t.shape[1]:
            raise ValueError("h_global and s_t must match on T and B dimensions")
        if s_t.shape[2] != 3:
            raise ValueError(
                f"Summary fusion expects 3 summary tokens, got {s_t.shape[2]}"
            )

        # ========= 三种融合方式 =========
        if self.mode == "crossattn":
            return self._forward_crossattn(h_global, s_t)
        elif self.mode == "gating":
            return self._forward_gating(h_global, s_t)
        elif self.mode == "film":
            return self._forward_film(h_global, s_t)

    def _forward_crossattn(
        self, h_global: torch.Tensor, s_t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Baseline：cross-attention 注入"""
        timesteps, batch_size, channels = h_global.shape

        q = h_global.reshape(timesteps * batch_size, 1, channels)
        kv = s_t.reshape(timesteps * batch_size, 3, channels)

        delta, _ = self.cross_attn(q, kv, kv, need_weights=False)
        delta = delta.view(timesteps, batch_size, channels)
        gate = torch.sigmoid(self.gate_logit).to(dtype=delta.dtype, device=delta.device)

        h_global_enh = self.out_norm(h_global + gate * delta)

        return {
            "delta": delta,
            "gate": gate,
            "h_global_enh": h_global_enh,
        }

    def _forward_gating(
        self, h_global: torch.Tensor, s_t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Summary gating：用 s_t 的平均值生成门控权重"""
        # s_t: [T, B, 3, D] -> mean -> [T, B, D]
        s_mean = s_t.mean(dim=2)
        gate = torch.sigmoid(self.gate_proj(s_mean))  # [T, B, D]

        # 用 (1 + gate) 做残差式增强，避免主干信号被过度衰减
        h_global_enh = self.out_norm(h_global * (1.0 + gate))
        delta = h_global_enh - h_global

        return {
            "delta": delta,
            "gate": gate.mean(),  # 返回标量，便于观察
            "h_global_enh": h_global_enh,
        }

    def _forward_film(
        self, h_global: torch.Tensor, s_t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """FiLM：mean(S_t) 后线性调制（缩放 + 偏移）"""
        s_mean = s_t.mean(dim=2)  # [T, B, D]

        scale = self.scale_proj(s_mean)  # [T, B, D]
        shift = self.shift_proj(s_mean)  # [T, B, D]

        h_global_enh = self.out_norm(h_global * scale + shift)
        delta = h_global_enh - h_global

        return {
            "delta": delta,
            "gate": scale.mean(),  # 用 scale 的平均值作为观察指标
            "h_global_enh": h_global_enh,
        }
