"""全局分支里的 summary fusion 模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SummaryFusion(nn.Module):
    """
    用 cross-attention 把 `S_t` 注入到 `H_global` 中。

    当前做法：
    - Q = H_global
    - K = S_t
    - V = S_t
    - 每个时间步独立做 attention
    - 最后做 residual + layer norm

    设计原因：
    - `H_global` 是时间主线，所以 query 必须来自它。
    - `S_t` 是结构摘要，所以 key/value 用它最自然。
    - 这里先做结构增强，再进入后面的 GQA transformer，符合方案图里
      “temporal tokens + group summary -> 结构增强的时间建模”的顺序。
    """

    def __init__(self, d_model: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        # 这里先用标准 MultiheadAttention 实现 summary fusion。
        # 注意它的职责不是替代 global 主干，而是先把结构摘要注入时间主线。
        # 第一版固定单头，优先保证行为清晰、便于排查。
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_logit = nn.Parameter(torch.tensor(-1.3862944))
        self.register_buffer('current_train_step', torch.zeros((), dtype=torch.long))
        self.warmup_start_step = 1000
        self.warmup_end_step = 10000
        self.warmup_start_scale = 0.25
        # 注入后做一次 layer norm，避免结构增强把时间主线数值尺度拉偏。
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, h_global: torch.Tensor, s_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        输入:
            h_global: [T, B, D]
            s_t:      [T, B, 3, D]

        输出:
            - `delta`:         [T, B, D]
            - `h_global_enh`:  [T, B, D]
        """
        if h_global.ndim != 3:
            raise ValueError(f'Expected 3D h_global, got shape {tuple(h_global.shape)}')
        if s_t.ndim != 4:
            raise ValueError(f'Expected 4D s_t, got shape {tuple(s_t.shape)}')
        if h_global.shape[0] != s_t.shape[0] or h_global.shape[1] != s_t.shape[1]:
            raise ValueError('h_global and s_t must match on T and B dimensions')
        if s_t.shape[2] != 3:
            raise ValueError(f'Summary fusion expects 3 summary tokens, got {s_t.shape[2]}')

        timesteps, batch_size, channels = h_global.shape

        # [T, B, D] -> [T*B, 1, D]
        # 每个时间步、每个样本都看成一个独立的 cross-attention 小问题。
        # 这样这里不会再次沿时间维混合信息，而是专注做“同一时刻的结构注入”。
        q = h_global.reshape(timesteps * batch_size, 1, channels)
        # [T, B, 3, D] -> [T*B, 3, D]
        # 每个时间步都对应 3 个 group summary token：左腿、右腿、躯干。
        kv = s_t.reshape(timesteps * batch_size, 3, channels)

        # 每个时间步上，时间主线 token 去查询同一时刻的 3 个 group summary。
        # 这一步的含义是：
        # “当前时刻的全局时间表示，去看这一时刻左腿/右腿/躯干分别在说什么。”
        delta, _ = self.cross_attn(q, kv, kv, need_weights=False)
        delta = delta.view(timesteps, batch_size, channels)
        gate = self.get_gate(dtype=delta.dtype, device=delta.device)

        # 残差 + 归一化：
        # 保证 summary 注入更像轻量增强，而不是直接重写时间主线。
        # 这对后面接 GQA 主干很重要，因为我们希望 GQA 看到的是“增强后的时间主线”，
        # 而不是完全被 summary 主导的表示。
        h_global_enh = self.out_norm(h_global + gate * delta)

        # 后续可尝试的替代结构：
        # 1. summary gating:
        #    如果后面发现 cross-attention 过重或收益不大，可以退回更简单的 gating。
        # 2. 两阶段 summary fusion:
        #    先 group 内摘要，再加一个跨组摘要，适合后面想增强结构信息层次时尝试。
        # 3. 带 FFN 的 summary fusion block:
        #    在 cross-attention 后加一个小 FFN，形成更完整的 transformer-style 小块。
        # 4. 更细粒度 summary token:
        #    后面如果手臂单独拆组，可以把 3 个 summary token 扩到 4/5 个继续复用。
        return {
            'delta': delta,
            'gate': gate,
            'h_global_enh': h_global_enh,
        }

    def set_train_step(self, step: int) -> None:
        self.current_train_step.fill_(int(step))

    def get_gate(self, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        raw_gate = torch.sigmoid(self.gate_logit)
        step = self.current_train_step.to(dtype=raw_gate.dtype, device=raw_gate.device)
        progress = (step - self.warmup_start_step) / (self.warmup_end_step - self.warmup_start_step)
        progress = progress.clamp(0.0, 1.0)
        scale = self.warmup_start_scale + (1.0 - self.warmup_start_scale) * progress
        gate = raw_gate * scale
        if dtype is not None or device is not None:
            gate = gate.to(dtype=dtype or gate.dtype, device=device or gate.device)
        return gate
