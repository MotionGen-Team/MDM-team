"""变体：GQA，q=8，kv=1（更轻的 KV 头）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGQA(nn.Module):
    """
    时间主线 GQA 模块变体（默认 q=8，kv=1）。

    作为 temporal_gqa.TemporalGQA 的可直接替换版本。
    """

    def __init__(self, d_model: int, num_query_heads: int = 8, num_kv_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        if d_model % num_query_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_query_heads={num_query_heads}")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.q_per_kv = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")

        timesteps, batch_size, _ = x.shape
        x_bt = x.permute(1, 0, 2).contiguous()

        q = self.q_proj(x_bt).view(batch_size, timesteps, self.num_query_heads, self.head_dim)
        k = self.k_proj(x_bt).view(batch_size, timesteps, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x_bt).view(batch_size, timesteps, self.num_kv_heads, self.head_dim)

        k = k.repeat_interleave(self.q_per_kv, dim=2)
        v = v.repeat_interleave(self.q_per_kv, dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, timesteps, self.d_model)
        attn_output = self.out_proj(attn_output)

        return attn_output.permute(1, 0, 2).contiguous()
