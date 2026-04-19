"""方案一 refine 层里的预测头模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class BasePredictionHead(nn.Module):
    """
    把融合后的 latent 特征映射回 motion space。

    当前结构：
        Linear(D -> D) -> GELU -> Linear(D -> C)
    """

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, f_t: torch.Tensor) -> torch.Tensor:
        if f_t.ndim != 3:
            raise ValueError(f'Expected 3D input, got shape {tuple(f_t.shape)}')
        return self.net(f_t)


class BaseToLatentProjector(nn.Module):
    """
    把基础预测 `y_base_raw [T,B,C]` 再投回统一 latent 宽度。

    这样后面就能和 `F_t [T,B,D]` 直接拼接。
    """

    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, y_base_raw: torch.Tensor) -> torch.Tensor:
        if y_base_raw.ndim != 3:
            raise ValueError(f'Expected 3D input, got shape {tuple(y_base_raw.shape)}')
        return self.out_norm(self.proj(y_base_raw))


class RefineHeads(nn.Module):
    """
    总装基础预测头相关逻辑。

    输出:
        y_base_raw     [T, B, C]
        y_base_latent  [T, B, D]
    """

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.base_head = BasePredictionHead(d_model=d_model, out_dim=out_dim)
        self.base_to_latent = BaseToLatentProjector(in_dim=out_dim, d_model=d_model)

    def forward(self, f_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        y_base_raw = self.base_head(f_t)
        y_base_latent = self.base_to_latent(y_base_raw)
        return {
            'y_base_raw': y_base_raw,
            'y_base_latent': y_base_latent,
        }
