"""方案一的融合与残差修正模块。"""

from .fusion import FusionBlock
from .heads import BasePredictionHead, BaseToLatentProjector, RefineHeads
from .residual_tcn import CoordinationResidualTCN, ResidualTCNBlock

__all__ = [
    'FusionBlock',
    'BasePredictionHead',
    'BaseToLatentProjector',
    'RefineHeads',
    'CoordinationResidualTCN',
    'ResidualTCNBlock',
]
