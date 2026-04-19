"""局部分支里的 body-group 切分定义。"""

from __future__ import annotations

from typing import Dict, List

import torch


# 当前这套索引与实现方案文档保持一致。
BODY_GROUPS: Dict[str, List[int]] = {
    'left_leg': [2, 5, 8, 11],
    'right_leg': [1, 4, 7, 10],
    'torso_upper': [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}


def split_body_groups(h_struct: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    按 body groups 切分结构侧输入。

    输入:
        h_struct: [T, B, 22, D_s]

    输出:
        dict(name -> [T, B, J_g, D_s])
    """
    if h_struct.ndim != 4:
        raise ValueError(f'Expected 4D h_struct, got shape {tuple(h_struct.shape)}')
    if h_struct.shape[2] != 22:
        raise ValueError(f'Expected 22 structured tokens, got {h_struct.shape[2]}')

    outputs: Dict[str, torch.Tensor] = {}
    for group_name, group_indices in BODY_GROUPS.items():
        index_tensor = torch.tensor(group_indices, device=h_struct.device, dtype=torch.long)
        outputs[group_name] = h_struct.index_select(dim=2, index=index_tensor)
    return outputs
