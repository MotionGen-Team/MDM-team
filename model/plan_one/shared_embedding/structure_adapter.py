"""Shared Embedding 的结构适配层。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

TORSO_JOINT_INDICES = {
    22: (0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
    21: (0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20),
}


@dataclass(frozen=True)
class HMLVecLayout:
    """HumanML/KIT 向量表示的固定切片规则。"""

    joints_num: int

    @property
    def root_dim(self) -> int:
        return 4

    @property
    def ric_dim(self) -> int:
        return (self.joints_num - 1) * 3

    @property
    def rot_dim(self) -> int:
        return (self.joints_num - 1) * 6

    @property
    def local_vel_dim(self) -> int:
        return self.joints_num * 3

    @property
    def foot_dim(self) -> int:
        return 4

    @property
    def total_dim(self) -> int:
        return self.root_dim + self.ric_dim + self.rot_dim + self.local_vel_dim + self.foot_dim


class StructureAdapter(nn.Module):
    """
    把原始动作输入转换成带关节结构的张量。

    设计说明：
    - 对 `hml_vec` 来说，输入本质上是展平后的特征向量，这里先把它拆回到逐关节 token。
    - 对 `xyz` 来说，输入本来就是关节结构，只需要整理成统一的 `[T, B, G_in, F_s]`。
    - `root_data` 会广播到每个关节 token 上，这样后续模块在看局部结构时也能拿到全局轨迹信息。
    """

    def __init__(
        self,
        data_rep: str = 'hml_vec',
        joints_num: int = 22,
        root_broadcast_mode: str = 'all_joints',
    ):
        super().__init__()
        self.data_rep = data_rep
        self.joints_num = joints_num
        self.layout = HMLVecLayout(joints_num=joints_num)
        self.root_broadcast_mode = root_broadcast_mode

        # 暴露给下游模块，用来确定第一层投影的输入维度。
        if self.data_rep == 'hml_vec':
            # root(4) + ric(3) + rot(6) + local_vel(3) + foot_contact(1)
            self.output_feat_dim = 17
        elif self.data_rep == 'xyz':
            self.output_feat_dim = 3
        else:
            raise ValueError(f'Unsupported data representation: {self.data_rep}')

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x_t:
                - `hml_vec`: [B, C, 1, T]，HumanML 为 263 维，KIT 为 251 维
                - `xyz`: [B, J, 3, T]

        输出：
            X_struct:
                - 形状为 [T, B, G_in, F_s]
                - 当前实现里 `G_in == joints_num`
        """
        if self.data_rep == 'hml_vec':
            return self._forward_hml_vec(x_t)
        if self.data_rep == 'xyz':
            return self._forward_xyz(x_t)
        raise ValueError(f'Unsupported data representation: {self.data_rep}')

    def _forward_hml_vec(self, x_t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError(f'Expected 4D input for hml_vec, got shape {tuple(x_t.shape)}')

        batch_size, channels, nfeats, nframes = x_t.shape
        if nfeats != 1:
            raise ValueError(f'hml_vec expects nfeats=1, got {nfeats}')
        if channels != self.layout.total_dim:
            raise ValueError(
                f'hml_vec channel mismatch: expected {self.layout.total_dim}, got {channels}'
            )

        # [B, C, 1, T] -> [T, B, C]
        x_seq = x_t.squeeze(2).permute(2, 0, 1).contiguous()

        root = x_seq[..., : self.layout.root_dim]

        ric_start = self.layout.root_dim
        ric_end = ric_start + self.layout.ric_dim
        ric = x_seq[..., ric_start:ric_end].view(nframes, batch_size, self.joints_num - 1, 3)

        rot_start = ric_end
        rot_end = rot_start + self.layout.rot_dim
        rot = x_seq[..., rot_start:rot_end].view(nframes, batch_size, self.joints_num - 1, 6)

        vel_start = rot_end
        vel_end = vel_start + self.layout.local_vel_dim
        local_vel = x_seq[..., vel_start:vel_end].view(nframes, batch_size, self.joints_num, 3)

        foot = x_seq[..., vel_end:].view(nframes, batch_size, self.layout.foot_dim)

        # 把 root 上下文复制到每个关节 token，这样后续分支在看局部结构时
        # 不需要重新回到扁平向量里读取全局轨迹信息。
        root_ctx = x_seq.new_zeros(nframes, batch_size, self.joints_num, self.layout.root_dim)
        if self.root_broadcast_mode == 'all_joints':
            root_ctx = root.unsqueeze(2).expand(-1, -1, self.joints_num, -1)
        elif self.root_broadcast_mode == 'torso_only':
            torso_indices = TORSO_JOINT_INDICES.get(self.joints_num)
            if torso_indices is None:
                raise ValueError(f'Unsupported joints_num for torso-only root injection: {self.joints_num}')
            root_ctx[..., torso_indices, :] = root.unsqueeze(2).expand(-1, -1, len(torso_indices), -1)
        elif self.root_broadcast_mode != 'none':
            raise ValueError(f'Unsupported root_broadcast_mode: {self.root_broadcast_mode}')

        # root 本身在 hml_vec 里没有 ric/rot 对应项，这里补零保持张量对齐。
        ric_full = x_seq.new_zeros(nframes, batch_size, self.joints_num, 3)
        ric_full[..., 1:, :] = ric

        rot_full = x_seq.new_zeros(nframes, batch_size, self.joints_num, 6)
        rot_full[..., 1:, :] = rot

        # 只有脚部关节持有接触标记，其它关节统一补零。
        foot_full = x_seq.new_zeros(nframes, batch_size, self.joints_num, 1)
        if self.joints_num == 22:
            # 这里沿用当前数据集已经验证过的脚部索引约定。
            left_foot_joints = (8, 11)
            right_foot_joints = (7, 10)
        elif self.joints_num == 21:
            # KIT 这里先保留一个保守映射，后面如果接 KIT 再单独核对。
            left_foot_joints = (19, 20)
            right_foot_joints = (14, 15)
        else:
            raise ValueError(f'Unsupported joints_num for foot mapping: {self.joints_num}')

        foot_full[..., right_foot_joints[0], 0] = foot[..., 0]
        foot_full[..., right_foot_joints[1], 0] = foot[..., 1]
        foot_full[..., left_foot_joints[0], 0] = foot[..., 2]
        foot_full[..., left_foot_joints[1], 0] = foot[..., 3]

        x_struct = torch.cat([root_ctx, ric_full, rot_full, local_vel, foot_full], dim=-1)
        return x_struct

    def _forward_xyz(self, x_t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError(f'Expected 4D input for xyz, got shape {tuple(x_t.shape)}')

        batch_size, joints_num, nfeats, nframes = x_t.shape
        if joints_num != self.joints_num:
            raise ValueError(f'xyz joints mismatch: expected {self.joints_num}, got {joints_num}')
        if nfeats != 3:
            raise ValueError(f'xyz input expects nfeats=3, got {nfeats}')

        # [B, J, 3, T] -> [T, B, J, 3]
        return x_t.permute(3, 0, 1, 2).contiguous()
