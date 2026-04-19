from copy import copy
import random

import numpy as np
import torch

from data_loaders.humanml.scripts.motion_process import (
    qrot,
    recover_from_ric,
    recover_root_rot_heading_ang,
)
from data_loaders.humanml_custom_utils import HML_EE_JOINT_NAMES, HML_JOINT_NAMES


def get_target_location(motion, mean, std, lengths, joints_num, all_goal_joint_names, target_joint_names, is_heading):
    assert (lengths == lengths[0]).all(), 'currently supporting only fixed length'
    batch_size = motion.shape[0]
    extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']

    target_loc = torch.zeros(
        (batch_size, len(extended_goal_joint_names), 3, lengths[0]),
        dtype=motion.dtype,
        device=motion.device,
    )

    joints_loc = hml_to_abs_loc(motion, mean, std, joints_num)
    pelvis_loc = HML_JOINT_NAMES.index('pelvis')
    joints_loc = torch.concat([joints_loc, joints_loc[:, pelvis_loc:pelvis_loc + 1]], dim=1)

    joint_names_with_traj = HML_JOINT_NAMES + ['traj']
    for sample_idx in range(batch_size):
        req_joint_idx_in = [joint_names_with_traj.index(name) for name in target_joint_names[sample_idx]]
        req_joint_idx_out = [extended_goal_joint_names.index(name) for name in target_joint_names[sample_idx]]
        target_loc[sample_idx, req_joint_idx_out] = joints_loc[sample_idx, req_joint_idx_in]

    target_loc[:, -2, 1] = 0

    heading = recover_root_rot_heading_ang(joints_loc)
    target_loc[:, -1:, 0][is_heading] = heading[is_heading]

    return target_loc[..., -1]


def hml_to_abs_loc(motion, mean, std, joints_num):
    unnormed_motion = (motion * std + mean).permute(0, 2, 3, 1).float()
    joints_loc = recover_from_ric(unnormed_motion, joints_num)
    joints_loc = joints_loc.view(-1, *joints_loc.shape[2:]).permute(0, 2, 3, 1)
    return joints_loc


def sample_goal(batch_size, device, force_joints=None):
    if force_joints is None:
        choices = np.array(['None', 'traj', 'pelvis'] + HML_EE_JOINT_NAMES)
        none_prob = 0.5
        probabilities = torch.ones(len(choices)) * (1 - none_prob) / (len(choices) - 1)
        probabilities[0] = none_prob
        assert probabilities.sum() - 1 < 1e-6, 'probabilities should sum to 1'
        max_goal_joints_per_sample = 2
        target_cond_idx = torch.multinomial(
            probabilities, max_goal_joints_per_sample * batch_size, replacement=True
        ).view(batch_size, max_goal_joints_per_sample)
        names = choices[target_cond_idx]
        names = np.array([np.unique(name) for name in names])
        names = np.array([np.delete(name, np.argwhere(name == 'None')) for name in names])
        is_heading = torch.bernoulli(torch.ones(batch_size, device=device) * 0.5).to(bool)
    else:
        options = get_allowed_joint_options(force_joints)
        names = [copy(random.choice(options)) for _ in range(batch_size)]
        is_heading = torch.zeros(batch_size, device=device).to(bool)
        for i, name_list in enumerate(names):
            if 'heading' in name_list:
                is_heading[i] = True
                del name_list[name_list.index('heading')]
    return names, is_heading


def get_allowed_joint_options(config_name):
    if config_name == 'DIMP_FULL':
        return [['pelvis', 'heading'], ['pelvis', 'head'], ['traj', 'heading'], ['right_wrist', 'heading'], ['left_wrist', 'heading'], ['right_foot', 'heading'], ['left_foot', 'heading']]
    if config_name == 'DIMP_FINAL':
        return [['pelvis', 'heading'], ['traj', 'heading'], ['right_wrist', 'heading'], ['left_wrist', 'heading'], ['right_foot', 'heading'], ['left_foot', 'heading'], []]
    if config_name == 'DIMP_SLIM':
        return [['pelvis', 'heading'], ['pelvis', 'head'], ['traj', 'heading'], ['left_wrist', 'heading'], ['left_foot', 'heading']]
    if config_name == 'DIMP_BENCH':
        return [['pelvis', 'heading'], ['pelvis', 'head']]
    if config_name == 'PURE_T2M':
        return [[]]
    return [config_name.split(',')]
