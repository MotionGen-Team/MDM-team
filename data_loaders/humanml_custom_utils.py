import numpy as np

# Custom joint convention aligned with the user's verified visualization order.
# The kinematic tree itself still matches paramUtil.t2m_kinematic_chain.
HML_JOINT_NAMES = [
    'pelvis',
    'right_hip',
    'left_hip',
    'spine1',
    'right_knee',
    'left_knee',
    'spine2',
    'right_ankle',
    'left_ankle',
    'spine3',
    'right_foot',
    'left_foot',
    'neck',
    'right_collar',
    'left_collar',
    'head',
    'right_shoulder',
    'left_shoulder',
    'right_elbow',
    'left_elbow',
    'right_wrist',
    'left_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)

HML_EE_JOINT_NAMES = ['left_foot', 'right_foot', 'left_wrist', 'right_wrist', 'head']
HML_LOWER_BODY_JOINTS = [
    HML_JOINT_NAMES.index(name)
    for name in [
        'pelvis',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        'left_foot',
        'right_foot',
    ]
]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]

# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS - 1))
HML_ROOT_MASK = np.concatenate((
    [True] * (1 + 2 + 1),
    HML_ROOT_BINARY[1:].repeat(3),
    HML_ROOT_BINARY[1:].repeat(6),
    HML_ROOT_BINARY.repeat(3),
    [False] * 4,
))
HML_ROOT_HORIZONTAL_MASK = np.concatenate((
    [True] * (1 + 2) + [False],
    np.zeros_like(HML_ROOT_BINARY[1:].repeat(3)),
    np.zeros_like(HML_ROOT_BINARY[1:].repeat(6)),
    np.zeros_like(HML_ROOT_BINARY.repeat(3)),
    [False] * 4,
))
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate((
    [True] * (1 + 2 + 1),
    HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
    HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
    HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
    [True] * 4,
))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK
