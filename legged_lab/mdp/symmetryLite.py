# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
import torch


# Examples of data augmentation for symmetry · Issue #64 · leggedrobotics/rsl_rl g1
# joint order ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 
# 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
#  'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
#  'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
#  'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint']

# Dex
# self.robot.joint_names: ['hip_pitch_l_joint', 'hip_pitch_r_joint', 'waist_yaw_joint', 'hip_roll_l_joint', 'hip_roll_r_joint', 
#  'waist_roll_joint', 'hip_yaw_l_joint', 'hip_yaw_r_joint', 'waist_pitch_joint', 'knee_pitch_l_joint', 'knee_pitch_r_joint', 'shoulder_pitch_l_joint', 'shoulder_pitch_r_joint', 
#  'ankle_pitch_l_joint', 'ankle_pitch_r_joint', 'shoulder_roll_l_joint', 'shoulder_roll_r_joint', 'ankle_roll_l_joint', 'ankle_roll_r_joint', 'shoulder_yaw_l_joint', 'shoulder_yaw_r_joint', 
#  'elbow_pitch_l_joint', 'elbow_pitch_r_joint']

# self.robot.joint_names: ['hip_roll_l_joint', 'hip_roll_r_joint', 'waist_yaw_joint', 'hip_pitch_l_joint', 'hip_pitch_r_joint', 
#  'shoulder_pitch_l_joint', 'shoulder_pitch_r_joint', 'hip_yaw_l_joint', 'hip_yaw_r_joint', 'shoulder_roll_l_joint', 'shoulder_roll_r_joint',
#  'knee_pitch_l_joint', 'knee_pitch_r_joint', 'shoulder_yaw_l_joint', 'shoulder_yaw_r_joint', 'ankle_pitch_l_joint', 'ankle_pitch_r_joint',  'elbow_pitch_l_joint', 'elbow_pitch_r_joint', 'ankle_roll_l_joint', 'ankle_roll_r_joint', 
#  ]
# # tiangong2_lite
# ['hip_roll_l_joint', 'hip_roll_r_joint', 'shoulder_pitch_l_joint', 'shoulder_pitch_r_joint', 'hip_pitch_l_joint', 'hip_pitch_r_joint', 
#  'shoulder_roll_l_joint', 'shoulder_roll_r_joint', 'hip_yaw_l_joint', 'hip_yaw_r_joint', 'shoulder_yaw_l_joint', 'shoulder_yaw_r_joint',
#  'knee_pitch_l_joint', 'knee_pitch_r_joint', 'elbow_pitch_l_joint', 'elbow_pitch_r_joint', 'ankle_pitch_l_joint', 'ankle_pitch_r_joint', 'ankle_roll_l_joint', 'ankle_roll_r_joint']

################################
#for tg2.0lite
################################


# 预计算镜像索引 - 模块级缓存，避免重复计算
_MIRROR_INDICES_CACHE = {}
ACTION_NUM = 20

def _get_mirror_indices(offset=0):
    """获取或创建镜像索引的缓存版本"""
    if offset not in _MIRROR_INDICES_CACHE:
        # 创建置换索引数组：对于23个关节，哪个位置应该从哪个位置读取
        perm = list(range(ACTION_NUM))
        # 交换对
        swap_pairs = [
            (0, 1), # hip_roll
            (4, 5), # hip_pitch
            (2, 3), # shoulder_pitch
            (8, 9), # hip_yaw
            (6, 7), # shoulder_roll
            (12, 13), # knee_pitch
            (10, 11), # shoulder_yaw
            (16, 17), # ankle_pitch
            (14, 15), # elbow_pitch
            (18, 19)] # ankle_roll
        for left, right in swap_pairs:
            perm[left], perm[right] = perm[right], perm[left]

        # 需要取反的索引
        negate_mask = [False] * ACTION_NUM
        negate_indices = [0, # left_hip_roll
                          1, # right_hip_roll
                          # 2, # waist_yaw
                          8, # left_hip_yaw
                          9, # right_hip_yaw
                          6, # left_shoulder_roll
                          7, # right_shoulder_roll
                          10, # left_shoulder_yaw
                          11, # right_shoulder_yaw
                          18, # left_ankle_roll
                          19 # right_ankle_roll
                        ]
        for idx in negate_indices:
            negate_mask[idx] = True

        _MIRROR_INDICES_CACHE[offset] = {
            'perm': torch.tensor(perm, dtype=torch.long),
            'negate': torch.tensor(negate_mask, dtype=torch.bool)
        }

    return _MIRROR_INDICES_CACHE[offset]


def mirror_joint_tensor(original: torch.Tensor, mirrored: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """快速镜像关节张量 - 使用预计算索引"""
    indices = _get_mirror_indices(offset)
    perm = indices['perm'].to(original.device)
    negate = indices['negate'].to(original.device)

    # 一次性置换所有关节
    joint_slice = slice(offset, offset + ACTION_NUM)
    mirrored[..., joint_slice] = original[..., offset + perm]
    # 一次性取反需要的关节
    mirrored[..., offset:offset + ACTION_NUM][..., negate] *= -1

def mirror_observation_policy(obs):
    """
    obs: (..., H * D)  actor history, H=10 frames, D=75 (or 76 for the carry task,
          which appends a scalar payload_mass channel).
    return: (..., 2 * H * D)  original obs concat with its mirror.
    """
    if obs is None:
        return obs

    *batch_shape, _ = obs.shape
    batch_size = obs.shape[0] if batch_shape else 1

    # Infer history length + per-frame dim dynamically so the same function
    # works for the walk task (D=75) and the carry task (D=76). The walk
    # task uses history=10; the carry task uses the same history length.
    total_dim = obs.shape[-1]
    if total_dim % 10 == 0:
        history_len = 10
        per_frame = total_dim // history_len
    else:
        # Fallback: single-frame obs.
        history_len = 1
        per_frame = total_dim

    # Pre-allocate output (avoid vstack).
    result = torch.empty(
        batch_size * 2, history_len * per_frame, device=obs.device, dtype=obs.dtype
    )
    result[:batch_size] = obs

    # Reshape for vectorized per-frame ops.
    obs_2d = obs.view(batch_size, history_len, per_frame)
    flipped_2d = obs_2d.clone()

    # base ang vel x,z
    flipped_2d[..., 0] = -obs_2d[..., 0]
    flipped_2d[..., 2] = -obs_2d[..., 2]
    # projected gravity y
    flipped_2d[..., 4] = -obs_2d[..., 4]
    # velocity commands y/z
    flipped_2d[..., 7] = -obs_2d[..., 7]
    flipped_2d[..., 8] = -obs_2d[..., 8]

    # Joint mirroring - flatten for batched processing.
    flipped_flat = flipped_2d.view(batch_size * history_len, per_frame)
    obs_flat = obs_2d.view(batch_size * history_len, per_frame)
    mirror_joint_tensor(obs_flat, flipped_flat, 9)
    mirror_joint_tensor(obs_flat, flipped_flat, 9 + ACTION_NUM)
    mirror_joint_tensor(obs_flat, flipped_flat, 9 + 2 * ACTION_NUM)

    # gait_clock and other swaps
    flipped_2d[..., 69], flipped_2d[..., 70] = obs_2d[..., 70].clone(), obs_2d[..., 69].clone()
    flipped_2d[..., 71], flipped_2d[..., 72] = obs_2d[..., 72].clone(), obs_2d[..., 71].clone()
    flipped_2d[..., 73], flipped_2d[..., 74] = obs_2d[..., 74].clone(), obs_2d[..., 73].clone()
    # Any extra channels beyond index 75 (e.g., the carry task's payload_mass
    # at index 75) are scalars that the mirror leaves untouched - they were
    # already copied by `flipped_2d = obs_2d.clone()` at the top.
    flipped_2d = flipped_flat.view(batch_size, history_len, per_frame)

    result[batch_size:] = flipped_2d.view(batch_size, history_len * per_frame)
    return result

def mirror_observation_critic(obs):
    """
    obs: (..., H * D)  critic history. Default per-frame dim D=80 (75 actor +
    3 root_lin_vel + 2 feet_contact). The carry task appends a scalar
    payload_mass channel at the end, so D=81.
    return: (..., 2 * H * D)  original obs concat with its mirror.
    """
    if obs is None:
        return obs

    *batch_shape, _ = obs.shape
    batch_size = obs.shape[0] if batch_shape else 1

    # Default per-frame dim for the base walk/run tasks is 80; the carry task
    # appends a scalar (81). We assume the obs layout is `actor_per_frame (75)
    # + root_lin_vel (3) + feet_contact (2) + [optional extras]` and detect the
    # base_dim from the total length and history_len (capped to 10 frames).
    total_dim = obs.shape[-1]
    base_per_frame = 9 + 3 * ACTION_NUM + 6 + 5  # 80
    if total_dim % 10 == 0 and total_dim // 10 >= base_per_frame:
        history_len = 10
        per_frame_dim = total_dim // history_len
    elif total_dim % base_per_frame == 0:
        history_len = total_dim // base_per_frame
        per_frame_dim = base_per_frame
    else:
        history_len = 1
        per_frame_dim = total_dim

    # 预分配输出
    result = torch.empty(batch_size * 2, history_len * per_frame_dim, device=obs.device, dtype=obs.dtype)
    result[:batch_size] = obs

    obs_3d = obs.view(batch_size, history_len, per_frame_dim)
    flipped_3d = obs_3d.clone()

    root_offset = 9+3*ACTION_NUM + 6
    feet_offset = root_offset + 3

    # 向量化镜像所有帧
    flipped_3d[..., 0] = -obs_3d[..., 0]   # base ang vel x
    flipped_3d[..., 2] = -obs_3d[..., 2]   # base ang vel z
    flipped_3d[..., 4] = -obs_3d[..., 4]   # projected gravity y
    flipped_3d[..., 7] = -obs_3d[..., 7]   # command y
    flipped_3d[..., 8] = -obs_3d[..., 8]   # command z

    # 关节镜像 - 展平批量处理
    flipped_flat = flipped_3d.view(batch_size * history_len, per_frame_dim)
    obs_flat = obs_3d.view(batch_size * history_len, per_frame_dim)
    mirror_joint_tensor(obs_flat, flipped_flat, 9)
    mirror_joint_tensor(obs_flat, flipped_flat, 9+ACTION_NUM)
    mirror_joint_tensor(obs_flat, flipped_flat, 9+2*ACTION_NUM)
    flipped_3d = flipped_flat.view(batch_size, history_len, per_frame_dim)

    # gait 交换
    flipped_3d[..., 69], flipped_3d[..., 70] = obs_3d[..., 70].clone(), obs_3d[..., 69].clone()
    flipped_3d[..., 71], flipped_3d[..., 72] = obs_3d[..., 72].clone(), obs_3d[..., 71].clone()
    flipped_3d[..., 73], flipped_3d[..., 74] = obs_3d[..., 74].clone(), obs_3d[..., 73].clone()

    # root_lin_vel：y 轴取反
    flipped_3d[..., root_offset + 1] = -obs_3d[..., root_offset + 1]

    # feet_contact：左右互换
    flipped_3d[..., feet_offset], flipped_3d[..., feet_offset + 1] = \
        obs_3d[..., feet_offset + 1].clone(), obs_3d[..., feet_offset].clone()

    result[batch_size:] = flipped_3d.view(batch_size, history_len * per_frame_dim)
    return result


def mirror_actions(actions):
    if actions is None:
        return None

    batch_size = actions.shape[0]
    # 预分配输出，避免 vstack
    result = torch.empty(batch_size * 2, actions.shape[-1], device=actions.device, dtype=actions.dtype)
    result[:batch_size] = actions

    # 直接在预分配空间中镜像
    mirror_joint_tensor(actions, result[batch_size:], offset=0)
    return result


def data_augmentation_func_g1(env, obs, actions, obs_type):


    if obs_type == "policy":
        obs_batch = mirror_observation_policy(obs)
    elif obs_type == "critic":
        obs_batch = mirror_observation_critic(obs)
    else:
        raise ValueError(f"Invalid observation type: {obs_type}")

    mean_actions_batch = mirror_actions(actions)
    return obs_batch, mean_actions_batch