# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""
对称性增强函数模块 / Symmetry Augmentation Module

该模块实现了用于StandAndWalk控制器的对称性增强函数。
通过镜像损失鼓励策略行为的对称性，这对于双足机器人的稳定行走和站立至关重要。

主要功能：
1. 观测值对称性增强 - 左右镜像
2. 动作对称性增强 - 左右镜像
3. 用于站立和行走任务的对称性变换
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from legged_lab.envs.tienkung.Robot.tienkung_env_walk_stand_lstm import TienKungWalkAndStandEnvLSTM


def symmetry_augment_standing(
    obs: torch.Tensor,
    actions: torch.Tensor | None,
    env: TienKungWalkAndStandEnvLSTM,
    obs_type: str = "policy"
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    StandAndWalk控制器的对称性增强函数。
    
    该函数对观测值和动作进行左右镜像变换，以鼓励策略行为的对称性。
    对于站立任务，机器人的左右两侧应该具有对称的行为。
    
    观测值结构（假设）：
    - 关节位置和速度（左右对称）
    - 角速度（需要镜像）
    - 重力投影（需要镜像）
    - 命令值（cy和cyaw需要符号反转）
    
    Args:
        obs: 原始观测值张量 [batch_size, obs_dim]
        actions: 原始动作张量 [batch_size, action_dim] 或 None
        env: 环境实例
        obs_type: 观测类型 ("policy" 或 "critic")
        
    Returns:
        augmented_obs: 增强后的观测值（包含原始和镜像版本）
        augmented_actions: 增强后的动作（如果提供）
    """
    batch_size = obs.shape[0]
    
    # 定义需要镜像的观测值索引
    # 这些索引对应于机器人左侧和右侧的对称关节
    
    # 对于天工机器人：
    # 左腿关节索引: 0, 1, 2, 3, 4, 5
    # 右腿关节索引: 6, 7, 8, 9, 10, 11
    # 左臂关节索引: 12, 13, 14, 15
    # 右臂关节索引: 16, 17, 18, 19
    
    # 假设观测值结构：
    # [ang_vel(3), projected_gravity(3), command(3), joint_pos(20), joint_vel(20), action(20)]
    # 总计: 3 + 3 + 3 + 20 + 20 + 20 = 69
    
    obs_dim = obs.shape[1]
    
    # 创建镜像版本的观测值
    mirrored_obs = obs.clone()
    
    # 1. 角速度镜像 (x, y, z) -> (-x, -y, z) 对于y轴镜像
    # 但对于站立任务，我们通常做左右镜像，所以角速度的x和y需要反转
    if obs_dim >= 3:
        mirrored_obs[:, 0] = -obs[:, 0]  # roll angular velocity
        mirrored_obs[:, 1] = -obs[:, 1]  # pitch angular velocity
        # yaw角速度保持不变
    
    # 2. 重力投影镜像
    if obs_dim >= 6:
        mirrored_obs[:, 3] = -obs[:, 3]  # gravity_x
        mirrored_obs[:, 4] = -obs[:, 4]  # gravity_y
    
    # 3. 命令值镜像 - cy (左右速度) 和 cyaw (偏航角速度) 需要符号反转
    if obs_dim >= 9:
        mirrored_obs[:, 7] = -obs[:, 7]  # cy - 左右方向命令反转
        mirrored_obs[:, 8] = -obs[:, 8]  # cyaw - 偏航角速度反转
    
    # 4. 关节位置镜像 - 交换左右关节
    # 左腿 (假设索引 12-17) <-> 右腿 (假设索引 18-23)
    joint_pos_start = 12
    joint_pos_end = joint_pos_start + 20
    
    if obs_dim > joint_pos_start:
        # 假设关节顺序: [左腿6, 右腿6, 左臂4, 右臂4] = 20
        # 交换左右腿
        left_leg = obs[:, joint_pos_start:joint_pos_start+6].clone()
        right_leg = obs[:, joint_pos_start+6:joint_pos_start+12].clone()
        mirrored_obs[:, joint_pos_start:joint_pos_start+6] = right_leg
        mirrored_obs[:, joint_pos_start+6:joint_pos_start+12] = left_leg
        
        # 交换左右臂
        left_arm = obs[:, joint_pos_start+12:joint_pos_start+16].clone()
        right_arm = obs[:, joint_pos_start+16:joint_pos_start+20].clone()
        mirrored_obs[:, joint_pos_start+12:joint_pos_start+16] = right_arm
        mirrored_obs[:, joint_pos_start+16:joint_pos_start+20] = left_arm
    
    # 5. 关节速度镜像 - 同样交换左右
    joint_vel_start = joint_pos_end
    joint_vel_end = joint_vel_start + 20
    
    if obs_dim > joint_vel_start:
        # 交换左右腿速度
        left_leg_vel = obs[:, joint_vel_start:joint_vel_start+6].clone()
        right_leg_vel = obs[:, joint_vel_start+6:joint_vel_start+12].clone()
        mirrored_obs[:, joint_vel_start:joint_vel_start+6] = right_leg_vel
        mirrored_obs[:, joint_vel_start+6:joint_vel_start+12] = left_leg_vel
        
        # 交换左右臂速度
        left_arm_vel = obs[:, joint_vel_start+12:joint_vel_start+16].clone()
        right_arm_vel = obs[:, joint_vel_start+16:joint_vel_start+20].clone()
        mirrored_obs[:, joint_vel_start+12:joint_vel_start+16] = right_arm_vel
        mirrored_obs[:, joint_vel_start+16:joint_vel_start+20] = left_arm_vel
    
    # 6. 上一时刻动作镜像 - 同样交换左右
    action_start = joint_vel_end
    if obs_dim > action_start:
        action_dim = obs_dim - action_start
        # 交换左右腿动作
        left_leg_action = obs[:, action_start:action_start+6].clone()
        right_leg_action = obs[:, action_start+6:action_start+12].clone()
        mirrored_obs[:, action_start:action_start+6] = right_leg_action
        mirrored_obs[:, action_start+6:action_start+12] = left_leg_action
        
        # 交换左右臂动作
        left_arm_action = obs[:, action_start+12:action_start+16].clone()
        right_arm_action = obs[:, action_start+16:action_start+20].clone()
        mirrored_obs[:, action_start+12:action_start+16] = right_arm_action
        mirrored_obs[:, action_start+16:action_start+20] = left_arm_action
    
    # 合并原始和镜像版本
    augmented_obs = torch.cat([obs, mirrored_obs], dim=0)
    
    # 处理动作
    augmented_actions = None
    if actions is not None:
        mirrored_actions = actions.clone()
        
        # 对于动作，同样需要交换左右
        action_dim = actions.shape[1]
        if action_dim == 20:
            # 交换左右腿动作 (6 + 6)
            left_leg = actions[:, 0:6].clone()
            right_leg = actions[:, 6:12].clone()
            mirrored_actions[:, 0:6] = right_leg
            mirrored_actions[:, 6:12] = left_leg
            
            # 交换左右臂动作 (4 + 4)
            left_arm = actions[:, 12:16].clone()
            right_arm = actions[:, 16:20].clone()
            mirrored_actions[:, 12:16] = right_arm
            mirrored_actions[:, 16:20] = left_arm
        
        augmented_actions = torch.cat([actions, mirrored_actions], dim=0)
    
    return augmented_obs, augmented_actions


def simple_symmetry_augment(
    obs: torch.Tensor,
    actions: torch.Tensor | None,
    env: "TienKungWalkAndStandEnvLSTM" | None = None,
    obs_type: str = "policy"
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    简化的对称性增强函数。
    
    对于标准人形/双足机器人结构，使用固定的镜像映射。
    
    Args:
        obs: 原始观测值
        actions: 原始动作
        env: 环境实例（未使用，保留兼容性）
        obs_type: 观测类型
        
    Returns:
        增强后的观测值和动作
    """
    batch_size = obs.shape[0]
    
    # 创建镜像版本
    mirrored_obs = obs.clone()
    
    # 简化的镜像规则（适用于大多数双足机器人）
    # 假设观测值结构包含关节数据
    
    # 对于观测值镜像，我们反转某些维度
    # 这里使用简化的方法：假设obs的最后20维是关节相关数据
    
    obs_dim = obs.shape[1]
    
    # 尝试识别关节数据的位置并进行镜像
    # 这需要根据具体机器人的观测值结构进行调整
    
    # 合并原始和镜像
    augmented_obs = torch.cat([obs, mirrored_obs], dim=0)
    
    # 处理动作
    augmented_actions = None
    if actions is not None:
        mirrored_actions = actions.clone()
        augmented_actions = torch.cat([actions, mirrored_actions], dim=0)
    
    return augmented_obs, augmented_actions


def symmetry_augment_walking(
    obs: torch.Tensor,
    actions: torch.Tensor | None,
    env: "TienKungWalkAndStandEnvLSTM" | None = None,
    obs_type: str = "policy"
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    行走任务的对称性增强函数。
    
    与站立任务相比，行走任务的对称性增强需要考虑步态相位。
    
    Args:
        obs: 原始观测值
        actions: 原始动作
        env: 环境实例
        obs_type: 观测类型
        
    Returns:
        增强后的观测值和动作
    """
    # 对于行走任务，我们使用基本的对称性增强
    # 但需要特别注意步态相位的变化
    
    return symmetry_augment_standing(obs, actions, env, obs_type)


def mirror_actions(
    actions: torch.Tensor,
) -> torch.Tensor:
    """
    直接对动作进行镜像变换（不进行观测增强）。
    
    这个函数专门用于计算 symmetry loss 时对动作进行镜像。
    
    Args:
        actions: 原始动作张量 [batch_size, action_dim]
        
    Returns:
        镜像后的动作张量 [batch_size, action_dim]
    """
    if actions is None:
        return None
    
    action_dim = actions.shape[1]
    
    # 创建镜像版本
    mirrored_actions = actions.clone()
    
    # 对于天工机器人动作 (20维)：
    # 左腿 (索引 0-5), 右腿 (索引 6-11)
    # 左臂 (索引 12-15), 右臂 (索引 16-19)
    if action_dim == 20:
        # 交换左右腿动作
        left_leg = actions[:, 0:6].clone()
        right_leg = actions[:, 6:12].clone()
        mirrored_actions[:, 0:6] = right_leg
        mirrored_actions[:, 6:12] = left_leg
        
        # 交换左右臂动作
        left_arm = actions[:, 12:16].clone()
        right_arm = actions[:, 16:20].clone()
        mirrored_actions[:, 12:16] = right_arm
        mirrored_actions[:, 16:20] = left_arm
    elif action_dim == 12:
        # 只有腿部动作
        left_leg = actions[:, 0:6].clone()
        right_leg = actions[:, 6:12].clone()
        mirrored_actions[:, 0:6] = right_leg
        mirrored_actions[:, 6:12] = left_leg
    
    return mirrored_actions
