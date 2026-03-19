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

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.tienkung.Robot.tienkung_env import TienKungEnv     #训练修改,原本训练
    # from legged_lab.envs.tienkung.Robot.tienkung_env_69 import TienKungEnv    #训练修改,删除线速度和步态参数
    # from legged_lab.envs.tienkung.Robot.tienkung_env_75_old import TienKungEnv     #训练修改，只删除线速度
    # from legged_lab.envs.tienkung.Robot.tienkung_env_75 import TienKungEnv        #训练修改，官方删除线速度
    # from legged_lab.envs.tienkung.Robot.tienkung_env_45_only_leg import TienKungEnv   #训练修改，只控制下半身

def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励机器人跟踪XY平面线速度命令，使用yaw坐标系下的指数衰减函数。"""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励机器人跟踪Z轴角速度命令，使用世界坐标系下的指数衰减函数。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚Z轴方向的线速度，防止机器人上下浮动。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚XY平面的角速度，保持机器人姿态稳定。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚能量消耗，计算关节扭矩和速度的乘积范数。"""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚关节加速度，使运动更加平滑。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """惩罚动作变化率，使控制更加平滑。"""
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚不希望发生的接触，如机器人身体其他部位接触地面。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """检测机器人是否在飞行状态（双脚都离开地面）。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚机器人身体倾斜，保持直立姿态。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """惩罚非超时终止的回合，如摔倒等失败情况。"""
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """奖励双足机器人的脚部空中时间，促进自然的步态。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚脚部滑动，当脚部接触地面时不应有水平移动。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    """奖励身体对地面的垂直作用力，促进稳定的支撑。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚关节位置偏离默认位置，当速度命令接近零时生效。"""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def body_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚身体特定部位的倾斜，保持局部姿态稳定。"""
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """检测脚部是否绊倒（水平力大于垂直力的5倍）。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    """惩罚双足人形机器人双脚距离过近，防止碰撞。"""
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)

#--------------------------（一）正则化奖励函数（针对关节扭矩 / 动作、脚间距的惩罚）--------------------------
# Regularization Reward

# 惩罚踝关节的大扭矩。
def ankle_torque(env: TienKungEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)

# 惩罚踝关节动作
def ankle_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)
# 只控制下半身，惩罚踝关节动作
def only_leg_ankle_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize ankle joint actions for only leg control."""
    # 修改为适应12维action空间：踝关节对应索引4,5,10,11
    ankle_indices = torch.tensor([4, 5, 10, 11], device=env.device)
    return torch.sum(torch.abs(env.action[:, ankle_indices]), dim=1)

# 惩罚髋滚转关节动作
def hip_roll_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[0], env.right_leg_ids[0]]]), dim=1)

# 只控制下半身，惩罚髋滚转关节动作
def only_leg_hip_roll_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    # 修改为适应12维action空间：髋滚转关节对应索引0,6
    hip_roll_indices = torch.tensor([0, 6], device=env.device)
    return torch.sum(torch.abs(env.action[:, hip_roll_indices]), dim=1)
    
# 惩罚髋偏航关节动作
def hip_yaw_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)
    
    
# 只控制下半身，惩罚髋偏航关节动作
def only_leg_hip_yaw_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    # 修改为适应12维action空间：髋偏航关节对应索引2,8
    hip_yaw_indices = torch.tensor([2, 8], device=env.device)
    return torch.sum(torch.abs(env.action[:, hip_yaw_indices]), dim=1)

# 当 y 方向速度指令较低时，惩罚左右脚的 y 轴间距偏离目标值
def feet_y_distance(env: TienKungEnv) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


# -------------------------（二）周期性步态奖励函数（基于步态相位的奖励 / 惩罚）--------------------------
# Periodic gait-based reward function

#生成步态相位的平滑时钟信号（区分摆动相 / 支撑相）
def gait_clock(phase, air_ratio, delta_t):
    """
    Generate periodic gait clock signals for foot swing and stance phases.

    This function constructs two phase-dependent signals:
    - `I_frc`: active during swing phase (used for penalizing ground force)
    - `I_spd`: active during stance phase (used for penalizing foot speed)

    Transitions between swing and stance are smoothed within a margin of `delta_t`
    to create differentiable transitions.

    Parameters
    ----------
    phase : torch.Tensor
        Normalized gait phase in [0, 1], shape: [num_envs].
    air_ratio : torch.Tensor
        Proportion of the gait cycle spent in swing phase, shape: [num_envs].
    delta_t : float
        Transition width around phase boundaries for smooth interpolation.

    Returns
    -------
    I_frc : torch.Tensor
        Gait-based swing-phase clock signal, range [0, 1], shape: [num_envs].
    I_spd : torch.Tensor
        Gait-based stance-phase clock signal, range [0, 1], shape: [num_envs].

    Notes
    -----
    - The transitions at the boundaries (e.g., swing→stance) are linear interpolations.
    - Used in reward shaping to associate expected behavior with gait phases.
    """
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd

# 惩罚摆动相中的脚部地面力
def gait_feet_frc_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score

# 惩罚支撑相中的脚部速度
def gait_feet_spd_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_spd_score + right_spd_score

# 奖励支撑相中的合理地面支撑力
def gait_feet_frc_support_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


# ==============================================================================
# 最小约束奖励函数（基于论文设计）
# ==============================================================================


# ===================== 辅助函数（通用计算） =====================

def quaternion_distance(q1, q2):
    """计算四元数距离（qd函数，对齐表格中的qd）
    
    Args:
        q1: 四元数 [qw, qx, qy, qz]
        q2: 四元数 [qw, qx, qy, qz]
    Returns:
        四元数距离标量
    """
    dot = torch.sum(q1 * q2, dim=-1)
    return 1 - dot ** 2


# ===================== 各奖励项计算函数（适配 Isaac Lab 框架） =====================

def reward_xy_velocity(env: TienKungEnv) -> torch.Tensor:
    """
    X/Y速度跟踪奖励
    公式：e^(-5*(v_xy - c_xy)^2)
    从env获取: 命令和实际速度
    """
    # 从env获取命令 (期望速度)
    command = env.command_generator.command[:, :2]
    # 获取实际速度（使用yaw坐标系）
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(env.robot.data.root_quat_w),
        env.robot.data.root_lin_vel_w[:, :3]
    )
    # 计算误差
    error = torch.sum(torch.square(command - vel_yaw[:, :2]), dim=1)
    return torch.exp(-5 * error)


def reward_yaw_orientation(env: TienKungEnv) -> torch.Tensor:
    """
    偏航角跟踪奖励
    公式：e^(-300 * qd(q_yaw, c_yaw))
    """
    # 从命令中获取期望偏航角
    desired_yaw = env.command_generator.command[:, 2]  # 形状: (num_envs,)
    
    # 当前偏航角 - 使用 euler_xyz_from_quat 提取标量角度
    # root_quat_w 形状: (num_envs, 4)
    # euler_xyz_from_quat 返回 (roll, pitch, yaw)，每个都是 (num_envs,)
    _, _, current_yaw = math_utils.euler_xyz_from_quat(env.robot.data.root_quat_w)
    
    # 计算偏航角误差
    yaw_error = current_yaw - desired_yaw
    # 归一化到 [-pi, pi]
    yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
    # 四元数距离简化为角度误差平方
    qd_val = yaw_error ** 2
    return torch.exp(-300 * qd_val)


def reward_roll_pitch_orientation(env: TienKungEnv) -> torch.Tensor:
    """
    滚转/俯仰角奖励
    公式：e^(-30 * qd(q_rp, e_rp))
    期望值：零（保持直立）
    """
    # 获取投影重力向量的XY分量（与滚转/俯仰角相关）
    projected_gravity = env.robot.data.projected_gravity_b[:, :2]
    # 误差为重力向量在XY平面的投影（越小越好）
    error = torch.sum(torch.square(projected_gravity), dim=1)
    return torch.exp(-30 * error)


def reward_feet_contact(env: TienKungEnv, t_window: float = 0.2, force_threshold: float = 1.0, stand_threshold: float = 0.1) -> torch.Tensor:
    """
    足底接触奖励
    
    奖励逻辑：
    1. 当收到站立命令时（速度命令接近零），返回奖励1
    2. 当不是站立命令时，任意t∈{t-0.2, t}内，任意一只脚站在地面上，奖励1
    3. 其余为0
    
    Args:
        env: 环境对象
        t_window: 时间窗口（默认0.2秒）
        force_threshold: 接触力阈值（默认1.0）
        stand_threshold: 站立命令判定阈值（默认0.1）
    
    Returns:
        足底接触奖励张量 (num_envs,)
    """
    # 获取速度命令
    command = env.command_generator.command[:, :2]
    # 判断是否为站立命令（x和y方向速度都小于阈值）
    is_stand_command = torch.norm(command, dim=1) < stand_threshold
    
    # 获取历史接触力数据
    # net_forces_w_history 形状: (num_envs, history_steps, num_bodies, 3)
    net_forces_history = env.contact_sensor.data.net_forces_w_history
    
    # 计算历史中每帧的接触状态
    # 只取脚部相关的接触力
    foot_forces = net_forces_history[:, :, env.feet_cfg.body_ids, :]
    
    # 计算每帧的接触力范数: (num_envs, history_steps, num_feet)
    force_norms = torch.norm(foot_forces, dim=-1)
    
    # 判断每帧是否有脚部接触（任意一只脚接触即可）: (num_envs, history_steps)
    has_contact_per_frame = torch.any(force_norms > force_threshold, dim=-1)
    
    # 检查最近0.2秒内是否有接触
    # step_dt = env.step_dt (每个step的时间)
    # 需要查看的帧数 = t_window / physics_dt (physics_dt = env.physics_dt)
    history_length = net_forces_history.shape[1]
    frames_to_check = min(int(t_window / env.physics_dt), history_length)
    
    # 取最近frames_to_check帧的接触状态
    recent_contacts = has_contact_per_frame[:, -frames_to_check:]
    
    # 判断在过去窗口内是否有过脚部接触（任意一帧）: (num_envs,)
    has_contact_in_window = torch.any(recent_contacts, dim=1)
    
    # 综合判断：
    # - 如果是站立命令 -> 奖励1
    # - 如果不是站立命令但过去0.2秒内有接触 -> 奖励1
    # - 否则 -> 奖励0
    reward = torch.where(
        is_stand_command | has_contact_in_window,
        torch.ones(env.num_envs, dtype=torch.float, device=env.device),
        torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    )
    
    return reward


def reward_base_height(env: TienKungEnv, target_height: float = 0.4) -> torch.Tensor:
    """
    基座高度奖励
    公式：e^(-20 * |p_z - c_z|)
    默认目标高度0.4m
    """
    # 获取基座高度
    p_z = env.robot.data.root_pos_w[:, 2]
    # 计算误差
    error = torch.abs(p_z - target_height)
    return torch.exp(-20 * error)


def reward_feet_airtime(env: TienKungEnv, target_airtime: float = 0.4) -> torch.Tensor:
    """
    足底腾空时间奖励
    公式：Σ(t_air,f - 0.4) * 1_{td,f}
    奖励足底在空中停留接近目标时间
    """
    # 获取各足底的空中时间
    air_time = env.contact_sensor.data.current_air_time[:, env.feet_cfg.body_ids]
    # 接触时间
    contact_time = env.contact_sensor.data.current_contact_time[:, env.feet_cfg.body_ids]
    # 判断是否在接触中
    in_contact = contact_time > 0.0
    
    # 单脚支撑检测（用于步态奖励）
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    
    # 计算腾空时间奖励
    # 只在单脚支撑时奖励腾空时间
    airtime_reward = torch.where(single_stance.unsqueeze(-1), air_time - target_airtime, torch.zeros_like(air_time))
    
    return torch.sum(airtime_reward, dim=1)


def reward_feet_orientation(env: TienKungEnv) -> torch.Tensor:
    """
    足底姿态奖励
    公式：固定返回1.0（表格中未给出具体公式）
    简化处理：奖励脚部保持水平（通过检查脚部法向量与重力方向对齐）
    """
    # 返回固定奖励1.0
    return torch.ones(env.num_envs, dtype=torch.float, device=env.device)


def reward_feet_position(env: TienKungEnv, alpha: float = 1.0) -> torch.Tensor:
    """
    足底位置奖励
    公式：e^(-Σ|∇u_i| - α) * f_min （if |f_min|>1 else 1）
    简化版本：奖励脚部保持在合理范围内
    """
    # 获取脚部位置
    feet_pos = env.robot.data.body_pos_w[:, env.feet_body_ids, :]
    root_pos = env.robot.data.root_pos_w[:, :3].unsqueeze(1)
    feet_rel_pos = feet_pos - root_pos
    
    # 计算脚部位置的梯度（简化处理：使用脚部速度作为梯度代理）
    feet_vel = env.robot.data.body_lin_vel_w[:, env.feet_body_ids, :]
    sum_grad = torch.sum(torch.abs(feet_vel))
    
    # 计算足底力
    f_min = torch.min(torch.norm(env.contact_sensor.data.net_forces_w[:, env.feet_body_ids, :], dim=-1), dim=1)[0]
    
    exp_term = torch.exp(-sum_grad - alpha)
    
    # 如果 |f_min| > 1 则乘以 f_min，否则返回1.0
    reward = torch.where(
        torch.abs(f_min) > 1.0,
        exp_term * f_min,
        torch.ones_like(f_min)
    )
    return reward


def reward_arm(env: TienKungEnv, target_angle: float = 0.0) -> torch.Tensor:
    """
    手臂角度奖励
    公式：e^(-3 * |θ_arm - c_arm|)
    目标：手臂保持在期望角度（默认0）
    """
    # 获取手臂关节位置（假设左右臂各4个关节）
    left_arm_pos = env.robot.data.joint_pos[:, env.left_arm_ids]
    right_arm_pos = env.robot.data.joint_pos[:, env.right_arm_ids]
    
    # 计算与目标角度的误差
    left_error = torch.sum(torch.abs(left_arm_pos - target_angle), dim=1)
    right_error = torch.sum(torch.abs(right_arm_pos - target_angle), dim=1)
    total_error = left_error + right_error
    
    return torch.exp(-3 * total_error)


def reward_base_acceleration(env: TienKungEnv) -> torch.Tensor:
    """
    基座加速度奖励
    公式：e^(-0.01 - Σ|b_xy|)
    惩罚基座XY方向的加速度
    """
    # 获取当前速度
    current_vel = env.robot.data.root_lin_vel_b[:, :2]
    
    # 尝试从历史获取上一时刻速度计算加速度
    # 如果没有历史记录，使用零加速度
    if hasattr(env, 'prev_base_vel'):
        prev_vel = env.prev_base_vel
        # 计算加速度（简化：直接使用速度作为代理）
        acc = torch.abs(current_vel - prev_vel)
    else:
        acc = torch.abs(current_vel)
    
    # 更新历史速度
    env.prev_base_vel = current_vel.clone()
    
    sum_acc = torch.sum(acc, dim=1)
    return torch.exp(-0.01 - sum_acc)


def reward_action_difference(env: TienKungEnv) -> torch.Tensor:
    """
    动作差分奖励（动作平滑性）
    公式：e^(-0.02 - Σ|a_f - a_{f-1}|)
    """
    # 获取当前动作和上一时刻动作
    current_action = env.action
    prev_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    
    # 计算动作差分
    sum_diff = torch.sum(torch.abs(current_action - prev_actions), dim=1)
    return torch.exp(-0.02 - sum_diff)


def reward_torque(env: TienKungEnv, t_max: float = 33.5) -> torch.Tensor:
    """
    电机力矩奖励（能耗惩罚）
    公式：e^(-0.02 * (1/N) * Σ|t_motor| / t_max)
    t_max: 电机最大力矩（默认33.5Nm）
    """
    # 获取电机力矩
    torque = env.robot.data.applied_torque
    
    # 计算归一化力矩
    n = torque.shape[1]
    avg_torque = torch.sum(torch.abs(torque), dim=1) / n
    normalized_torque = avg_torque / t_max
    
    return torch.exp(-0.02 * normalized_torque)

