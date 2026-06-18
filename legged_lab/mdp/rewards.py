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
import numpy as np
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.dex.dex_env import DexEnv

def gaussian(x, value_at_1):
    scale = np.sqrt(-2 * np.log(value_at_1))
    return torch.exp(-0.5 * (x*scale)**2)

def tolerance(x, bounds=(0.0, 0.0), margin=0.1, value_at_margin=0.1):
    lower, upper = bounds 
    assert lower < upper
    assert margin >= 0

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, 1.0, gaussian(d.double(), value_at_margin))

    return value

def episode_progress_gate(env: BaseEnv | DexEnv, threshold: float = 700.0) -> torch.Tensor:
    """
    Returns a per-env gate that activates once the env's own episode length exceeds the threshold.

    P0.1 (carry-task fix): previously this was a global mean gate
    (``(episode_lengths.mean() >= threshold).float() * ones``) which forced the cube rewards
    to 0 for every env whenever the buffer-wide mean episode length was below the threshold.
    Under heavy early-training termination this created a chicken-and-egg deadlock:
    terminate early -> mean episode short -> gate never opens -> no cube gradient -> never
    learn to walk. The fix is to compare per-env episode_length_buf against the threshold
    so each env's cube reward activates independently as soon as *that env* has progressed.
    """
    episode_lengths = getattr(env, "episode_length_buf", None)
    if episode_lengths is None:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float)
    max_len = getattr(env, "max_episode_length", None)
    effective_threshold = threshold
    if max_len is not None:
        # If the desired threshold is not achievable, do not gate the rewards.
        if threshold >= float(max_len):
            return torch.ones_like(episode_lengths, dtype=torch.float)
        effective_threshold = min(threshold, float(max_len))

    return (episode_lengths.float() >= effective_threshold).float()

def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | DexEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | DexEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_z = torch.square(asset.data.root_lin_vel_b[:, 2])
    vel_z = torch.nan_to_num(vel_z, nan=0.0, posinf=0.0, neginf=0.0)
    return vel_z.clamp(max=25.0)


def ang_vel_xy_l2(env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_xy = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    ang_xy = torch.nan_to_num(ang_xy, nan=0.0, posinf=0.0, neginf=0.0)
    return ang_xy.clamp(max=400.0)

def body_ang_vel_xy_l2(env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # waist_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    waist_ang_vel = asset.data.body_ang_vel_w[:,asset_cfg.body_ids[0], :]
    root_quat_w = asset.data.root_quat_w[:,:]
    waist_angle_vel_b = math_utils.quat_apply(math_utils.quat_conjugate(root_quat_w),waist_ang_vel)
    return torch.sum(torch.square(waist_angle_vel_b[:, :2]), dim=1)


def body_ang_acc_xy_l2(env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    waist_ang_acc = asset.data.body_ang_acc_w[:, asset_cfg.body_ids[0], :]
    root_quat_w = asset.data.root_quat_w[:, :]
    waist_ang_acc_b = math_utils.quat_apply(math_utils.quat_conjugate(root_quat_w), waist_ang_acc)
    acc_xy = torch.sum(torch.square(waist_ang_acc_b[:, :2]), dim=1)
    acc_xy = torch.nan_to_num(acc_xy, nan=0.0, posinf=0.0, neginf=0.0)
    return acc_xy.clamp(max=1e6)


def energy(env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    acc = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
    acc = torch.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
    return acc.clamp(max=1e4)


def action_rate_l2(env: BaseEnv | DexEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    ).clamp(max=4.0)


def undesired_contacts(env: BaseEnv | DexEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | DexEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv | DexEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def alive_reward(env: BaseEnv | DexEnv) -> torch.Tensor:
    """Reward for staying alive and not terminating early."""
    return (~env.reset_buf).float()


def feet_air_time_positive_biped(
    env: BaseEnv | DexEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
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
    env: BaseEnv | DexEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | DexEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward = (reward - threshold).clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag

def waist_joint_deviation_l1(env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

def waist_joint_velocity_l1(env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(vel), dim=1)

def body_orientation_l2(
    env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv | DexEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


# Regularization Reward
def ankle_torque(env: DexEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: DexEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(env: DexEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    y_zero_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_zero_flag * torch.sum(torch.abs(env.action[:, [env.left_leg_ids[0], env.right_leg_ids[0]]]), dim=1)


def hip_roll_vel(env: DexEnv) -> torch.Tensor:
    """Penalize hip roll joint velocities when commanded y-velocity is near zero."""
    y_zero_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    hip_roll_joint_vel = env.robot.data.joint_vel[:, [env.left_leg_ids[0], env.right_leg_ids[0]]]
    return y_zero_flag * torch.sum(torch.abs(hip_roll_joint_vel), dim=1)


def hip_yaw_action(env: DexEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def feet_y_distance(env: DexEnv) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


# ========================================
# New gait-free reward functions for running
# ========================================

def feet_contact_alternation(env: BaseEnv | DexEnv, sensor_cfg: SceneEntityCfg, threshold: float = 10.0) -> torch.Tensor:
    """Reward alternating foot contacts (one foot in air while other is grounded).
    
    Encourages natural bipedal gait without explicit phase tracking.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    left_contact = contact_forces[:, 0] > threshold
    right_contact = contact_forces[:, 1] > threshold
    
    # Reward when exactly one foot is in contact (XOR)
    single_stance = (left_contact ^ right_contact).float()
    
    # Only apply when moving
    moving_flag = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:,2]) > 0.2
    return single_stance * moving_flag


def feet_air_time_reward(env: BaseEnv | DexEnv, sensor_cfg: SceneEntityCfg, target_time: float = 0.3) -> torch.Tensor:
    """Reward appropriate air time for each foot.
    
    Encourages dynamic running motion without phase tracking.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # [num_envs, 2]
    
    # Reward air time close to target (gaussian reward)
    air_time_error = torch.abs(air_time - target_time)
    reward = torch.exp(-10.0 * air_time_error).sum(dim=1)
    
    # Only apply when moving
    moving_flag = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:,2]) > 0.2
    return reward * moving_flag


def feet_contact_forces_balanced(env: BaseEnv | DexEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize unbalanced contact forces between feet.
    
    Prevents asymmetric gait patterns.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    
    # Penalize large force difference 
    left_force = contact_forces[:, 0]
    right_force = contact_forces[:, 1]
    
    # Only check when both feet have some contact
    both_contact = (left_force > 5.0) & (right_force > 5.0)
    force_diff = torch.abs(left_force - right_force) / (left_force + right_force + 1e-6)
    penalty = force_diff * both_contact.float()
    
    # Only apply when moving
    moving_flag = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:,2]) > 0.2
    return penalty * moving_flag


def forward_velocity_reward(env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward forward velocity progress beyond command tracking.
    
    Encourages dynamic forward motion during movinging.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_b = asset.data.root_lin_vel_b[:, :]
    
    # Reward forward velocity when command is forward
    forward_cmd = torch.clamp(env.command_generator.command[:, 0], min=0)
    forward_vel = torch.clamp(vel_b[:, 0], min=0)
    
    # Bonus for exceeding minimal velocity (encourages dynamic motion)
    velocity_bonus = torch.clamp(forward_vel - 0.3, min=0)
    reward = velocity_bonus * (forward_cmd > 0.2).float()
    
    return reward


def feet_clearance(env: BaseEnv | DexEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, min_height: float = 0.05) -> torch.Tensor:
    """Reward lifting feet off ground during swing (cm).
    
    Prevents dragging feet.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # [num_envs, 2, 3]
    root_height = asset.data.root_link_pos_w[:, 2:3]
    feet_height = feet_pos_w[:, :, 2] - root_height  # relative to root
    
    # Only check height when foot is in air
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_air = contact_forces < 5.0
    
    # Reward height above minimum when in air
    clearance = torch.clamp(feet_height - min_height, min=0)
    reward = (clearance * in_air.float()).sum(dim=1)
    
    # Only apply when moving
    moving_flag = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:,2])> 0.2
    return reward * moving_flag


def contact_no_slip(env: BaseEnv | DexEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize foot velocity when in contact (slipping).
    
    Similar to feet_slide but with better scaling.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_contact = contact_forces > 10.0
    
    feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]  # xy velocity
    feet_speed = feet_vel.norm(dim=-1)
    
    # Penalize speed when in contact
    slip_penalty = (feet_speed * in_contact.float()).sum(dim=1)
    
    return slip_penalty


def step_frequency_reward(env: BaseEnv | DexEnv, sensor_cfg: SceneEntityCfg, target_freq: float = 2.5) -> torch.Tensor:
    """Reward appropriate step frequency based on velocity.
    
    Higher velocity should result in appropriate stride frequency.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Use contact and air time to estimate cycle period
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    
    # Estimate cycle time (contact + air)
    cycle_time = contact_time + air_time
    cycle_time = torch.clamp(cycle_time.mean(dim=1), min=0.2, max=2.0)
    
    # Estimate frequency
    est_freq = 1.0 / (cycle_time + 1e-6)
    
    # Target frequency scales with velocity command
    cmd_vel = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:,2])
    target_freq_scaled = target_freq * torch.clamp(cmd_vel / 0.8, min=0.5, max=1.5)
    
    # Reward frequency close to target
    freq_error = torch.abs(est_freq - target_freq_scaled)
    reward = torch.exp(-freq_error)
    
    # Only apply when moving
    moving_flag = cmd_vel > 0.2
    return reward * moving_flag


# ========================================
# Legacy gait functions (deprecated)
# ========================================

def gait_clock(phase, air_ratio, delta_t):
    """DEPRECATED: Use new gait-free rewards instead."""
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


def gait_feet_frc_perio(env: DexEnv, delta_t: float = 0.02) -> torch.Tensor:
    """DEPRECATED: Use feet_contact_alternation instead."""
    non_zero_flag = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2]) > 0.1
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-100.0 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-100.0 * torch.square(env.avg_feet_force_per_step[:, 1])))
    reward = left_frc_score + right_frc_score
    return reward * non_zero_flag 

def gait_feet_spd_perio(env: DexEnv, delta_t: float = 0.02) -> torch.Tensor:
    """DEPRECATED: Use contact_no_slip instead."""
    non_zero_flag = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2]) > 0.1
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100.0 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100.0 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    reward = left_spd_score + right_spd_score
    return reward * non_zero_flag


def gait_feet_frc_support_perio(env: DexEnv, delta_t: float = 0.02) -> torch.Tensor:
    """DEPRECATED: Use feet_air_time_reward instead."""
    non_zero_flag = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2]) > 0.1
    try:
        left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
        right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
        left_frc_score = left_frc_support_mask * (1 - torch.exp(-100.0 * torch.square(env.avg_feet_force_per_step[:, 0])))
        right_frc_score = right_frc_support_mask * (1 - torch.exp(-100.0 * torch.square(env.avg_feet_force_per_step[:, 1])))
        reward = left_frc_score + right_frc_score
        return reward * non_zero_flag
    except AttributeError:
        # Fallback if gait_phase not available
        return torch.zeros(env.num_envs, device=env.device)




def stand_still(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    zero_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize joint deviation for a specified joint group when command is near zero.

    The optional ``zero_threshold`` provides a small deadband on command magnitude to reduce
    over-tightening posture when nearly stationary (helps keep arms from collapsing inward).
    """
    zero_flag = torch.norm(env.command_generator.command[:, :2], dim=1)+ torch.abs(env.command_generator.command[:, 2]) <= zero_threshold
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1) * zero_flag 

def stand_still_exp(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    zero_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize joint deviation for a specified joint group when command is near zero.

    The optional ``zero_threshold`` provides a small deadband on command magnitude to reduce
    over-tightening posture when nearly stationary (helps keep arms from collapsing inward).
    """
    zero_flag = torch.norm(env.command_generator.command[:, :2], dim=1)+ torch.abs(env.command_generator.command[:, 2]) <= zero_threshold
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-1.0*torch.sum(torch.abs(angle), dim=1)) * zero_flag 


def stand_still_vel_exp(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    zero_flag = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2]) <= 0.1
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, :]
    return zero_flag * torch.exp(-1.0 * torch.sum(torch.abs(vel), dim=1))


def stand_still_vel(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), zero_threshold: float = 0.1
) -> torch.Tensor:
    """Penalize arm joint velocities when command magnitude is near zero (standing still)."""
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
        <= zero_threshold
    )
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(vel), dim=1) * zero_flag


def stand_still_feet_motion_penalty(env: DexEnv) -> torch.Tensor:
    """Penalize foot motion when command is near zero to prevent treading in place."""
    zero_flag = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2]) <= 0.1
    feet_speed_penalty = torch.sum(env.avg_feet_speed_per_step, dim=1)
    return feet_speed_penalty * zero_flag

def stand_still_double_support(
    env: BaseEnv | DexEnv,
    sensor_cfg: SceneEntityCfg,
    zero_threshold: float = 0.1,
    contact_threshold: float = 10.0,
) -> torch.Tensor:
    """Reward both feet being in contact when standing still.
    
    This reward encourages the robot to maintain stable double support (both feet
    on the ground) when the command is near zero, promoting better balance and
    reducing unnecessary single-leg standing or hopping during static postures.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for contact sensor with body_ids for both feet.
        zero_threshold: Command magnitude threshold below which standing is expected.
        contact_threshold: Force threshold (N) to consider a foot as being in contact.
    
    Returns:
        Reward of 1.0 when both feet are in contact during standing, 0.0 otherwise.
    """
    # Check if command is near zero (should be standing)
    cmd_magnitude = torch.norm(env.command_generator.command[:, :2], dim=1) + \
                    torch.abs(env.command_generator.command[:, 2])
    zero_flag = cmd_magnitude <= zero_threshold
    
    # Get contact forces for both feet
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # [num_envs, 2]
    
    # Check if both feet are in contact
    left_contact = contact_forces[:, 0] > contact_threshold
    right_contact = contact_forces[:, 1] > contact_threshold
    both_feet_contact = (left_contact & right_contact).float()
    
    # Only reward during standing
    return both_feet_contact * zero_flag.float()


def stand_still_body_lin_vel(
    env: BaseEnv | DexEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    zero_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize base linear velocity when standing still.
    
    Reduces body swaying by penalizing any translational movement of the base
    when the command is near zero.
    
    Args:
        env: The environment instance.
        asset_cfg: The scene entity configuration for the robot.
        zero_threshold: Command magnitude threshold below which standing is expected.
    
    Returns:
        Penalty based on linear velocity magnitude.
    """
    # Check if command is near zero (should be standing)
    cmd_magnitude = torch.norm(env.command_generator.command[:, :2], dim=1) + \
                    torch.abs(env.command_generator.command[:, 2])
    zero_flag = cmd_magnitude <= zero_threshold
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Penalize all linear velocities (xyz)
    lin_vel_penalty = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :3]), dim=1)

    # Only apply penalty when standing
    return lin_vel_penalty * zero_flag.float()


# -----------------------------------------------------------------------------
# Carry-task reward terms (used by lite_carry).
# Each term guards on `hasattr(env, "object")` so the cfg can be safely shared
# with non-carry tasks (or used during early init before the object is spawned).
# -----------------------------------------------------------------------------

def _carry_target_world(env) -> torch.Tensor:
    """Compute the world-frame carry target: midpoint of two hands + root forward 18cm.

    Returns:
        Tensor of shape (num_envs, 3) in world frame.
    """
    bs = env.robot.data.body_state_w
    lh = bs[:, env.elbow_body_ids[0], :3] + math_utils.quat_apply(
        bs[:, env.elbow_body_ids[0], 3:7], env.left_arm_local_vec)
    rh = bs[:, env.elbow_body_ids[1], :3] + math_utils.quat_apply(
        bs[:, env.elbow_body_ids[1], 3:7], env.right_arm_local_vec)
    mid = 0.5 * (lh + rh)
    fwd = math_utils.quat_apply(
        env.robot.data.root_quat_w,
        torch.tensor([0.18, 0.0, 0.0], device=env.device).expand(env.num_envs, 3),
    )
    return mid + fwd


def keep_object_in_hand(
    env: "BaseEnv | DexEnv",
    dist_scale: float = 0.10,
    gate_threshold: float = 200.0,
    use_gate: bool = True,
) -> torch.Tensor:
    """Reward for keeping the carried object close to the carry target between the hands.

    P0.2 (carry-task fix):
      * ``dist_scale`` widened 0.05 -> 0.10 so the Gaussian still has gradient when
        ``_sync_carry_frame`` introduces small teleportation errors.
      * ``gate_threshold`` reduced 400 -> 200 to coordinate with the per-env gate
        in ``episode_progress_gate`` (P0.1). Each env now activates independently
        after 200 of its own steps rather than waiting for the buffer-wide mean.
      * ``use_gate`` lets the carry cfg turn the gate off entirely (initial P0
        validation uses this to inspect the raw dist signal distribution).

    Args:
        env: The environment instance.
        dist_scale: Length scale of the Gaussian (smaller = stricter). Distance in meters
            at which the reward drops to ~exp(-1) ≈ 0.37.
        gate_threshold: Episode-progress threshold before the reward activates.
        use_gate: When False, the episode_progress_gate is skipped (raw reward only).
    """
    if not hasattr(env, "object") or env.object is None:
        return torch.zeros(env.num_envs, device=env.device)
    target = _carry_target_world(env)
    dist = torch.norm(env.object.data.root_pos_w - target, dim=-1)
    reward = torch.exp(-(dist ** 2) / (dist_scale ** 2))
    if use_gate:
        reward = reward * episode_progress_gate(env, threshold=gate_threshold)
    return reward


def object_orientation_keep(
    env: "BaseEnv | DexEnv",
    tilt_scale: float = 0.1,
    gate_threshold: float = 200.0,
    use_gate: bool = True,
) -> torch.Tensor:
    """Reward for keeping the carried object upright (projected gravity on its xy plane small).

    P0.4 (carry-task fix):
      * Coordinates with P0.1 (per-env gate) and P0.3 (cube now has real tilt
        noise) so this reward can actually deliver gradient.
      * ``use_gate`` lets the carry cfg turn the gate off for raw-signal
        validation.

    Args:
        env: The environment instance.
        tilt_scale: Tilt length scale (smaller = stricter). The reward drops to ~0.37
            when the in-plane gravity component magnitude equals this value.
        gate_threshold: Episode-progress threshold before the reward activates.
        use_gate: When False, the episode_progress_gate is skipped (raw reward only).
    """
    if not hasattr(env, "object") or env.object is None:
        return torch.zeros(env.num_envs, device=env.device)
    quat = env.object.data.root_quat_w
    # Project world +Z into object frame: yields the gravity direction in object frame.
    g_obj = math_utils.quat_apply(
        quat, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    )
    xy_norm = torch.norm(g_obj[:, :2], dim=-1)
    reward = torch.exp(-(xy_norm ** 2) / (tilt_scale ** 2))
    if use_gate:
        reward = reward * episode_progress_gate(env, threshold=gate_threshold)
    return reward


def object_not_dropping(
    env: "BaseEnv | DexEnv",
    z_threshold: float = 0.55,
    terminate_threshold: float = 0.20,
    gate_threshold: float = 200.0,
    use_gate: bool = True,
) -> torch.Tensor:
    """Soft reward for keeping the carried object above a height floor.

    P0.4 (carry-task fix):
      * The reward is now a smooth Gaussian: ``exp(-(z_thr - z).clamp(min=0) * 5)``.
        It peaks at 1.0 when ``z >= z_thr`` and decays continuously toward 0
        as the cube drops, replacing the previous hard 0/1 step.
      * The reward function no longer mutates ``env.reset_buf`` (separation of
        concerns: rewards give gradient, ``check_reset()`` decides termination).
        Instead it stashes a per-env "carry dropped" flag on
        ``env._carry_dropped_buf`` which :class:`TienKungCarryEnv` then OR's
        into ``reset_buf`` in its ``reset()`` hook. This keeps the
        cube-drop termination visible to TensorBoard through the normal
        ``Episode_Termination/*`` channels.
      * ``z_threshold`` and ``terminate_threshold`` are decoupled; the latter
        is now the *soft* floor at which the env will be reset on the next
        step.
      * ``use_gate`` lets the carry cfg turn the gate off for raw-signal
        validation (mirrors :func:`keep_object_in_hand`).

    Args:
        env: The environment instance.
        z_threshold: World-frame z above which the object is fully "held" (reward = 1).
        terminate_threshold: World-frame z below which the env is force-reset on
            the next step (recorded in ``env._carry_dropped_buf``).
        gate_threshold: Episode-progress threshold before the reward activates.
        use_gate: When False, the episode_progress_gate is skipped (raw reward only).
    """
    if not hasattr(env, "object") or env.object is None:
        return torch.zeros(env.num_envs, device=env.device)
    z = env.object.data.root_pos_w[:, 2]
    # Soft reward: 1.0 at z >= z_threshold, decays continuously below.
    reward = torch.exp(-((z_threshold - z).clamp(min=0.0)) * 5.0)
    # Mark "carry dropped" on the env so the reset hook can OR it into
    # ``reset_buf``. We never touch ``env.reset_buf`` here directly.
    if not hasattr(env, "_carry_dropped_buf"):
        env._carry_dropped_buf = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._carry_dropped_buf = (z < terminate_threshold)
    if use_gate:
        reward = reward * episode_progress_gate(env, threshold=gate_threshold)
    return reward


def arm_pose_l1(
    env: "BaseEnv | DexEnv",
    target_q_l: torch.Tensor | None = None,
    target_q_r: torch.Tensor | None = None,
    use_default_if_none: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Carry-pose reward: pull the 8 arm joints toward target joint positions.

    P0.5 (carry-task addition): the inherited :func:`joint_deviation_l1`
    always uses the *default* joint pose (arms hanging at the sides) as the
    target, which directly fights the carry task (which needs the shoulders
    pitched forward and the elbows bent into a U-shape). This reward instead
    lets the user supply a *carry-specific* target — for P0 we default to the
    robot's default joint positions (so the reward is initially a no-op while
    we verify the rest of the pipeline); P3.2 will replace the target with
    the result of an offline IK calibration that solves for the joint angles
    putting both hands at the carry frame.

    The joint order expected (8 joints total, must be in this order):

        [*_shoulder_pitch_*_joint, *_shoulder_roll_*_joint,
         *_shoulder_yaw_*_joint,   *_elbow_pitch_*_joint]  for both l and r

    Convention used by the carry cfg: first 4 are left, last 4 are right.
    The 8 names are passed in via ``asset_cfg.joint_names`` in the carry cfg.

    Returns ``-sum(|q - q_target|)`` (always non-positive; weight the cfg
    accordingly). The ``zero_flag`` from :func:`joint_deviation_l1` is
    intentionally NOT inherited: we want the carry pose to be active even
    while the robot stands in place (it must hold the cube in the carry
    frame regardless of command).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    ids = asset_cfg.joint_ids
    if ids is None or len(ids) != 8:
        # Misconfigured asset_cfg; the reward manager should have caught this
        # at resolve time. Return zeros so the training run does not crash.
        return torch.zeros(env.num_envs, device=env.device)

    if use_default_if_none or target_q_l is None:
        target_l = asset.data.default_joint_pos[:, ids[:4]]
    else:
        # ``target_q_l`` is shape [4]; broadcast to [num_envs, 4].
        target_l = target_q_l.to(env.device).expand(env.num_envs, 4)
    if use_default_if_none or target_q_r is None:
        target_r = asset.data.default_joint_pos[:, ids[4:]]
    else:
        target_r = target_q_r.to(env.device).expand(env.num_envs, 4)

    err_l = q[:, ids[:4]] - target_l
    err_r = q[:, ids[4:]] - target_r
    return -(torch.sum(torch.abs(err_l), dim=1) + torch.sum(torch.abs(err_r), dim=1))

