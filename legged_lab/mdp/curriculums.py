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

# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from legged_lab.envs.base.command_curriculum import GridAdaptiveCurriculum


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)


# -----------------------------------------------------------------------------
# Grid-adaptive command curriculum (Isaac Lab friendly)
# -----------------------------------------------------------------------------


def grid_adaptive_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_names: Sequence[str],
    success_thresholds: Sequence[float],
) -> None:
    """Grid-based adaptive command curriculum that runs on reset.

    - Keeps per-env bin indices and a shared GridAdaptiveCurriculum instance on the env.
    - Uses configurable reward terms to decide success; falls back to zero if terms missing.
    - Samples new commands for the resetting envs and logs means for TensorBoard.
    """

    if len(env_ids) == 0:
        return

    # Lazy init shared state
    if not hasattr(env, "_grid_cmd_curriculum"):
        cfg = getattr(env.cfg.commands, "command_curriculum_cfg", None)
        if cfg is None:
            return
        env._grid_cmd_curriculum = GridAdaptiveCurriculum(cfg)
        env._grid_cmd_bins = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    curriculum = env._grid_cmd_curriculum

    # Collect per-term episode sums for the resetting envs
    task_rewards = []
    for name in reward_term_names:
        term_sums = env.reward_manager._episode_sums.get(name, None)
        if term_sums is None:
            task_rewards.append(torch.zeros(len(env_ids), device=env.device))
        else:
            task_rewards.append(term_sums[env_ids].detach().clone())

    # Align thresholds length
    thresholds = list(success_thresholds)
    if len(thresholds) < len(task_rewards):
        thresholds += [thresholds[-1]] * (len(task_rewards) - len(thresholds))

    if len(task_rewards) > 0:
        curriculum.update_success_rate(
            env._grid_cmd_bins[env_ids].cpu().numpy(),
            task_rewards,
            thresholds[: len(task_rewards)],
        )
        curriculum.update_weights()

    # Sample new commands for the resetting envs
    cmds_np, bin_inds = curriculum.sample(len(env_ids))
    cmds = torch.tensor(cmds_np, device=env.device, dtype=torch.float)
    env.command_generator.command[env_ids, 0] = cmds[:, 0]
    env.command_generator.command[env_ids, 1] = cmds[:, 1]
    env.command_generator.command[env_ids, 2] = cmds[:, 2]
    env._grid_cmd_bins[env_ids] = torch.as_tensor(bin_inds, device=env.device, dtype=torch.long)

    # TensorBoard-friendly logs
    if not hasattr(env, "extras"):
        env.extras = {}
    env.extras.setdefault("log", {})
    env.extras["log"]["Curriculum/max_cmd_x"] = env.command_generator.command[:, 0].max()
    env.extras["log"]["Curriculum/min_cmd_x"] = env.command_generator.command[:, 0].min()
    env.extras["log"]["Curriculum/max_cmd_y"] = env.command_generator.command[:, 1].max()
    env.extras["log"]["Curriculum/min_cmd_y"] = env.command_generator.command[:, 1].min()
    env.extras["log"]["Curriculum/max_cmd_yaw"] = env.command_generator.command[:, 2].max()
    env.extras["log"]["Curriculum/min_cmd_yaw"] = env.command_generator.command[:, 2].min()
    env.extras["log"]["Curriculum/weight_mean"] = torch.as_tensor(curriculum.weights.mean())


# -----------------------------------------------------------------------------
# Carry-task mass curriculum (lite_carry). Three phases are advanced based on
# running mean rewards collected from the parent class's reset() call:
#
#   Phase A: mass ~ U(0, 0.01)   - "empty hands", let the policy learn to walk
#   Phase B: mass ~ U(0, 5)      - advance when track_lin_vel reward is stable
#   Phase C: mass ~ U(5, 10)     - advance when keep_object_in_hand is stable
#
# Phase transitions require the gate condition to hold for `gate_hold_resets`
# consecutive reset cycles (avoids single-iter noise flipping the phase).
# The cube mass is written directly to the PhysX view in lock-step with the
# env's payload_mass buffer (used by the actor observation), so the policy
# always sees the same weight it is actually carrying.
# -----------------------------------------------------------------------------


def weight_curriculum(
    env,
    env_ids: torch.Tensor,
    reward_term_names: Sequence[str],
    success_thresholds: Sequence[float],
    gate_hold_resets: int = 200,
    mean_reward_threshold: float | None = None,
    mean_reward_window: int = 100,
) -> None:
    """Adaptive mass curriculum for the carry task.

    Args:
        env: The env instance (TienKungCarryEnv).
        env_ids: Indices of environments being reset this call.
        reward_term_names: Names of the reward terms used to gate phase transitions.
            Length must be >= 2: index 0 is the track gate, index 1 is the keep gate.
        success_thresholds: Per-term thresholds to upgrade. ``thresholds[0]`` is the
            track-reward threshold for A->B, ``thresholds[1]`` is the keep-reward
            threshold for B->C.
        gate_hold_resets: How many consecutive reset cycles a candidate phase must
            hold before the curriculum actually advances. Avoids flicker.
        mean_reward_threshold: If set, the A->B transition is additionally gated on
            the rolling Mean reward (Train/mean_reward equivalent computed from
            per-env episode total reward) exceeding this threshold. The gate is
            COMPOUND: both the per-term ``track_rew`` AND the Mean reward must
            hold. Set to ``None`` to disable (legacy per-term-only behavior).
        mean_reward_window: Size of the rolling deque of per-episode total rewards.
            Defaults to 100, matching the rsl_rl runner's ``rewbuffer`` so the
            env-side running mean is numerically equivalent to the TensorBoard
            ``Train/mean_reward`` scalar.
    """
    if len(env_ids) == 0:
        return
    if not hasattr(env, "object") or env.object is None:
        return

    # Lazy-init persistent curriculum state on the env.
    if not hasattr(env, "_carry_phase"):
        env._carry_phase = 0
        env._carry_max_mass_seen = torch.zeros(env.num_envs, device=env.device)
        env._carry_gate_counter = 0

    # Lazy-init rolling Mean reward state.
    if not hasattr(env, "_mean_reward_history"):
        env._mean_reward_history = deque(maxlen=mean_reward_window)
        env._running_mean_reward = 0.0

    # Update rolling Mean reward from the per-env total rewards captured in
    # TienKungCarryEnv.reset() BEFORE super().reset() zeroed the per-term sums.
    # Each env in env_ids produced one completed episode; we push that
    # episode's total reward into the deque and recompute the mean. The
    # buffer is consumed (deleted) so a stray re-call doesn't double-count.
    if hasattr(env, "_last_reset_episode_totals"):
        totals = env._last_reset_episode_totals.detach()
        for t in totals.cpu().tolist():
            env._mean_reward_history.append(float(t))
        if env._mean_reward_history:
            env._running_mean_reward = sum(env._mean_reward_history) / len(env._mean_reward_history)
        del env._last_reset_episode_totals

    # Read mean rewards over the resetting envs.
    track_rew = torch.zeros((), device=env.device)
    keep_rew = torch.zeros((), device=env.device)
    for i, name in enumerate(reward_term_names):
        s = env.reward_manager._episode_sums.get(name, None)
        if s is None:
            v = torch.zeros(len(env_ids), device=env.device)
        else:
            v = s[env_ids].detach()
        if i == 0:
            track_rew = v.mean()
        elif i == 1:
            keep_rew = v.mean()

    thr_a_to_b = float(success_thresholds[0]) if len(success_thresholds) > 0 else 1.5
    thr_b_to_c = float(success_thresholds[1]) if len(success_thresholds) > 1 else 0.6

    # Decide whether we want to advance phase (gate requires sustained hold).
    # A->B is COMPOUND-gated: per-term track_rew AND (if configured) the
    # rolling Mean reward must both hold. B->C is unchanged from before.
    candidate = env._carry_phase
    if env._carry_phase == 0:
        mean_ok = mean_reward_threshold is None or env._running_mean_reward > mean_reward_threshold
        if mean_ok and track_rew > thr_a_to_b:
            candidate = 1
    elif env._carry_phase == 1 and keep_rew > thr_b_to_c and env._carry_max_mass_seen.max() > 4.8:
        candidate = 2

    if candidate != env._carry_phase:
        env._carry_gate_counter += 1
        if env._carry_gate_counter >= gate_hold_resets:
            env._carry_phase = candidate
            env._carry_gate_counter = 0
    else:
        env._carry_gate_counter = 0

    # Sample new mass for the resetting envs from the current phase's range.
    mass_ranges = [(0.0, 0.01), (0.0, 5.0), (5.0, 10.0)]
    phase = min(env._carry_phase, len(mass_ranges) - 1)
    low, high = mass_ranges[phase]
    new_mass = torch.rand(len(env_ids), device=env.device) * (high - low) + low
    env.payload_mass[env_ids] = new_mass
    env._carry_max_mass_seen[env_ids] = torch.maximum(env._carry_max_mass_seen[env_ids], new_mass)

    # Write the per-env mass to the PhysX view of the cube. Reset the body mass
    # back to its USD default first, then overwrite, mirroring what
    # randomize_rigid_body_mass does so the inertia recompute stays consistent.
    asset = env.object
    # root_physx_view tensors live on CPU. The env_ids arg is on the RL device
    # (cuda), so move it to cpu for indexing.
    env_ids_cpu = env_ids.detach().to("cpu")
    body_ids = slice(None)
    masses = asset.root_physx_view.get_masses()
    default_mass = asset.data.default_mass[env_ids_cpu, body_ids].clone()
    masses[env_ids_cpu, body_ids] = default_mass
    new_mass_cpu = new_mass.detach().to(masses.device).unsqueeze(-1)
    masses[env_ids_cpu, body_ids] = new_mass_cpu
    asset.root_physx_view.set_masses(masses, env_ids_cpu)

    # Recompute inertia (matches the recompute_inertia=True path of
    # randomize_rigid_body_mass). Note: RigidObject's `inertias` view is flat
    # [num_envs, 9] (1 body, 9 inertia components), NOT [num_envs, num_bodies, 9]
    # like an Articulation, so the indexing differs from the Articulation path
    # in events.py. We use 2D indexing here.
    ratios = new_mass_cpu / default_mass.clamp(min=1e-6)  # [num_envs, 1]
    inertias = asset.root_physx_view.get_inertias()
    default_inertia = asset.data.default_inertia[env_ids_cpu].clone()  # [num_envs, 9]
    inertias[env_ids_cpu] = default_inertia * ratios  # [num_envs, 9] * [num_envs, 1] = [num_envs, 9]
    asset.root_physx_view.set_inertias(inertias, env_ids_cpu)

    # TensorBoard logs.
    if not hasattr(env, "extras"):
        env.extras = {}
    env.extras.setdefault("log", {})
    env.extras["log"]["Curriculum/phase"] = float(env._carry_phase)
    env.extras["log"]["Curriculum/max_mass"] = env.payload_mass.max()
    env.extras["log"]["Curriculum/min_mass"] = env.payload_mass.min()
    env.extras["log"]["Curriculum/track_rew"] = track_rew
    env.extras["log"]["Curriculum/keep_rew"] = keep_rew
    env.extras["log"]["Curriculum/mean_reward"] = env._running_mean_reward
