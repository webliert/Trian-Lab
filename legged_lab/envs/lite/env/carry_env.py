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

"""TienKungCarryEnv: subclass of :class:`TienKungWalkEnv` for the carry-with-weights task.

The carry task reuses everything in :class:`TienKungWalkEnv` (gait, observations, rewards,
domain randomization) and only adds a few hooks:

* On init, it grabs the cube ``RigidObject`` that :class:`SceneCfg` registered
  when ``BaseSceneCfg.enable_object=True``, allocates a per-env ``payload_mass``
  buffer, and binds the mass-adaptive curriculum.
* On every ``step()``, it teleports the cube to a "carry frame" computed from the
  two hand link positions plus a small forward offset. This fakes the effect of
  the hands holding the cube in place.
* On every ``reset(env_ids)``, it advances the mass curriculum (Phase A
  0 kg -> Phase B 0-5 kg -> Phase C 5-10 kg).
* On every ``compute_current_observations()``, it appends the normalized
  ``payload_mass`` (1 dim) to both actor and critic obs, so the policy can
  condition on the weight it is currently carrying.

We deliberately do not modify the parent class; all carry-specific behavior
lives in this file and the matching :mod:`legged_lab.mdp.curriculums` /
:mod:`legged_lab.mdp.rewards` additions.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from legged_lab.envs.lite.env.walk_env import TienKungWalkEnv

if TYPE_CHECKING:
    from legged_lab.envs.lite.config.carry_cfg import TienKungCarryFlatEnvCfg


class TienKungCarryEnv(TienKungWalkEnv):
    """TienKung 2 Lite env with a hand-bound cube and a mass-adaptive curriculum."""

    def __init__(self, cfg: TienKungCarryFlatEnvCfg, headless):
        # IMPORTANT: parent __init__ builds the scene, event_manager,
        # reward_manager, and calls self.reset(env_ids) once at the end. That
        # internal reset call goes through Python's MRO and hits OUR override
        # below, so we must initialize every attribute that `reset()` reads
        # BEFORE we call super().__init__(). `_curriculum_callable` defaults
        # to None here; the real binding happens after super returns. Same for
        # `object` / `payload_mass` (they're read in the carry-frame sync and
        # in compute_current_observations, both of which guard on hasattr).
        self._curriculum_callable: object | None = None

        # Parent __init__ builds the scene, event_manager, reward_manager, and
        # calls self.reset(env_ids) once. We add the carry-specific state AFTER
        # super returns so the parent's internal reset() call doesn't observe
        # payload_mass / object yet.
        super().__init__(cfg, headless)

        # The cube is registered by SceneCfg only when BaseSceneCfg.enable_object
        # is True. Check the cfg directly rather than `"object" in self.scene`
        # because InteractiveScene.__contains__ ends up calling __getitem__ on
        # an env-id key in some IsaacLab versions and raises spurious KeyErrors.
        if getattr(self.cfg.scene, "enable_object", False):
            self.object: RigidObject = self.scene["object"]
            self.payload_mass = torch.full((self.num_envs,), 0.0, device=self.device, dtype=torch.float32)
            self._prev_carry_pos_w = self.object.data.root_pos_w.clone()
            # Extend the observation noise vector by one zero entry so the
            # payload-mass channel stays noise-free (mass is a deterministic
            # function of the curriculum, not a sensor reading).
            if self.add_noise and hasattr(self, "noise_scale_vec"):
                extra = torch.zeros(1, device=self.device, dtype=self.noise_scale_vec.dtype)
                self.noise_scale_vec = torch.cat([self.noise_scale_vec, extra], dim=-1)

        # Bind the curriculum term. We use functools.partial so the cfg can
        # supply reward_term_names / success_thresholds at construction time
        # without forcing the term to look them up at every reset.
        self._curriculum_callable = self._make_weight_curriculum(cfg)

        # Sync the cube's PhysX mass to payload_mass once, so the very first
        # control step (before any user-triggered reset) sees consistent
        # observation / physics. Without this, the cube would briefly be at
        # its USD-default mass (1.0 kg) while the obs says 0.0.
        if self._curriculum_callable is not None and getattr(self, "object", None) is not None:
            self._curriculum_callable(torch.arange(self.num_envs, device=self.device))

    # ------------------------------------------------------------------ helpers

    def _make_weight_curriculum(self, cfg: TienKungCarryFlatEnvCfg):
        """Late-import the curriculum function to avoid an import cycle."""
        from legged_lab.mdp.curriculums import weight_curriculum

        if not getattr(cfg, "weight_curriculum_cfg", None):
            return None
        # Bind `self` as the first positional so the partial only needs env_ids.
        # Capturing self here is intentional: the partial is rebuilt only when
        # __init__ runs, so the self reference is stable for the env's lifetime.
        return partial(
            weight_curriculum,
            self,
            reward_term_names=cfg.weight_curriculum_cfg["reward_term_names"],
            success_thresholds=cfg.weight_curriculum_cfg["success_thresholds"],
            mean_reward_threshold=cfg.weight_curriculum_cfg.get("mean_reward_threshold"),
            mean_reward_window=cfg.weight_curriculum_cfg.get("mean_reward_window", 100),
        )

    def _sync_carry_frame(self) -> None:
        """Teleport the cube to the carry frame between the two hands.

        Called once per control step, after the parent's decimation loop. The
        carry frame is the midpoint of the two elbow links plus an 18 cm forward
        offset in the root yaw frame. ``lin_vel`` is the finite-difference of
        consecutive carry-frame positions, so PhysX sees a smooth teleport
        rather than a zero-velocity snap.

        P0.3 (carry-task fix): the cube orientation is no longer hard-bound to
        the robot's root orientation (which made ``object_orientation_keep``
        trivially == 1.0). Instead we build a yaw-only quat from the
        carry-frame's mid-point velocity direction, then add a small
        per-step uniform roll/pitch noise (±5° = ±0.087 rad). The policy must
        now actively keep the cube from tilting to collect the orientation
        reward.
        """
        bs = self.robot.data.body_state_w
        lh = bs[:, self.elbow_body_ids[0], :3] + quat_apply(bs[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        rh = bs[:, self.elbow_body_ids[1], :3] + quat_apply(
            bs[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec
        )
        mid = 0.5 * (lh + rh)
        fwd = quat_apply(
            self.robot.data.root_quat_w,
            torch.tensor([0.18, 0.0, 0.0], device=self.device).expand(self.num_envs, 3),
        )
        pos = mid + fwd
        vel = (pos - self._prev_carry_pos_w) / self.step_dt
        # Build a yaw-only quat from the carry-frame velocity direction. Use a
        # small floor on the speed magnitude so atan2 stays well-defined when
        # the robot is momentarily stationary.
        speed = torch.norm(vel[:, :2], dim=-1, keepdim=True).clamp(min=1e-3)
        yaw = torch.atan2(vel[:, 1], vel[:, 0])
        roll = (torch.rand(self.num_envs, device=self.device) - 0.5) * 0.174  # ±5°
        pitch = (torch.rand(self.num_envs, device=self.device) - 0.5) * 0.174  # ±5°
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        # root_state layout: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        root_state = torch.cat([pos, quat, vel, torch.zeros_like(vel)], dim=-1)
        self.object.write_root_state_to_sim(root_state)
        self._prev_carry_pos_w = pos.clone()

    # --------------------------------------------------------- overridden hooks

    def compute_current_observations(self):
        """Append normalized ``payload_mass`` to the actor/critic obs."""
        actor_obs, critic_obs = super().compute_current_observations()
        if hasattr(self, "payload_mass"):
            scale = max(float(self.cfg.scene.carry_mass_max), 1e-6)
            mass_norm = (self.payload_mass / scale).unsqueeze(-1)
            actor_obs = torch.cat([actor_obs, mass_norm], dim=-1)
            critic_obs = torch.cat([critic_obs, mass_norm], dim=-1)
        return actor_obs, critic_obs

    def step(self, actions: torch.Tensor):
        """Run a control step and then snap the cube to the carry frame."""
        result = super().step(actions)
        if getattr(self, "object", None) is not None:
            self._sync_carry_frame()
        return result

    def reset(self, env_ids):
        """Run the parent reset, then advance the mass curriculum for the reset envs.

        P0.4 (carry-task fix): after the parent reset cleared ``self.reset_buf``,
        we OR the per-env "carry dropped" flag (set by the reward function
        :func:`legged_lab.mdp.rewards.object_not_dropping`) back into
        ``reset_buf`` so cube-drop terminations are still visible to
        TensorBoard's ``Episode_Termination/*`` channels and ``time_out``
        ratios remain accurate. We do this *after* ``super().reset(env_ids)``
        to avoid double-counting.

        P_gate (Mean reward gate): ``super().reset(env_ids)`` invokes
        :meth:`RewardManager.reset` which zeroes
        ``self.reward_manager._episode_sums[name][env_ids]``. The
        curriculum (called *after* super().reset) needs the just-finished
        episode's per-env total reward to update the rolling Mean reward,
        so we capture the sum across all reward terms for the resetting
        envs *before* calling super().reset and stash it on
        ``self._last_reset_episode_totals``. The curriculum reads and
        consumes that buffer.
        """
        if len(env_ids) == 0:
            return
        # CAPTURE per-env total reward BEFORE super().reset() zeroes the
        # per-term sums. Only attempt this once the carry state and the
        # reward_manager are both bound (the parent's __init__ triggers an
        # internal reset before we finish binding — that one must no-op).
        if getattr(self, "_curriculum_callable", None) is not None and getattr(self, "object", None) is not None:
            total = torch.zeros(len(env_ids), device=self.device, dtype=torch.float32)
            for term_sum in self.reward_manager._episode_sums.values():
                total = total + term_sum[env_ids].detach()
            self._last_reset_episode_totals = total

        super().reset(env_ids)
        # P0.4: bring forward the "carry dropped" termination flag into
        # ``reset_buf`` for any envs that crossed the soft floor in the
        # previous step. We can't use the env_ids here because the dropped
        # flag is per-env (set in the reward), so we just OR the whole tensor.
        if getattr(self, "_carry_dropped_buf", None) is not None:
            self.reset_buf = self.reset_buf | self._carry_dropped_buf
            # P0.9 (debug instrumentation): log the carry-dropped fraction for
            # the resetting envs. Independent of the base terminate_contacts /
            # timeout channels that ``super().reset()`` already logged; an env
            # can be in multiple categories (e.g. cube drop AND pelvis contact
            # in the same step). Computed before ``_carry_dropped_buf.zero_()``
            # so the per-step fraction is correct.
            self.extras["log"]["Episode_Termination/carry_dropped"] = (
                self._carry_dropped_buf[env_ids].float().mean()
            )
            self._carry_dropped_buf.zero_()
        # Use getattr with a default so the very first reset (the one the parent
        # __init__ triggers before our carry state is bound) is a safe no-op.
        if getattr(self, "_curriculum_callable", None) is not None and getattr(self, "object", None) is not None:
            self._curriculum_callable(env_ids)
