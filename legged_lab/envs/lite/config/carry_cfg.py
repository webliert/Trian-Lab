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

"""Configuration for the ``lite_carry`` task (hands-cradled cube, mass-adaptive PPO).

The carry cfg mirrors :mod:`legged_lab.envs.lite.walk_cfg` but:

* Algorithm is plain PPO (no AMP, no mocap, no discriminator).
* :attr:`BaseSceneCfg.enable_object` is True, so :class:`SceneCfg` registers the
  cube RigidObject and :class:`TienKungCarryEnv` binds it to the hands.
* :attr:`reward` is :class:`LiteRewardCfgCarry` which adds four reward terms
  (three cube-related + one arm pose) on top of the base walking rewards.
* :attr:`weight_curriculum_cfg` is consumed by
  :func:`legged_lab.mdp.curriculums.weight_curriculum` (invoked from
  :meth:`TienKungCarryEnv.reset`) to advance Phase A (0 kg) -> Phase B (0-5 kg)
  -> Phase C (5-10 kg) over a single training run.
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.tienkung2_lite import TIENKUNG2LITE_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.envs.lite.config.walk_ppo_cfg import GaitCfg, LiteRewardCfg
from legged_lab.mdp.symmetryLite import data_augmentation_func_g1
from legged_lab.terrains import ROUGH_TERRAINS_CFG


@configclass
class TienkungEventCfg(EventCfg):
    """TienKung 环境的扩展事件配置:在 EventCfg 基础上增加 4 个可选项,供 carry_cfg 按需填入。

    该类原位于 :mod:`legged_lab.envs.lite.config.walk_stand_cfg`,因 carry 已切到 walk
    体系,这里内联以避免对 walk_stand 的循环依赖。4 个新槽默认 ``None``,IsaacLab
    configclass 视 ``None`` 为"不覆盖",因此不会引入未定义行为。
    """

    randomize_pd_gains: EventTerm = None
    randomize_apply_external_force_torque: EventTerm = None
    randomize_rigid_body_com: EventTerm = None
    randomize_joint_params: EventTerm = None


@configclass
class LiteRewardCfgCarry(LiteRewardCfg):
    """Walking rewards (inherited) + cube-related rewards + pose reward.

    The base class is :class:`legged_lab.envs.lite.config.walk_ppo_cfg.LiteRewardCfg`,
    so all inherited reward weights / thresholds follow the vanilla-PPO walk
    task's baseline (e.g. ``track_lin_vel_xy_exp = 1.0``, ``alive_reward = 0.5``,
    ``feet_too_near = -5.0``, ``termination_penalty = -200.0``,
    ``joint_deviation_legs = -0.02``).

    The class adds three deltas on top of the inherited set:

    * **Three cube rewards** (``keep_object_in_hand``, ``object_orientation_keep``,
      ``object_not_dropping``) with their carry-task weights; ``use_gate=False``
      so the raw signals are observable in early training, and the per-env
      ``episode_progress_gate`` is added back automatically when the cube-gate
      default ``gate_threshold`` is reached.
    * **Carry-specific ``arm_pose_l1``** (weight=-0.5, target=default pose).
      This shadows the inherited walk_ppo ``arm_pose_l1`` (weight=+1.0) and is
      a placeholder until the offline-IK carry-pose target is wired in
      (P3.2 deliverable). The negative weight on a non-positive function
      reward is a deliberate "arms away from default" signal so the policy
      can find a non-default carry-pose configuration; the cube-state rewards
      anchor the actual pose.
    * **``joint_deviation_arms = None``** — the inherited term targets
      ``shoulder_roll``/``shoulder_yaw`` default joint pos (arms at sides),
      which is the opposite of the carry pose. Disabling it here only
      affects the carry task; the walk/run tasks still inherit the original
      term from :class:`walk_ppo_cfg.LiteRewardCfg`.

    The P0.6 weakened values for ``termination_penalty`` and
    ``feet_y_distance`` have been reverted to walk_ppo defaults; the
    Mean reward gate in :func:`legged_lab.mdp.curriculums.weight_curriculum`
    now protects against early-training instability, so the previous
    safety net is no longer needed.
    """

    # Cube rewards (carry-task additions on top of inherited walking rewards).
    # keep_object_in_hand = RewTerm(
    #     func=mdp.keep_object_in_hand, weight=5.0, params={"dist_scale": 0.10, "use_gate": False}
    # )
    keep_object_in_hand = None
    
    # object_orientation_keep = RewTerm(func=mdp.object_orientation_keep, weight=1.0, params={"use_gate": False})
    object_orientation_keep = None
    
    # object_not_dropping = RewTerm(func=mdp.object_not_dropping, weight=2.0, params={"use_gate": False})
    object_not_dropping = None

    # Carry-pose reward. Targets default for now; P3.2 will swap in
    # the offline-IK result. The weight=-0.5 sign is a placeholder
    # (see class docstring).
    arm_pose_l1 = RewTerm(
        func=mdp.arm_pose_l1,
        weight=1.0,
        params={
            "use_default_if_none": True,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "shoulder_pitch_l_joint",
                    "shoulder_roll_l_joint",
                    "shoulder_yaw_l_joint",
                    "elbow_pitch_l_joint",
                    "shoulder_pitch_r_joint",
                    "shoulder_roll_r_joint",
                    "shoulder_yaw_r_joint",
                    "elbow_pitch_r_joint",
                ],
            ),
        },
    )

    # Disable the inherited ``joint_deviation_arms`` (weight=-0.2, default
    # joint-pos target) because the carry pose has shoulders rolled forward;
    # the inherited target is the opposite of what the carry task wants.
    # Disabling it here only affects the carry task — the walk/run tasks
    # still inherit the original term from ``walk_ppo_cfg.LiteRewardCfg``.
    joint_deviation_arms = None


@configclass
class TienKungCarryFlatEnvCfg:
    """Top-level env cfg for the ``lite_carry`` task."""

    device: str = "cuda:0"

    # Parent TienKungWalkStandEnv unconditionally instantiates an AMPLoaderDisplay in
    # its __init__ using this attribute. lite_carry is a pure PPO task, so
    # the loader is never actually exercised during training (no AMP), but
    # AMPLoaderDisplay crashes on an empty file list (torch.vstack([])), so
    # we point it at an existing walk mocap as a benign placeholder.
    amp_motion_files_display: list = ["legged_lab/envs/lite/datasets/motion_visualization/walk.txt"]

    # -- scene --------------------------------------------------------------
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=TIENKUNG2LITE_CFG,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        # Carry-task specific:
        enable_object=True,
        default_carry_mass=1.0,
        carry_mass_max=10.0,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )

    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=[
            "knee_pitch.*",
            "shoulder_roll.*",
            "elbow_pitch.*",
            "pelvis",
        ],
        feet_body_names=["ankle_roll.*"],
    )
    reward = LiteRewardCfgCarry()
    gait = GaitCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.2),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi),
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )

    # -- domain randomization ----------------------------------------------
    # Cube mass is NOT randomized here; it is driven by weight_curriculum
    # (see TienKungCarryEnv.reset) so that the policy always observes the
    # exact payload_mass that the physics is integrating against.
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=TienkungEventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "yaw": (-3.14, 3.14),
                    },
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            ),
            # P0.8 (carry-task fix): disable the per-reset external-force/torque
            # kicker. set_external_force_and_torque is PERSISTENT in the PhysX
            # buffer, so mode="reset" with force_range=(-20, 20) applies a
            # constant ~20N / ~5Nm to the pelvis for the entire episode. On a
            # 50 kg humanoid that tops the policy in ~35 steps (= 0.355 s at
            # step_dt=0.01 s) -- exactly the observed mean episode length. The
            # other 3 carry-only DR terms are one-shot (startup) or modest
            # scale-only (pd_gains 0.75-1.25) and are not the dominant cause.
            # Re-introduce a bounded perturbation in a follow-up if desired.
            randomize_apply_external_force_torque=None,
            randomize_joint_params=None,
            randomize_rigid_body_com=None,
            randomize_pd_gains=None,
        ),
        #     randomize_pd_gains=EventTerm(
        #         func=mdp.randomize_actuator_gains,
        #         mode="reset",
        #         params={
        #             "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        #             "stiffness_distribution_params": (0.75, 1.25),
        #             "damping_distribution_params": (0.75, 1.25),
        #             "operation": "scale",
        #             "distribution": "uniform",
        #         },
        #     ),
        #     randomize_joint_params=EventTerm(
        #         func=mdp.randomize_joint_parameters,
        #         mode="startup",
        #         params={
        #             "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        #             "friction_distribution_params": (0.001, 0.6),
        #             "armature_distribution_params": (0.002, 0.060),
        #             "operation": "abs",
        #             "distribution": "uniform",
        #         },
        #     ),
        #     randomize_rigid_body_com=EventTerm(
        #         func=mdp.randomize_rigid_body_com,
        #         mode="startup",
        #         params={
        #             "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"]),
        #             "com_range": {
        #                 "x": (-0.05, 0.05),
        #                 "y": (-0.05, 0.05),
        #                 "z": (0.0, 0.0),
        #             },
        #         },
        #     ),
        # ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )

    sim: SimCfg = SimCfg(
        dt=0.005,
        decimation=4,
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )

    # -- mass curriculum ----------------------------------------------------
    # reward_term_names[0] is the Phase A->B per-term gate (track_lin_vel_xy_exp),
    # reward_term_names[1] is the Phase B->C per-term gate (keep_object_in_hand).
    # The A->B transition is COMPOUND-gated: both the per-term threshold AND
    # the rolling "Mean reward" (Train/mean_reward equivalent) must hold for
    # ``gate_hold_resets`` consecutive reset cycles. The Mean reward gate
    # ensures the policy is globally good before box-mass is introduced.
    weight_curriculum_cfg: dict = {
        "reward_term_names": ["track_lin_vel_xy_exp", "keep_object_in_hand"],
        "success_thresholds": [1.5, 0.6],
        "mean_reward_threshold": 30.0,
        "mean_reward_window": 100,
    }


@configclass
class TienKungCarryAgentCfg(RslRlOnPolicyRunnerCfg):
    """PPO agent cfg for ``lite_carry`` (no AMP, no mocap)."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 30000
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=100,
            data_augmentation_func=data_augmentation_func_g1,
        ),
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 500
    runner_class_name = "OnPolicyRunner"

    experiment_name = "lite_carry"
    run_name = ""
    # P3.3 (brought forward into P0): use wandb so the P0 checkpoint
    # validation can compare the four diagnostic curves side-by-side with
    # future P1/P2 runs in a single dashboard.
    logger = "wandb"
    neptune_project = "lite_carry"
    wandb_project = "lite_carry"
    swanlab_project = "lite_carry"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
