"""
天工机器人奔跑环境配置模块 / TienKung Robot Running Environment Configuration Module

该模块定义了天工机器人奔跑环境的相关配置类，包括步态配置、简化奖励配置和奔跑环境配置。
This module defines configuration classes for TienKung robot running environment, including gait configuration, simplified reward configuration, and running environment configuration.
"""

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

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
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
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401


@configclass
class GaitCfg:
    """步态配置类 / Gait Configuration Class
    
    定义机器人奔跑时的步态参数，包括空中相位比例、相位偏移和步态周期。
    Defines gait parameters for robot running, including air phase ratio, phase offset, and gait cycle.
    """
    
    # 左腿空中相位比例 / Left leg air phase ratio
    gait_air_ratio_l: float = 0.6
    
    # 右腿空中相位比例 / Right leg air phase ratio
    gait_air_ratio_r: float = 0.6
    
    # 左腿相位偏移 / Left leg phase offset
    gait_phase_offset_l: float = 0.6
    
    # 右腿相位偏移 / Right leg phase offset
    gait_phase_offset_r: float = 0.1
    
    # 步态周期 / Gait cycle
    gait_cycle: float = 0.5


@configclass
class LiteRewardCfg:
    """简化奖励配置类 / Simplified Reward Configuration Class
    
    定义奔跑任务中的简化奖励函数配置，包括各种奖励项及其权重。
    Defines simplified reward function configuration for running task, including various reward terms and their weights.
    """
    
    # 跟踪线性速度XY指数奖励 / Track linear velocity XY exponential reward
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    
    # 跟踪角速度Z指数奖励 / Track angular velocity Z exponential reward
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    
    # 线性速度Z L2惩罚 / Linear velocity Z L2 penalty
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    
    # 角速度XY L2惩罚 / Angular velocity XY L2 penalty
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    
    # 能量消耗惩罚 / Energy consumption penalty
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    
    # 关节加速度L2惩罚 / Joint acceleration L2 penalty
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    
    # 动作变化率L2惩罚 / Action rate L2 penalty
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # 非期望接触惩罚 / Undesired contacts penalty
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )
    
    # 身体朝向L2惩罚 / Body orientation L2 penalty
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}, weight=-2.0
    )
    
    # 平坦朝向L2惩罚 / Flat orientation L2 penalty
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    
    # 终止惩罚 / Termination penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # 脚部滑动惩罚 / Feet slide penalty
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    
    # 脚部力惩罚 / Feet force penalty
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    
    # 脚部过近惩罚 / Feet too near penalty
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]), "threshold": 0.2},
    )
    
    # 脚部绊倒惩罚 / Feet stumble penalty
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])},
    )
    
    # 关节位置限制惩罚 / Joint position limits penalty
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    
    # 髋部关节偏差惩罚 / Hip joint deviation penalty
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_yaw_.*_joint",
                    "hip_roll_.*_joint",
                    "shoulder_pitch_.*_joint",
                    "elbow_pitch_.*_joint",
                ],
            )
        },
    )
    
    # 手臂关节偏差惩罚 / Arm joint deviation penalty
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_roll_.*_joint", "shoulder_yaw_.*_joint"])},
    )
    
    # 腿部关节偏差惩罚 / Leg joint deviation penalty
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_pitch_.*_joint",
                    "knee_pitch_.*_joint",
                    "ankle_pitch_.*_joint",
                    "ankle_roll_.*_joint",
                ],
            )
        },
    )

    # 步态脚力周期奖励 / Gait feet force periodic reward
    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0)
    
    # 步态脚速度周期奖励 / Gait feet speed periodic reward
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0)
    
    # 步态脚力支撑周期奖励 / Gait feet force support periodic reward
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.6)

    # 踝关节扭矩惩罚 / Ankle torque penalty
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    
    # 踝关节动作惩罚 / Ankle action penalty
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    
    # 髋关节滚转动作惩罚 / Hip roll action penalty
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.0)
    
    # 髋关节偏航动作惩罚 / Hip yaw action penalty
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.0)
    
    # 脚部Y距离惩罚 / Feet Y distance penalty
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0)


@configclass
class TienKungRunFlatEnvCfg:
    """天工机器人奔跑平坦环境配置类 / TienKung Robot Running Flat Environment Configuration Class
    
    定义天工机器人在平坦地形上奔跑的环境配置参数。
    Defines environment configuration parameters for TienKung robot running on flat terrain.
    """
    
    # AMP运动文件显示路径 / AMP motion files display path
    amp_motion_files_display = ["legged_lab/envs/tienkung/datasets/motion_visualization/run.txt"]
    
    # 计算设备 / Computing device
    device: str = "cuda:0"
    
    # 场景配置 / Scene configuration
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=TIENKUNG2LITE_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        # terrain_type="plane",
        # terrain_generator= None,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
    )
    
    # 机器人配置 / Robot configuration
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
        feet_body_names=["ankle_roll.*"],
    )
    
    # 奖励配置 / Reward configuration
    reward = LiteRewardCfg()
    
    # 步态配置 / Gait configuration
    gait = GaitCfg()
    
    # 归一化配置 / Normalization configuration
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
    
    # 命令配置 / Commands configuration
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )
    
    # 噪声配置 / Noise configuration
    noise: NoiseCfg = NoiseCfg(
        add_noise=False,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    
    # 域随机化配置 / Domain randomization configuration
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
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
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    
    # 仿真配置 / Simulation configuration
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class TienKungRunAgentCfg(RslRlOnPolicyRunnerCfg):
    """天工机器人奔跑智能体配置类 / TienKung Robot Running Agent Configuration Class
    
    定义奔跑任务的强化学习智能体配置参数，继承自RSL-RL策略运行器配置。
    Defines reinforcement learning agent configuration parameters for running task, inheriting from RSL-RL policy runner configuration.
    """
    
    # 随机种子 / Random seed
    seed = 42
    
    # 计算设备 / Computing device
    device = "cuda:0"
    
    # 每个环境的步数 / Number of steps per environment
    num_steps_per_env = 24
    
    # 最大迭代次数 / Maximum iterations
    max_iterations = 50000
    
    # 经验归一化 / Empirical normalization
    empirical_normalization = False
    
    # 策略配置 / Policy configuration
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    # 算法配置 / Algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
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
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    
    # 动作裁剪 / Action clipping
    clip_actions = None
    
    # 保存间隔 / Save interval
    save_interval = 100
    
    # 运行器类名 / Runner class name
    runner_class_name = "AmpOnPolicyRunner"
    
    # 实验名称 / Experiment name
    experiment_name = "run"
    
    # 运行名称 / Run name
    run_name = ""
    
    # 日志器 / Logger
    logger = "tensorboard"
    
    # Neptune项目 / Neptune project
    neptune_project = "run"
    
    # WandB项目 / WandB project
    wandb_project = "run"
    
    # 是否恢复训练 / Whether to resume training
    resume = False
    
    # 加载运行 / Load run
    load_run = ".*"
    
    # 加载检查点 / Load checkpoint
    load_checkpoint = "model_.*.pt"

    # AMP参数 / AMP parameters
    amp_reward_coef = 0.3
    amp_motion_files = ["legged_lab/envs/tienkung/datasets/motion_amp_expert/run.txt"]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 20
