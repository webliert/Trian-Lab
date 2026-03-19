"""
天工机器人站立环境配置模块 / TienKung Robot Standing Environment Configuration Module

该模块定义了天工机器人站立任务的环境配置，包括简化奖励配置和站立环境配置。
专门设计用于让机器人快速学会站立，然后通过课程学习过渡到行走。
This module defines configuration classes for TienKung robot standing task, including simplified reward configuration 
and standing environment configuration. Designed for fast learning of standing with curriculum learning for walk transition.
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


@configclass
class StandRewardCfg:
    """站立奖励配置类 / Stand Reward Configuration Class
    
    定义站立任务中的简化奖励函数配置，重点关注保持身体直立和稳定。
    Defines simplified reward function configuration for standing task, focusing on body upright and stability.
    """
    
    # 平坦朝向L2惩罚 - 保持身体直立（增加权重）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    
    # 身体朝向L2惩罚 - 保持pelvis水平
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}, weight=-5.0
    )
    
    # 线性速度Z L2惩罚 - 防止上下浮动
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    
    # 角速度XY L2惩罚 - 防止身体摇晃（增加权重）
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0)
    
    # 能量消耗惩罚
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    
    # 关节加速度L2惩罚 - 使运动更平滑
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    
    # 动作变化率L2惩罚 - 使控制更平滑
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # 关节位置限制惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    
    # 髋部关节偏差惩罚 - 站立时保持稳定姿态
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
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
    
    # 手臂关节偏差惩罚
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_roll_.*_joint", "shoulder_yaw_.*_joint"])},
    )
    
    # 腿部关节偏差惩罚 - 站立时更关键
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
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
    
    # 踝关节扭矩惩罚
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    
    # 踝关节动作惩罚
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    
    # 髋关节滚转动作惩罚
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.0)
    
    # 髋关节偏航动作惩罚
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.0)
    
    # 终止惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # 非期望接触惩罚
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
    
    # 脚部滑动惩罚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    
    # 脚部绊倒惩罚
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])},
    )


@configclass
class TienKungStandFlatEnvCfg:
    """天工机器人站立平坦环境配置类 / TienKung Robot Standing Flat Environment Configuration Class
    
    定义天工机器人在平坦地形上站立的环境配置参数。
    专注于快速学习站立，通过课程学习机制逐渐增加行走任务比例。
    Defines environment configuration parameters for TienKung robot standing on flat terrain.
    Focuses on fast standing learning with curriculum learning to gradually increase walking task ratio.
    """
    
    # 计算设备 / Computing device
    device: str = "cuda:0"
    
    # 场景配置 / Scene configuration
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=TIENKUNG2LITE_CFG,
        # 使用平坦地形，更容易学习站立
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
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
    reward = StandRewardCfg()
    
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
    # 初始阶段90%站立环境，10%低速行走环境
    # 随着训练进行，可以调整这个比例
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        # 课程学习：初始90%站立，逐渐增加行走比例
        rel_standing_envs=0.9,  # 90% 站立命令
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            # 允许低速行走环境，用于课程学习过渡
            lin_vel_x=(-0.3, 0.3),  # 缩小速度范围，初始以站立为主
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.5, 0.5),  # 减小角速度范围
            heading=(-math.pi, math.pi)
        ),
    )
    
    # 噪声配置 / Noise configuration
    # 减少噪声以帮助初始站立学习
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.1,  # 减少速度噪声
            ang_vel=0.1,  # 减少角速度噪声
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.0,  # 减少关节速度噪声
            height_scan=0.1,
        ),
    )
    
    # 域随机化配置 / Domain randomization configuration
    # 减少初始扰动，帮助站立学习
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.8, 1.0),  # 增加摩擦力范围
                    "dynamic_friction_range": (0.6, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-2.0, 2.0),  # 减小质量扰动
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    # 减小初始位置和姿态扰动
                    "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-1.0, 1.0)},
                    "velocity_range": {
                        "x": (-0.1, 0.1),
                        "y": (-0.1, 0.1),
                        "z": (-0.1, 0.1),
                        "roll": (-0.1, 0.1),
                        "pitch": (-0.1, 0.1),
                        "yaw": (-0.1, 0.1),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.8, 1.2),  # 减小关节位置扰动
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(15.0, 20.0),  # 增加推力间隔，减少初始干扰
                params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},  # 减小推力
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    
    # 仿真配置 / Simulation configuration
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class TienKungStandAgentCfg(RslRlOnPolicyRunnerCfg):
    """天工机器人站立智能体配置类 / TienKung Robot Standing Agent Configuration Class
    
    定义站立任务的强化学习智能体配置参数。
    Defines reinforcement learning agent configuration parameters for standing task.
    """
    
    # 随机种子 / Random seed
    seed = 42
    
    # 计算设备 / Computing device
    device = "cuda:0"
    
    # 每个环境的步数 / Number of steps per environment
    num_steps_per_env = 24
    
    # 最大迭代次数 / Maximum iterations
    # 站立任务收敛较快，可以适当减少
    max_iterations = 30000
    
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
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    
    # 动作裁剪 / Action clipping
    clip_actions = None
    
    # 保存间隔 / Save interval
    save_interval = 100
    
    # 运行器类名 / Runner class name
    runner_class_name = "OnPolicyRunner"
    
    # 实验名称 / Experiment name
    experiment_name = "stand"
    
    # 运行名称 / Run name
    run_name = ""
    
    # 日志器 / Logger
    logger = "tensorboard"
    
    # Neptune项目 / Neptune project
    neptune_project = "stand"
    
    # WandB项目 / WandB project
    wandb_project = "stand"
    
    # 是否恢复训练 / Whether to resume training
    resume = False
    
    # 加载运行 / Load run
    load_run = ".*"
    
    # 加载检查点 / Load checkpoint
    load_checkpoint = "model_.*.pt"
