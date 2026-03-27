"""
天工机器人行走+站立环境配置模块 / TienKung Robot Walk and Stand Environment Configuration Module

该模块定义了天工机器人行走和站立两种模式的环境配置，包括步态配置、站立奖励配置、行走奖励配置和综合环境配置。
This module defines configuration classes for TienKung robot walk and stand modes, including gait configuration, 
stand reward configuration, walk reward configuration, and integrated environment configuration.
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


# Import SaW controller modules
from rsl_rl.modules.actor_critic_saw import ActorCriticSaW

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
from legged_lab.utils.video_recorder import VideoRecorderCfg, VideoRecorderCameraCfg


@configclass
class WalkAndStandMlpRewardCfg:
    """行走和站立奖励配置类 / Walk and Stand Reward Configuration Class
    
    定义行走和站立任务中的奖励函数配置，包括基于论文设计的核心奖励项和原有兼容奖励项。
    Defines reward function configuration for walk and stand tasks, including paper-based core rewards and original compatible rewards.
    
    ===============================================================================
    基于论文的最小约束奖励函数配置
    Minimal Constraint Reward Function Configuration based on Paper Design
    ===============================================================================
    
    REWARD_WEIGHTS = {
        "xy_velocity": 0.15,          # XY速度跟踪权重
        "yaw_orientation": 0.1,       # 偏航角跟踪权重
        "roll_pitch_orientation": 0.2,  # 滚转/俯仰角权重
        "feet_contact": 0.1,          # 足底接触权重
        "base_height": 0.05,         # 基座高度权重
        "feet_airtime": 1.0,         # 足底腾空时间权重（正向奖励）
        "feet_orientation": 0.05,     # 足底姿态权重
        "feet_position": 0.05,        # 足底位置权重
        "arm": 0.03,                 # 手臂角度权重
        "base_acceleration": 0.1,     # 基座加速度权重
        "action_difference": 0.02,    # 动作差分权重
        "torque": 0.02               # 电机力矩权重
    }
    """
    
    # ==============================================================================
    # 基于论文设计的核心奖励函数（最小约束）
    # Core Reward Functions Based on Paper Design (Minimal Constraint)
    # ==============================================================================
    
    # 正向奖励项 / Positive Reward Terms
    # XY速度跟踪权重
    xy_velocity = RewTerm(func=mdp.reward_xy_velocity, weight=0.15)
    # 偏航角跟踪权重
    yaw_orientation = RewTerm(func=mdp.reward_yaw_orientation, weight=0.1)
    # 滚转/俯仰角权重
    roll_pitch_orientation = RewTerm(func=mdp.reward_roll_pitch_orientation, weight=0.2)
    # 足底接触权重
    feet_contact = RewTerm(func=mdp.reward_feet_contact, weight=0.1)
    # 基座高度权重
    base_height = RewTerm(func=mdp.reward_base_height, weight=0.05)
    # 足底腾空时间权重（正向奖励）
    feet_airtime = RewTerm(func=mdp.reward_feet_airtime, weight=1.0)
    # 足底姿态权重
    feet_orientation = RewTerm(func=mdp.reward_feet_orientation, weight=0.05)
    # 足底位置权重
    feet_position = RewTerm(func=mdp.reward_feet_position, weight=0.05)
    # 手臂角度权重
    arm = RewTerm(func=mdp.reward_arm, weight=0.03)
    
    # 惩罚项 / Penalty Terms
    # 基座加速度权重
    base_acceleration = RewTerm(func=mdp.reward_base_acceleration, weight=0.1)
    # 动作差分权重
    action_difference = RewTerm(func=mdp.reward_action_difference, weight=0.02)
    # 电机力矩权重
    torque = RewTerm(func=mdp.reward_torque, weight=0.02)


@configclass
class TienKungSawMlpFlatEnvCfg:
    """天工机器人行走+站立平坦环境配置类 / TienKung Robot Walk and Stand Flat Environment Configuration Class
    
    定义天工机器人在平坦地形上行走和站立的环境配置参数。
    Defines environment configuration parameters for TienKung robot walking and standing on flat terrain.
    """
    
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
    reward = WalkAndStandMlpRewardCfg()
    
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
    # 注意：包含站立命令（速度为0）和行走命令 / Note: includes stand commands (velocity=0) and walk commands
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        # 增加站立环境比例 - 让更多环境处于站立状态 / Increase stand environment ratio - let more environments be in standing state
        # 修改为50%站立，50%行走，确保策略学习到良好的站立能力
        rel_standing_envs=0.5,  # 50% 环境站立，50% 环境行走 / 50% environments stand, 50% environments walk
        rel_heading_envs=0.5,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,  # 启用命令可视化 / Enable command visualization
        ranges=CommandRangesCfg(
            # 包含零速度（站立）和非零速度（行走）/ Include zero velocity (stand) and non-zero velocity (walk)
            lin_vel_x=(-0.6, 1.0), 
            lin_vel_y=(-0.5, 0.5), 
            ang_vel_z=(-1.57, 1.57), 
            heading=(-math.pi, math.pi)
        ),
    )
    
    # 噪声配置 / Noise configuration
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
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
    
    # # 视频录制配置 / Video recording configuration
    # # 用于在无头训练模式下间隔录制训练视频
    # video_recorder: VideoRecorderCfg = VideoRecorderCfg(
    #     enable=True,  # 默认关闭，可通过命令行启用
    #     interval=500,  # 每500步录制一次
    #     num_frames=500,  # 每次录制500帧
    #     output_dir="videos",  # 输出目录
    #     fps=30,  # 帧率
    #     width=1280,  # 宽度
    #     height=720,  # 高度
    #     camera=VideoRecorderCameraCfg(
    #         position=(2.0, -2.0, 1.8),  # 左前方位置
    #         look_at=(0.0, 0.0, 0.5),  # 对准机器人中心
    #         name="main_camera",
    #         prim_path="/World/recording_camera"
    #     )
    # )


@configclass
class TienKungSawMlpAgentCfg(RslRlOnPolicyRunnerCfg):
    """天工机器人行走+站立智能体配置类 / TienKung Robot Walk and Stand Agent Configuration Class
    
    定义行走+站立任务的强化学习智能体配置参数，继承自RSL-RL策略运行器配置。
    Defines reinforcement learning agent configuration parameters for walk+stand task, inheriting from RSL-RL policy runner configuration.
    
    ===============================================================================
    StandAndWalk控制器配置（基于论文设计）
    StandAndWalk Controller Configuration (Based on Paper Design)
    ===============================================================================
    
    SaW控制器特性：
    - 架构：(64, 64)双层LSTM循环神经网络
    - 输入：机器人状态（关节速度、位置、躯干方向）+ 用户命令 cu=[cx, cy, cyaw]
    - 输出：20个关节空间的PD设定点
    - 运行频率：50Hz (SaW控制器)，2kHz (PD控制器)
    - 训练算法：近端策略优化 (PPO) + 镜像损失
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
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    
    # 动作裁剪 / Action clipping
    clip_actions = None
    
    # 保存间隔 / Save interval
    save_interval = 1000
    
    # 运行器类名 / Runner class name
    runner_class_name = "OnPolicyRunner"
    
    # 实验名称 / Experiment name
    experiment_name = "walk_and_stand_mlp"
    
    # 运行名称 / Run name
    run_name = ""
    
    # 日志器 / Logger
    logger = "tensorboard"
    
    # Neptune项目 / Neptune project
    neptune_project = "walk_and_stand_mlp"
    
    # WandB项目 / WandB project
    wandb_project = "walk_and_stand_mlp"
    
    # 是否恢复训练 / Whether to resume training
    resume = False
    
    # 加载运行 / Load run
    load_run = ".*"
    
    # 加载检查点 / Load checkpoint
    load_checkpoint = "model_.*.pt"


@configclass
class SaWCommandConfig:
    """StandAndWalk命令配置类 / StandAndWalk Command Configuration Class
    
    定义用户命令类别和采样策略。
    Defines user command categories and sampling strategies.
    
    论文描述的五种命令类别：
    1. Standing (站立): cu = [0, 0, 0]
    2. Walking in sagittal plane (矢状面行走): cx变化
    3. Walking laterally (侧向行走): cy变化
    4. Rotating in place (原地旋转): cyaw变化
    5. Omnidirectional walking (全向行走): cx, cy, cyaw同时变化
    """
    
    # 命令类别枚举
    COMMAND_CATEGORIES = [
        "standing",           # 站立
        "sagittal_walk",     # 矢状面行走
        "lateral_walk",      # 侧向行走
        "rotation",           # 原地旋转
        "omnidirectional",   # 全向行走
    ]
    
    # 命令范围（论文指定）
    # Command ranges (as specified in paper)
    COMMAND_RANGES = {
        "cx": (-0.5, 2.0),    # m/s, 前后方向速度
        "cy": (-0.5, 0.5),    # m/s, 左右方向速度
        "cyaw": (-0.5, 0.5),  # rad/s, 偏航角速度
    }
    
    # 命令重采样时间范围（2-6秒）
    resampling_time_range = (2.0, 6.0)  # seconds
    
    # 每个类别的采样概率（均匀分布）
    category_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # 均匀分布


@configclass
class SaWRandomPushConfig:
    """StandAndWalk随机推力配置类 / StandAndWalk Random Push Configuration Class
    
    定义随机推力参数，用于增强扰动 rejection能力。
    Defines random push parameters for disturbance rejection capability.
    
    论文描述：
    - 每帧1%概率受到随机推力
    - 推力范围：200N到800N
    - 持续时间：单个timestep (20ms)
    """
    
    # 是否启用随机推力
    enable = True
    
    # 推力概率（每帧）
    push_probability = 0.01  # 1%
    
    # 推力范围（N）
    force_range = (200.0, 800.0)  # N
    
    # 推力持续时间（timesteps）
    force_duration_steps = 1  # 1 step = 20ms
    
    # 推力方向范围（弧度）
    force_angle_range = (0.0, 2 * math.pi)  # 360度均匀分布


#先不更改，看看LSTM是什么原因
"""
    force_duration_steps = 20  # 1 step = 20ms
        force_range = (20.0, 200.0)  # N
"""