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

"""基础环境配置组合模块 / Base environment configuration combination module

该模块整合基础配置并定义智能体训练参数，将各个配置类组合成完整的环境配置

This module integrates base configurations and defines agent training parameters,
combining various configuration classes into a complete environment configuration.
"""

import math
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg as EventTerm
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

from .base_config import (
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
    RewardCfg,
    RobotCfg,
    SimCfg,
)


@configclass
class BaseEnvCfg:
    """基础环境配置基类 / Base environment configuration class
    
    整合所有基础配置参数，构成完整的环境配置
    
    Integrates all base configuration parameters to form a complete environment configuration.
    """
    device: str = "cuda:0"  # 计算设备 / Computing device
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,  # 最大回合时长（秒） / Maximum episode length (seconds)
        num_envs=4096,  # 环境数量 / Number of environments
        env_spacing=2.5,  # 环境间距 / Environment spacing
        robot=MISSING,  # 机器人配置 / Robot configuration
        terrain_type=MISSING,  # 地形类型 / Terrain type
        terrain_generator=None,  # 地形生成器配置 / Terrain generator configuration
        max_init_terrain_level=5,  # 最大初始地形等级 / Maximum initial terrain level
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,  # 是否启用高度扫描 / Enable height scanning
            prim_body_name=MISSING,  # 主要扫描体名称 / Primary scanning body name
            resolution=0.1,  # 扫描分辨率 / Scanning resolution
            size=(1.6, 1.0),  # 扫描区域尺寸 / Scanning area size
            debug_vis=False,  # 调试可视化 / Debug visualization
            drift_range=(0.0, 0.0),  # 漂移范围 / Drift range
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,  # Actor观察历史长度 / Actor observation history length
        critic_obs_history_length=10,  # Critic观察历史长度 / Critic observation history length
        action_scale=0.25,  # 动作缩放比例 / Action scaling factor
        terminate_contacts_body_names=MISSING,  # 终止接触体名称 / Termination contact body names
        feet_body_names=MISSING,  # 脚部体名称 / Foot body names
    )
    reward = RewardCfg()  # 奖励配置 / Reward configuration
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,  # 线速度缩放比例 / Linear velocity scaling factor
            ang_vel=1.0,  # 角速度缩放比例 / Angular velocity scaling factor
            projected_gravity=1.0,  # 投影重力缩放比例 / Projected gravity scaling factor
            commands=1.0,  # 命令缩放比例 / Command scaling factor
            joint_pos=1.0,  # 关节位置缩放比例 / Joint position scaling factor
            joint_vel=1.0,  # 关节速度缩放比例 / Joint velocity scaling factor
            actions=1.0,  # 动作缩放比例 / Action scaling factor
            height_scan=1.0,  # 高度扫描缩放比例 / Height scan scaling factor
        ),
        clip_observations=100.0,  # 观察值裁剪范围 / Observation clipping range
        clip_actions=100.0,  # 动作裁剪范围 / Action clipping range
        height_scan_offset=0.5,  # 高度扫描偏移量 / Height scan offset
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),  # 重采样时间范围 / Resampling time range
        rel_standing_envs=0.2,  # 相对站立环境比例 / Relative standing environments ratio
        rel_heading_envs=1.0,  # 相对航向环境比例 / Relative heading environments ratio
        heading_command=True,  # 航向命令 / Heading command
        heading_control_stiffness=0.5,  # 航向控制刚度 / Heading control stiffness
        debug_vis=True,  # 调试可视化 / Debug visualization
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0),  # X轴线速度范围 / X-axis linear velocity range
            lin_vel_y=(-0.5, 0.5),  # Y轴线速度范围 / Y-axis linear velocity range
            ang_vel_z=(-1.57, 1.57),  # Z轴角速度范围 / Z-axis angular velocity range
            heading=(-math.pi, math.pi),  # 航向角范围 / Heading angle range
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,  # 添加噪声 / Add noise
        noise_scales=NoiseScalesCfg(
            ang_vel=0.2,  # 角速度噪声缩放比例 / Angular velocity noise scaling factor
            projected_gravity=0.05,  # 投影重力噪声缩放比例 / Projected gravity noise scaling factor
            joint_pos=0.01,  # 关节位置噪声缩放比例 / Joint position noise scaling factor
            joint_vel=1.5,  # 关节速度噪声缩放比例 / Joint velocity noise scaling factor
            height_scan=0.1,  # 高度扫描噪声缩放比例 / Height scan noise scaling factor
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),  # 静摩擦系数范围 / Static friction coefficient range
                    "dynamic_friction_range": (0.4, 0.8),  # 动摩擦系数范围 / Dynamic friction coefficient range
                    "restitution_range": (0.0, 0.005),  # 恢复系数范围 / Restitution coefficient range
                    "num_buckets": 64,  # 桶数量 / Number of buckets
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                    "mass_distribution_params": (-5.0, 5.0),  # 质量分布参数 / Mass distribution parameters
                    "operation": "add",  # 操作类型 / Operation type
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},  # 姿态范围 / Pose range
                    "velocity_range": {
                        "x": (-0.5, 0.5),  # X轴速度范围 / X-axis velocity range
                        "y": (-0.5, 0.5),  # Y轴速度范围 / Y-axis velocity range
                        "z": (-0.5, 0.5),  # Z轴速度范围 / Z-axis velocity range
                        "roll": (-0.5, 0.5),  # 滚转角速度范围 / Roll angular velocity range
                        "pitch": (-0.5, 0.5),  # 俯仰角速度范围 / Pitch angular velocity range
                        "yaw": (-0.5, 0.5),  # 偏航角速度范围 / Yaw angular velocity range
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),  # 位置范围 / Position range
                    "velocity_range": (0.0, 0.0),  # 速度范围 / Velocity range
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),  # 间隔时间范围 / Interval time range
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},  # 速度范围 / Velocity range
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),  # 动作延迟配置 / Action delay configuration
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))  # 仿真配置 / Simulation configuration

    def __post_init__(self):
        """后初始化方法 / Post-initialization method
        
        在配置类初始化后执行的自定义逻辑
        
        Custom logic executed after configuration class initialization.
        """
        pass


@configclass
class BaseAgentCfg(RslRlOnPolicyRunnerCfg):
    """基础智能体配置 / Base agent configuration
    
    定义智能体训练的相关参数，继承自RSL-RL的OnPolicyRunner配置
    
    Defines agent training parameters, inherits from RSL-RL's OnPolicyRunner configuration.
    """
    seed = 42  # 随机种子 / Random seed
    device = "cuda:0"  # 计算设备 / Computing device
    num_steps_per_env = 24  # 每个环境的步数 / Number of steps per environment
    max_iterations = 50000  # 最大迭代次数 / Maximum iterations
    empirical_normalization = False  # 经验归一化 / Empirical normalization
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",  # 类名 / Class name
        init_noise_std=1.0,  # 初始噪声标准差 / Initial noise standard deviation
        noise_std_type="scalar",  # 噪声标准差类型 / Noise standard deviation type
        actor_hidden_dims=[512, 256, 128],  # Actor隐藏层维度 / Actor hidden layer dimensions
        critic_hidden_dims=[512, 256, 128],  # Critic隐藏层维度 / Critic hidden layer dimensions
        activation="elu",  # 激活函数 / Activation function
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",  # 类名 / Class name
        value_loss_coef=1.0,  # 价值损失系数 / Value loss coefficient
        use_clipped_value_loss=True,  # 使用裁剪价值损失 / Use clipped value loss
        clip_param=0.2,  # 裁剪参数 / Clip parameter
        entropy_coef=0.005,  # 熵系数 / Entropy coefficient
        num_learning_epochs=5,  # 学习轮数 / Number of learning epochs
        num_mini_batches=4,  # 小批量数量 / Number of mini-batches
        learning_rate=1.0e-3,  # 学习率 / Learning rate
        schedule="adaptive",  # 调度策略 / Schedule strategy
        gamma=0.99,  # 折扣因子 / Discount factor
        lam=0.95,  # GAE参数 / GAE parameter
        desired_kl=0.01,  # 期望KL散度 / Desired KL divergence
        max_grad_norm=1.0,  # 最大梯度范数 / Maximum gradient norm
        normalize_advantage_per_mini_batch=False,  # 每个小批量归一化优势 / Normalize advantage per mini-batch
        symmetry_cfg=None,  # 对称配置 / Symmetry configuration
        rnd_cfg=None,  # RND配置 / RND configuration
    )
    clip_actions = None  # 动作裁剪 / Action clipping
    save_interval = 100  # 保存间隔 / Save interval
    experiment_name = ""  # 实验名称 / Experiment name
    run_name = ""  # 运行名称 / Run name
    logger = "wandb"  # 日志器 / Logger
    neptune_project = "leggedlab"  # Neptune项目 / Neptune project
    wandb_project = "leggedlab"  # WandB项目 / WandB project
    resume = False  # 是否恢复训练 / Resume training
    load_run = ".*"  # 加载运行模式 / Load run pattern
    load_checkpoint = "model_.*.pt"  # 加载检查点模式 / Load checkpoint pattern

    def __post_init__(self):
        """后初始化方法 / Post-initialization method
        
        在配置类初始化后执行的自定义逻辑
        
        Custom logic executed after configuration class initialization.
        """
        pass
