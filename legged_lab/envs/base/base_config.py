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

"""基础环境配置模块 / Base configuration module

该模块定义了基础环境的配置类，包括奖励配置、场景配置、机器人配置等
用于标准化机器人仿真环境的配置结构

This module defines base environment configuration classes including reward configuration,
scene configuration, robot configuration, etc. Used for standardizing robot simulation
environment configuration structures.
"""

import math
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.sensors.camera import TiledCameraCfg
from legged_lab.sensors.lidar import LidarCfg


@configclass
class RewardCfg:
    """奖励配置基类 / Base reward configuration class
    
    用于定义奖励函数的配置参数，子类需要具体实现奖励函数
    
    Used for defining reward function configuration parameters, subclasses need to
    implement specific reward functions.
    """
    pass


@configclass
class HeightScannerCfg:
    """高度扫描仪配置 / Height scanner configuration
    
    配置高度扫描传感器的参数
    
    Configuration for height scanning sensor parameters.
    """
    enable_height_scan: bool = False  # 是否启用高度扫描功能 / Enable height scanning functionality
    prim_body_name: str = MISSING  # 主要扫描体名称 / Primary scanning body name
    resolution: float = 0.1  # 扫描分辨率（米） / Scanning resolution (meters)
    size: tuple = (1.6, 1.0)  # 扫描区域尺寸（宽，高） / Scanning area size (width, height)
    debug_vis: bool = False  # 是否启用调试可视化 / Enable debug visualization
    drift_range: tuple = (0.0, 0.0)  # 漂移范围（最小，最大） / Drift range (min, max)


@configclass
class BaseSceneCfg:
    """基础场景配置 / Base scene configuration
    
    定义仿真场景的基本参数
    
    Defines basic parameters for simulation scene.
    """
    max_episode_length_s: float = 20.0  # 最大回合时长（秒） / Maximum episode length (seconds)
    num_envs: int = 4096  # 环境数量 / Number of environments
    env_spacing: float = 2.5  # 环境间距 / Environment spacing
    robot: ArticulationCfg = MISSING  # 机器人配置 / Robot configuration
    terrain_type: str = MISSING  # 地形类型 / Terrain type
    terrain_generator: TerrainGeneratorCfg = None  # 地形生成器配置 / Terrain generator configuration
    max_init_terrain_level: int = 5  # 最大初始地形等级 / Maximum initial terrain level
    height_scanner: HeightScannerCfg = HeightScannerCfg()  # 高度扫描仪配置 / Height scanner configuration
    lidar: LidarCfg = LidarCfg()  # 激光雷达配置 / LiDAR configuration
    depth_camera: TiledCameraCfg = TiledCameraCfg()  # 深度相机配置 / Depth camera configuration


@configclass
class RobotCfg:
    """机器人配置 / Robot configuration
    
    定义机器人的相关参数
    
    Defines robot-related parameters.
    """
    actor_obs_history_length: int = 10  # Actor观察历史长度 / Actor observation history length
    critic_obs_history_length: int = 10  # Critic观察历史长度 / Critic observation history length
    action_scale: float = 0.25  # 动作缩放比例 / Action scaling factor
    terminate_contacts_body_names: list = []  # 终止接触的体名称列表 / List of body names for termination contacts
    feet_body_names: list = []  # 脚部体名称列表 / List of foot body names


@configclass
class ObsScalesCfg:
    """观察缩放配置 / Observation scaling configuration
    
    定义各种观察值的缩放比例
    
    Defines scaling factors for various observations.
    """
    lin_vel: float = 1.0  # 线速度缩放比例 / Linear velocity scaling factor
    ang_vel: float = 1.0  # 角速度缩放比例 / Angular velocity scaling factor
    projected_gravity: float = 1.0  # 投影重力缩放比例 / Projected gravity scaling factor
    commands: float = 1.0  # 命令缩放比例 / Command scaling factor
    joint_pos: float = 1.0  # 关节位置缩放比例 / Joint position scaling factor
    joint_vel: float = 1.0  # 关节速度缩放比例 / Joint velocity scaling factor
    actions: float = 1.0  # 动作缩放比例 / Action scaling factor
    height_scan: float = 1.0  # 高度扫描缩放比例 / Height scan scaling factor


@configclass
class NormalizationCfg:
    """归一化配置 / Normalization configuration
    
    定义观察值和动作的归一化参数
    
    Defines normalization parameters for observations and actions.
    """
    obs_scales: ObsScalesCfg = ObsScalesCfg()  # 观察缩放配置 / Observation scaling configuration
    clip_observations: float = 100.0  # 观察值裁剪范围 / Observation clipping range
    clip_actions: float = 100.0  # 动作裁剪范围 / Action clipping range
    height_scan_offset: float = 0.5  # 高度扫描偏移量 / Height scan offset


@configclass
class CommandRangesCfg:
    """命令范围配置 / Command ranges configuration
    
    定义各种命令的范围
    
    Defines ranges for various commands.
    """
    lin_vel_x: tuple = (-0.6, 1.0)  # X轴线速度范围（米/秒） / X-axis linear velocity range (m/s)
    lin_vel_y: tuple = (-0.5, 0.5)  # Y轴线速度范围（米/秒） / Y-axis linear velocity range (m/s)
    ang_vel_z: tuple = (-1.0, 1.0)  # Z轴角速度范围（弧度/秒） / Z-axis angular velocity range (rad/s)
    heading: tuple = (-math.pi, math.pi)  # 航向角范围（弧度） / Heading angle range (radians)


@configclass
class CommandsCfg:
    """命令配置 / Commands configuration
    
    定义命令生成器的参数
    
    Defines command generator parameters.
    """
    resampling_time_range: tuple = (10.0, 10.0)  # 重采样时间范围（秒） / Resampling time range (seconds)
    rel_standing_envs: float = 0.2  # 相对站立环境比例 / Relative standing environments ratio
    rel_heading_envs: float = 1.0  # 相对航向环境比例 / Relative heading environments ratio
    heading_command: bool = True  # 是否启用航向命令 / Enable heading command
    heading_control_stiffness: float = 0.5  # 航向控制刚度 / Heading control stiffness
    debug_vis: bool = True  # 是否启用调试可视化 / Enable debug visualization
    ranges: CommandRangesCfg = CommandRangesCfg()  # 命令范围配置 / Command ranges configuration


@configclass
class NoiseScalesCfg:
    """噪声缩放配置 / Noise scaling configuration
    
    定义各种观察值的噪声缩放比例
    
    Defines noise scaling factors for various observations.
    """
    lin_vel: float = 0.2  # 线速度噪声缩放比例 / Linear velocity noise scaling factor
    ang_vel: float = 0.2  # 角速度噪声缩放比例 / Angular velocity noise scaling factor
    projected_gravity: float = 0.05  # 投影重力噪声缩放比例 / Projected gravity noise scaling factor
    joint_pos: float = 0.01  # 关节位置噪声缩放比例 / Joint position noise scaling factor
    joint_vel: float = 1.5  # 关节速度噪声缩放比例 / Joint velocity noise scaling factor
    height_scan: float = 0.1  # 高度扫描噪声缩放比例 / Height scan noise scaling factor


@configclass
class NoiseCfg:
    """噪声配置 / Noise configuration
    
    定义是否添加噪声及噪声缩放参数
    
    Defines whether to add noise and noise scaling parameters.
    """
    add_noise: bool = True  # 是否添加噪声 / Whether to add noise
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()  # 噪声缩放配置 / Noise scaling configuration


@configclass
class EventCfg:
    """事件配置 / Event configuration
    
    定义各种事件（如物理材料随机化、质量随机化等）的参数
    
    Defines parameters for various events (e.g., physics material randomization, mass randomization, etc.).
    """
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),  # 静摩擦系数范围 / Static friction coefficient range
            "dynamic_friction_range": (0.4, 0.8),  # 动摩擦系数范围 / Dynamic friction coefficient range
            "restitution_range": (0.0, 0.005),  # 恢复系数范围 / Restitution coefficient range
            "num_buckets": 64,  # 桶数量 / Number of buckets
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-5.0, 5.0),  # 质量分布参数 / Mass distribution parameters
            "operation": "add",  # 操作类型 / Operation type
        },
    )
    reset_base = EventTerm(
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
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),  # 位置范围 / Position range
            "velocity_range": (0.0, 0.0),  # 速度范围 / Velocity range
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),  # 间隔时间范围（秒） / Interval time range (seconds)
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},  # 速度范围 / Velocity range
    )


@configclass
class ActionDelayCfg:
    """动作延迟配置 / Action delay configuration
    
    定义动作延迟的参数
    
    Defines action delay parameters.
    """
    enable: bool = False  # 是否启用动作延迟 / Enable action delay
    params: dict = {"max_delay": 5, "min_delay": 0}  # 延迟参数 / Delay parameters


@configclass
class DomainRandCfg:
    """域随机化配置 / Domain randomization configuration
    
    定义域随机化的各种事件和参数
    
    Defines various events and parameters for domain randomization.
    """
    events: EventCfg = EventCfg()  # 事件配置 / Event configuration
    action_delay: ActionDelayCfg = ActionDelayCfg()  # 动作延迟配置 / Action delay configuration


@configclass
class PhysxCfg:
    """PhysX物理引擎配置 / PhysX physics engine configuration
    
    定义PhysX物理引擎的参数
    
    Defines PhysX physics engine parameters.
    """
    gpu_max_rigid_patch_count: int = 10 * 2**15  # GPU最大刚体补丁数量 / GPU maximum rigid patch count


@configclass
class SimCfg:
    """仿真配置 / Simulation configuration
    
    定义仿真参数
    
    Defines simulation parameters.
    """
    dt: float = 0.005  # 时间步长（秒） / Time step (seconds)
    decimation: int = 4  # 降采样因子 / Decimation factor
    physx: PhysxCfg = PhysxCfg()  # PhysX配置 / PhysX configuration
