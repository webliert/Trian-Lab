# Isaac Lab仿真相关模块 / Isaac Lab simulation related modules
import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore

# 数值计算库 / Numerical computation libraries
import numpy as np
import torch
import math

# Isaac Lab资产和场景模块 / Isaac Lab assets and scene modules
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext

# Isaac Lab工具模块 / Isaac Lab utility modules
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate_inverse, yaw_quat

# 科学计算库 / Scientific computing library
from scipy.spatial.transform import Rotation

# 天工实验室配置模块 / TienKung Lab configuration modules
from legged_lab.envs.tienkung.Experiment.walk_and_stand_cfg import (
    TienKungWalkAndStandFlatEnvCfg,
    SaWCommandConfig,
    SaWRandomPushConfig
)
from legged_lab.utils.env_utils.scene import SceneCfg

# RSL-RL核心模块 / RSL-RL core modules
from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay


class TienKungWalkAndStandEnv(VecEnv):
    def __init__(
        self,
        cfg: TienKungWalkAndStandFlatEnvCfg,
        headless: bool,
    ):
        self.cfg : TienKungWalkAndStandFlatEnvCfg = cfg # 保存配置引用 / Save configuration reference
        self.headless = headless  # 是否无头模式 / Whether in headless mode
        self.device = self.cfg.device  # CUDA设备 / CUDA device
        # 时间参数设置 / Time parameter settings
        # physics_dt: 物理仿真步长(默认0.005秒=5ms) / Physics simulation timestep
        self.physics_dt = self.cfg.sim.dt
        # step_dt: 决策步长 = decimation * physics_dt (默认4*0.005=0.02秒=20ms) / Decision step size
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs # 环境数量 / Number of environments
        self.seed(cfg.scene.seed)   # 设置随机种子 / Set random seed

        """
        物理仿真配置 / Physics Simulation Configuration
        -----------------------------------------------
        创建仿真上下文，设置物理参数和渲染参数。
        Create simulation context with physics and rendering parameters.
        """
        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,  # 计算设备 / Computing device
            dt=cfg.sim.dt,      # 物理步长 / Physics timestep
            render_interval=cfg.sim.decimation,  # 渲染间隔 / Render interval
            # PhysX物理引擎配置 / PhysX physics engine configuration
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            # 物理材质配置 - 高摩擦系数确保足式机器人稳定站立
            # Physics material configuration - high friction coefficient for stable standing
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",     # 摩擦系数乘法组合 / Friction coefficient multiply combination
                restitution_combine_mode="multiply", # 恢复系数乘法组合 / Restitution coefficient multiply combination
                static_friction=1.0,                 # 静摩擦系数 / Static friction coefficient
                dynamic_friction=1.0,                # 动摩擦系数 / Dynamic friction coefficient
            ),
        )
        
        # 创建Isaac Sim仿真上下文 / Create Isaac Sim simulation context
        self.sim = SimulationContext(sim_cfg)

        """
        交互式场景创建 / Interactive Scene Creation
        -----------------------------------------
        根据场景配置创建包含机器人、地形、传感器的仿真场景。
        Create simulation scene with robot, terrain, and sensors based on scene configuration.
        """
        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        
        # 初始化仿真器 / Initialize simulator
        self.sim.reset()

        # 获取机器人关节式 articulation 对象 / Get robot articulation object
        self.robot: Articulation = self.scene["robot"]
        
        # 获取接触力传感器 / Get contact force sensor
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        """
        高度扫描传感器初始化 (可选) / Height Scanner Sensor Initialization (Optional)
        --------------------------------------------------------------------------------
        用于地形感知，通过射线检测获取前方地形高度信息。
        Used for terrain perception, obtains terrain height information through ray casting.
        """
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        """
        传感器初始化 / Sensor Initialization
        -------------------------------------
        根据配置可选地初始化激光雷达和深度相机传感器。
        Optionally initialize LiDAR and depth camera sensors based on configuration.
        """
        # 激光雷达传感器 / LiDAR sensor
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        
        # 深度相机传感器 / Depth camera sensor
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

        """
        命令生成器初始化 / Command Generator Initialization
        ----------------------------------------------------
        创建速度命令生成器，用于生成随机的目标速度命令。
        Create velocity command generator for generating random target velocity commands.
        """
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,  # 命令重采样时间范围
            rel_standing_envs=self.cfg.commands.rel_standing_envs,           # 站立环境比例
            rel_heading_envs=self.cfg.commands.rel_heading_envs,             # 航向控制环境比例
            heading_command=self.cfg.commands.heading_command,               # 是否启用航向命令
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,  # 航向控制刚度
            debug_vis=self.cfg.commands.debug_vis,                           # 调试可视化
            ranges=self.cfg.commands.ranges,                                # 命令范围
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        
        """
        奖励管理器初始化 / Reward Manager Initialization
        ----------------------------------------------
        创建奖励管理器，根据奖励配置计算每个 timestep 的奖励值。
        Create reward manager to compute reward values for each timestep based on reward configuration.
        """
        self.reward_manager = RewardManager(self.cfg.reward, self)

        """
        初始化缓冲区 / Initialize Buffers
        ---------------------------------
        初始化用于存储观察、动作、步态参数等的缓冲区。
        Initialize buffers for storing observations, actions, gait parameters, etc.
        """
        self.init_buffers()

        """
        事件管理器初始化 / Event Manager Initialization
        -----------------------------------------------
        创建事件管理器，处理域随机化事件(如物理参数随机化、推进机器人等)。
        Create event manager to handle domain randomization events (e.g., physics parameter randomization, robot pushing).
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        
        # 应用启动时事件 / Apply startup events
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        
        # 初始化环境 / Initialize environments
        self.reset(env_ids)

    """
    天工机器人环境实现模块 / TienKung Robot Environment Implementation Module
    
    该模块实现了天工机器人的强化学习环境，包含运动控制、传感器数据处理、奖励计算等功能。
    This module implements the reinforcement learning environment for TienKung robot, 
    including motion control, sensor data processing, reward calculation, etc.
    """

    def init_buffers(self):
        """
        初始化各类缓冲区 / Initialize Various Buffers
        
        创建并初始化环境运行所需的所有缓冲区，包括:
        Create and initialize all buffers required for environment operation:
        - 观察缓冲区 (历史观察值)
        - 动作缓冲区 (动作延迟)
        - 步态参数缓冲区
        - 仿真状态缓冲区
        
        - Observation buffer (historical observations)
        - Action buffer (action delay)
        - Simulation state buffer
        """
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["ankle_roll_l_link", "ankle_roll_r_link"], preserve_order=True
        )
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["elbow_pitch_l_link", "elbow_pitch_r_link"], preserve_order=True
        )
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "hip_roll_l_joint",
                "hip_pitch_l_joint",
                "hip_yaw_l_joint",
                "knee_pitch_l_joint",
                "ankle_pitch_l_joint",
                "ankle_roll_l_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "hip_roll_r_joint",
                "hip_pitch_r_joint",
                "hip_yaw_r_joint",
                "knee_pitch_r_joint",
                "ankle_pitch_r_joint",
                "ankle_roll_r_joint",
            ],
            preserve_order=True,
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "shoulder_pitch_l_joint",
                "shoulder_roll_l_joint",
                "shoulder_yaw_l_joint",
                "elbow_pitch_l_joint",
            ],
            preserve_order=True,
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=[
                "shoulder_pitch_r_joint",
                "shoulder_roll_r_joint",
                "shoulder_yaw_r_joint",
                "elbow_pitch_r_joint",
            ],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["ankle_pitch_l_joint", "ankle_pitch_r_joint", "ankle_roll_l_joint", "ankle_roll_r_joint"],
            preserve_order=True,
        )

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.left_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))
        self.right_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))

        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.init_obs_buffer()


    def compute_current_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算当前时刻的观察值 / Compute Current Observations
        
        构建当前时刻的强化学习观察向量，包含机器人本体感知信息和任务命令信息。
        Constructs reinforcement learning observation vectors for the current timestep, including
        robot proprioceptive information and task command information.
        
        观察值组成 / Observation Composition:
        ---------------------------------
        Actor观察值 (当前单步):
        - 角速度 (3维): 机器人基座在机体系下的角速度
        - 投影重力 (3维): 重力向量在机体系下的投影
        - 命令 (3维): 目标线速度和角速度命令
        - 关节位置 (20维): 相对于默认位置的关节角度
        - 关节速度 (20维): 关节角速度
        - 动作 (20维): 上一时刻执行的动作
        
        Actor Observations (single step):
        - Angular velocity (3D): Robot base angular velocity in body frame
        - Projected gravity (3D): Gravity vector projected onto body frame
        - Commands (3D): Target linear and angular velocity commands
        - Joint positions (20D): Joint angles relative to default positions
        - Joint velocities (20D): Joint angular velocities
        - Actions (20D): Actions executed at previous timestep
        
        Critic观察值:
        在Actor观察值基础上额外添加:
        - 线速度 (3维): 机器人基座在机体系下的线速度
        - 脚部接触 (2维): 左右脚是否与地面接触
        
        Critic Observations:
        Additional over Actor:
        - Linear velocity (3D): Robot base linear velocity in body frame
        - Foot contact (2D): Whether left/right feet are in contact with ground
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (actor_observations, critic_observations)
                                             / (Actor观察值, Critic观察值)
        """
        # 获取机器人数据 / Get robot data
        robot = self.robot
        # 获取接触力历史数据 / Get contact force history data
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # 提取各种观察分量 / Extract various observation components
        # 基座角速度 (机体系) / Base angular velocity (body frame)
        ang_vel = robot.data.root_ang_vel_b
        # 投影重力向量 / Projected gravity vector
        projected_gravity = robot.data.projected_gravity_b
        # 速度命令 / Velocity commands
        command = self.command_generator.command
        # 关节位置 (相对于默认位置) / Joint positions (relative to default positions)
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        # 关节速度 (相对于默认速度) / Joint velocities (relative to default velocities)
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        # 上一时刻的动作 / Previous timestep action
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        # 基座线速度 (机体系) / Base linear velocity (body frame)
        root_lin_vel = robot.data.root_ang_vel_b
        # 脚部接触状态 / Foot contact status
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5

        # 拼接Actor观察值 / Concatenate Actor observations
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,  # 3
                projected_gravity * self.obs_scales.projected_gravity,  # 3
                command * self.obs_scales.commands,  # 3
                joint_pos * self.obs_scales.joint_pos,  # 20
                joint_vel * self.obs_scales.joint_vel,  # 20
                action * self.obs_scales.actions,  # 20
            ],
            dim=-1,
        )
        
        # 拼接Critic观察值 (在Actor基础上添加线速度和脚部接触) / Concatenate Critic observations
        current_critic_obs = torch.cat([current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1)

        return current_actor_obs, current_critic_obs

    def compute_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算完整观察值 / Compute Complete Observations
        
        基于当前观察构建历史观察序列，并可选地添加传感器数据(高度扫描、深度相机)和噪声。
        Builds historical observation sequences based on current observations, and optionally
        adds sensor data (height scan, depth camera) and noise.
        
        此方法完成以下工作:
        1. 获取当前单步观察值
        2. 可选地添加观测噪声
        3. 将观察值加入历史缓冲区
        4. 可选地添加高度扫描数据
        5. 可选地添加深度相机数据
        6. 对观察值进行裁剪
        
        This method:
        1. Gets current single-step observations
        2. Optionally adds observation noise
        3. Adds observations to history buffer
        4. Optionally adds height scan data
        5. Optionally adds depth camera data
        6. Clips observations
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (actor_observations, critic_observations)
                                             / (Actor观察值, Critic观察值)
        """
        # 获取当前观察值 / Get current observations
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        
        # 可选地添加噪声 / Optionally add noise
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        # 将当前观察加入历史缓冲区 / Add current observations to history buffer
        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        # 重塑历史观察值 / Reshape historical observations
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        
        # 可选地添加高度扫描数据 / Optionally add height scan data
        if self.cfg.scene.height_scanner.enable_height_scan:
            # 计算地形高度差 / Calculate terrain height difference
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            # 添加到Critic观察 / Add to Critic observations
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            # 可选地添加高度扫描噪声 / Optionally add height scan noise
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            # 添加到Actor观察 / Add to Actor observations
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        # 可选地添加深度相机数据 / Optionally add depth camera data
        if self.cfg.scene.depth_camera.enable_depth_camera:
            # 获取深度图像 / Get depth image
            depth_image = self.depth_camera.data.output["distance_to_image_plane"]

            # 展平深度图像: (num_envs, height, width, 1) --> (num_envs, height * width)
            # Flatten depth image: (num_envs, height, width, 1) --> (num_envs, height * width)
            flattened_depth = depth_image.view(self.num_envs, -1)

            # 将展平后的深度数据添加到观察向量末尾
            # Append flattened depth data to observation vector ends
            actor_obs = torch.cat([actor_obs, flattened_depth], dim=-1)
            critic_obs = torch.cat([critic_obs, flattened_depth], dim=-1)

        # 裁剪观察值到指定范围 / Clip observations to specified range
        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids: torch.Tensor):
        """
        重置指定环境 / Reset Specified Environments
        
        当环境需要重置时调用(回合结束或发生异常终止情况)。重置过程包括:
        Called when environments need to be reset (episode ended or abnormal termination occurred). Reset includes:
        - 清零脚部力和速度累积缓冲区
        - 地形课程学习更新(如果启用)
        - 场景重置
        - 应用重置事件
        - 奖励管理器重置
        - 命令生成器重置
        - 观察和动作缓冲区重置
        - 回合长度计数器重置
        
        - Zero foot force and velocity accumulation buffers
        - Terrain curriculum learning update (if enabled)
        - Scene reset
        - Apply reset events
        - Reward manager reset
        - Command generator reset
        - Observation and action buffer reset
        - Episode length counter reset
        
        Args:
            env_ids: 需要重置的环境ID列表 / List of environment IDs to reset
        """
        if len(env_ids) == 0:
            return

        # 清零脚部力和速度缓冲区 / Zero foot force and speed buffers
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        # 初始化日志字典 / Initialize log dictionary
        self.extras["log"] = dict()
        
        # 地形课程学习更新 / Terrain curriculum learning update
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        # 重置场景 / Reset scene
        self.scene.reset(env_ids)
        
        # 应用重置事件 / Apply reset events
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        # 重置奖励管理器 / Reset reward manager
        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        # 重置命令生成器 / Reset command generator
        self.command_generator.reset(env_ids)
        
        # 重置观察和动作缓冲区 / Reset observation and action buffers
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        
        # 重置回合长度计数器 / Reset episode length counter
        self.episode_length_buf[env_ids] = 0

        # 写入仿真数据 / Write simulation data
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        执行一步仿真 / Execute One Simulation Step
        
        这是强化学习环境的核心方法，处理智能体的一个动作并返回下一状态。
        This is the core method of the RL environment, processes one action from the agent and returns the next state.
        
        执行流程 / Execution Flow:
        -----------------------
        1. 动作处理:
           - 应用动作延迟(如果启用)
           - 裁剪动作到指定范围
           - 转换为目标关节位置
        
        2. 物理仿真循环 (decimation次):
           - 设置关节位置目标
           - 写入仿真数据
           - 执行物理仿真步骤
           - 更新场景状态
           - 累积脚部力和速度数据
        
        3. 后处理:
           - 计算平均脚部力和速度
           - 渲染图形(如果非headless模式)
           - 更新回合长度
           - 更新步态参数
           - 更新命令生成器
           - 应用间隔事件
        
        4. 奖励计算:
           - 检查是否需要重置
           - 计算奖励值
           - 重置需要重置的环境
        
        1. Action Processing:
           - Apply action delay (if enabled)
           - Clip actions to specified range
           - Convert to target joint positions
        
        2. Physics Simulation Loop (decimation times):
           - Set joint position targets
           - Write simulation data
           - Execute physics simulation step
           - Update scene state
           - Accumulate foot force and velocity data
        
        3. Post-processing:
           - Calculate average foot force and velocity
           - Render graphics (if non-headless mode)
           - Update episode length
           - Update gait parameters
           - Update command generator
           - Apply interval events
        
        4. Reward Calculation:
           - Check if reset is needed
           - Calculate reward values
           - Reset environments that need reset
        
        Args:
            actions: 智能体输出的动作张量 / Action tensor output by the agent
                    形状: (num_envs, num_actions) / Shape: (num_envs, num_actions)
                    
        Returns:
            tuple: (observations, rewards, dones, extras)
                - observations: 观察值 / Observations
                - rewards: 奖励值 / Rewards
                - dones: 是否结束 / Dones
                - extras: 额外信息 / Extra information
        """
        # 1. 动作处理 / Action processing
        # 应用动作延迟 / Apply action delay
        delayed_actions = self.action_buffer.compute(actions)
        # 裁剪动作 / Clip actions
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        # 转换为目标关节位置: action * scale + default_pos
        # Convert to target joint position: action * scale + default_pos
        processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos

        # 2. 物理仿真循环 / Physics simulation loop
        # 初始化脚部力和速度累加器 / Initialize foot force and velocity accumulators
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # 执行decimation次物理仿真 / Execute decimation times of physics simulation
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            # 设置关节位置目标 (PD位置控制) / Set joint position targets (PD position control)
            self.robot.set_joint_position_target(processed_actions)
            # 写入仿真数据 / Write simulation data
            self.scene.write_data_to_sim()
            # 执行物理仿真步骤 / Execute physics simulation step
            self.sim.step(render=False)
            # 更新场景状态 / Update scene state
            self.scene.update(dt=self.physics_dt)

            # 累积脚部接触力 / Accumulate foot contact forces
            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            # 累积脚部线速度 / Accumulate foot linear velocities
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        # 计算平均值 / Calculate average values
        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        # 3. 后处理 / Post-processing
        # 渲染图形(如果非headless模式) / Render graphics (if non-headless mode)
        if not self.headless:
            self.sim.render()

        # 更新回合长度 / Update episode length
        self.episode_length_buf += 1

        # 更新命令生成器 / Update command generator
        self.command_generator.compute(self.step_dt)
        # 应用间隔事件 / Apply interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 4. 奖励计算 / Reward calculation
        # 检查是否需要重置 / Check if reset is needed
        self.reset_buf, self.time_out_buf = self.check_reset()
        # 计算奖励 / Compute rewards
        reward_buf = self.reward_manager.compute(self.step_dt)
        # 获取需要重置的环境ID / Get environment IDs that need reset
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # 重置这些环境 / Reset these environments
        self.reset(self.reset_env_ids)

        # 计算下一状态的观察值 / Compute observations for next state
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        检查是否需要重置环境 / Check if Environment Reset is Needed
        
        判断环境是否需要重置，主要基于以下条件:
        Determines if environment needs to be reset, mainly based on:
        1. 异常接触力: 身体部位(非脚部)与地面发生接触
        2. 超时: 回合长度达到最大限制
        
        1. Abnormal contact force: Body parts (non-foot) contact with ground
        2. Timeout: Episode length reaches maximum limit
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (reset_buffer, time_out_buffer)
                - reset_buffer: 是否需要重置的标志 / Flag indicating whether reset is needed
                - time_out_buffer: 是否超时的标志 / Flag indicating whether timeout occurred
        """
        # 获取接触力历史数据 / Get contact force history data
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # 检查非脚部是否有异常接触力 / Check if abnormal contact forces exist on non-foot body parts
        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        
        # 检查是否超时 / Check if timeout
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        # 合并条件 / Merge conditions
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        """
        初始化观察缓冲区 / Initialize Observation Buffers
        
        创建用于存储历史观察值的循环缓冲区。
        Creates circular buffers for storing historical observation values.
        
        如果启用噪声，还需初始化噪声缩放向量。
        If noise is enabled, also initializes noise scaling vectors.
        """
        # 如果启用噪声 / If noise is enabled
        if self.add_noise:
            # 获取当前观察值以确定维度 / Get current observations to determine dimensions
            actor_obs, _ = self.compute_current_observations()
            
            # 创建噪声缩放向量 / Create noise scale vector
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            
            # 设置各维度的噪声缩放 / Set noise scales for each dimension
            noise_vec[:3] = noise_scales.lin_vel * self.obs_scales.lin_vel
            noise_vec[3:6] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[6:9] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[9:12] = 0  # 命令不添加噪声 / Commands not added with noise
            noise_vec[12 : 12 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[12 + self.num_actions : 12 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[12 + self.num_actions * 2 : 12 + self.num_actions * 3] = 0.0
            noise_vec[12 + self.num_actions * 3 : 18 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

            # 高度扫描噪声缩放 / Height scan noise scale
            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        # 创建Actor观察历史缓冲区 / Create Actor observation history buffer
        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        # 创建Critic观察历史缓冲区 / Create Critic observation history buffer
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids: torch.Tensor) -> dict:
        """
        更新地形等级 / Update Terrain Levels
        
        基于课程学习(Curriculum Learning)策略，根据机器人位置动态调整地形难度。
        Dynamically adjusts terrain difficulty based on robot position using curriculum learning strategy.
        
        当机器人远离环境中心时，将其移动到更高难度的地形区域。
        When robot moves away from environment center, move it to higher difficulty terrain area.
        
        Args:
            env_ids: 需要更新地形的环境ID列表 / List of environment IDs to update terrain
            
        Returns:
            dict: 包含地形等级信息的字典 / Dictionary containing terrain level information
        """
        # 计算机器人到环境中心的距离 / Calculate distance from robot to environment center
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        
        # 判断是否需要升级地形 / Determine if terrain needs to be upgraded
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        
        # 判断是否需要降级地形 / Determine if terrain needs to be downgraded
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up  # 确保不会同时升级和降级 / Ensure not upgrading and downgrading at the same time
        
        # 更新环境原点 / Update environment origins
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        
        # 返回地形等级信息 / Return terrain level information
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """
        获取观察值 / Get Observations
        
        返回当前环境的观察值供策略使用。
        Returns current environment observations for policy use.
        
        Returns:
            tuple[torch.Tensor, dict]: (observations, extras)
                - observations: Actor观察值 / Actor observations
                - extras: 额外信息，包含Critic观察值 / Extra information, including Critic observations
        """
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        """
        设置随机种子 / Set Random Seed
        
        设置随机种子以确保实验可复现性。
        Sets random seed to ensure experiment reproducibility.
        
        Args:
            seed: 随机种子值，-1表示使用随机值 / Random seed value, -1 means use random value
            
        Returns:
            int: 设置的随机种子值 / The set random seed value
        """
        # 尝试设置Omniverse Replicator的种子 / Try to set Omniverse Replicator seed
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        
        # 设置PyTorch和NumPy的种子 / Set PyTorch and NumPy seeds
        return torch_utils.set_seed(seed)


    # ==============================================================================
    # StandAndWalk (SaW) 控制器特定功能
    # StandAndWalk (SaW) Controller Specific Functions
    # ==============================================================================
    
    def init_saw_buffers(self):
        """
        初始化SaW控制器专用缓冲区。
        Initialize buffers specific to SaW controller.
        
        包括：
        - 命令类别缓冲区
        - 命令重采样计时器
        - 随机推力状态
        """
        # 命令类别配置
        self.command_categories = SaWCommandConfig.COMMAND_CATEGORIES
        self.command_ranges = SaWCommandConfig.COMMAND_RANGES
        self.command_resample_range = SaWCommandConfig.resampling_time_range
        self.category_weights = SaWCommandConfig.category_weights
        
        # 初始化命令类别（每个环境一个类别）
        self.command_category_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        
        # 初始化命令重采样计时器
        self.command_timer_buf = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        
        # 随机推力配置
        self.push_config = SaWRandomPushConfig()
        
        # 初始化推力状态
        self.current_push_force = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )
        self.push_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        
        # 初始化躯干方向观察（用于SaW控制器的输入）
        self.torso_orientation = torch.zeros(
            (self.num_envs, 4), dtype=torch.float, device=self.device  # 四元数
        )
    
    def sample_command_category(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        从五种命令类别中均匀采样一个新类别。
        Sample a new category uniformly from five command categories.
        
        五种命令类别：
        1. Standing (站立): cu = [0, 0, 0]
        2. Walking in sagittal plane (矢状面行走): cx变化
        3. Walking laterally (侧向行走): cy变化
        4. Rotating in place (原地旋转): cyaw变化
        5. Omnidirectional walking (全向行走): cx, cy, cyaw同时变化
        
        Args:
            env_ids: 需要采样类别的环境ID
            
        Returns:
            采样的类别索引
        """
        num_categories = len(self.command_categories)
        samples = torch.randint(
            0, num_categories, 
            size=(len(env_ids),), 
            device=self.device
        )
        self.command_category_buf[env_ids] = samples
        return samples
    
    def sample_command_for_category(self, category: int, num_envs: int) -> torch.Tensor:
        """
        根据指定类别生成对应的命令值。
        Generate command values based on specified category.
        
        Args:
            category: 命令类别索引 (0-4)
            num_envs: 需要生成命令的环境数量
            
        Returns:
            命令张量 [num_envs, 3] - [cx, cy, cyaw]
        """
        commands = torch.zeros((num_envs, 3), device=self.device)
        cx_range = self.command_ranges["cx"]
        cy_range = self.command_ranges["cy"]
        cyaw_range = self.command_ranges["cyaw"]
        
        if category == 0:  # Standing
            # cu = [0, 0, 0]
            commands[:, 0] = 0.0
            commands[:, 1] = 0.0
            commands[:, 2] = 0.0
            
        elif category == 1:  # Sagittal walk (前后行走)
            # cx变化, cy=0, cyaw=0
            commands[:, 0] = torch.rand(num_envs, device=self.device) * (cx_range[1] - cx_range[0]) + cx_range[0]
            commands[:, 1] = 0.0
            commands[:, 2] = 0.0
            
        elif category == 2:  # Lateral walk (侧向行走)
            # cx=0, cy变化, cyaw=0
            commands[:, 0] = 0.0
            commands[:, 1] = torch.rand(num_envs, device=self.device) * (cy_range[1] - cy_range[0]) + cy_range[0]
            commands[:, 2] = 0.0
            
        elif category == 3:  # Rotation (原地旋转)
            # cx=0, cy=0, cyaw变化
            commands[:, 0] = 0.0
            commands[:, 1] = 0.0
            commands[:, 2] = torch.rand(num_envs, device=self.device) * (cyaw_range[1] - cyaw_range[0]) + cyaw_range[0]
            
        elif category == 4:  # Omnidirectional (全向行走)
            # cx, cy, cyaw同时变化
            commands[:, 0] = torch.rand(num_envs, device=self.device) * (cx_range[1] - cx_range[0]) + cx_range[0]
            commands[:, 1] = torch.rand(num_envs, device=self.device) * (cy_range[1] - cy_range[0]) + cy_range[0]
            commands[:, 2] = torch.rand(num_envs, device=self.device) * (cyaw_range[1] - cyaw_range[0]) + cyaw_range[0]
        
        return commands
    
    def resample_commands(self, env_ids: torch.Tensor):
        """
        为指定环境重新采样命令。
        Resample commands for specified environments.
        
        1. 采样新的命令类别
        2. 根据类别生成对应的命令值
        3. 重置命令计时器
        
        Args:
            env_ids: 需要重新采样命令的环境ID
        """
        # 采样新的命令类别
        new_categories = self.sample_command_category(env_ids)
        
        # 为每个环境根据其类别生成命令
        num_envs = len(env_ids)
        new_commands = torch.zeros((num_envs, 3), device=self.device)
        
        cx_range = self.command_ranges["cx"]
        cy_range = self.command_ranges["cy"]
        cyaw_range = self.command_ranges["cyaw"]
        
        # 遍历每个环境，根据其类别生成命令
        for i, env_id in enumerate(env_ids):
            category = new_categories[i].item()  # 转换为Python int
            
            if category == 0:  # Standing
                new_commands[i] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
            elif category == 1:  # Sagittal walk
                cx = torch.rand(1, device=self.device) * (cx_range[1] - cx_range[0]) + cx_range[0]
                new_commands[i] = torch.cat([cx, torch.zeros(2, device=self.device)])
            elif category == 2:  # Lateral walk
                cy = torch.rand(1, device=self.device) * (cy_range[1] - cy_range[0]) + cy_range[0]
                new_commands[i] = torch.tensor([0.0, cy.item(), 0.0], device=self.device)
            elif category == 3:  # Rotation
                cyaw = torch.rand(1, device=self.device) * (cyaw_range[1] - cyaw_range[0]) + cyaw_range[0]
                new_commands[i] = torch.tensor([0.0, 0.0, cyaw.item()], device=self.device)
            elif category == 4:  # Omnidirectional
                cx = torch.rand(1, device=self.device) * (cx_range[1] - cx_range[0]) + cx_range[0]
                cy = torch.rand(1, device=self.device) * (cy_range[1] - cy_range[0]) + cy_range[0]
                cyaw = torch.rand(1, device=self.device) * (cyaw_range[1] - cyaw_range[0]) + cyaw_range[0]
                new_commands[i] = torch.cat([cx, cy, cyaw])
        
        # 更新命令生成器
        self.command_generator.command[env_ids] = new_commands
        
        # 重置命令计时器（在2-6秒范围内随机）
        new_timer = torch.rand(len(env_ids), device=self.device) * (
            self.command_resample_range[1] - self.command_resample_range[0]
        ) + self.command_resample_range[0]
        self.command_timer_buf[env_ids] = new_timer
    
    def apply_random_push(self):
        """
        应用随机推力以增强扰动 rejection 能力。
        Apply random pushes to enhance disturbance rejection capability.
        
        论文描述：
        - 每帧1%概率受到随机推力
        - 推力范围：200N到800N
        - 持续时间：单个timestep (20ms)
        - 方向：360度均匀分布
        
        这个函数应该在每个step的物理仿真循环中调用。
        """
        if not self.push_config.enable:
            return
        
        num_envs = self.num_envs
        
        # 为每个环境生成随机数来决定是否施加推力
        push_rand = torch.rand(num_envs, device=self.device)
        
        # 识别需要施加新推力的环境（1%概率）
        new_push_envs = push_rand < self.push_config.push_probability
        
        # 为新推力环境生成推力值
        if new_push_envs.any():
            new_push_indices = new_push_envs.nonzero(as_tuple=False).flatten()
            num_new = len(new_push_indices)
            
            # 生成推力大小 (200-800N)
            force_magnitude = torch.rand(num_new, device=self.device) * (
                self.push_config.force_range[1] - self.push_config.force_range[0]
            ) + self.push_config.force_range[0]
            
            # 生成推力方向 (360度)
            force_angle = torch.rand(num_new, device=self.device) * 2 * math.pi
            
            # 在x-y平面内计算推力向量
            force_x = force_magnitude * torch.cos(force_angle)
            force_y = force_magnitude * torch.sin(force_angle)
            
            # 设置推力（沿用之前的z方向）
            self.current_push_force[new_push_indices, 0] = force_x
            self.current_push_force[new_push_indices, 1] = force_y
            self.current_push_force[new_push_indices, 2] = 0.0
            
            # 激活推力
            self.push_active[new_push_indices] = True
        
        # 应用推力到机器人基座
        if self.push_active.any():
            push_env_ids = self.push_active.nonzero(as_tuple=False).flatten()
            
            # 获取机器人根节点的位置
            root_pos = self.robot.data.root_pos_w[push_env_ids]
            
            # 对每个环境应用推力
            for i, env_id in enumerate(push_env_ids):
                force = self.current_push_force[env_id]
                # 在世界坐标系下应用力
                self.robot.set_external_force_at_body(
                    force,
                    body_name="pelvis",
                    position=torch.zeros(3, device=self.device),  # 力作用在质心
                    env_ids=[env_id.item()]
                )
            
            # 推力只持续一个timestep，因此之后关闭
            self.push_active[push_env_ids] = False
    
    def update_saw_commands(self, dt: float):
        """
        更新SaW控制器命令。
        Update SaW controller commands.
        
        检查命令计时器，对需要重采样的环境进行命令重采样。
        
        Args:
            dt: 时间步长
        """
        # 更新命令计时器
        self.command_timer_buf -= dt
        
        # 找出需要重采样的环境（计时器 <= 0）
        resample_mask = self.command_timer_buf <= 0
        
        if resample_mask.any():
            resample_env_ids = resample_mask.nonzero(as_tuple=False).flatten()
            self.resample_commands(resample_env_ids)
    
    def compute_saw_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算SaW控制器专用的观察值。
        Compute SaW controller specific observations.
        
        SaW控制器的输入包括：
        1. 机器人状态：
           - 关节速度 (20维)
           - 关节位置 (20维)
           - 躯干方向 (4维四元数)
        2. 用户命令：
           - cu = [cx, cy, cyaw] (3维)
        
        输出：Actor观察值和Critic观察值
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (actor_observations, critic_observations)
        """
        robot = self.robot
        
        # 1. 机器人状态
        # 关节速度 (20维)
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        
        # 关节位置 (相对于默认位置，20维)
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        
        # 躯干方向 (4维四元数)
        torso_quat = robot.data.root_quat_w
        
        # 2. 用户命令
        command = self.command_generator.command  # [cx, cy, cyaw]
        
        # 拼接Actor观察值
        # [关节速度(20), 关节位置(20), 躯干方向(4), 命令(3)] = 47维
        current_actor_obs = torch.cat([
            joint_vel * self.obs_scales.joint_vel,      # 20
            joint_pos * self.obs_scales.joint_pos,      # 20
            torso_quat,                                  # 4
            command * self.obs_scales.commands,         # 3
        ], dim=-1)
        
        # Critic观察值在Actor基础上添加线速度
        root_lin_vel = robot.data.root_lin_vel_b
        current_critic_obs = torch.cat([
            current_actor_obs,
            root_lin_vel * self.obs_scales.lin_vel,   # 3
        ], dim=-1)
        
        return current_actor_obs, current_critic_obs

