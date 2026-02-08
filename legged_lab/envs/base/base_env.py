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

"""基础环境实现模块 / Base environment implementation module

该模块实现机器人仿真环境的核心功能，包括环境初始化、观察计算、重置和步进等
继承自RSL-RL的VecEnv类，提供标准化的强化学习环境接口

This module implements core robot simulation environment functionality including
environment initialization, observation computation, reset, and step operations.
Inherits from RSL-RL's VecEnv class, providing standardized reinforcement learning
environment interface.
"""

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer

from legged_lab.envs.base.base_env_config import BaseEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv


class BaseEnv(VecEnv):
    """基础环境类 / Base environment class
    
    实现机器人仿真环境的核心功能，包括初始化、步进、重置等操作
    
    Implements core robot simulation environment functionality including
    initialization, stepping, resetting, etc.
    """

    def __init__(self, cfg: BaseEnvCfg, headless):
        """环境初始化 / Environment initialization
        
        Args:
            cfg: 环境配置 / Environment configuration
            headless: 是否无头模式 / Whether headless mode
        """
        self.cfg: BaseEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        # 仿真配置 / Simulation configuration
        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",  # 摩擦组合模式 / Friction combine mode
                restitution_combine_mode="multiply",  # 恢复组合模式 / Restitution combine mode
                static_friction=1.0,  # 静摩擦系数 / Static friction coefficient
                dynamic_friction=1.0,  # 动摩擦系数 / Dynamic friction coefficient
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        # 场景配置 / Scene configuration
        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]  # 机器人对象 / Robot object
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]  # 接触传感器 / Contact sensor
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]  # 高度扫描仪 / Height scanner

        # 命令生成器配置 / Command generator configuration
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)  # 奖励管理器 / Reward manager

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)  # 事件管理器 / Event manager
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

    def init_buffers(self):
        """初始化缓冲区 / Initialize buffers
        
        初始化各种数据缓冲区，包括观察缓冲区、动作缓冲区等
        
        Initialize various data buffers including observation buffers, action buffers, etc.
        """
        self.extras = {}  # 额外信息 / Extra information

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s  # 最大回合时长 / Maximum episode length
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)  # 最大回合步数 / Maximum episode steps
        self.num_actions = self.robot.data.default_joint_pos.shape[1]  # 动作数量 / Number of actions
        self.clip_actions = self.cfg.normalization.clip_actions  # 动作裁剪范围 / Action clipping range
        self.clip_obs = self.cfg.normalization.clip_observations  # 观察裁剪范围 / Observation clipping range

        self.action_scale = self.cfg.robot.action_scale  # 动作缩放比例 / Action scaling factor
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

        self.robot_cfg = SceneEntityCfg(name="robot")  # 机器人配置 / Robot configuration
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )  # 终止接触配置 / Termination contact configuration
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)  # 脚部配置 / Feet configuration
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales  # 观察缩放比例 / Observation scaling factors
        self.add_noise = self.cfg.noise.add_noise  # 是否添加噪声 / Whether to add noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 回合长度缓冲区 / Episode length buffer
        self.sim_step_counter = 0  # 仿真步计数器 / Simulation step counter
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # 超时缓冲区 / Timeout buffer
        self.init_obs_buffer()

    def compute_current_observations(self):
        """计算当前观察值 / Compute current observations
        
        计算当前时间步的观察值，包括Actor和Critic的观察
        
        Compute observations for current timestep, including both Actor and Critic observations.
        
        Returns:
            current_actor_obs: Actor当前观察值 / Actor current observations
            current_critic_obs: Critic当前观察值 / Critic current observations
        """
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history  # 净接触力 / Net contact forces

        ang_vel = robot.data.root_ang_vel_b  # 角速度 / Angular velocity
        projected_gravity = robot.data.projected_gravity_b  # 投影重力 / Projected gravity
        command = self.command_generator.command  # 命令 / Command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos  # 关节位置 / Joint position
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel  # 关节速度 / Joint velocity
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]  # 动作 / Action
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
            ],
            dim=-1,
        )

        root_lin_vel = robot.data.root_lin_vel_b  # 根线速度 / Root linear velocity
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5  # 脚部接触 / Feet contact
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        """计算观察值 / Compute observations
        
        计算完整的观察值，包括历史观察和噪声处理
        
        Compute complete observations including historical observations and noise processing.
        
        Returns:
            actor_obs: Actor观察值 / Actor observations
            critic_obs: Critic观察值 / Critic observations
        """
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)  # 裁剪Actor观察值 / Clip actor observations
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)  # 裁剪Critic观察值 / Clip critic observations

        return actor_obs, critic_obs

    def reset(self, env_ids):
        """重置环境 / Reset environment
        
        重置指定环境ID的环境状态
        
        Reset environment states for specified environment IDs.
        
        Args:
            env_ids: 环境ID列表 / Environment IDs
        """
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)  # 重置奖励管理器 / Reset reward manager
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)  # 重置命令生成器 / Reset command generator
        self.actor_obs_buffer.reset(env_ids)  # 重置Actor观察缓冲区 / Reset actor observation buffer
        self.critic_obs_buffer.reset(env_ids)  # 重置Critic观察缓冲区 / Reset critic observation buffer
        self.action_buffer.reset(env_ids)  # 重置动作缓冲区 / Reset action buffer
        self.episode_length_buf[env_ids] = 0  # 重置回合长度 / Reset episode length

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        """执行一步环境仿真 / Perform one environment step
        
        执行一步环境仿真，包括动作处理、物理仿真、观察计算等
        
        Perform one environment step including action processing, physics simulation, observation computation, etc.
        
        Args:
            actions: 动作张量 / Action tensor
            
        Returns:
            actor_obs: Actor观察值 / Actor observations
            reward_buf: 奖励缓冲区 / Reward buffer
            reset_buf: 重置缓冲区 / Reset buffer
            extras: 额外信息 / Extra information
        """
        delayed_actions = self.action_buffer.compute(actions)  # 延迟动作 / Delayed actions

        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)  # 裁剪动作 / Clip actions
        processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos  # 处理后的动作 / Processed actions

        # 执行多个物理步 / Execute multiple physics steps
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()  # 渲染 / Render

        self.episode_length_buf += 1  # 更新回合长度 / Update episode length
        self.command_generator.compute(self.step_dt)  # 计算命令 / Compute commands
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()  # 检查重置条件 / Check reset conditions
        reward_buf = self.reward_manager.compute(self.step_dt)  # 计算奖励 / Compute rewards
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        """检查重置条件 / Check reset conditions
        
        检查是否需要重置环境，包括接触终止和超时条件
        
        Check if environment needs to be reset, including contact termination and timeout conditions.
        
        Returns:
            reset_buf: 重置缓冲区 / Reset buffer
            time_out_buf: 超时缓冲区 / Timeout buffer
        """
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

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
        )  # 接触终止条件 / Contact termination condition
        time_out_buf = self.episode_length_buf >= self.max_episode_length  # 超时条件 / Timeout condition
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        """初始化观察缓冲区 / Initialize observation buffers
        
        初始化观察缓冲区，包括噪声向量的计算
        
        Initialize observation buffers including noise vector computation.
        """
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel  # 角速度噪声 / Angular velocity noise
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity  # 投影重力噪声 / Projected gravity noise
            noise_vec[6:9] = 0  # 命令噪声（通常为0） / Command noise (usually 0)
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos  # 关节位置噪声 / Joint position noise
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )  # 关节速度噪声 / Joint velocity noise
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0  # 动作噪声（通常为0） / Action noise (usually 0)
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan  # 高度扫描噪声 / Height scan noise
                self.height_scan_noise_vec = height_scan_noise_vec

        # 初始化循环缓冲区 / Initialize circular buffers
        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        """更新地形等级 / Update terrain levels
        
        根据机器人的移动更新地形等级，用于课程学习
        
        Update terrain levels based on robot movement, used for curriculum learning.
        
        Args:
            env_ids: 环境ID列表 / Environment IDs
            
        Returns:
            extras: 额外信息 / Extra information
        """
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2  # 向上移动条件 / Move up condition
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )  # 向下移动条件 / Move down condition
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())  # 地形等级统计 / Terrain level statistics
        return extras

    def get_observations(self):
        """获取观察值 / Get observations
        
        获取当前环境的观察值
        
        Get current environment observations.
        
        Returns:
            actor_obs: Actor观察值 / Actor observations
            extras: 额外信息 / Extra information
        """
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        """设置随机种子 / Set random seed
        
        设置随机种子以确保实验的可重复性
        
        Set random seed to ensure experiment reproducibility.
        
        Args:
            seed: 随机种子值 / Random seed value
            
        Returns:
            设置的随机种子 / Set random seed
        """
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
