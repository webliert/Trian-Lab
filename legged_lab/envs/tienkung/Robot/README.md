# TienKung Robot Environment Module

## 概述 / Overview

本目录包含天工(TienKung)机器人强化学习仿真环境的实现代码。该环境基于Isaac Sim和RSL-RL框架构建，支持足式机器人的运动控制训练。

This directory contains the implementation code for TienKung robot reinforcement learning simulation environment. The environment is built on Isaac Sim and RSL-RL framework, supporting locomotion control training for legged robots.

## 文件结构 / File Structure

```
Robot/
├── tienkung_env.py           # 主环境实现 (带步态参数)
├── tienkung_env_stand.py     # 站立环境实现 (无步态参数)
├── tienkung_env_69.py        # 变体 (删除线速度和步态参数)
├── tienkung_env_75.py        # 变体 (官方删除线速度)
├── tienkung_env_75_old.py    # 变体 (旧版本删除线速度)
├── tienkung_env_45_only_leg.py  # 变体 (仅控制下半身)
└── README.md                 # 本文档
```

## 环境类说明 / Environment Class Description

### TienKungEnv (tienkung_env.py)

主环境类，继承自`VecEnv`，实现天工机器人的强化学习仿真环境。

#### 核心功能

1. **仿真环境搭建** - 基于Isaac Sim构建物理仿真环境
2. **传感器管理** - 支持接触传感器、高度扫描仪、激光雷达、深度相机
3. **运动控制** - 实现PD位置控制驱动的关节控制
4. **奖励计算** - 集成奖励管理器计算多目标奖励
5. **步态生成** - 内置相位参数生成周期性步态
6. **AMP支持** - 支持对抗性运动先验(Adversarial Motion Prior)

#### 主要方法

| 方法名                         | 功能描述                                 |
| ------------------------------ | ---------------------------------------- |
| `__init__`                     | 环境初始化，创建仿真器、场景、传感器等   |
| `init_buffers`                 | 初始化各类缓冲区                         |
| `step`                         | 执行一步仿真，返回观察值、奖励、终止标志 |
| `reset`                        | 重置指定环境                             |
| `compute_observations`         | 计算完整观察值                           |
| `compute_current_observations` | 计算当前时刻观察值                       |
| `check_reset`                  | 检查是否需要重置环境                     |
| `visualize_motion`             | AMP运动可视化                            |
| `get_amp_obs_for_expert_trans` | 获取AMP专家观察值                        |
| `_calculate_gait_para`         | 计算步态参数                             |

## 代码执行流程 / Code Execution Flow

### 1. 训练流程 (Training Flow)

```
┌─────────────────────────────────────────────────────────────┐
│                      train.py                               │
│                  (legged_lab/scripts/)                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              task_registry.get_cfgs(task_name)              │
│         获取环境配置和智能体配置                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              task_registry.get_task_class(task_name)       │
│              获取环境类 (TienKungEnv)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              env = TienKungEnv(env_cfg, headless)           │
│                     环境初始化                               │
│                  (见下文详细流程)                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              runner = AmpOnPolicyRunner(...)               │
│              创建训练运行器                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              runner.learn(num_learning_iterations)          │
│                     开始训练                                 │
└─────────────────────────────────────────────────────────────┘
```

### 2. 环境初始化流程 (Environment Initialization Flow)

```
TienKungEnv.__init__(cfg, headless)
│
├─► 1. 设置基本参数
│   - device, physics_dt, step_dt, num_envs
│   - seed(cfg.scene.seed)
│
├─► 2. 创建仿真上下文 (SimulationContext)
│   - sim_cfg = SimulationCfg(...)
│   - self.sim = SimulationContext(sim_cfg)
│
├─► 3. 创建交互式场景 (InteractiveScene)
│   - scene_cfg = SceneCfg(...)
│   - self.scene = InteractiveScene(scene_cfg)
│   - self.sim.reset()
│
├─► 4. 获取机器人对象
│   - self.robot = self.scene["robot"]
│
├─► 5. 初始化传感器
│   - contact_sensor (接触力传感器)
│   - height_scanner (高度扫描仪，可选)
│   - lidar (激光雷达，可选)
│   - depth_camera (深度相机，可选)
│
├─► 6. 创建命令生成器
│   - self.command_generator = UniformVelocityCommand(...)
│
├─► 7. 创建奖励管理器
│   - self.reward_manager = RewardManager(...)
│
├─► 8. 初始化缓冲区
│   - init_buffers()
│   - init_obs_buffer()
│
├─► 9. 创建事件管理器
│   - self.event_manager = EventManager(...)
│   - self.event_manager.apply(mode="startup")
│
└─► 10. 初始化环境
    - self.reset(env_ids)
```

### 3. 仿真步骤流程 (Simulation Step Flow)

```
env.step(actions)
│
├─► 1. 动作处理
│   ├─► delayed_actions = action_buffer.compute(actions)
│   │       (应用动作延迟)
│   │
│   └─► self.action = torch.clip(delayed_actions, ...)
│       processed_actions = self.action * action_scale + default_joint_pos
│
├─► 2. 物理仿真循环 (decimation次)
│   │
│   ├─► for i in range(decimation):
│   │   │
│   │   ├─► self.robot.set_joint_position_target(processed_actions)
│   │   │       (设置PD位置控制目标)
│   │   │
│   │   ├─► self.scene.write_data_to_sim()
│   │   │       (写入仿真数据)
│   │   │
│   │   ├─► self.sim.step(render=False)
│   │   │       (执行物理仿真)
│   │   │
│   │   ├─► self.scene.update(dt=physics_dt)
│   │   │       (更新场景状态)
│   │   │
│   │   └─► 累积脚部力和速度数据
│   │       - avg_feet_force_per_step
│   │       - avg_feet_speed_per_step
│   │
│   └─► 计算平均值
│
├─► 3. 后处理
│   ├─► self.episode_length_buf += 1
│   ├─► _calculate_gait_para() (更新步态参数)
│   ├─► command_generator.compute() (更新命令)
│   └─► event_manager.apply(mode="interval")
│
├─► 4. 奖励计算
│   ├─► reset_buf, time_out_buf = check_reset()
│   │       - 检查异常接触力
│   │       - 检查回合超时
│   │
│   ├─► reward_buf = reward_manager.compute()
│   │       - 计算多目标奖励
│   │
│   └─► reset(reset_env_ids)
│           - 重置需要重置的环境
│
└─► 5. 返回结果
    - actor_obs, reward_buf, reset_buf, extras
```

### 4. 观察值计算流程 (Observation Computation Flow)

```
compute_observations()
│
├─► compute_current_observations()
│   │
│   ├─► 获取机器人数据
│   │   - ang_vel (角速度)
│   │   - projected_gravity (投影重力)
│   │   - command (命令)
│   │   - joint_pos (关节位置)
│   │   - joint_vel (关节速度)
│   │   - action (上一动作)
│   │   - feet_contact (脚部接触)
│   │
│   ├─► 构建Actor观察值 (55维)
│   │   ├─ ang_vel * scale (3)
│   │   ├─ projected_gravity * scale (3)
│   │   ├─ command * scale (3)
│   │   ├─ joint_pos * scale (20)
│   │   ├─ joint_vel * scale (20)
│   │   ├─ action * scale (20)
│   │   ├─ sin(2π * gait_phase) (2)
│   │   └─ cos(2π * gait_phase) (2)
│   │       phase_ratio (2)
│   │
│   └─► 构建Critic观察值
│       - current_actor_obs + lin_vel + feet_contact
│
├─► 添加噪声 (可选)
│   - actor_obs += noise * noise_scale
│
├─► 添加到历史缓冲区
│   - actor_obs_buffer.append()
│   - critic_obs_buffer.append()
│
├─► 添加传感器数据 (可选)
│   ├─► 高度扫描数据
│   └─► 深度相机数据
│
└─► 裁剪并返回
    - actor_obs = clip(actor_obs, -clip_obs, clip_obs)
    - critic_obs = clip(critic_obs, -clip_obs, clip_obs)
```

## 观察值维度 / Observation Dimensions

### Actor观察值 (单步)

| 索引范围 | 维度 | 描述                    |
| -------- | ---- | ----------------------- |
| 0:3      | 3    | 基座角速度 (机体系)     |
| 3:6      | 3    | 投影重力向量            |
| 6:9      | 3    | 速度命令                |
| 9:29     | 20   | 关节位置 (相对默认位置) |
| 29:49    | 20   | 关节速度                |
| 49:69    | 20   | 上一时刻动作            |
| 69:71    | 2    | 步态相位正弦            |
| 71:73    | 2    | 步态相位余弦            |
| 73:75    | 2    | 相位比例                |

**总计: 75维 (单步)**

### Critic观察值

在Actor基础上添加:
- 基座线速度 (3维)
- 脚部接触状态 (2维)

**总计: 80维 (单步)**

### 历史观察值

如果历史长度=10:
- Actor观察: 75 × 10 = 750维
- Critic观察: 80 × 10 = 800维

## 支持的任务 / Supported Tasks

| 任务名称             | 描述               | 配置文件                  |
| -------------------- | ------------------ | ------------------------- |
| `walk`               | 行走任务           | walk_cfg.py               |
| `run`                | 奔跑任务           | run_cfg.py                |
| `walk_with_sensor`   | 带传感器融合的行走 | walk_with_sensor_cfg.py   |
| `run_with_sensor`    | 带传感器融合的奔跑 | run_with_sensor_cfg.py    |
| `walk_only_leg`      | 仅腿部控制行走     | walk_cfg_only_leg.py      |
| `unitree_style_walk` | 标准PPO行走        | unitree_style_walk_cfg.py |
| `walk_and_stand`     | 行走+站立          | walk_and_stand_cfg.py     |
| `stand`              | 站立任务           | stand_cfg.py              |

## 快速开始 / Quick Start

### 训练

```bash
# 训练行走任务
python legged_lab/scripts/train.py --task=walk

# 训练奔跑任务
python legged_lab/scripts/train.py --task=run

# 使用传感器训练
python legged_lab/scripts/train.py --task=walk_with_sensor
```

### 推理

```bash
# 使用训练好的策略运行
python legged_lab/scripts/play.py --task=walk --checkpoint=Exported_policy/walk.pt
```

## 关键配置参数 / Key Configuration Parameters

### 仿真参数 (SimCfg)

```python
dt: 0.005          # 物理步长 (秒)
decimation: 4     # 降采样因子
```

### 机器人参数 (RobotCfg)

```python
actor_obs_history_length: 10   # Actor观察历史长度
critic_obs_history_length: 10  # Critic观察历史长度
action_scale: 0.25            # 动作缩放比例
```

### 步态参数 (GaitCfg)

```python
gait_air_ratio_l: 0.38       # 左腿空中相位比例
gait_air_ratio_r: 0.38       # 右腿空中相位比例
gait_phase_offset_l: 0.38    # 左腿相位偏移
gait_phase_offset_r: 0.88    # 右腿相位偏移
gait_cycle: 0.85             # 步态周期 (秒)
```

## 奖励函数 / Reward Functions

主要奖励项包括:

- `track_lin_vel_xy_exp`: 跟踪线性速度 (指数奖励)
- `track_ang_vel_z_exp`: 跟踪角速度 (指数奖励)
- `lin_vel_z_l2`: Z轴速度惩罚
- `ang_vel_xy_l2`: XY轴角速度惩罚
- `energy`: 能量消耗惩罚
- `dof_acc_l2`: 关节加速度惩罚
- `action_rate_l2`: 动作变化率惩罚
- `undesired_contacts`: 非期望接触惩罚
- `body_orientation_l2`: 身体朝向惩罚
- `feet_slide`: 脚部滑动惩罚
- `gait_feet_frc_perio`: 步态脚力周期奖励
- `gait_feet_spd_perio`: 步态脚速周期奖励

## 注意事项 / Notes

1. **硬件要求**: 需要NVIDIA GPU和Isaac Sim环境
2. **依赖项**: isaaclab, rsl_rl, torch, numpy
3. **噪声添加**: 默认启用观测噪声以提高策略鲁棒性
4. **域随机化**: 包含物理参数随机化和外力推动等域随机化策略
5. **课程学习**: 支持地形难度渐进增加

## 许可证 / License

本代码基于BSD-3-Clause许可证发布，原始代码来自RSL-RL、Isaac Lab和Legged Lab项目。

This code is released under BSD-3-Clause license, original code from RSL-RL, Isaac Lab and Legged Lab projects.
