# StandAndWalk控制器实现文档

## 概述

本实现基于您提供的论文描述，完成了StandAndWalk (SaW) 控制器的所有功能。

## 实现的功能

### 1. StandAndWalk控制器架构

**Actor-Critic模块 (`rsl_rl/rsl_rl/modules/actor_critic_saw.py`)**
- (64, 64) 双层LSTM循环神经网络
- 输入: 机器人状态(关节速度、位置、躯干方向) + 用户命令 cu=[cx, cy, cyaw]
- 输出: 20个关节空间的PD设定点
- 运行频率: 50Hz (SaW控制器), 2kHz (PD控制器)
- 训练算法: PPO + 镜像损失

### 2. 训练流程

**命令类别采样 (`legged_lab/envs/tienkung/Robot/tienkung_env_walk_stand.py`)**
- 五种命令类别:
  1. Standing (站立): cu = [0, 0, 0]
  2. Walking in sagittal plane (矢状面行走): cx变化
  3. Walking laterally (侧向行走): cy变化
  4. Rotating in place (原地旋转): cyaw变化
  5. Omnidirectional walking (全向行走): cx, cy, cyaw同时变化

**命令重采样**
- 每2-6秒重新采样一次命令
- 均匀分布的五种类别选择

**命令范围**
- cx: [-0.5, 2.0] m/s (前后方向)
- cy: [-0.5, 0.5] m/s (左右方向)
- cyaw: [-0.5, 0.5] rad/s (偏航角速度)

### 3. 随机推力

**扰动增强 (`SaWRandomPushConfig`)**
- 每帧1%概率受到随机推力
- 推力范围: 200N到800N
- 持续时间: 单个timestep (20ms)
- 方向: 360度均匀分布

### 4. 对称性增强

**镜像损失 (`legged_lab/mdp/symmetry_augmentation.py`)**
- `symmetry_augment_standing`: 站立任务的对称性增强
- `symmetry_augment_walking`: 行走任务的对称性增强
- `simple_symmetry_augment`: 简化的对称性增强

## 关键文件

1. **控制器实现**: `rsl_rl/rsl_rl/modules/actor_critic_saw.py`
2. **环境实现**: `legged_lab/envs/tienkung/Robot/tienkung_env_walk_stand.py`
3. **配置文件**: `legged_lab/envs/tienkung/Experiment/walk_and_stand_cfg.py`
4. **对称性增强**: `legged_lab/mdp/symmetry_augmentation.py`

## 配置参数

### SaW命令配置 (`SaWCommandConfig`)
```python
COMMAND_CATEGORIES = [
    "standing",           # 站立
    "sagittal_walk",     # 矢状面行走
    "lateral_walk",      # 侧向行走
    "rotation",           # 原地旋转
    "omnidirectional",   # 全向行走
]

COMMAND_RANGES = {
    "cx": (-0.5, 2.0),    # m/s
    "cy": (-0.5, 0.5),    # m/s
    "cyaw": (-0.5, 0.5),  # rad/s
}

resampling_time_range = (2.0, 6.0)  # seconds
```

### 随机推力配置 (`SaWRandomPushConfig`)
```python
enable = True
push_probability = 0.01  # 1%
force_range = (200.0, 800.0)  # N
force_duration_steps = 1  # 1 step = 20ms
force_angle_range = (0.0, 2 * math.pi)  # 360度
```

## 使用方法

### 训练
```bash
python legged_lab/scripts/train.py --task TienKungWalkAndStand --headless
```

### 推理
```bash
python legged_lab/scripts/play.py --policy_path Exported_policy/walk_and_stand.pt
```

## 注意事项

1. 确保 `actor_obs_history_length` 设置合理（建议10-15）
2. SaW控制器运行在50Hz，而PD控制器运行在2kHz
3. 镜像损失系数建议在0.1左右
4. 随机推力有助于提高扰动 rejection 能力
