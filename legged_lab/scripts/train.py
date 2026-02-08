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

import argparse  # 用于解析命令行参数

from isaaclab.app import AppLauncher  # 从IsaacLab导入应用启动器

from legged_lab.utils import task_registry  # 导入任务注册表，用于获取环境配置和类
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner  # 导入PPO训练运行器

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip  # 导入命令行参数处理工具

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# 添加任务名称参数
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# 添加环境数量参数
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# 添加随机种子参数
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# 追加RSL-RL相关的命令行参数
cli_args.add_rsl_rl_args(parser)
# 追加AppLauncher相关的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli, hydra_args = parser.parse_known_args()
# 如果任务名称包含"sensor"，则启用相机渲染
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# 启动Omniverse应用程序
app_launcher = AppLauncher(args_cli)
# 获取模拟应用实例
simulation_app = app_launcher.app
# 导入所需的Python标准库
import os
from datetime import datetime

# 导入PyTorch和相关工具
import torch
from isaaclab.utils.io import dump_yaml  # 用于保存YAML配置文件
from isaaclab_tasks.utils import get_checkpoint_path  # 用于获取检查点路径

from legged_lab.envs import *  # noqa:F401, F403  # 导入所有环境类
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # 用于更新RSL-RL配置

# 设置PyTorch相关性能优化选项
torch.backends.cuda.matmul.allow_tf32 = True  # 允许CUDA矩阵乘法使用TF32格式以提高性能
torch.backends.cudnn.allow_tf32 = True  # 允许cuDNN使用TF32格式
torch.backends.cudnn.deterministic = False  # 禁用确定性算法以提高性能
torch.backends.cudnn.benchmark = False  # 禁用cuDNN自动寻找最佳卷积算法


def train():
    # 定义运行器类型注解
    runner: OnPolicyRunner | AmpOnPolicyRunner

    # 获取任务类名
    env_class_name = args_cli.task
    # 从任务注册表获取环境配置和智能体配置
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    # 从任务注册表获取环境类
    env_class = task_registry.get_task_class(env_class_name)

    # 如果指定了环境数量，则覆盖配置中的环境数量
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # 使用命令行参数更新智能体配置
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    # 设置环境场景的随机种子
    env_cfg.scene.seed = agent_cfg.seed

    # 如果使用分布式训练
    if args_cli.distributed:
        # 设置CUDA设备ID
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # 为不同线程设置不同的种子，增加多样性
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    # 创建环境实例
    env = env_class(env_cfg, args_cli.headless)

    # 设置日志根目录
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # 创建当前时间格式的日志目录
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 如果指定了运行名称，则添加到日志目录名中
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    # 根据配置的运行器类名创建运行器实例
    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # 如果配置了从检查点恢复训练
    if agent_cfg.resume:
        # 获取之前训练的检查点路径
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # 加载之前训练的模型
        runner.load(resume_path)

    # 将环境配置和智能体配置保存到日志目录
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 开始训练，指定最大学习迭代次数，并设置随机初始回合长度
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    # 执行训练函数
    train()
    # 关闭模拟应用
    simulation_app.close()
