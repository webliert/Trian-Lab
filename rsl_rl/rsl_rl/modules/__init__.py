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

"""
RSL-RL Modules

该模块包含强化学习的策略和价值函数网络模块。
This module contains policy and value function network modules for reinforcement learning.
"""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .actor_critic_saw import ActorCriticSaW, ActorCriticRecurrentSaW
from .discriminator import Discriminator
from .normalizer import EmpiricalNormalization
from ..utils import Normalizer
from .rnd import RandomNetworkDistillation
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "ActorCriticSaW",  # StandAndWalk控制器
    "ActorCriticRecurrentSaW",
    "Discriminator",
    "EmpiricalNormalization",
    "Normalizer",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",
]
