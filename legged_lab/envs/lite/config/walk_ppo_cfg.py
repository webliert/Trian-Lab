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

"""Vanilla-PPO walker cfg — same env / env-cfg / gait / reward as :mod:`walk_cfg`.

This file exists so that ``lite_walk`` (AMP-PPO) and ``lite_walk_ppo`` (vanilla
PPO) can be trained side-by-side for direct comparison. The only difference vs.
``TienKungWalkAgentCfg`` is the algorithm + runner selection:

* ``runner_class_name``: ``"AmpOnPolicyRunner"`` -> ``"OnPolicyRunner"``
* ``algorithm.class_name``: ``"AMPPPO"`` -> ``"PPO"``
* All six AMP-specific keys (``amp_*`` / ``min_normalized_std``) are dropped.

The env cfg (``TienKungWalkFlatEnvCfg``), gait (``GaitCfg``) and reward class
(``LiteRewardCfg``) are imported and re-used as-is. ``TienKungWalkEnv`` will
still construct an ``AMPLoaderDisplay`` in ``__init__`` (the field is on
``TienKungWalkFlatEnvCfg`` and points at ``motion_visualization/walk.txt`` as a
benign placeholder), but the vanilla PPO runner never calls the AMP hooks
(``get_amp_obs_for_expert_trans`` / ``reset_env_ids``), so the loader is dead
weight at runtime.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TienKungWalkPPOAgentCfg(RslRlOnPolicyRunnerCfg):
    """Vanilla PPO agent cfg for the ``lite_walk_ppo`` task.

    Mirrors :class:`legged_lab.envs.lite.config.walk_cfg.TienKungWalkAgentCfg`
    (same hyperparameters, same ActorCritic architecture) but drops the
    discriminator/motion-file machinery.
    """

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
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
        symmetry_cfg=None,  # keep walk_cfg behaviour (no mirror loss)
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 500
    runner_class_name = "OnPolicyRunner"
    experiment_name = "walk_ppo"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk_ppo"
    wandb_project = "walk_ppo"
    swanlab_project = "walk_ppo"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
