# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TienKung-Lab is an RL-based locomotion control framework for the full-sized TienKung humanoid robot (which won the first Humanoid Robot Half Marathon). It integrates AMP-style rewards with periodic gait rewards on top of IsaacLab, supports Sim2Sim transfer to MuJoCo, and incorporates ray-casting-based sensors (LiDAR, depth camera) for perception. The framework is also validated on the real robot.

**Stack**: IsaacSim 4.5.0, IsaacLab 2.1.0, RSL_RL 2.3.1, Python 3.10, Ubuntu 22.04, MuJoCo 3.3.2 (for sim2sim).

## Installation

```bash
# Install TienKung-Lab (assumes Isaac Lab is already installed via conda)
cd TienKung-Lab
pip install -e .

# Install the rsl_rl fork in-tree
cd rsl_rl
pip install -e .
```

Repository must live **outside** the IsaacLab directory. If Pylance misses extension indexing, extend `python.analysis.extraPaths` in `.vscode/settings.json`.

## Python Execution (conda env requirement)

All Python execution in this project — `python`, `python3`, `pip`, `pip3` — MUST happen with the `tienkung_lab` conda env activated. The env lives under `/home/szxx/Downloads/New_miniconda3/envs/tienkung_lab` and ships the pinned IsaacSim 4.5.0 / IsaacLab 2.1.0 / RSL-RL 2.3.1 / Python 3.10 stack. Running against base conda (or any other env) silently picks up the wrong interpreter and missing/mismatched deps.

Standard prefix for any python/pip command:

```bash
source /home/szxx/Downloads/New_miniconda3/etc/profile.d/conda.sh && \
conda activate tienkung_lab && \
<your-command>
```

This is enforced by a PreToolUse hook at [.claude/hooks/check-conda-tienkung-lab.sh](.claude/hooks/check-conda-tienkung-lab.sh), wired up via `.claude/settings.local.json`. The hook intercepts every Bash tool invocation: if it detects `python`/`python3`/`pip`/`pip3` at a command position (start-of-line or after `;`/`&&`/`||`/`|`/`&`) AND the same shell line does NOT contain the literal substring `conda activate tienkung_lab`, it denies the call with a fix-it message. Commands where `python` appears only inside arguments (`find -name "*python*"`, `grep python file`, `which python`) pass through unaffected.

To enable the hook on a new clone/machine, copy `.claude/hooks/check-conda-tienkung-lab.sh` into your `.claude/settings.local.json` `hooks.PreToolUse` entry — `settings.local.json` itself is gitignored (it carries personal Read allow rules).

## Common Commands

All entry points live in `legged_lab/scripts/`. They auto-launch the Omniverse app via `AppLauncher`. Tasks with `sensor` in the name set `--enable_cameras=True` automatically.

```bash
# Train (AMP PPO)
python legged_lab/scripts/train.py --task=lite_walk --headless --logger=tensorboard --num_envs=4096
python legged_lab/scripts/train.py --task=lite_run  --headless --logger=tensorboard --num_envs=4096
python legged_lab/scripts/train.py --task=lite_swing --headless --logger=tensorboard --num_envs=4096
python legged_lab/scripts/train.py --task=dex_walk  --headless --logger=tensorboard --num_envs=4096
python legged_lab/scripts/train.py --task=dex_run   --headless --logger=tensorboard --num_envs=4096

# Play (loads checkpoint, exports JIT/ONNX policy to logs/.../exported/)
python legged_lab/scripts/play.py --task=lite_walk --num_envs=1
python legged_lab/scripts/play.py --task=lite_run  --num_envs=1

# Visualize AMP motion (loads motion_visualization/<task>.txt)
python legged_lab/scripts/play_amp_animation.py --task=walk --num_envs=1
python legged_lab/scripts/play_amp_animation.py --task=walk_with_sensor --num_envs=1

# Sim2Sim in MuJoCo (uses Exported_policy/{walk,run}.pt by default)
python legged_lab/scripts/sim2sim.py --task walk --policy Exported_policy/walk.pt --duration 100
python legged_lab/scripts/sim2sim.py --task run  --policy Exported_policy/run.pt  --duration 100

# Convert GMR retargeted motion (.pkl) to the txt format used for visualization
python legged_lab/scripts/gmr_data_conversion.py --input_pkl <pkl> --output_txt legged_lab/envs/lite/datasets/motion_visualization/motion.txt

# Convert visualization txt into expert motion for AMP training
python legged_lab/scripts/play_amp_animation.py --task=walk --num_envs=1 --save_path legged_lab/envs/lite/datasets/motion_amp_expert/motion.txt --fps 30.0
```

Loggers supported: `tensorboard`, `wandb`, `neptune`, `swanlab` (selected via `--logger`). Resume via `--resume --load_run <run_name> --checkpoint model_<N>.pt`. Use `--distributed` for multi-GPU.

## Code Formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

Pre-commit runs: `black --line-length 120 --preview`, `flake8` (max line 120, max complexity 18), `isort --profile black`, `pyupgrade --py37-plus`, `codespell`, and license-header insertion from `.github/LICENSE_HEADER.txt`. All new Python files need the standard multi-project BSD-3-Clause header (see `LICENSE_HEADER.txt`).

## Architecture

### Repository layout

```
TienKung-Lab/
├── legged_lab/                  # Main Isaac Lab extension (this repo's contribution)
│   ├── envs/                    # VecEnv subclasses + per-task configs
│   ├── mdp/                     # Reward terms, event randomization, symmetry, curriculums
│   ├── scripts/                 # Entry points (train, play, sim2sim, play_amp_animation, …)
│   ├── assets/                  # Robot USD/URDF/MJCF (tienkung2_lite, tienkung2_pro, EVT2, tiangong_dex)
│   ├── sensors/                 # Camera (TiledCamera) and LiDAR configs
│   ├── terrains/                # Terrain generator cfgs (ROUGH_TERRAINS_CFG, GRAVEL_TERRAINS_CFG)
│   └── utils/                   # task_registry, CLI args, keyboard, scene assembly
├── rsl_rl/                      # In-tree fork of leggedrobotics/rsl_rl with AMP support
│   └── rsl_rl/
│       ├── algorithms/          # PPO, AMPPPO, Distillation
│       ├── runners/             # OnPolicyRunner, AmpOnPolicyRunner
│       ├── modules/             # ActorCritic, Discriminator, EmpiricalNormalization, RND, StudentTeacher
│       ├── storage/             # RolloutStorage, ReplayBuffer
│       └── utils/               # AMPLoader, AMPLoaderDisplay, motion loaders
├── gmr/                         # GMR motion-retargeting outputs (.pkl) and converted txts
├── Exported_policy/             # Pretrained JIT policies (walk.pt, run.pt)
├── docs/                        # GIFs and figures
└── logs/                        # Training runs, tensorboard, swanlab outputs
```

### Task registry

Tasks are registered in `legged_lab/envs/__init__.py` against the global `task_registry` (`legged_lab/utils/task_registry.py`). Each entry maps `name → (VecEnv class, EnvCfg, AgentCfg)`. The five registered tasks:

| Name | Env class | Variant |
| --- | --- | --- |
| `lite_walk` | `TienKungEnv` (`legged_lab/envs/lite/tienkung_env.py`) | TienKung 2 Lite, walk, AMP |
| `lite_run` | `TienKungEnv` | TienKung 2 Lite, run, AMP |
| `lite_swing` | `TienKungSwingEnv` (`legged_lab/envs/lite/robot/tienkung_env.py`) | TienKung 2 Lite, swing (prefab) |
| `dex_walk` | `DexEnv` (`legged_lab/envs/dex/dex_env.py`) | TienKung 2 Pro / Dex |
| `dex_run` | `DexEnv` | TienKung 2 Pro / Dex |

`scripts/train.py` and `scripts/play.py` resolve `--task` to `(env_class, env_cfg, agent_cfg)` via `task_registry.get_cfgs(name)`, then dispatch to the runner class determined by `agent_cfg.runner_class_name` (e.g. `OnPolicyRunner` or `AmpOnPolicyRunner`).

### Environment construction

`BaseEnv` (`legged_lab/envs/base/base_env.py`) is the **non-AMP** VecEnv base. It is overridden by `TienKungEnv` in `legged_lab/envs/lite/tienkung_env.py` to add:

- **Gait parameterization** (`self.gait_phase`, `gait_cycle`, `phase_ratio`, `phase_offset`) — `_calculate_gait_para()` updates phase from `episode_length_buf * step_dt / gait_cycle`, with per-env randomization if `gait.gait_cycle_lower/upper` is set.
- **Joint-body mapping caches** (`left_leg_ids`, `right_leg_ids`, `left_arm_ids`, `right_arm_ids`, `feet_body_ids`, `elbow_body_ids`, `ankle_joint_ids`) — used by rewards and AMP observation extraction.
- **AMP motion playback** via `AMPLoaderDisplay` (`rsl_rl/utils/motion_loader_for_display.py`) — `visualize_motion(time)` writes joint pos/vel + root state, advances the sim, and returns the AMP observation tensor (`get_amp_obs_for_expert_trans()` is the live-robot equivalent).
- **Optional GridAdaptiveCurriculum** for velocity commands (`legged_lab/envs/base/command_curriculum.py`) — enabled when `cfg.command_curriculum_cfg` is set; tracks per-env success on `(lin_vel_x, lin_vel_y, ang_vel_z)` bins and resamples when threshold is met.
- **DelayBuffer** for action delay and **CircularBuffer** for actor/critic observation history (`actor_obs_history_length`, `critic_obs_history_length`).
- **Optional sensors** added in scene: height scanner (RayCaster), LiDAR, depth camera (`TiledCamera`).

`step()` runs `cfg.sim.decimation` physics sub-steps per RL step (`step_dt = dt * decimation`), applies domain-randomization events (`startup` / `reset` / `interval` modes), checks reset (contact-force termination + timeout), then computes the next observation.

### Configs

- `legged_lab/envs/base/base_config.py` — primitive `@configclass` building blocks: `BaseSceneCfg`, `RobotCfg`, `RewardCfg`, `DomainRandCfg` (`EventCfg` with `physics_material`, `add_base_mass`, `reset_base`, `reset_robot_joints`, `push_robot`), `CommandsCfg`, `NormalizationCfg`, `ObsScalesCfg`, `NoiseCfg`, `SimCfg`.
- `legged_lab/envs/base/base_env_config.py` — `BaseEnvCfg` (composes the primitives) and `BaseAgentCfg` (RSL-RL PPO + ActorCritic defaults: hidden dims `[512, 256, 128]`, lr `1e-3`, gamma `0.99`, lam `0.95`, KL `0.01`, 5 epochs, 4 minibatches, 24 steps/env).
- `legged_lab/envs/lite/walk_cfg.py` / `run_cfg.py` / `walk_with_sensor_cfg.py` / `run_with_sensor_cfg.py` — concrete `TienKungWalkFlatEnvCfg` etc. that subclass / replace these and add `LiteRewardCfg` plus a `GaitCfg`. `TienkungEventCfg` adds `randomize_pd_gains`, `randomize_apply_external_force_torque`, `randomize_rigid_body_com`, `randomize_joint_params`.
- `legged_lab/envs/lite/experiment/swing_cfg.py` — `lite_swing` task config.
- `legged_lab/envs/dex/{walk,run}_cfg.py` — dexterous variants.

### Rewards & symmetry

- `legged_lab/mdp/rewards.py` — task reward terms: `track_lin_vel_xy_yaw_frame_exp`, `track_ang_vel_z_world_exp`, `lin_vel_z_l2`, `ang_vel_xy_l2`, `energy`, `joint_acc_l2`, `action_rate_l2`, `undesired_contacts`, `body_orientation_l2`, `flat_orientation_l2`, `feet_slide`, `body_force`, `feet_too_near_humanoid`, `feet_stumble`, `joint_pos_limits`, `joint_deviation_l1`, `gait_feet_frc_perio`, `gait_feet_spd_perio`, `gait_feet_frc_support_perio`, `ankle_torque`, `ankle_action`, `hip_roll_action`, `hip_yaw_action`, `feet_y_distance`, `hip_roll_vel`, `alive_reward`, `stand_still`, `stand_still_exp`, `stand_still_vel`, `stand_still_feet_motion_penalty`, `stand_still_double_support`, `is_terminated`, plus an `episode_progress_gate` helper.
- `legged_lab/mdp/events.py` — startup / reset / interval randomization (mass, friction, PD gains, COM, external force, joint params, etc.).
- `legged_lab/mdp/symmetryLite.py` — left/right mirror augmentation for observations (`mirror_observation_policy`, `mirror_observation_critic`) and actions (`mirror_actions`), wired up as `data_augmentation_func_g1` for RSL-RL's `RslRlSymmetryCfg`. `ACTION_NUM = 20` (matches 20-DoF TienKung 2 Lite).
- `legged_lab/mdp/curriculums.py` — `command_levels_lin_vel`, `command_levels_ang_vel`, and `grid_adaptive_command_curriculum` (uses `GridAdaptiveCurriculum`).
- `legged_lab/mdp/symmetryDex.py` — mirror augmentation for the Dex/Pro variant.

### RSL-RL integration

`rsl_rl/` is an in-tree fork with AMP support. Key entry points:

- `rsl_rl/rsl_rl/runners/on_policy_runner.py` — PPO runner.
- `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` — AMP-PPO runner (uses `AMPLoader` from `rsl_rl/utils/motion_loader.py`).
- `rsl_rl/rsl_rl/algorithms/amp_ppo.py` — AMP discriminator loss added to PPO.
- `rsl_rl/rsl_rl/modules/discriminator.py` — motion discriminator (MLP).
- `rsl_rl/rsl_rl/utils/motion_loader.py` / `motion_loader_for_display.py` — txt motion file loaders. `AMPLoader` is the training-time expert buffer; `AMPLoaderDisplay` is the playback loader used by `play_amp_animation.py` and `TienKungEnv.visualize_motion`.

AMP expert motion format: rows of `[dof_pos(20), dof_vel(20), end-effector positions(12)]`. Visualization motion format: `[root_pos(3), euler(3), dof_pos(20), root_lin_vel(3), root_ang_vel(3), dof_vel(20)]`. The two-stage conversion (GMR pkl → visualization txt → expert txt) lives in `legged_lab/scripts/gmr_data_conversion.py` and `legged_lab/scripts/play_amp_animation.py --save_path`.

### Sim2Sim (MuJoCo)

`legged_lab/scripts/sim2sim.py` loads a JIT policy and runs the robot under MuJoCo for cross-simulation validation. Hard-coded mappings:

- `SimToSimCfg.sim` mirrors the training `BaseEnvCfg.sim` and observation layout (20 actions, 75-dim per-step observation × 10 history).
- `SimToSimCfg.robot` carries the gait parameters (air ratio, phase offset, cycle) selected by `--task walk|run`. These must match the training `GaitCfg` of the corresponding task.
- `MujocoRunner.mujoco_to_isaac_idx` / `isaac_to_mujoco_idx` define the joint-order remap (joints in MuJoCo MJCF are interleaved left/right, not leg/arm-blocked like the policy expects).
- `MujocoRunner.get_obs()` builds the 75-dim observation: `[ang_vel(3), projected_gravity(3), command_vel(3), dof_pos(20), dof_vel(20), action(20), sin(2π·phase)(2), cos(2π·phase)(2), phase_ratio(2)]`.
- The pynput keyboard listener (`8/2/4/6/7/9`) adjusts `command_vel` at runtime.

The MuJoCo model path defaults to `legged_lab/assets/tienkung2_lite/mjcf/tienkung.xml`; the exported policy defaults to `Exported_policy/<task>.pt`.

### Robot assets

- `legged_lab/assets/tienkung2_lite/` — 20-DoF TienKung 2 Lite. `tienkung.py` defines `TIENKUNG2LITE_CFG` (`ArticulationCfg`) with three `ImplicitActuatorCfg` groups: `legs`, `feet`, `arms`. `__init__.py` re-exports this as the canonical Lite config. USD lives in `usd/`, MJCF in `mjcf/`, URDF+meshes in `urdf/` and `meshes/`.
- `legged_lab/assets/tienkung2_pro/mjcf/` — TienKung 2 Pro (Dex variant, no Isaac cfg shipped).
- `legged_lab/assets/tiangong_dex_urdf_EVT2/` and `legged_lab/assets/EVT2/` — URDF + meshes for the Dex robot and the EVT2 hardware reference.

### Visualizing motion with sensors

`walk_with_sensor` and `run_with_sensor` task configs enable `height_scanner`, `lidar`, and/or `depth_camera` on the robot. Camera rendering is enabled at startup by `train.py` / `play.py` / `play_amp_animation.py` when the task name contains `sensor`. The depth-camera data is flattened and concatenated to actor/critic observations in `TienKungEnv.compute_observations()`.

### CLI args

- `legged_lab/utils/cli_args.py` — `add_rsl_rl_args()` adds `--max_iterations`, `--experiment_name`, `--run_name`, `--resume`, `--load_run`, `--checkpoint`, `--logger`, `--log_project_name`, `--swanlab_project`, `--distributed`. `update_rsl_rl_cfg()` applies them to the agent config.
- `legged_lab/utils/task_registry.py` — global `task_registry` (singleton) used by all scripts.
- `legged_lab/utils/env_utils/scene.py` — `SceneCfg(InteractiveSceneCfg)` assembles terrain, robot, contact sensor, lights, and the optional height-scanner / LiDAR / depth-camera sensors.
- `legged_lab/utils/keyboard.py` — `Keyboard(DeviceBase)` for in-sim keyboard (R = reset envs).
