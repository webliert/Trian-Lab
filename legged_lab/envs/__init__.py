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

from legged_lab.envs.tienkung.Experiment.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.Experiment.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung.Robot.tienkung_env import TienKungEnv     #训练修改,原本训练
# from legged_lab.envs.tienkung.Robot.tienkung_env_69 import TienKungEnv    #训练修改,删除线速度和步态参数
# from legged_lab.envs.tienkung.Robot.tienkung_env_75_old import TienKungEnv     #训练修改，只删除线速度
# from legged_lab.envs.tienkung.Robot.tienkung_env_75 import TienKungEnv    #训练修改，官方删除线速度
# from legged_lab.envs.tienkung.Robot.tienkung_env_45_only_leg import TienKungEnv   #训练修改，只控制下半身

from legged_lab.envs.tienkung.Experiment.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.tienkung.Experiment.walk_cfg_only_leg import (
    TienKungWalkAgentCfg_OnlyLeg,
    TienKungWalkFlatEnvCfg_OnlyLeg,
)
from legged_lab.envs.tienkung.Experiment.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)
# Unitree-RL-Lab style configuration (standard PPO, no AMP)
from legged_lab.envs.tienkung.Experiment.unitree_style_walk_cfg import (
    TienKungUnitreeStyleEnvCfg,
    TienKungUnitreeStyleAgentCfg,
)
# Walk and Stand LSTM configuration
from legged_lab.envs.tienkung.Experiment.walk_and_stand_cfg_lstm import (
    TienKungSawLstmAgentCfg,
    TienKungSawLstmFlatEnvCfg,
)

# Walk and Stand MLP configuration
from legged_lab.envs.tienkung.Experiment.walk_and_stand_cfg_mlp import (
    TienKungSawMlpAgentCfg,
    TienKungSawMlpFlatEnvCfg,
)

# Stand configuration (for fast standing learning with curriculum)
from legged_lab.envs.tienkung.Experiment.stand_cfg import (
    TienKungStandAgentCfg,
    TienKungStandFlatEnvCfg,
)
from legged_lab.envs.tienkung.Robot.tienkung_env_stand import TienKungStandEnv
from legged_lab.envs.tienkung.Robot.tienkung_env_walk_stand_lstm import TienKungWalkAndStandEnvLSTM
from legged_lab.envs.tienkung.Robot.tienkung_env_walk_stand_mlp import TienKungWalkAndStandEnvMLP

from legged_lab.utils.task_registry import task_registry

task_registry.register("walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
task_registry.register("run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
task_registry.register(
    "walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg()
)
task_registry.register(
    "run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg()
)
task_registry.register(
    "walk_only_leg", TienKungEnv, TienKungWalkFlatEnvCfg_OnlyLeg(), TienKungWalkAgentCfg_OnlyLeg()
)
# Register Unitree-RL-Lab style task (standard PPO, no AMP)
task_registry.register(
    "unitree_style_walk", TienKungEnv, TienKungUnitreeStyleEnvCfg(), TienKungUnitreeStyleAgentCfg()
)
# Register Walk and Stand LSTM task
task_registry.register(
    "walk_and_stand_lstm", TienKungWalkAndStandEnvLSTM, TienKungSawLstmFlatEnvCfg(), TienKungSawLstmAgentCfg()
)

# Register Walk and Stand MLP task
task_registry.register(
    "walk_and_stand_mlp", TienKungWalkAndStandEnvMLP, TienKungSawMlpFlatEnvCfg(), TienKungSawMlpAgentCfg()
)

# Register Stand task (fast standing learning, uses TienKungStandEnv without gait)
task_registry.register(
    "stand", TienKungStandEnv, TienKungStandFlatEnvCfg(), TienKungStandAgentCfg()
)
