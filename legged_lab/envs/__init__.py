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


# import different env，每个文件用唯一类名，无需别名
from legged_lab.envs.dex.dex_env import DexEnv
from legged_lab.envs.lite.env.walk_env import TienKungWalkEnv
from legged_lab.envs.lite.env.walk_stand_env import TienKungWalkStandEnv
from legged_lab.envs.lite.env.swing_env import TienKungSwingEnv
from legged_lab.envs.lite.env.carry_env import TienKungCarryEnv

# import tienkung lite different config config
from legged_lab.envs.lite.config.walk_cfg import TienKungWalkAgentCfg, TienKungWalkFlatEnvCfg
from legged_lab.envs.lite.config.walk_ppo_cfg import TienKungWalkPPOAgentCfg, TienKungWalkPPOFlatEnvCfg
from legged_lab.envs.lite.config.walk_stand_cfg import TienKungWalkStandAgentCfg, TienKungWalkStandFlatEnvCfg
from legged_lab.envs.lite.config.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.lite.config.walk_with_sensor_cfg import TienKungWalkWithSensorAgentCfg, TienKungWalkWithSensorFlatEnvCfg
from legged_lab.envs.lite.config.run_with_sensor_cfg import TienKungRunWithSensorAgentCfg, TienKungRunWithSensorFlatEnvCfg
from legged_lab.envs.lite.config.swing_cfg import TienKungSwingAgentCfg, TienKungSwingFlatEnvCfg
from legged_lab.envs.lite.config.carry_cfg import TienKungCarryAgentCfg, TienKungCarryFlatEnvCfg

# import tienkung dex different config config
from legged_lab.envs.dex.run_cfg import DexRunAgentCfg, DexRunFlatEnvCfg
from legged_lab.envs.dex.walk_cfg import DexWalkAgentCfg, DexWalkFlatEnvCfg


from legged_lab.utils.task_registry import task_registry

# lite_walk / lite_run share the same TienKungWalkEnv — they only differ in cfg.
task_registry.register("lite_walk",              TienKungWalkEnv,      TienKungWalkFlatEnvCfg,             TienKungWalkAgentCfg)
task_registry.register("lite_walk_ppo",          TienKungWalkEnv,      TienKungWalkPPOFlatEnvCfg,          TienKungWalkPPOAgentCfg)
task_registry.register("lite_walk_stand",        TienKungWalkStandEnv, TienKungWalkStandFlatEnvCfg,        TienKungWalkStandAgentCfg)
task_registry.register("lite_run",               TienKungWalkEnv,      TienKungRunFlatEnvCfg,              TienKungRunAgentCfg)
task_registry.register("lite_swing",             TienKungSwingEnv,     TienKungSwingFlatEnvCfg,            TienKungSwingAgentCfg)
task_registry.register("lite_carry",             TienKungCarryEnv,     TienKungCarryFlatEnvCfg,            TienKungCarryAgentCfg)
task_registry.register("lite_walk_with_sensor",  TienKungWalkEnv,      TienKungWalkWithSensorFlatEnvCfg,   TienKungWalkWithSensorAgentCfg)
task_registry.register("lite_run_with_sensor",   TienKungWalkEnv,      TienKungRunWithSensorFlatEnvCfg,    TienKungRunWithSensorAgentCfg)
task_registry.register("dex_walk",               DexEnv,               DexWalkFlatEnvCfg,                  DexWalkAgentCfg)
task_registry.register("dex_run",                DexEnv,               DexRunFlatEnvCfg,                   DexRunAgentCfg)
