# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Balance-Bot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_env_cfg:BalanceBotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotPPORunnerCfg",
    },
)

gym.register(
    id="Template-Balance-Bot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_env_cfg:BalanceBotEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotPPORunnerCfg",
    },
)

gym.register(
    id="Template-Balance-Bot-StandUp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_standup_env_cfg:BalanceBotStandUpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotStandUpPPORunnerCfg",
    },
)

gym.register(
    id="Template-Balance-Bot-StandUp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_standup_env_cfg:BalanceBotStandUpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotStandUpPPORunnerCfg",
    },
)

gym.register(
    id="Template-Balance-Bot-Hold-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_env_cfg:BalanceBotHoldEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotPPORunnerCfg",
    },
)

gym.register(
    id="Template-Balance-Bot-Hold-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_env_cfg:BalanceBotHoldEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotPPORunnerCfg",
    },
)

gym.register(
    id="Template-Balance-Bot-RealMass-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_env_cfg:BalanceBotRealMassEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotPPORunnerCfg",
    },
)

gym.register(
    id="Template-Balance-Bot-RealMass-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_env_cfg:BalanceBotRealMassEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotPPORunnerCfg",
    },
)

gym.register(
    id="Template-Balance-Bot-Duty2kg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_env_cfg:BalanceBotDuty2kgEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotPPORunnerCfg",
    },
)

gym.register(
    id="Template-Balance-Bot-Duty2kg-Fast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_env_cfg:BalanceBotDuty2kgFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotFastPPORunnerCfg",
    },
)

gym.register(
    id="Template-Balance-Bot-Duty2kg-Fast-SlowFilter-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_env_cfg:BalanceBotDuty2kgFastSlowFilterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotFastPPORunnerCfg",
    },
)

gym.register(
    id="Template-Balance-Bot-Duty2kg-Coast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_bot_env_cfg:BalanceBotDuty2kgCoastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BalanceBotPPORunnerCfg",
    },
)
