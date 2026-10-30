# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stand-up task: get up from lying on the ground, balance, and drive with velocity commands.

Differences to the balance task:

* episodes start anywhere between upright and lying flat on the front / back,
* falling over does not end the episode, so the policy learns to recover,
* a wide upright reward gives a learning signal even while lying down, and velocity tracking is only rewarded
  while upright (see :func:`mdp.track_lin_vel_upright_exp`).

Train the balance task first and fine-tune from it (``--resume``), or train this task from scratch.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .balance_bot_env_cfg import BalanceBotEnvCfg


@configclass
class BalanceBotStandUpEnvCfg(BalanceBotEnvCfg):
    """Stand up from the ground, balance and drive."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # start anywhere from upright to lying flat on the front / back (the robot rests at 90 deg, with the wheels
        # just off the ground); lift a little so a lying body does not start inside the ground
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.01, 0.03),
            "roll": (-0.5 * math.pi, 0.5 * math.pi),
            "yaw": (-math.pi, math.pi),
        }
        # stronger pushes, so the robot also gets knocked over during the episode
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.8, 0.8)}

        # falling over is part of the task now
        self.terminations.fell_over = None
        self.rewards.terminated = None
        self.rewards.alive = None

        # coarse term: gradient while lying down; fine term (inherited, std 0.25): precise balancing
        self.rewards.get_up = RewTerm(func=mdp.upright_reward, weight=2.0, params={"std": 1.0})


@configclass
class BalanceBotStandUpEnvCfg_PLAY(BalanceBotStandUpEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        # smaller scene for play
        self.scene.num_envs = 32
        self.scene.env_spacing = 1.5
        # disable observation noise, keep pushes so the recovery behaviour is visible
        self.observations.policy.enable_corruption = False
