# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def uprightness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Cosine of the base tilt angle, clipped to [0, 1]: 1 when upright, 0 when lying flat or upside down."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)


def upright_gate(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Smooth gate: 0 at tilt cosine <= ``threshold``, rising linearly to 1 when fully upright."""
    return torch.clamp((uprightness(env, asset_cfg) - threshold) / (1.0 - threshold), 0.0, 1.0)


def upright_reward(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Dense reward for standing up, using exp(-tilt^2 / std^2).

    Unlike a squared-gravity penalty, this keeps a useful gradient when the robot is lying on the ground.
    """
    tilt = torch.acos(torch.clamp(-env.scene[asset_cfg.name].data.projected_gravity_b[:, 2], -1.0, 1.0))
    return torch.exp(-torch.square(tilt) / std**2)


def track_lin_vel_upright_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    gate_threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the commanded planar velocity (base frame), only rewarded while the robot is upright.

    Gating stops the policy from collecting tracking reward by dragging itself along the ground while lying down.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(cmd[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / std**2) * upright_gate(env, gate_threshold, asset_cfg)


def track_ang_vel_z_upright_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    gate_threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the commanded yaw rate (world z), only rewarded while the robot is upright."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    error = torch.square(cmd[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-error / std**2) * upright_gate(env, gate_threshold, asset_cfg)


def pitch_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize rocking back and forth: the angular velocity about the wheel axle (base x-axis)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 0])
