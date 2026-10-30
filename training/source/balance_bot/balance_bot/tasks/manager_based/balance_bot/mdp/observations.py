# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def measured_wheel_vel(env: ManagerBasedRLEnv, action_name: str = "wheel_vel") -> torch.Tensor:
    """Wheel speeds as the ESP32 reports them (hall-sensor estimate of :class:`HoverboardWheelAction`)."""
    return env.action_manager.get_term(action_name).measured_velocity
