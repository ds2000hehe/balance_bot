# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def measured_wheel_vel(env: ManagerBasedRLEnv, action_name: str = "wheel_vel") -> torch.Tensor:
    """Wheel speeds as the ESP32 reports them (hall-sensor estimate of :class:`HoverboardWheelAction`)."""
    return env.action_manager.get_term(action_name).measured_velocity


class FilteredGravity(ManagerTermBase):
    """Gravity direction as the robot estimates it: a complementary filter on a simulated gyro + accelerometer.

    Same filter as ``GravityEstimator`` in the robot's policy node (jetson/.../policy_core.py): the gyro propagates
    the estimate, the accelerometer (specific force, which includes the body's own acceleration) pulls it back with
    time constant ``tau`` and is ignored while its magnitude is more than 3 m/s^2 away from g. So, like on the robot,
    the estimate lags and is pulled off by accelerations: driving the wheels hard reads as a false tilt.

    Per environment at reset: an IMU mounting error (rotation about the wheel axle, up to ``mount_error`` rad) and a
    gyro bias (up to ``gyro_bias`` rad/s per axis). ``accel_noise`` [m/s^2] and ``gyro_noise`` [rad/s] are added every
    step (uniform).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n = env.num_envs
        self.g = torch.zeros(n, 3, device=env.device)
        self.g[:, 2] = -1.0
        self.prev_vel = torch.zeros(n, 3, device=env.device)
        self.initialized = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.mount = torch.zeros(n, device=env.device)
        self.bias = torch.zeros(n, 3, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        p = self.cfg.params
        self.initialized[env_ids] = False
        self.mount[env_ids] = torch.empty_like(self.mount[env_ids]).uniform_(-p.get("mount_error", 0.0), p.get("mount_error", 0.0))
        b = p.get("gyro_bias", 0.0)
        self.bias[env_ids] = torch.empty_like(self.bias[env_ids]).uniform_(-b, b)

    def _mount(self, v: torch.Tensor) -> torch.Tensor:
        """Rotate base-frame vectors into the (slightly misaligned) IMU frame: about the wheel axle (base x)."""
        c, s = torch.cos(self.mount), torch.sin(self.mount)
        return torch.stack((v[:, 0], c * v[:, 1] - s * v[:, 2], s * v[:, 1] + c * v[:, 2]), dim=1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tau: float = 0.5,
        mount_error: float = 0.0,
        gyro_bias: float = 0.0,
        accel_noise: float = 0.0,
        gyro_noise: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        dt = env.step_dt
        vel = asset.data.root_lin_vel_w
        acc_w = torch.where(self.initialized.unsqueeze(1), (vel - self.prev_vel) / dt, torch.zeros_like(vel))
        self.prev_vel = vel.clone()
        acc_w = acc_w.clone()
        acc_w[:, 2] += 9.81  # specific force: what an accelerometer measures
        accel = self._mount(math_utils.quat_apply_inverse(asset.data.root_quat_w, acc_w))
        accel = accel + torch.empty_like(accel).uniform_(-accel_noise, accel_noise)
        gyro = self._mount(asset.data.root_ang_vel_b) + self.bias
        gyro = gyro + torch.empty_like(gyro).uniform_(-gyro_noise, gyro_noise)

        norm = torch.linalg.norm(accel, dim=1, keepdim=True).clamp_min(1e-3)
        g_acc = -accel / norm
        g = self.g - dt * torch.cross(gyro, self.g, dim=1)
        alpha = min(1.0, dt / tau)
        use_acc = (norm - 9.81).abs() < 3.0
        g = torch.where(use_acc, (1.0 - alpha) * g + alpha * g_acc, g)
        g = g / torch.linalg.norm(g, dim=1, keepdim=True).clamp_min(1e-6)
        # first step after a reset: start from the accelerometer, like the robot's filter
        self.g = torch.where(self.initialized.unsqueeze(1), g, g_acc)
        self.initialized[:] = True
        return self.g
