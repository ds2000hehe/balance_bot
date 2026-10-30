# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wheel actions: velocity targets with a random control delay (IMU -> ROS -> policy -> ESP32), optionally driven
through a model of the real ESP32 + BLDC driver + hoverboard motor (:mod:`.hoverboard`)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointVelocityAction, JointVelocityActionCfg
from isaaclab.utils import configclass

from .hoverboard import HoverboardDrive, HoverboardDriveCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedJointVelocityAction(JointVelocityAction):
    """Joint velocity targets that reach the motors a random number of physics steps late.

    Each environment draws its delay (in physics steps) from ``cfg.delay_range`` at reset. A policy trained with
    zero delay acts like a very high-gain controller and becomes unstable with the few tens of milliseconds a real
    sensor -> computer -> motor driver loop adds.
    """

    cfg: DelayedJointVelocityActionCfg

    def __init__(self, cfg: DelayedJointVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._decimation = env.cfg.decimation
        if not 0 <= cfg.delay_range[0] <= cfg.delay_range[1] <= 2 * self._decimation:
            raise ValueError(f"delay_range must be within [0, {2 * self._decimation}] physics steps")
        # processed targets of the current and the two previous policy steps
        self._history = torch.zeros(3, self.num_envs, self.action_dim, device=self.device)
        self._delay = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self._substep = 0
        self.reset()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._history = torch.roll(self._history, 1, dims=0)
        self._history[0] = self._processed_actions
        self._substep = 0

    def _delayed_target(self) -> torch.Tensor:
        # the newest target that is at least `delay` physics steps old
        k = self._substep
        self._substep += 1
        return torch.where(
            self._delay <= k,
            self._history[0],
            torch.where(self._delay <= k + self._decimation, self._history[1], self._history[2]),
        )

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self._delayed_target(), joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._history[:, env_ids] = 0.0
        lo, hi = self.cfg.delay_range
        self._delay[env_ids] = torch.randint(lo, hi + 1, self._delay[env_ids].shape, device=self.device)


@configclass
class DelayedJointVelocityActionCfg(JointVelocityActionCfg):
    """Configuration for :class:`DelayedJointVelocityAction`."""

    class_type: type = DelayedJointVelocityAction

    delay_range: tuple[int, int] = (0, 6)
    """Range of the control delay in physics steps (inclusive), drawn per environment at reset."""


class HoverboardWheelAction(DelayedJointVelocityAction):
    """Delayed wheel speed targets, executed by a model of the ESP32 speed loop, BLDC driver and hoverboard motor.

    The model computes the wheel torques every physics step (the wheel joints need zero stiffness and damping).
    :attr:`measured_velocity` is the firmware's hall-sensor speed estimate, which the policy observes.
    """

    cfg: HoverboardWheelActionCfg

    def __init__(self, cfg: HoverboardWheelActionCfg, env: ManagerBasedEnv):
        self.drive = None
        super().__init__(cfg, env)
        self.drive = HoverboardDrive(cfg.drive, self.num_envs, self.action_dim, self.device)
        self._physics_dt = env.physics_dt
        # PhysX wraps the angle of a joint without limits at +-2 pi; the hall sensors need a continuous angle
        self._prev_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._angle = self._prev_pos.clone()

    @property
    def measured_velocity(self) -> torch.Tensor:
        """Wheel speeds as the ESP32 measures them [rad/s], in joint coordinates."""
        return self.drive.velocity

    def _update_sensors(self) -> torch.Tensor:
        pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        delta = torch.remainder(pos - self._prev_pos + math.pi, 2.0 * math.pi) - math.pi
        self._prev_pos = pos
        prev_angle = self._angle
        self._angle = self._angle + delta
        self.drive.update_sensors(self._angle, prev_angle, self._physics_dt)
        return self._asset.data.joint_vel[:, self._joint_ids]

    def drive_wheels(self, target: torch.Tensor):
        """Run the drive model for one physics step with wheel speed ``target`` (joint coordinates)."""
        omega = self._update_sensors()
        torque = self.drive.torque(target, omega, self._physics_dt)
        self._asset.set_joint_effort_target(torque, joint_ids=self._joint_ids)

    def drive_wheels_open_loop(self, duty: torch.Tensor):
        """Run the drive model for one physics step with a raw duty (the firmware's "D" test command)."""
        omega = self._update_sensors()
        self._asset.set_joint_effort_target(self.drive.open_loop_torque(duty, omega), joint_ids=self._joint_ids)

    def apply_actions(self):
        target = self._delayed_target()
        if self.cfg.control == "duty":
            # duty = target / MOTOR_MAX_RAD_S: the speed loop's feed-forward alone (same action scale as "speed")
            self.drive_wheels_open_loop(torch.clamp(target / self.drive.cfg.max_rad_s, -1.0, 1.0))
        else:
            self.drive_wheels(target)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self.drive is not None:
            self.drive.reset(env_ids)


@configclass
class HoverboardWheelActionCfg(DelayedJointVelocityActionCfg):
    """Configuration for :class:`HoverboardWheelAction`."""

    class_type: type = HoverboardWheelAction

    drive: HoverboardDriveCfg = HoverboardDriveCfg()

    control: str = "speed"
    """"speed": the ESP32 runs its PI speed loop on the hall speed estimate ("V" command). "duty": the action is
    sent as a motor duty (``target / max_rad_s``, the firmware's "D" command), with no speed loop."""
