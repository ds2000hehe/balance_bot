# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Model of the real wheel drive: ESP32 firmware + hall-sensored BLDC driver + hoverboard hub motor.

The policy's wheel speed targets go to the wheel controller (arduino/balance_bot_atmega), which closes a PI speed loop on a
speed estimate from the motor's hall sensors and drives a BLDC driver board with PWM + DIR only (BRAKE and STOP are
not used: the driver changes direction and speed quickly by itself). This module reproduces that chain per physics step so the policy trains against the real behaviour:

* **Speed estimate**: 90 hall counts per wheel turn (15 pole pairs x 6 states) with hall placement errors, speed
  from the time between hall edges, bounded between edges and zero after ``standstill_s`` without an edge. This is
  also what the policy observes as wheel speed, like on the robot.
* **Firmware speed loop** (same constants as the sketch): ``duty = target / max_rad_s + kp * e + ki * integral(e)``,
  minimum duty kick, anti-windup.
* **Driver + motor**: the driver applies ``duty x battery voltage`` (sign = DIR) to the motor and drives current both
  ways, so it speeds a wheel up and slows it down: current ``(duty * V - ke * omega) / R``, limited to the driver's
  current limit. Duties in the dead zone switch the output off (the wheel coasts).

Battery voltage, motor constants, resistance, current limit, dead zone, friction and hall errors are randomized per
environment at reset.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass


@configclass
class HoverboardDriveCfg:
    """Parameters of the wheel drive. Ranges are sampled uniformly per environment at reset."""

    # -- hall sensors
    counts_per_rev: int = 90
    """Hall counts per wheel turn: 6 x pole pairs (hoverboard motors: 15 pole pairs)."""
    hall_error: float = 0.08
    """Max placement error of each hall edge, as a fraction of one count."""

    # -- firmware (keep equal to arduino/balance_bot_atmega/balance_bot_atmega.ino)
    kp: float = 0.01
    ki: float = 0.2
    max_rad_s: float = 60.0
    """MOTOR_MAX_RAD_S: feed-forward scale of the speed loop."""
    min_duty: float = 0.0
    """MIN_DUTY: kick added in the target's direction (0 = let the integral overcome the dead zone)."""
    standstill_s: float = 0.1
    max_measured_rad_s: float = 100.0
    """Speed estimates beyond this are clipped (two hall edges microseconds apart would give absurd speeds)."""
    max_target_rad_s: float = 30.0

    # -- driver + motor
    battery_voltage_range: tuple[float, float] = (33.0, 41.0)
    """36 V hoverboard pack (10s Li-ion), empty to full [V]."""
    ke_range: tuple[float, float] = (0.55, 0.65)
    """Back-EMF constant [V s/rad], equal to the torque constant [N m/A] (hoverboard motors: ~16 RPM/V)."""
    resistance_range: tuple[float, float] = (0.3, 0.6)
    """Winding resistance (phase to phase) plus wiring [ohm]."""
    current_limit_range: tuple[float, float] = (12.0, 18.0)
    """Driver current limit [A]."""
    dead_zone_range: tuple[float, float] = (0.0, 0.06)
    """Duty below which the driver does not drive. The ZS-X11H responds from ~0.5/255 with the wheel in the air; the
    wider range covers friction and cogging on the ground and keeps the policy calmer (trained with (0, 0.01), the
    policy wobbled more)."""
    friction_range: tuple[float, float] = (0.0, 0.3)
    """Coulomb friction of the hub motor [N m]."""


class HoverboardDrive:
    """Batched drive model for ``num_envs x num_wheels`` wheels (wheel speeds in joint coordinates)."""

    def __init__(self, cfg: HoverboardDriveCfg, num_envs: int, num_wheels: int, device: str):
        self.cfg = cfg
        self.device = device
        shape = (num_envs, num_wheels)
        zeros = lambda: torch.zeros(shape, device=device)  # noqa: E731
        # hall position state (in counts) and speed estimate
        self.time = torch.zeros(num_envs, 1, device=device)
        self.count = torch.zeros(shape, dtype=torch.long, device=device)
        self.last_edge_t = zeros()
        self.prev_count = torch.zeros(shape, dtype=torch.long, device=device)
        self.prev_edge_t = zeros()
        self.last_dir = torch.zeros(shape, dtype=torch.long, device=device)  # direction of the last counted edge(s)
        self.velocity = zeros()  # firmware speed estimate [rad/s]
        self.integral = zeros()
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # randomized parameters
        self.edge_offset = torch.zeros(num_envs, num_wheels, 6, device=device)  # hall edge position within a count
        self.voltage, self.ke, self.resistance = zeros(), zeros(), zeros()
        self.current_limit, self.dead_zone, self.friction = zeros(), zeros(), zeros()
        self.reset()

    def _uniform(self, rng: tuple[float, float], n: int, like: torch.Tensor) -> torch.Tensor:
        return torch.empty((n,) + like.shape[1:], device=self.device).uniform_(*rng)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        n = self.voltage[env_ids].shape[0]
        c = self.cfg
        # same battery for both wheels, separate motors
        self.voltage[env_ids] = self._uniform(c.battery_voltage_range, n, self.voltage[:, :1]).expand(-1, self.voltage.shape[1])
        self.ke[env_ids] = self._uniform(c.ke_range, n, self.ke)
        self.resistance[env_ids] = self._uniform(c.resistance_range, n, self.resistance)
        self.current_limit[env_ids] = self._uniform(c.current_limit_range, n, self.current_limit)
        self.dead_zone[env_ids] = self._uniform(c.dead_zone_range, n, self.dead_zone)
        self.friction[env_ids] = self._uniform(c.friction_range, n, self.friction)
        self.edge_offset[env_ids] = 0.5 + self._uniform((-c.hall_error, c.hall_error), n, self.edge_offset)
        self.integral[env_ids] = 0.0
        self.velocity[env_ids] = 0.0
        self.last_dir[env_ids] = 0
        self.time[env_ids] = 0.0
        self.initialized[env_ids] = False

    # -- hall sensors

    def _counts(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hall count at joint position ``pos``; also returns the position in counts."""
        x = pos * (self.cfg.counts_per_rev / (2.0 * math.pi))
        m = torch.floor(x)
        edge = torch.gather(self.edge_offset, 2, torch.remainder(m.long(), 6).unsqueeze(-1)).squeeze(-1)
        return m.long() + (x - m >= edge).long(), x

    def _edge_position(self, count: torch.Tensor) -> torch.Tensor:
        """Position (in counts) of the edge where the count becomes ``count``."""
        m = count - 1
        edge = torch.gather(self.edge_offset, 2, torch.remainder(m, 6).unsqueeze(-1)).squeeze(-1)
        return m.float() + edge

    def update_sensors(self, pos: torch.Tensor, prev_pos: torch.Tensor, dt: float):
        """Advance time by ``dt`` while the wheels moved from ``prev_pos`` to ``pos`` and update the speed estimate."""
        # keep the clocks small: float32 edge timing must stay well below a millisecond
        shift = torch.floor(self.time / 100.0) * 100.0
        if (shift > 0).any():
            self.time -= shift
            self.last_edge_t -= shift
            self.prev_edge_t -= shift
        t0 = self.time.clone()
        self.time += dt
        t1 = self.time
        count, x1 = self._counts(pos)
        new_envs = ~self.initialized
        if new_envs.any():
            self.count[new_envs] = count[new_envs]
            self.prev_count[new_envs] = count[new_envs]
            self.last_edge_t[new_envs] = t1.expand_as(self.last_edge_t)[new_envs]
            self.prev_edge_t[new_envs] = t1.expand_as(self.prev_edge_t)[new_envs]
            self.initialized[:] = True
        moved = count != self.count
        if moved.any():
            # time of the latest edge, interpolated within the step
            _, x0 = self._counts(prev_pos)
            boundary = torch.where(count > self.count, self._edge_position(count), self._edge_position(count + 1))
            frac = torch.clamp((boundary - x0) / torch.where((x1 - x0).abs() > 1e-9, x1 - x0, torch.ones_like(x1)), 0, 1)
            edge_t = t0 + frac * dt
            self.last_edge_t = torch.where(moved, edge_t, self.last_edge_t)
            self.count = count

        # firmware: speed from the time between edges, bounded between edges
        rad_per_count = 2.0 * math.pi / self.cfg.counts_per_rev
        new_counts = self.count - self.prev_count
        has_new = new_counts != 0
        span = torch.clamp(self.last_edge_t - self.prev_edge_t, 1e-6, self.cfg.standstill_s)
        v_new = new_counts.float() * rad_per_count / span
        # an edge against the previous direction means the wheel rocked back over the same hall boundary: the time
        # since the last edge is not the time for one count, and the wheel has just passed through zero speed
        new_dir = torch.sign(new_counts)
        reversed_ = has_new & (self.last_dir != 0) & (new_dir != self.last_dir)
        v_new = torch.where(reversed_, torch.zeros_like(v_new), v_new)
        self.last_dir = torch.where(has_new, new_dir, self.last_dir)
        since = t1 - self.last_edge_t
        bound = rad_per_count / (since + 1e-6)
        v_hold = torch.where(since >= self.cfg.standstill_s, torch.zeros_like(self.velocity),
                             torch.clamp(self.velocity, -bound, bound))
        v_new = torch.clamp(v_new, -self.cfg.max_measured_rad_s, self.cfg.max_measured_rad_s)
        self.velocity = torch.where(has_new, v_new, v_hold)
        self.prev_count = torch.where(has_new, self.count, self.prev_count)
        self.prev_edge_t = torch.where(has_new, self.last_edge_t, self.prev_edge_t)

    # -- firmware speed loop + driver + motor

    def torque(self, target: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
        """Wheel torque for speed ``target`` at true wheel speed ``omega`` (call after :meth:`update_sensors`)."""
        c = self.cfg
        target = torch.clamp(target, -c.max_target_rad_s, c.max_target_rad_s)
        error = target - self.velocity
        duty = target / c.max_rad_s + c.kp * error + c.ki * self.integral
        duty = duty + torch.where(target.abs() > 0.2, torch.sign(target) * c.min_duty, torch.zeros_like(duty))
        duty = torch.clamp(duty, -1.0, 1.0)
        # anti-windup: hold the integral while saturated in the direction of the error
        saturated = ((duty >= 1.0) & (error > 0)) | ((duty <= -1.0) & (error < 0))
        self.integral = torch.where(saturated, self.integral, self.integral + error * dt)
        return self.driver_torque(duty, omega)

    def open_loop_torque(self, duty: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """Wheel torque for a raw duty (the firmware's "D" command; 0 = output off, the wheel coasts)."""
        self.integral.zero_()
        return self.driver_torque(torch.clamp(duty, -1.0, 1.0), omega)

    def driver_torque(self, duty: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """BLDC driver + motor: torque for ``duty`` at true wheel speed ``omega``."""
        # duty x battery voltage with the sign of DIR; current flows both ways, so the driver also slows the wheel.
        # Duties in the dead zone switch the output off and the wheel coasts.
        on = duty.abs() > self.dead_zone
        current = (duty * self.voltage - self.ke * omega) / self.resistance
        current = torch.clamp(current, -self.current_limit, self.current_limit)
        current = torch.where(on, current, torch.zeros_like(current))
        friction = self.friction * torch.tanh(omega / 0.5)
        return self.ke * current - friction
