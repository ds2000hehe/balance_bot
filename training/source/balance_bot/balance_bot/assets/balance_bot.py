# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the two-wheeled balance bot.

Robot layout (from the URDF / USD):

* ``base_link``: articulation root, rigidly attached to ``body``. Its frame is the "base" frame used by all
  observations, rewards and commands: **+Z is up, the wheel axle is along X and the robot drives along Y**.
* ``body``: the chassis.
* ``left_wheel_joint`` / ``right_wheel_joint``: continuous revolute joints driving ``left_wheel`` / ``right_wheel``.

The USD is produced by ``scripts/tools/prepare_usd.py`` (see the README). Set the ``BALANCE_BOT_USD`` environment
variable to train with a different file.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

BALANCE_BOT_USD = os.environ.get(
    "BALANCE_BOT_USD", os.path.join(os.path.dirname(__file__), "data", "balance_bot.usd")
)
"""Path to the robot USD."""

WHEEL_RADIUS = 0.0825
"""Wheel radius [m] (6.5" hoverboard wheels). The USD is prepared with the same radius (prepare_usd.py --wheel_radius)."""

WHEEL_MAX_TORQUE = 40.0
"""Safety cap on the wheel torque [N.m]. The real limits (driver current limit x torque constant, ~7-12 N.m for
hoverboard motors) come from the drive model in ``mdp/hoverboard.py``."""

WHEEL_MAX_SPEED = 30.0
"""Largest wheel speed target the policy can command [rad/s] (the ESP32 firmware's MAX_TARGET_RAD_S)."""

##
# Configuration
##

BALANCE_BOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BALANCE_BOT_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # the axle sits 4 mm above base_link; spawn the wheels ~5 mm above the ground and let them settle
        pos=(0.0, 0.0, WHEEL_RADIUS + 0.001),
        joint_pos={".*_wheel_joint": 0.0},
        joint_vel={".*_wheel_joint": 0.0},
    ),
    actuators={
        # torque-driven wheels: the wheel action (mdp.HoverboardWheelAction) computes the torque of the real drive
        # (ESP32 speed loop + BLDC driver + hoverboard motor) every physics step
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=WHEEL_MAX_TORQUE,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the balance bot with torque-driven wheels (see ``mdp.HoverboardWheelAction``)."""
