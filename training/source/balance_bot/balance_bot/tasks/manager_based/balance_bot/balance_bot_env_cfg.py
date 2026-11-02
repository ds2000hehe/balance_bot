# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Balance-and-drive task: keep the two-wheeled robot upright while tracking velocity commands.

Frame convention: the robot's base frame has +Z up, the wheel axle along X and the driving direction along Y.
The velocity command therefore uses ``lin_vel_y`` for driving speed and keeps ``lin_vel_x`` (sideways) at zero.
The robot's front is -Y, so driving forward is a negative ``lin_vel_y``.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

from balance_bot.assets import BALANCE_BOT_CFG, WHEEL_MAX_SPEED  # isort:skip


##
# Scene definition
##


@configclass
class BalanceBotSceneCfg(InteractiveSceneCfg):
    """Configuration for the balance bot scene."""

    # ground plane (procedural, no asset download needed)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = BALANCE_BOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.2,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),  # sideways: a wheeled robot cannot do this
            lin_vel_y=(-1.5, 1.5),  # forward / backward [m/s]
            ang_vel_z=(-2.0, 2.0),  # turning [rad/s]
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # wheel speed targets reach the wheel controller 5-40 ms late (1-8 physics steps of 5 ms: IMU -> ROS -> policy
    # -> USB serial), like on the real robot (without the delay the policy is unstable on hardware), and are executed
    # by a model of the ATmega firmware (speed pulses without direction, PI speed loop, throttle ramp, safe
    # reversals), the ZS-X11H driver and the hoverboard motor (dead zone, current limit)
    wheel_vel = mdp.HoverboardWheelActionCfg(
        asset_name="robot",
        joint_names=[".*_wheel_joint"],
        scale=WHEEL_MAX_SPEED,
        use_default_offset=False,
        clip={".*": (-WHEEL_MAX_SPEED, WHEEL_MAX_SPEED)},
        delay_range=(1, 8),
        control="speed",
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy. Only quantities the IMU and the ESP32 (hall sensors) measure on the robot."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1), scale=0.25)
        # tilt as the robot's policy node estimates it: complementary filter on gyro + accelerometer (the real IMU's
        # own orientation output is unusable), so accelerating reads as a false tilt, plus mounting error and bias
        projected_gravity = ObsTerm(
            func=mdp.FilteredGravity,
            params={"tau": 0.5, "mount_error": 0.026, "gyro_bias": 0.01, "accel_noise": 0.2, "gyro_noise": 0.02},
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        wheel_vel = ObsTerm(func=mdp.measured_wheel_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        # distance behind / ahead of the forward command from wheel odometry (None: policies without it)
        position_error = None

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the critic (simulation only)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        wheel_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        base_height = ObsTerm(func=mdp.base_pos_z)
        position_error = None

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events (domain randomization and resets)."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "roll" is the rotation about the wheel axle (base x-axis), i.e. leaning forward / backward
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "roll": (-0.2, 0.2), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"y": (-0.2, 0.2), "roll": (-0.3, 0.3)},
        },
    )

    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-math.pi, math.pi), "velocity_range": (0.0, 0.0)},
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.4, 0.4)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    upright = RewTerm(func=mdp.upright_reward, weight=1.0, params={"std": 0.25})
    track_lin_vel = RewTerm(
        func=mdp.track_lin_vel_upright_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.35}
    )
    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_upright_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # -- penalties
    terminated = RewTerm(func=mdp.is_terminated, weight=-20.0)
    # -- smoothness: no rocking, no jumps in the wheel commands, no jerky wheel accelerations
    pitch_rate = RewTerm(func=mdp.pitch_rate_l2, weight=-0.3)
    wheel_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-3)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.6)
    wheel_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.0e-6, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint")}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # same angle as the robot's tilt cut-off (policy node max_tilt_deg): beyond it the motors stop
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(30.0)})


##
# Environment configuration
##


@configclass
class BalanceBotEnvCfg(ManagerBasedRLEnvCfg):
    """Balance the robot and drive it with velocity commands."""

    # Scene settings
    scene: BalanceBotSceneCfg = BalanceBotSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings: 200 Hz physics, 50 Hz policy
        self.decimation = 4
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.1)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class BalanceBotEnvCfg_PLAY(BalanceBotEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        # smaller scene for play
        self.scene.num_envs = 32
        self.scene.env_spacing = 1.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class BalanceBotHoldEnvCfg(BalanceBotEnvCfg):
    """Raw throttle (duty) policy that also holds its position.

    smooth6's settings (duty output, 5-80 ms delay, wheel drag up to 0.6 N m for carpet, pitch_rate -0.6,
    action_rate -1.0) with the full throttle range (action = duty, robot: action_scale = motor_max_rad_s), pushes up
    to 1 m/s, episodes ending at 40 deg, wide physics randomization (mass, centre of mass, inertia), plus one more
    observation, the distance behind / ahead of the forward command from wheel
    odometry (``PositionError``, appended after the last actions), and a reward for keeping it small. The robot's
    policy node detects the extra input from the policy's input size.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.wheel_vel.control = "duty"
        self.actions.wheel_vel.delay_range = (1, 16)
        self.actions.wheel_vel.drive.friction_range = (0.0, 0.6)
        self.rewards.pitch_rate.weight = -0.6
        self.rewards.action_rate.weight = -1.0
        self.observations.policy.position_error = ObsTerm(func=mdp.PositionError, params={"limit": 0.5})
        self.observations.critic.position_error = ObsTerm(func=mdp.position_error_state, params={"limit": 0.5})
        self.rewards.position_hold = RewTerm(func=mdp.position_hold_exp, weight=2.0, params={"std": 0.1})
        # full throttle range: the action is the duty directly (scale = max_rad_s), not limited to 30/68 = 44 %
        self.actions.wheel_vel.scale = self.actions.wheel_vel.drive.max_rad_s
        self.actions.wheel_vel.clip = {".*": (-self.actions.wheel_vel.drive.max_rad_s, self.actions.wheel_vel.drive.max_rad_s)}
        # harder pushes, and episodes run to 40 deg so recovering from big leans is practised
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-1.0, 1.0)}
        self.terminations.fell_over.params["limit_angle"] = math.radians(40.0)
        # physics of the real robot not known exactly: wider mass, centre of mass (forward / back and height matter
        # most for balancing), separate inertia; tyres grip (no slip on the real floor)
        # the real body (everything but the wheels) weighs 2 kg; the model's two body parts are 3 kg each (6 kg):
        # set each to 0.7-1.4 kg (body 1.4-2.8 kg; inertia scales with the mass, then body_inertia on top)
        self.events.add_body_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["body", "mate_connector_1"])
        self.events.add_body_mass.params["mass_distribution_params"] = (0.7, 1.4)
        self.events.add_body_mass.params["operation"] = "abs"
        self.events.body_com.params["com_range"] = {"x": (-0.01, 0.01), "y": (-0.02, 0.02), "z": (-0.04, 0.04)}
        self.events.body_inertia = EventTerm(
            func=mdp.randomize_rigid_body_inertia,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="body"), "scale_range": (0.7, 1.5)},
        )
        self.events.physics_material.params["static_friction_range"] = (1.0, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.9, 1.3)


@configclass
class BalanceBotHoldEnvCfg_PLAY(BalanceBotHoldEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class BalanceBotRealMassEnvCfg(BalanceBotEnvCfg):
    """smooth3's setup (wheel speed targets through the firmware's PI speed loop, no raw throttle, 5-40 ms delay,
    falls end at 30 deg) with the real robot's body: 2 kg instead of the model's 6 kg, and the centre of mass
    randomized further forward / back and up / down (the real position is not known exactly).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        # the model's two body parts are 3 kg each: set each to 0.7-1.4 kg (body 1.4-2.8 kg, real 2 kg)
        self.events.add_body_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["body", "mate_connector_1"])
        self.events.add_body_mass.params["mass_distribution_params"] = (0.7, 1.4)
        self.events.add_body_mass.params["operation"] = "abs"
        # centre of mass: +-2 cm forward / back, +-4 cm height
        self.events.body_com.params["com_range"] = {"x": (-0.01, 0.01), "y": (-0.02, 0.02), "z": (-0.04, 0.04)}
        # forward / back pushes up to 1 m/s (with 0.4 m/s, smooth12 fell on 87 % of 1 m/s forward pushes), and
        # delay up to 50 ms (measured on the robot: ~45 ms from command to gyro)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-1.0, 1.0)}
        self.actions.wheel_vel.delay_range = (1, 10)


@configclass
class BalanceBotRealMassEnvCfg_PLAY(BalanceBotRealMassEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
