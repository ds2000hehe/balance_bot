# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stand-in for the real robot: one simulated balance bot that talks ROS 2 like the real IMU + ESP32.

Run the ``balance_bot_policy`` ROS 2 node against it to test the whole deployment pipeline in simulation:

    publishes   /imu        sensor_msgs/Imu              IMU mounted x forward, y left, z up (REP-103)
                /wheel_vel  std_msgs/Float32MultiArray   wheel speeds [left, right] rad/s as the ESP32 measures them
                                                         (hall sensors), forward positive
    subscribes  /wheel_cmd  std_msgs/Float32MultiArray   wheel speed targets [left, right] rad/s, forward positive
                /wheel_duty std_msgs/Float32MultiArray   raw duty [left, right] (the ESP32's "D" test command)

The robot's front is the -Y side of base_link; its left wheel is ``right_wheel_joint``.
                /sim/reset  std_msgs/Empty               put the robot back at its start pose

The simulation runs in real time. Physics and sensors run at 200 Hz and a new /wheel_cmd is applied within one 5 ms
physics step, like the ESP32 that acts on commands as they arrive, while the policy node runs at its own 50 Hz.
The wheels are driven by the same model of the ESP32 speed loop, BLDC drivers and hoverboard motors that the policy
is trained with (``mdp.HoverboardWheelAction``).
The window is drawn at ``--render_hz`` with the time left over, so a slow GPU lowers the frame rate, not the
simulation speed.
Without a recent /wheel_cmd (0.1 s) the motor outputs are switched off (the wheels coast), like the ESP32's command
timeout. The robot is
never reset automatically, so it stays where it falls.

Start it with ``test/sim/run_sim_bridge.sh`` (or all of the test: ``test/run_sim_test.sh``) (sets up Isaac Sim's bundled ROS 2 libraries).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simulated balance bot with ROS 2 IMU / wheel topics.")
parser.add_argument("--task", type=str, default="Template-Balance-Bot-Play-v0", help="Task that defines the scene.")
parser.add_argument("--pushes", action="store_true", default=False, help="Keep the random pushes of the task.")
parser.add_argument(
    "--no_orientation", action="store_true", default=False,
    help="Publish only gyro + accelerometer (orientation_covariance[0] = -1), like a raw IMU.",
)
parser.add_argument("--imu_topic", type=str, default="/imu")
parser.add_argument("--wheel_state_topic", type=str, default="/wheel_vel")
parser.add_argument("--wheel_cmd_topic", type=str, default="/wheel_cmd")
parser.add_argument("--cmd_timeout", type=float, default=0.1, help="Brake the wheels after this long without a command.")
parser.add_argument(
    "--lockstep", action="store_true", default=False,
    help="Wait for a new /wheel_cmd every 20 ms of sim time instead of running in real time (no control latency).",
)
parser.add_argument(
    "--start_tilt", type=float, default=None,
    help="Start tilted forward by this angle [rad] (1.57 = lying on the front) instead of the task's reset pose.",
)
parser.add_argument(
    "--hold", action="store_true", default=False,
    help="Hang the robot upright in the air (like a test stand), so the wheels spin freely: for wheel_check.",
)
parser.add_argument("--render_hz", type=float, default=30.0, help="Window frame rate (skipped when behind).")
parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds (0 = run until closed).")
AppLauncher.add_app_launcher_args(parser)
# one robot runs faster on CPU physics than GPU physics (GPU calls have a fixed per-call cost)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import math
import time

import gymnasium as gym
import rclpy
import torch
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty, Float32MultiArray

import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

import balance_bot.tasks  # noqa: F401

# the real robot's conventions, matching the defaults in jetson/config/robot.yaml:
# topic [left, right] = joints [right_wheel_joint, left_wheel_joint] with forward positive; driving forward (-Y)
# turns right_wheel_joint negative and left_wheel_joint positive
TOPIC_JOINTS = ["right_wheel_joint", "left_wheel_joint"]
TOPIC_SIGNS = torch.tensor([-1.0, 1.0])
# IMU orientation in the base frame: IMU x forward (base -Y), y left (base +X), z up = base rotated -90 deg about z
BASE_TO_IMU_QUAT = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, -math.sin(math.pi / 4)]])  # (w, x, y, z)
GRAVITY = 9.81


@torch.inference_mode()
def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.terminations.fell_over = None
    env_cfg.episode_length_s = 1.0e6
    if not args_cli.pushes:
        env_cfg.events.push_robot = None
    # follow the robot with the camera, looking at its front
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (1.0, -1.0, 0.6)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.1)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot, sim, scene = env.scene["robot"], env.sim, env.scene
    device = env.device
    physics_dt = env.physics_dt
    steps_per_control = env_cfg.decimation  # physics steps per policy step in training
    wheel_term = env.action_manager.get_term("wheel_vel")  # the hoverboard drive model
    max_wheel_speed = wheel_term.cfg.scale
    base_body = robot.find_bodies("base_link")[0][0]
    wheel_ids = robot.find_joints(TOPIC_JOINTS, preserve_order=True)[0]  # sim joint index of each topic entry
    term_ids = wheel_term._joint_ids
    term_ids = list(range(robot.num_joints)) if isinstance(term_ids, slice) else list(term_ids)
    topic_to_term = [term_ids.index(j) for j in wheel_ids]  # drive-model column of each topic entry
    signs = TOPIC_SIGNS.to(device)
    base_to_imu = BASE_TO_IMU_QUAT.to(device)
    has_gui = sim.has_gui()

    def reset():
        env.reset()
        if args_cli.start_tilt is not None:
            # tilt about the wheel axle (base x-axis); lift a little so the body does not start inside the ground
            tilt = torch.tensor([args_cli.start_tilt], device=device)
            quat = math_utils.quat_from_euler_xyz(tilt, torch.zeros_like(tilt), torch.zeros_like(tilt))
            pos = robot.data.default_root_state[:, :3] + scene.env_origins
            pos[:, 2] += 0.02 * min(abs(args_cli.start_tilt), 1.0)
            robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

    hold_pose = None

    def physics_step(joint_vel_target, duty=None):
        # what env.step does per physics step, without the observation / reward / termination managers: the wheel
        # speed targets go straight to the drive model (ESP32 speed loop + BLDC driver + hoverboard motor); with
        # ``duty`` the speed loop is bypassed (the ESP32's "D" command, 0 = outputs off)
        if duty is not None:
            wheel_term.drive_wheels_open_loop(duty[term_ids].unsqueeze(0))
        else:
            wheel_term.drive_wheels(joint_vel_target[term_ids].unsqueeze(0))
        if hold_pose is not None:
            robot.write_root_pose_to_sim(hold_pose)
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt=physics_dt)
        if args_cli.pushes:
            env.event_manager.apply(mode="interval", dt=physics_dt)

    # the first steps are slow (shader / physics warm-up); get them done before announcing the robot
    if args_cli.hold:
        # upright, wheels 15 cm above the ground
        hold_pose = torch.cat([robot.data.default_root_state[:, :3] + scene.env_origins, robot.data.default_root_state[:, 3:7]], dim=-1)
        hold_pose[:, 2] += 0.15
    reset()
    for _ in range(20):
        physics_step(torch.zeros(robot.num_joints, device=device))
        if has_gui:
            sim.render()
    reset()

    rclpy.init()
    node = rclpy.create_node("balance_bot_sim")
    imu_pub = node.create_publisher(Imu, args_cli.imu_topic, qos_profile_sensor_data)
    wheel_pub = node.create_publisher(Float32MultiArray, args_cli.wheel_state_topic, qos_profile_sensor_data)
    state = {"cmd": torch.zeros(2, device=device), "cmd_time": -1.0e9, "reset": False, "new_cmd": False,
             "duty": torch.zeros(2, device=device), "duty_time": -1.0e9}

    def on_wheel_cmd(msg):
        if len(msg.data) == 2:
            state["cmd"] = torch.tensor(list(msg.data), device=device)
            state["cmd_time"] = time.monotonic()
            state["new_cmd"] = True

    def on_wheel_duty(msg):
        if len(msg.data) == 2:
            state["duty"] = torch.clamp(torch.tensor(list(msg.data), device=device), -1.0, 1.0)
            state["duty_time"] = time.monotonic()

    def on_reset(_msg):
        state["reset"] = True

    node.create_subscription(Float32MultiArray, args_cli.wheel_cmd_topic, on_wheel_cmd, 10)
    node.create_subscription(Float32MultiArray, "/wheel_duty", on_wheel_duty, 10)
    node.create_subscription(Empty, "/sim/reset", on_reset, 10)
    print(
        f"[SIM] simulated robot up: publishing {args_cli.imu_topic}, {args_cli.wheel_state_topic}; "
        f"listening on {args_cli.wheel_cmd_topic}, /sim/reset",
        flush=True,
    )

    steps_per_second = round(1.0 / physics_dt)
    start_time = next_time = last_render = time.monotonic()
    step = 0
    while simulation_app.is_running():
        rclpy.spin_once(node, timeout_sec=0.0)
        if args_cli.lockstep and step % steps_per_control == 0:
            # wait (up to 0.5 s) for the controller's answer to the last published state
            deadline = time.monotonic() + 0.5
            while not state["new_cmd"] and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.005)
            state["new_cmd"] = False
        if state["reset"]:
            reset()
            state["reset"] = False
            print("[SIM] robot reset", flush=True)

        # wheel command (topic convention) -> joint velocity targets; outputs off if the command is stale
        now_t = time.monotonic()
        fresh = now_t - state["cmd_time"] < args_cli.cmd_timeout
        duty_fresh = now_t - state["duty_time"] < args_cli.cmd_timeout and state["duty_time"] > state["cmd_time"]
        target = torch.zeros(robot.num_joints, device=device)
        if duty_fresh:
            duty = torch.zeros(robot.num_joints, device=device)
            duty[wheel_ids] = state["duty"] * signs
            physics_step(target, duty=duty)
            fresh = True
        elif fresh:
            target[wheel_ids] = torch.clamp(state["cmd"] * signs, -max_wheel_speed, max_wheel_speed)
            physics_step(target)
        else:
            physics_step(target, duty=torch.zeros(robot.num_joints, device=device))

        # IMU: orientation, gyro and accelerometer (specific force) in the IMU frame
        quat_w = math_utils.quat_mul(robot.data.root_quat_w, base_to_imu)
        gyro = math_utils.quat_apply_inverse(base_to_imu, robot.data.root_ang_vel_b)[0]
        acc_w = robot.data.body_lin_acc_w[:, base_body] + torch.tensor([0.0, 0.0, GRAVITY], device=device)
        accel = math_utils.quat_apply_inverse(quat_w, acc_w)[0]
        imu = Imu()
        imu.header.stamp = node.get_clock().now().to_msg()
        imu.header.frame_id = "imu_link"
        if args_cli.no_orientation:
            imu.orientation_covariance[0] = -1.0
        else:
            w, x, y, z = quat_w[0].tolist()
            imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = w, x, y, z
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = gyro.tolist()
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = accel.tolist()
        imu_pub.publish(imu)
        wheel_vel = wheel_term.measured_velocity[0, topic_to_term] * signs
        wheel_pub.publish(Float32MultiArray(data=wheel_vel.tolist()))

        step += 1
        if step % steps_per_second == 0:
            tilt = math.acos(max(-1.0, min(1.0, -robot.data.projected_gravity_b[0, 2].item())))
            v_fwd = -robot.data.root_lin_vel_b[0, 1].item()
            yaw_rate = robot.data.root_ang_vel_b[0, 2].item()
            print(
                f"[SIM] t={step * physics_dt:6.1f}s  tilt {math.degrees(tilt):5.1f} deg  forward {v_fwd:+.2f} m/s"
                f"  yaw rate {yaw_rate:+.2f} rad/s  wheel cmd {'live' if fresh else 'none (outputs off)'}"
                f"  (real-time factor {step * physics_dt / (time.monotonic() - start_time):.2f})",
                flush=True,
            )
            if args_cli.duration and step * physics_dt >= args_cli.duration:
                break

        # real time: physics first, then draw frames with whatever time is left
        now = time.monotonic()
        if args_cli.lockstep:
            if has_gui and now - last_render >= 1.0 / args_cli.render_hz:
                sim.render()
                last_render = now
            continue
        next_time += physics_dt
        if next_time - now > 0:
            if has_gui and now - last_render >= 1.0 / args_cli.render_hz:
                sim.render()
                last_render = now
            sleep = next_time - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
        elif now - next_time > 0.1:
            # too far behind (e.g. window dragged): drop the backlog instead of fast-forwarding
            next_time = now

    node.destroy_node()
    rclpy.try_shutdown()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
