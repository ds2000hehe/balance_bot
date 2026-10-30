# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained balance bot policy: how many robots get (and stay) upright.

All robots start tilted by ``--start_tilt`` radians, randomly forward or backward (1.0 rad = lying on the ground,
0.0 = upright), with zero velocity commands and no pushes. The wheels are first held still for ``--settle_time``
seconds so the robots fall and come to rest on the ground; then the policy takes over. The script prints the fraction of robots that are
upright (tilt < ``--upright_tilt``) once per second and exits.

.. code-block:: bash

    ./isaaclab.sh -p scripts/rsl_rl/evaluate.py --task Template-Balance-Bot-StandUp-v0 --headless --start_tilt 1.0

"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import balance_bot.balance_bot.training.scripts.rsl_rl.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a balance bot checkpoint.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--start_tilt", type=float, default=1.0, help="Initial tilt [rad]; 1.0 is lying on the ground.")
parser.add_argument("--upright_tilt", type=float, default=0.2, help="Tilt [rad] below which a robot counts as upright.")
parser.add_argument("--duration", type=float, default=5.0, help="Evaluation duration [s].")
parser.add_argument(
    "--settle_time", type=float, default=1.0, help="Time [s] with braked wheels before the policy takes over."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import math
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import balance_bot.tasks  # noqa: F401

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate the RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # clean evaluation conditions: stand still, no pushes, no observation noise, no early resets
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.terminations.fell_over = None
    env_cfg.episode_length_s = args_cli.settle_time + args_cli.duration + 1.0

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and os.path.isabs(args_cli.checkpoint):
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # place every robot at the start tilt (about the wheel axle = base x-axis), forward or backward at random
    robot = env.unwrapped.scene["robot"]
    n, device = env.num_envs, env.unwrapped.device
    sign = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    roll = sign * args_cli.start_tilt
    yaw = torch.rand(n, device=device) * 2 * math.pi
    quat = math_utils.quat_from_euler_xyz(roll, torch.zeros_like(roll), yaw)
    pos = robot.data.default_root_state[:, :3] + env.unwrapped.scene.env_origins
    pos[:, 2] += 0.02 * min(args_cli.start_tilt, 1.0)
    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))
    obs = env.get_observations()

    steps_per_s = round(1.0 / env.unwrapped.step_dt)
    with torch.inference_mode():
        for _ in range(int(args_cli.settle_time * steps_per_s)):
            obs, _, _, _ = env.step(torch.zeros(n, env.num_actions, device=device))
    tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    print(f"\n[EVAL] after settling: tilt mean {tilt.mean().item():.2f} rad, min {tilt.min().item():.2f} rad")

    cos_upright = math.cos(args_cli.upright_tilt)
    ever_upright = torch.zeros(n, dtype=torch.bool, device=device)
    print(f"[EVAL] start tilt {args_cli.start_tilt:.2f} rad, {n} robots")
    with torch.inference_mode():
        for step in range(1, int(args_cli.duration * steps_per_s) + 1):
            obs, _, _, _ = env.step(policy(obs))
            upright = -robot.data.projected_gravity_b[:, 2] > cos_upright
            ever_upright |= upright
            if step % steps_per_s == 0:
                print(
                    f"[EVAL] t={step / steps_per_s:4.1f}s  upright now: {upright.float().mean().item():6.1%}"
                    f"  ever upright: {ever_upright.float().mean().item():6.1%}"
                )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
