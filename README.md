# Balance Bot

Two-wheeled self-balancing robot (hoverboard hub motors, ZS-X11H drivers) controlled by a reinforcement learning
policy trained in Isaac Lab.

## Where things run

| Folder | Runs on | What it is | Guide |
|---|---|---|---|
| `training/` | training PC | Isaac Lab extension: robot model, drive model, tasks, training and export scripts, `logs/` | [training/README.md](training/README.md) |
| `test/` | training PC | drive a policy in Isaac Sim with the Jetson's ROS nodes before using it on the robot | [test/README.md](test/README.md) |
| `jetson/` | robot computer | Docker containers: serial bridge, policy, PS4 controller; policy library; ROS packages | [jetson/README.md](jetson/README.md) |
| `arduino/` | Arduino (Uno / Nano) | wheel controller firmware: hall sensors, speed loop, ZS-X11H drivers | [arduino/README.md](arduino/README.md) |

## Typical loop

```bash
cd training
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Template-Balance-Bot-StandUp-v0 --headless --run_name <run>
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task Template-Balance-Bot-StandUp-Play-v0 \
    --checkpoint logs/rsl_rl/balance_bot_standup/<date>_<run>/model_<N>.pt        # watch + export
cd ..
jetson/add_policy.sh training/logs/rsl_rl/balance_bot_standup/<date>_<run> <name>   # add to the library
test/run_sim_test.sh --policy <name>                                                  # drive it in Isaac Sim
rsync -av --exclude __pycache__ --exclude .env jetson/ user@jetson:~/balance_bot/   # copy to the robot
# on the Jetson: POLICY_NAME=<name> in ~/balance_bot/.env, then docker compose up -d
```

The drive model used in training (`training/source/.../mdp/hoverboard.py`) mirrors the Arduino firmware's speed
loop: change both together, then retrain.
