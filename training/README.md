# Training: Isaac Lab extension for the balance bot

Reinforcement learning environments for a two-wheeled self-balancing robot (inverted pendulum on two driven
wheels), built as an Isaac Lab external project (generated with Isaac Lab's template, `./isaaclab.sh --new`).

Everything here runs on the training PC, from this folder (Isaac Lab writes `logs/` into the current directory):

```bash
cd ~/RL/balance_bot/training
```

Trained policies go to the robot through the policy library: `../jetson/add_policy.sh logs/rsl_rl/<experiment>/<run>
<name>`, then test them in Isaac Sim with `../test/run_sim_test.sh --policy <name>`.

## Robot and frame conventions

* Articulation root: `base_link` (rigidly attached to `body`). **+Z is up, the wheel axle is along X, and the
  robot drives along Y. The robot's front is -Y**, so its left wheel is `right_wheel_joint`.
* Because of this, the velocity command uses `lin_vel_y` for driving speed (±1.5 m/s; forward is negative) and
  keeps `lin_vel_x` at 0. The ROS 2 node in `../jetson/ros2_ws/` converts to the usual ROS convention (x forward).
  `ang_vel_z` is the turn rate (±2 rad/s).
* Actions: target velocity for `left_wheel_joint` and `right_wheel_joint` (scaled by `WHEEL_MAX_SPEED`). They reach
  the wheels 0-30 ms late and are executed by a model of the real drive (`mdp/hoverboard.py`): the Arduino firmware's
  PI speed loop on a 90-count hall-sensor speed estimate, a BLDC driver controlled by PWM + DIR that drives current
  both ways (BRAKE / STOP not used), and a hoverboard hub motor (back-EMF, current limit, dead zone, friction),
  randomized per robot.
* Policy observations are only quantities a real robot can measure (IMU angular velocity and gravity direction,
  wheel speeds as the Arduino measures them, command, last action). The critic also gets privileged base velocity and height (asymmetric
  actor-critic).

The drive model's parameters (`HoverboardDriveCfg` in `mdp/hoverboard.py`) must match the real motors and the
firmware constants in `../arduino/balance_bot_atmega/balance_bot_atmega.ino` (gains, `MOTOR_MAX_RAD_S`, `MIN_DUTY`, ...):
change both together. The wheel radius (6.5" hoverboard wheels, 0.0825 m) is set in `assets/balance_bot.py` and in
the USD (`prepare_usd.py --wheel_radius`).

## Installation

1. Install Isaac Lab ([guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)).
2. Install this extension in editable mode:

    ```bash
    ~/IsaacLab/isaaclab.sh -p -m pip install -e source/balance_bot
    ```

3. Prepare the robot USD. This writes a cleaned copy to `source/balance_bot/balance_bot/assets/data/balance_bot.usd`
   (already done for `~/Desktop/balance_bot1.usd`; re-run it whenever you change the robot):

    ```bash
    ~/IsaacLab/isaaclab.sh -p scripts/tools/prepare_usd.py ~/Desktop/balance_bot1.usd --wheel_radius 0.0825
    ```

    The script fixes the `metersPerUnit` tag, gives massless CAD helper links a tiny mass (PhysX would otherwise add
    1 kg to each), recomputes inertias that no longer match hand-edited masses, and replaces the polygonal
    wheel colliders with smooth spheres (`--keep_mesh_wheels` to disable). Set the environment variable
    `BALANCE_BOT_USD=/path/to/file.usd` to use a different file.

## Training

```bash
# list the tasks
~/IsaacLab/isaaclab.sh -p scripts/list_envs.py

# 1) balance + drive (~10 min on an RTX 5070)
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Template-Balance-Bot-v0 --headless

# 2) stand up + balance + drive, warm-started from the balance policy
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Template-Balance-Bot-StandUp-v0 --headless \
    --resume --load_run <balance_run_folder> --experiment_name balance_bot
#    (or train it from scratch by leaving out the --resume arguments)

# watch a trained policy (also exports policy.pt / policy.onnx to logs/.../exported)
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task Template-Balance-Bot-StandUp-Play-v0

# training curves
~/IsaacLab/isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl
```

Logs and checkpoints are written to `logs/rsl_rl/<experiment_name>/<date_time>/`.

## Layout

```
scripts/
  rsl_rl/train.py, play.py        # standard Isaac Lab RSL-RL scripts
  rsl_rl/evaluate.py              # stand-up success rate of a checkpoint
  tools/prepare_usd.py            # USD clean-up for training
  list_envs.py, zero_agent.py, random_agent.py
source/balance_bot/balance_bot/
  assets/balance_bot.py           # ArticulationCfg (USD path, actuators, motor limits)
  assets/data/balance_bot.usd     # prepared robot USD
  tasks/manager_based/balance_bot/
    __init__.py                   # gym registrations
    balance_bot_env_cfg.py        # balance + drive task (scene, obs, actions, rewards, events, terminations)
    balance_bot_standup_env_cfg.py# stand-up task
    mdp/rewards.py                # custom rewards (upright, gated velocity tracking, pitch rate)
    mdp/actions.py                # delayed wheel targets + HoverboardWheelAction (drives the drive model)
    mdp/hoverboard.py             # model of the Arduino speed loop + BLDC driver + hoverboard motor
    mdp/observations.py           # wheel speeds as the Arduino measures them
    agents/rsl_rl_ppo_cfg.py      # PPO hyperparameters
```

## Tuning tips

* **Getting up from lying flat is limited by the robot's design.** The body (8.75 cm from axle to front/back face) is
  deeper than the wheel radius (8.25 cm), so when it lies flat the wheels are ~0.5 cm off the ground. The only way up
  is the reaction-wheel trick: spin the wheels up, then reverse them hard. How often this works depends on wheel
  inertia, the motors' top speed and the driver's current limit. Bigger wheels (or a body whose faces sit inside the wheel radius)
  make standing up much easier.
* Check a checkpoint with `scripts/rsl_rl/evaluate.py`: it lays every robot down, lets it settle, and reports the
  fraction that gets up:

    ```bash
    ~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/evaluate.py --task Template-Balance-Bot-StandUp-v0 --headless
    ```
* Domain randomization (friction, body mass ±20 %, centre of mass ±1 cm, pushes) lives in `EventCfg`. Widen it
  for sim-to-real transfer.
* Reward weights are in `RewardsCfg`; the per-term `Episode_Reward/*` values in the training log show which
  terms dominate.

## Template notes

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/balance_bot/balance_bot/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/balance_bot"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```