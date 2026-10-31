# Policy library

Each folder is one trained policy: `<name>/policy.onnx` runs on the Jetson, `<name>/policy.pt` in the PC sim test
(`test/run_sim_test.sh --policy <name>`). Pick one with `POLICY_NAME` in `../.env`, or for one start:

```bash
POLICY_NAME=scratch docker compose up -d       # recreates the balance_bot container with that policy
docker compose run --rm balance_bot policies   # list the library
```

All of these were trained for the current drive: ZS-X11H with speed + DIR, the firmware's speed loop
(`output_mode: "speed"`, KP 0.01 / KI 0.2, `MOTOR_MAX_RAD_S` 60). Policies from older drive models (brake, duty)
are not in the library: they do not match the firmware.

Wobble = pitch-rate std in Isaac Lab (256 robots, measured ZS-X11H dead zone), lower is calmer.

| Name | Training run | Wobble standing / driving | Speed error | Notes |
|---|---|---|---|---|
| `smooth3` | `2026-10-30_23-12-57_smooth3` | see below | see below | **default**. Balance only (does not stand up, follows speed commands poorly), trained on the realistic model: SC speed pulses without direction, firmware slew / safe reversal, gyro+accel tilt filter with mount error and bias, 5-40 ms delay, falls end at 30 deg. Stronger smoothness penalties (pitch_rate -0.3, action_rate -0.6, wheel_acc -2e-6) |
| `smooth2` | `2026-10-25_23-20-18_smooth2` | **0.25** / **0.37** rad/s | 0.082 m/s | From scratch, stronger smoothness penalties (action_rate -0.3, pitch_rate -0.15). Sometimes cannot get up from lying flat (~7% in Isaac, seen once in the sim test): stand it up by hand |
| `scratch` | `2026-10-25_13-47-08_scratch` | 0.38 / 0.45 rad/s | **0.071** m/s | from scratch, original penalties: tracks speed best, wobbles more, always gets up from lying flat |
| `pwm_dir` | `2026-10-25_11-45-12_pwm_dir` | 0.35 / 0.45 rad/s | 0.119 m/s | fine-tuned through all earlier drive models: falls least when pushed, tracks speed worst |

Add a new one from the training PC (then copy `jetson/` to the robot again):

```bash
jetson/add_policy.sh training/logs/rsl_rl/balance_bot_standup/<run> <name>
```

Realistic-model comparison (scratchpad eval_smooth.py, 512 robots, after the first 2 s; stand = no command or
pushes, drive = random commands with pushes; a fall = tilt beyond 30 deg):

| | falls stand / drive | rocking stand / drive | tilt wobble stand | wheel accel stand | speed error drive |
|---|---|---|---|---|---|
| `smooth2` | 38% / 97% | 0.50 / 0.53 rad/s | 1.85 deg | 30 rad/s2 | 0.20 m/s |
| `smooth3` | **12%** / **21%** | **0.34** / **0.40** rad/s | **1.14** deg | **22** rad/s2 | 0.79 m/s |
