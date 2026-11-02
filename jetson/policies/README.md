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
| `smooth21` | `2026-11-02_19-18-58_smooth21_ft_revbrake` | see below | see below | **Needs the reversal-braking firmware** (REVERSE_MAX_RAD_S 100, REVERSE_BRAKE). smooth14 fine-tuned (1200 it) on the drivers as measured: lowering the throttle coasts, reversing brakes hard. In Isaac at 40-50 ms delay: falls braking from 1 m/s 6 / 3 % (smooth14 on the current firmware 63 / 75 %), standing wander 1.0 m; drives backward slowly (0.29 of 0.5 m/s; earlier eval labels had forward / backward swapped); 1 m/s pushes still 85-93 % |
| `smooth15` | `2026-11-02_10-54-54_smooth15_duty_2kg_100hz` | see below | see below | `smooth14` at **100 Hz** (params.yaml: duty, control_rate 100; needs the IMU at >= 100 Hz). Task Template-Balance-Bot-Duty2kg-Fast-v0. In Isaac at 40-50 ms delay: best push recovery of the duty policies (1 m/s push fwd / back: 10 / 21 % falls), brakes from 1 m/s with 7 / 4 % falls |
| `smooth14` | `2026-11-02_10-35-30_smooth14_duty_2kg` | see below | see below | From scratch: `smooth4_duty` (raw throttle, action scale 30) with the real 2 kg body (1.4-2.8 kg) and 15-50 ms delay (measured ~45 ms). Task Template-Balance-Bot-Duty2kg-v0. In Isaac: calmest driving (rock 0.11-0.15 rad/s), but falls on most 1 m/s pushes (55 / 97 %) |
| `smooth12` | `2026-11-02_09-48-18_smooth12_speed_2kg` | see below | see below | Back to `smooth3`'s setup (wheel speed targets through the Arduino PI loop, no params.yaml, 5-40 ms delay, pushes up to 0.4 m/s) with the real 2 kg body (1.4-2.8 kg) and centre of mass +-2 cm fwd / +-4 cm height. Task Template-Balance-Bot-RealMass-v0 before the push / delay change. In Isaac (2 kg physics): wanders less than smooth3 (0.68 vs 1.01 m), but falls on 87 % of 1 m/s forward pushes (smooth3 7 %), drives forward slower than commanded (0.37 of 0.5 m/s) |
| `smooth11` | `2026-11-01_15-06-28_smooth11_2kg` | see below | see below | **default**. Task Template-Balance-Bot-Hold-v0: raw throttle with the full range (params.yaml: action_scale 68), holds its position (14th input: distance from wheel odometry, policy_node detects it), body mass corrected to the real 2 kg, wide randomization of centre of mass (+-2 cm fwd, +-4 cm height), inertia (0.7-1.5x), high tyre grip, pushes up to 1 m/s, episodes to 40 deg |
| `smooth7` | `2026-11-01_13-12-11_smooth7_track` | see below | see below | `smooth6` with tighter speed tracking (track_lin_vel weight 3, std 0.15): wanders half as far standing, wobbles a bit more |
| `smooth6` | `2026-11-01_12-28-45_smooth6_robust` | see below | see below | Like `smooth4_duty` (raw throttle, params.yaml), trained for 5-80 ms control delay (was 5-40), wheel drag up to 0.6 N m (carpet), pitch_rate -0.6, action_rate -1.0. In Isaac: calmer than smooth4_duty at 60-80 ms delay, worse at 20-40 ms |
| `smooth4_duty` | `2026-11-01_10-33-48_smooth4_duty` | see below | see below | Like `smooth3`, but outputs raw throttle (`output_mode: duty`, set by its params.yaml): no Arduino PI loop on the coarse speed pulses. Balance only |
| `smooth3` | `2026-10-30_23-12-57_smooth3` | see below | see below | Balance only (does not stand up, follows speed commands poorly), trained on the realistic model: SC speed pulses without direction, firmware slew / safe reversal, gyro+accel tilt filter with mount error and bias, 5-40 ms delay, falls end at 30 deg. Stronger smoothness penalties (pitch_rate -0.3, action_rate -0.6, wheel_acc -2e-6) |
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
| `smooth3` | 12% / 21% | 0.34 / 0.40 rad/s | 1.14 deg | 22 rad/s2 | 0.79 m/s |
| `smooth4_duty` | **8%** / **18%** | **0.31** / **0.35** rad/s | **0.86** deg | **18** rad/s2 | 0.84 m/s |

A policy folder may hold a `params.yaml`: its values override `config/robot.yaml` for that policy (entrypoint.sh).

Standing, fixed control delay (rocking rad/s / falls):

| delay | `smooth4_duty` | `smooth6` |
|---|---|---|
| 20 ms | **0.33** / **10%** | 0.39 / 14% |
| 40 ms | **0.41** / **20%** | 0.46 / 27% |
| 60 ms | 0.61 / **30%** | **0.50** / 40% |
| 80 ms | 1.09 / 49% | **0.62** / 46% |

Standing 10 s (Isaac, 5-40 ms delay): distance wandered (median / 90th percentile), rocking, falls:

| | wander | rocking | falls |
|---|---|---|---|
| `smooth6` | 0.88 / 1.74 m | **0.35** rad/s | **14%** |
| `smooth7` | **0.47 / 0.96** m | 0.41 rad/s | 17% |

policy_node's drift correction (drift_ki) changes neither: the policies hardly respond to its small command offsets.

On the corrected physics (2 kg body, randomized centre of mass / inertia), falls counted at 50 deg (the robot's cut-off):

| | wander 10 s standing (median / p90) | falls standing | falls after push 1.0 m/s fwd / back | 1.5 m/s fwd / back |
|---|---|---|---|---|
| `smooth6` | 0.90 / 1.74 m | 7.8% | 13% / 10% | 50% / 31% |
| `smooth11` | **0.21 / 0.95 m** | **5.1%** | 11% / 13% | **31%** / 34% |

Duty policies on the 2 kg body at the robot's measured delay (40-50 ms), 512 robots, fall = past 50 deg
(scratchpad eval_speed.py; push = 1 m/s shove; brake = 1 m/s, then a stop command):

| | stand: falls / wander | push fwd / back falls | drive rock | brake fwd / back: falls / stop distance |
|---|---|---|---|---|
| `smooth4_duty` | 7 % / 0.91 m | 10 / 43 % | 0.36 rad/s | 3 / 4 % , 2.7 / 2.5 m |
| `smooth6` | 10 % / 0.80 m | 19 / 13 % | 0.23 rad/s | 10 / 11 % , 1.9 / 1.7 m |
| `smooth14` | 4 % / 0.89 m | 55 / 97 % | 0.15 rad/s | 4 / 6 % , 2.5 / 1.9 m |
| `smooth15` (100 Hz) | 6 % / 0.81 m | 10 / 21 % | 0.18 rad/s | 7 / 5 % , 1.9 / 2.0 m |
