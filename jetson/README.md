# Jetson: what runs on the robot computer (Docker)

This folder is everything the robot computer needs: copy it to the Jetson and `docker compose up`. The Arduino
firmware is in `../arduino/`, the PC sim test in `../test/`.

Two containers run everything on the robot computer and start on every boot:

```
PS4 controller ─Bluetooth─► host (bluez) ─► /dev/input/js0
                                                  │
              ┌──────── joystick container ───────┼────────────────┐
              │ joy_node ──/joy──► teleop_twist_joy ──/cmd_vel──┐  │  (only while R2 is held)
              │               └──► joy_enable ──/balance_bot/enable│  (Circle = start, Cross = stop)
              └─────────────────────────────────────────────────┼──┘
IMU driver ──/imu──►  ┌──────────── balance_bot container ────────▼────┐
                      │ policy_node ──/wheel_cmd──► wheel_serial_bridge │──USB──► Arduino ──► ZS-X11H drivers
                      │ (policies/$POLICY_NAME; MODE=wheel_test: wheel_teleop instead)  │
                      │      ▲                            │             │
                      │      └──────────/wheel_vel────────┘             │◄──USB── hall speeds
                      └─────────────────────────────────────────────────┘
```

The IMU driver is not in the containers: run it as before (on the host or as another service), on the same
`ROS_DOMAIN_ID`. The containers use the host network, so they see all ROS topics of the robot and the LAN.

## Files

| File | What to set |
|---|---|
| `.env` | `SERIAL_PORT` (Arduino USB port), `BAUD`, `MODE` (`balance` / `wheel_test` / `bridge`), `POLICY_NAME`, `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION` |
| `config/robot.yaml` | topics (`imu_topic`, `cmd_vel_topic`, ...), IMU mounting (`imu_to_body`), `wheel_signs`, speed limits, drift correction, `start_enabled`; `wheel_teleop` section for `MODE=wheel_test` (`track_width`) |
| `config/joystick.yaml` | PS4 controller: start / stop buttons, dead-man button, which sticks drive and turn, speeds |
| `policies/<name>/policy.onnx` | the policy library, see [policies/README.md](policies/README.md). `add_policy.sh` adds a training run |

## Install (once, on the robot computer)

Any 64-bit Linux (x86_64 or arm64, e.g. Raspberry Pi 5 / Jetson / NUC).

1. Install Docker and make it start on boot:

    ```bash
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER          # then log out and back in, so docker works without sudo
    sudo systemctl enable --now docker
    ```

2. Copy this folder to the Jetson, from the PC:

    ```bash
    rsync -av --exclude __pycache__ ~/RL/balance_bot/jetson/ user@jetson:~/balance_bot/
    ```

    To update it later without overwriting the Jetson's own settings, add `--exclude .env`.
3. Flash the Arduino (`../arduino/README.md`) and plug it into the Jetson.
4. Find its port and put it into `.env`:

    ```bash
    ls -l /dev/serial/by-id/     # e.g. usb-Silicon_Labs_CP2102_..._0001-if00-port0 -> ../../ttyUSB0
    ```

    Use the `/dev/serial/by-id/...` path: it stays the same when you replug the Arduino or add other USB devices.
5. Check `config/robot.yaml`: `imu_topic` = what your IMU driver publishes (`ros2 topic list`), `imu_to_body` = how
   the IMU is mounted.
6. Pair the PS4 controller (below).
7. Build and start:

    ```bash
    cd ~/balance_bot
    docker compose up -d --build
    docker compose logs -f        # "Loaded policy from /policies/smooth2/policy.onnx", "opened /dev/serial/...",
                                  # "/joy button 1 -> start balancing ..."
    ```

Both containers start again after a reboot or crash (`restart: unless-stopped`). `docker compose down` stops them
(and they stay stopped after a reboot until the next `docker compose up -d`).

## Pairing the PS4 controller (once)

Bluetooth runs on the host (not in a container): once paired and trusted, the controller reconnects by itself
whenever you press its PS button, also after a reboot.

```bash
sudo apt install -y bluez
sudo systemctl enable --now bluetooth
bluetoothctl
  power on
  agent on
  default-agent
  scan on
  # on the controller: hold Share + PS until the light bar flashes quickly (pairing mode)
  # wait for "Wireless Controller" with its address, e.g. [NEW] Device A4:AE:12:34:56:78 Wireless Controller
  pair A4:AE:12:34:56:78
  trust A4:AE:12:34:56:78
  connect A4:AE:12:34:56:78
  scan off
  exit
ls /dev/input/js*               # /dev/input/js0 = the controller is connected
```

The light bar goes solid when connected. A wired USB cable works too, without pairing. The controller needs the
`hid_playstation` driver, included in Linux 5.12 and newer (Ubuntu 22.04+, Raspberry Pi OS bookworm).

## Driving

With `start_enabled: false` (the default) the robot does nothing after boot until you press **Circle**.

| PS4 controller | |
|---|---|
| **PS** | connect the controller (after boot, or after it went to sleep) |
| **Circle** | start balancing. From lying on its front or back the robot gets up by itself |
| **Cross** | stop: motors off, **the robot falls over** (hold it, or stop near the ground) |
| hold **R2** | dead-man: the sticks only drive while R2 is held. Release it and the robot stops and stands |
| left stick up / down (with R2) | forward / back (0.5 m/s at full stick) |
| right stick left / right (with R2) | turn (1.5 rad/s) |

Speeds: `scale_linear` / `scale_angular` in `config/joystick.yaml` (the policy is trained up to 1.5 m/s, 2 rad/s).

If the controller disconnects (flat battery, out of range), the robot keeps balancing in place: `/cmd_vel` goes
quiet and the policy ramps its command to zero within a second. Reconnect with PS and press Cross to stop.

Check the buttons and sticks of your controller with `docker compose exec joystick ros2 topic echo /joy` and change
`config/joystick.yaml` if they differ (`docker compose restart joystick`).

Without the controller, from any machine on the same network and `ROS_DOMAIN_ID`:

```bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 topic pub --once /balance_bot/enable std_msgs/msg/Bool "{data: true}"    # hold the robot upright first
ros2 run teleop_twist_keyboard teleop_twist_keyboard                          # or any /cmd_vel source
ros2 topic pub --once /balance_bot/enable std_msgs/msg/Bool "{data: false}"   # stop
```

## Choosing a policy

```bash
docker compose run --rm balance_bot policies   # list the library (policies/)
POLICY_NAME=scratch docker compose up -d       # run another one (until the next plain "up -d")
```

Set `POLICY_NAME` in `.env` to make the choice permanent (also after reboots).

## Calibration and bench tests (wheels off the ground)

The policy assumes that both wheels turn forward at the speed it commands. Check this once after wiring, and after
changing motors, drivers, battery or firmware. Robot on a stand, wheels free:

1. **Wheel check** (automatic): directions, speed per duty (`MOTOR_MAX_RAD_S`), dead zone, speed loop response:

    ```bash
    MODE=bridge docker compose up -d
    docker compose exec balance_bot ros2 run balance_bot_policy wheel_check
    docker compose cp balance_bot:/root/. ./wheel_check_logs     # CSV logs (wheel_check_<date>.csv)
    ```

    Fix what it reports in the sketch (`*_MOTOR_INVERT`, `*_FEEDBACK_INVERT`, `MOTOR_MAX_RAD_S`) and flash again.
2. **Pulses per turn** (do this first, before `wheel_check`): Arduino Serial Monitor, send `?`, turn a wheel exactly
   one turn by hand, send `?` again. The change in count is the SC pulses per turn: set `COUNTS_PER_REV` in the
   sketch to it and flash again (`../arduino/README.md`). A wrong value scales every measured wheel speed.
3. **Wheel radius**: measure the tyre diameter; `wheel_radius` in `config/robot.yaml` (0.0825 m = 6.5" wheels).
4. **Drive the wheels with the controller** (no balancing):

    ```bash
    MODE=wheel_test docker compose up -d
    docker compose logs -f balance_bot          # "target [...] measured [...]" once a second
    ```

    Hold R2 and push the left stick: both wheels forward, measured close to target. Right stick: the wheels turn
    opposite ways (right stick left = left wheel back, right wheel forward). The speed limit is
    `wheel_teleop.max_wheel_speed` (10 rad/s).
5. Back to balancing: `docker compose up -d` (MODE from `.env`).

If the measured motor values are far outside what the training assumed (wheel_check says so), the policy needs
retraining with the drive model (`mdp/hoverboard.py`) set to them.

## Updating

* New policy: `jetson/add_policy.sh <run dir> <name>` on the training PC, rsync `jetson/` to the robot, then
  `POLICY_NAME=<name> docker compose up -d`.
* Changed `config/robot.yaml` / `config/joystick.yaml`: `docker compose restart balance_bot` / `joystick`.
* Changed `.env`: `docker compose up -d` (recreates the containers).
* Changed the ROS package code: `docker compose up -d --build`.

## Troubleshooting

* `cannot open /dev/...`: wrong `SERIAL_PORT`, or the Arduino is unplugged (the bridge keeps retrying).
* `no recent IMU data`: the IMU driver is not running, publishes on another topic (`imu_topic`), another
  `ROS_DOMAIN_ID`, or below 100 Hz.
* Circle does nothing: `docker compose logs joystick` should show "start button: balancing on". If not,
  `ls /dev/input/js*` on the host (controller connected?) and `docker compose exec joystick ros2 topic echo /joy`
  (button numbers). If it shows but the robot does not start, `docker compose logs balance_bot` says why (no IMU or
  wheel data).
* Topics visible with `ros2 topic list` but no data arrives: make the IMU driver use the same `RMW_IMPLEMENTATION`.
  The containers use Fast DDS over UDP only (`fastdds_udp.xml`). They run as root, and Fast DDS's shared-memory
  transport does not work between root and normal-user processes.
* Safety: if the balance_bot container stops, the Arduino switches the motors off 200 ms after the last command (the
  wheels coast, so the robot falls over: hold it or keep it near the ground).
