# balance_bot_policy (ROS 2)

Runs the Isaac Lab policy on the real robot at 50 Hz.

```
 IMU (sensor_msgs/Imu) ──────────┐
 /cmd_vel (geometry_msgs/Twist) ─┼──► policy_node ──► /wheel_cmd  [left, right] rad/s ──► ESP32 
 /wheel_vel [left, right] rad/s ─┘                                                         │
        ▲                                                                                  │
        └──────────────────────────── measured wheel speeds (hall sensors) ◄───────────────┘
```

## Build and run

```bash
cd ~/RL/balance_bot/jetson/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch balance_bot_policy policy.launch.py            # uses config/policy.yaml
# or: ros2 launch balance_bot_policy policy.launch.py params_file:=/path/to/my.yaml
```

Drive it with anything that publishes `/cmd_vel`, e.g.
`ros2 run teleop_twist_keyboard teleop_twist_keyboard`, or the smoother `ros2 run balance_bot_policy teleop_key`
(keys set a speed instead of having to be held: w/x faster forward/backward, a/d turn, s or space stop).
`linear.x` is the forward speed (clipped to ±1.5 m/s) and `angular.z` the turn rate (clipped to ±2 rad/s): the
ranges the policy was trained on. Changes are ramped (`max_lin_acc`, `max_ang_acc`). Without `/cmd_vel` for 1 s the
robot slows down and keeps balancing in place.

The policy alone settles a few cm/s off its command (it crept backwards at zero command), so the node adds an
integral loop on wheel odometry: `ki * ∫(cmd − v_wheels) dt` is added to the forward command (`drift_ki`, limited to
`drift_max_correction`, paused while the robot is tilted more than ~25°). Set `wheel_radius` to the real wheels;
`drift_ki: 0.0` turns it off.

Emergency stop: `ros2 topic pub --once /balance_bot/enable std_msgs/msg/Bool "{data: false}"`.

## Wheel controller (Arduino over USB)

`~/RL/balance_bot/arduino/balance_bot_atmega/balance_bot_atmega.ino` (Arduino Uno / Nano, ZS-X11H drivers; the older
`arduino/esp32_legacy/balance_bot_esp32` speaks the same protocol)
drives two sensored BLDC motors through driver boards with PWM + DIR only (BRAKE / STOP unused). It runs a 500 Hz velocity
loop per wheel from the motors' hall sensors and talks plain text over USB serial (115200 baud): `V <left> <right>`
sets targets in rad/s, the controller answers `W <left> <right>` (measured, 100 Hz). It switches the motors off (the wheels
coast) when no command arrives for 200 ms. Set the pins and `POLE_PAIRS` at the top of the sketch; the comment there explains the wiring, the invert
flags, the open-loop `D` test command and how to tune `KP`/`KI` from the Serial Monitor.

`wheel_serial_bridge` relays `/wheel_cmd` to the ESP32 and publishes its `W` lines on `/wheel_vel`. On the robot,
start both nodes with:

```bash
ros2 launch balance_bot_policy robot.launch.py port:=/dev/ttyUSB0   # or /dev/ttyACM0 on boards with native USB
```

(Add yourself to the `dialout` group once to access the port: `sudo usermod -aG dialout $USER`, then log in again.)
The policy needs a **fast, stiff wheel velocity loop**: in simulation the wheels reached their target almost
instantly. A slow or soft PID is the most common reason a sim-trained balancer fails on hardware.

## Bench test before the first run: `wheel_check`

With the **wheels off the ground** and only the serial bridge running, `wheel_check` drives the wheels through the
same topics and `wheel_signs` as the policy node and checks that they behave the way the policy was trained to
expect:

```bash
ros2 launch balance_bot_policy robot.launch.py mode:=bridge port:=/dev/ttyUSB0
ros2 run balance_bot_policy wheel_check            # --yes skips the steps that need your hands / eyes
```

It asks you to turn each wheel one turn by hand (hall counts and sign), to say whether each wheel spins forward,
and to hold / lean / rock the robot (IMU axes and gyro sign). It measures the driver's dead zone, the speed at full
duty (-> `MOTOR_MAX_RAD_S`), whether lowering the duty slows the wheel (as the training model assumes), the speed
loop's delay, step, reversal, slow-speed and back-and-forth response, stopping, and that the motors switch off when
the commands stop. In-air speed-loop results are compared with what the training's drive model does in the same test
(a free wheel is rough at low speed; on the ground the robot's weight smooths it out). Every command and
measurement is written to `~/wheel_check_<date>.csv`: send it along if anything is WARN or FAIL, so the drive model
can be matched to the robot and the policy retrained.

The simulated robot can be tested the same way: `test/sim/run_sim_bridge.sh --hold` hangs it in the air.

## Settings to check before the first run (`config/policy.yaml`)

1. **`imu_to_body`**: rotation from your IMU's axes to the robot's axes (x forward, y left, z up). The default
   (identity) assumes the IMU is mounted with x pointing to the robot's front, y to its left and z up.
   If the IMU's x axis points backward or sideways on your robot, change this matrix.
2. **`wheel_signs`**: `[1, 1]` assumes that on the ESP32 positive speed means "drive forward" for both wheels
   (`[left, right]`, seen from behind the robot), which the sketch's invert flags take care of.
   Check with the robot held off the ground: a forward lean must spin both wheels forward.
3. **`use_imu_orientation`**: if your IMU publishes a fused orientation, it is used for the gravity direction.
   Otherwise (orientation_covariance[0] = -1) a gyro + accelerometer filter estimates it. A gyro alone is not
   enough: the policy needs to know which way is down.
4. **`policy_path`**: the exported `policy.pt` (or `policy.onnx`, needs `onnxruntime`).

## First test on hardware

Hold the robot (or lift its wheels) with the e-stop ready. Then tilt it slowly forward and backward and watch
`ros2 topic echo /wheel_cmd`: the wheels must turn the way the robot leans. Only then let it balance on the
ground, with `/cmd_vel` at zero.

## Wheel test without balancing

The separate package `balance_bot_wheel_teleop` turns `/cmd_vel` straight into wheel speed targets on `/wheel_cmd`
(differential drive: `wheel_radius`, `track_width`), for checking the wheels with the robot on a stand:

```bash
ros2 launch balance_bot_policy robot.launch.py mode:=wheel_test port:=/dev/ttyUSB0   # bridge + wheel_teleop
ros2 launch balance_bot_policy joystick.launch.py                                     # PS4: hold R2 and drive
```

Never run it together with `policy_node` (both command the wheels).

