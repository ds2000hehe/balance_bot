# Test: drive the policy in Isaac Sim on the PC

Before putting a policy on the robot, drive it here. Isaac Sim plays the robot (same IMU and wheel topics, same
drive model as in training), and the Jetson's own ROS nodes and config run against it:

```
Isaac Sim (sim/sim_bridge.py) ──/imu, /wheel_vel──► policy_node ──/wheel_cmd──► Isaac Sim
                                                        ▲
                         keyboard teleop or PS4 ──/cmd_vel
```

Needs this PC's setup: Isaac Lab in `~/IsaacLab`, ROS 2 Jazzy in `/opt/ros/jazzy`, `ros-jazzy-joy` and
`ros-jazzy-teleop-twist-joy` for `--joy`.

```bash
test/run_sim_test.sh                    # policy POLICY_NAME from jetson/.env, keyboard teleop
test/run_sim_test.sh --policy scratch   # another policy from jetson/policies/
test/run_sim_test.sh --joy              # PS4 controller connected to the PC (Circle, Cross, hold R2 to drive)
```

The Isaac Sim window opens after ~30 s. The keyboard teleop starts balancing right away (it has no start button):

| Key | |
|---|---|
| w / x | faster forward / backward |
| a / d | turn left / right |
| s / space | stop |
| q | quit (stops everything) |

The robot falls over and stays down? Reset it from another terminal:

```bash
export ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_fastrtps_cpp; source /opt/ros/jazzy/setup.bash
ros2 topic pub --once /sim/reset std_msgs/msg/Empty "{}"
```

Everything runs on `ROS_DOMAIN_ID` 42 (`--domain N`), so it never reaches the real robot (domain 0). Logs are in
`test/logs/` (`sim.log`, `policy.log`, `build.log`).

`sim/` holds the simulated robot; `sim/run_sim_bridge.sh --help` lists its options (`--hold` hangs the robot in the
air for `wheel_check`, `--pushes` keeps the random pushes, `--start_tilt 1.57` starts lying down).
