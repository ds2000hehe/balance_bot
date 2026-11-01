#!/usr/bin/env bash
# Drive the trained policy in Isaac Sim on this PC, with the same ROS nodes and config as the Jetson:
#
#   Isaac Sim (sim/sim_bridge.py: simulated robot, publishes /imu and /wheel_vel, takes /wheel_cmd)
#     + policy_node with jetson/config/robot.yaml and jetson/policies/<name>/policy.pt
#     + keyboard teleop (default) or the PS4 controller (--joy, jetson/config/joystick.yaml)
#
#   test/run_sim_test.sh                    # policy POLICY_NAME from jetson/.env, keyboard
#   test/run_sim_test.sh --policy scratch   # another policy from jetson/policies/
#   test/run_sim_test.sh --joy              # PS4 controller on this PC: Circle start, Cross stop, hold R2 to drive
#
# Runs on ROS_DOMAIN_ID 42 (--domain N) so it never talks to the real robot. Ctrl-C (or q in the keyboard teleop)
# stops everything. Logs: test/logs/.
set -e
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JETSON="$TEST_DIR/../jetson"
POLICY_NAME=$(sed -n 's/^POLICY_NAME=//p' "$JETSON/.env")
DOMAIN=42
JOY=false
while [ $# -gt 0 ]; do
    case "$1" in
        --policy) POLICY_NAME=$2; shift 2 ;;
        --joy) JOY=true; shift ;;
        --domain) DOMAIN=$2; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option $1 (--policy NAME, --joy, --domain N)"; exit 1 ;;
    esac
done
POLICY_FILE="$JETSON/policies/$POLICY_NAME/policy.pt"
if [ ! -f "$POLICY_FILE" ]; then
    echo "$POLICY_FILE not found. Policies:"; ls "$JETSON/policies" | grep -v README; exit 1
fi

# system ROS 2 (not conda's Python: rclpy needs /usr/bin/python3)
unset CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PATH=/usr/bin:$PATH ROS_DOMAIN_ID=$DOMAIN RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/jazzy/setup.bash
LOGS="$TEST_DIR/logs"; BUILD="$TEST_DIR/.build"; mkdir -p "$LOGS"

# build the Jetson's ROS packages here (outside jetson/, so that folder stays clean for copying)
echo "building jetson/ros2_ws ..."
colcon --log-base "$BUILD/log" build --base-paths "$JETSON/ros2_ws/src" --build-base "$BUILD/build" \
    --install-base "$BUILD/install" > "$LOGS/build.log" 2>&1 || { tail -20 "$LOGS/build.log"; exit 1; }
source "$BUILD/install/setup.bash"

# the robot's config with the PC paths; the keyboard teleop has no start button, so start balancing right away
python3 - "$JETSON/config/robot.yaml" "$LOGS/robot_sim.yaml" "$POLICY_FILE" "$JOY" <<'EOF'
import sys, yaml
src, dst, policy, joy = sys.argv[1:]
params = yaml.safe_load(open(src))
p = params["balance_bot_policy"]["ros__parameters"]
p["policy_path"] = policy
p["start_enabled"] = joy != "true"
yaml.safe_dump(params, open(dst, "w"), sort_keys=False)
EOF

PIDS=()
cleanup() {
    echo; echo "stopping ..."
    for pid in "${PIDS[@]}"; do kill -INT -- "-$pid" 2>/dev/null || true; done
    sleep 3
    for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "starting Isaac Sim (window opens in ~30 s, log: test/logs/sim.log) ..."
ROS_DOMAIN_ID=$DOMAIN setsid "$TEST_DIR/sim/run_sim_bridge.sh" > "$LOGS/sim.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 180); do
    timeout 3 ros2 topic list 2>/dev/null | grep -q "^/imu$" && break
    kill -0 "${PIDS[0]}" 2>/dev/null || { echo "Isaac Sim exited:"; tail -20 "$LOGS/sim.log"; exit 1; }
    sleep 1
done
timeout 10 ros2 topic pub --once -w 1 /sim/reset std_msgs/msg/Empty "{}" > /dev/null || true  # start from the reset pose

echo "starting policy_node with $POLICY_NAME (log: test/logs/policy.log) ..."
setsid ros2 launch balance_bot_policy policy.launch.py params_file:="$LOGS/robot_sim.yaml" > "$LOGS/policy.log" 2>&1 &
PIDS+=($!)
sleep 3
grep -m1 "Loaded policy" "$LOGS/policy.log" || { echo "policy_node failed:"; tail -20 "$LOGS/policy.log"; exit 1; }

echo
echo "reset the robot if it falls:  ROS_DOMAIN_ID=$DOMAIN ros2 topic pub --once /sim/reset std_msgs/msg/Empty '{}'"
if [ "$JOY" = true ]; then
    echo "PS4 controller: Circle = start balancing, Cross = stop, hold R2 + sticks = drive. Ctrl-C quits."
    ros2 launch balance_bot_policy joystick.launch.py params_file:="$JETSON/config/joystick.yaml"
else
    ros2 run balance_bot_policy teleop_key
fi
