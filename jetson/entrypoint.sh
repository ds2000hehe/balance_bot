#!/usr/bin/env bash
# Commands:
#   robot              (default) USB serial bridge + MODE (.env):
#                        balance     the policy /policies/$POLICY_NAME/policy.onnx (self-balancing)
#                        wheel_test  cmd_vel straight to wheel speeds, no balancing (robot on a stand)
#                        bridge      bridge only (for wheel_check)
#                      configured by /config/robot.yaml
#   joystick           PS4 controller: joy_node + teleop_twist_joy (/cmd_vel, R2 = dead-man) + Circle/Cross
#                      start/stop, configured by /config/joystick.yaml
#   policies           list the policy library (/policies)
#   <anything else>    run it inside the ROS environment, e.g. "ros2 run balance_bot_policy wheel_check"
set -e
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash

list_policies() {
    echo "policies in jetson/policies/ (pick one with POLICY_NAME in .env):"
    for f in /policies/*/policy.onnx; do
        [ -f "$f" ] || continue
        name=$(basename "$(dirname "$f")")
        mark=" "; [ "$name" = "${POLICY_NAME:-}" ] && mark="*"
        echo " $mark $name"
    done
}

case "${1:-robot}" in
    joystick)
        JOY_PARAMS=/config/joystick.yaml
        [ -f "$JOY_PARAMS" ] || { echo "missing $JOY_PARAMS (mount jetson/config)"; exit 1; }
        ls /dev/input/js* >/dev/null 2>&1 || echo "no gamepad yet (/dev/input/js*): press the PS button; it is picked up when it connects"
        echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} RMW=$RMW_IMPLEMENTATION joystick"
        exec ros2 launch balance_bot_policy joystick.launch.py params_file:="$JOY_PARAMS"
        ;;
    policies)
        list_policies
        exit 0
        ;;
    robot)
        ;;
    *)
        exec "$@"
        ;;
esac

PARAMS=/config/robot.yaml
[ -f "$PARAMS" ] || { echo "missing $PARAMS (mount jetson/config)"; exit 1; }
MODE=${MODE:-balance}
case "$MODE" in
    balance|wheel_test|bridge) ;;
    *) echo "MODE=$MODE: use balance, wheel_test or bridge"; exit 1 ;;
esac

if [ "$MODE" = "balance" ]; then
    POLICY_FILE=/policies/${POLICY_NAME:?set POLICY_NAME in .env}/policy.onnx
    if [ ! -f "$POLICY_FILE" ]; then
        echo "POLICY_NAME=$POLICY_NAME: $POLICY_FILE not found"
        list_policies
        exit 1
    fi
    # robot.yaml with policy_path pointing at the chosen policy, plus the policy's own settings
    # (/policies/<name>/params.yaml, e.g. output_mode: duty) on top
    RUN_PARAMS=/tmp/robot.yaml
    python3 - "$PARAMS" "$RUN_PARAMS" "$POLICY_FILE" <<'EOF'
import os, sys, yaml
src, dst, policy = sys.argv[1:]
params = yaml.safe_load(open(src))
node = params["balance_bot_policy"]["ros__parameters"]
node["policy_path"] = policy
extra = os.path.join(os.path.dirname(policy), "params.yaml")
if os.path.exists(extra):
    overrides = yaml.safe_load(open(extra)) or {}
    node.update(overrides)
    print(f"policy settings from {extra}: {overrides}")
yaml.safe_dump(params, open(dst, "w"), sort_keys=False)
EOF
    PARAMS=$RUN_PARAMS
fi
if [ ! -e "$SERIAL_PORT" ]; then
    echo "WARNING: $SERIAL_PORT does not exist yet (wheel controller unplugged?); the bridge keeps retrying"
fi

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} RMW=$RMW_IMPLEMENTATION serial=$SERIAL_PORT @ $BAUD mode=$MODE" \
     "${POLICY_FILE:+policy=$POLICY_FILE}"
exec ros2 launch balance_bot_policy robot.launch.py \
    port:="$SERIAL_PORT" baud:="$BAUD" params_file:="$PARAMS" mode:="$MODE"
