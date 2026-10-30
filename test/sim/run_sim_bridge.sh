#!/usr/bin/env bash
# Start the simulated robot (sim_bridge.py) with Isaac Sim's bundled ROS 2 Jazzy libraries.
# Isaac Sim uses Python 3.11 and cannot load /opt/ros/jazzy (Python 3.12), so the system ROS environment is removed
# here. Any arguments are passed on, e.g.:  test/sim/run_sim_bridge.sh --task Template-Balance-Bot-StandUp-Play-v0
set -e
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
ROS_LIBS="$ISAACLAB_PATH/_isaac_sim/exts/isaacsim.ros2.bridge/jazzy"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PYTHON_VERSION ROS_VERSION
# isaaclab.sh prefers an active conda env / venv over Isaac Sim's own Python; Isaac Lab is installed in the latter
unset CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
export PYTHONPATH="$ROS_LIBS/rclpy"
export LD_LIBRARY_PATH="$ROS_LIBS/lib"
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd "$SCRIPT_DIR/../.."
exec "$ISAACLAB_PATH/isaaclab.sh" -p "$SCRIPT_DIR/sim_bridge.py" "$@"
