"""Wheel test without balancing: cmd_vel -> /wheel_cmd. Run with wheel_serial_bridge, never with policy_node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default = os.path.join(get_package_share_directory("balance_bot_wheel_teleop"), "config", "wheel_teleop.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default),
            Node(
                package="balance_bot_wheel_teleop",
                executable="wheel_teleop",
                name="wheel_teleop",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
