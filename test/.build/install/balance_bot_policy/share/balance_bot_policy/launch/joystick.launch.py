"""PS4 controller: joy_node (gamepad -> /joy), teleop_twist_joy (sticks -> /cmd_vel) and joy_enable
(Circle -> start balancing, Cross -> stop). All parameters are in ``params_file`` (default config/joystick.yaml).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("balance_bot_policy")
    params = LaunchConfiguration("params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=os.path.join(share, "config", "joystick.yaml")),
            Node(package="joy", executable="joy_node", name="joy_node", output="screen", parameters=[params]),
            Node(
                package="teleop_twist_joy",
                executable="teleop_node",
                name="teleop_twist_joy_node",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="balance_bot_policy",
                executable="joy_enable",
                name="joy_enable",
                output="screen",
                parameters=[params],
            ),
        ]
    )
