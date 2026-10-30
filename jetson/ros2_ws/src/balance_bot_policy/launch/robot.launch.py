"""Real robot: the USB serial bridge to the wheel controller plus, depending on ``mode``:

* ``balance`` (default): the policy node (self-balancing, driven by cmd_vel),
* ``wheel_test``: balance_bot_wheel_teleop (cmd_vel straight to wheel speeds, no balancing: robot on a stand),
* ``bridge``: nothing else, e.g. for ``ros2 run balance_bot_policy wheel_check``.

All nodes read ``params_file`` (sections balance_bot_policy, wheel_serial_bridge, wheel_teleop).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("balance_bot_policy")
    mode = LaunchConfiguration("mode")
    params = LaunchConfiguration("params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument("port", default_value="/dev/ttyUSB0", description="wheel controller serial port"),
            DeclareLaunchArgument("baud", default_value="115200"),
            DeclareLaunchArgument("params_file", default_value=os.path.join(share, "config", "policy.yaml")),
            DeclareLaunchArgument("mode", default_value="balance", description="balance | wheel_test | bridge"),
            Node(
                package="balance_bot_policy",
                executable="wheel_serial_bridge",
                name="wheel_serial_bridge",
                output="screen",
                parameters=[params, {"port": LaunchConfiguration("port"), "baud": LaunchConfiguration("baud")}],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(share, "launch", "policy.launch.py")),
                launch_arguments={"params_file": params}.items(),
                condition=IfCondition(PythonExpression(["'", mode, "' == 'balance'"])),
            ),
            Node(
                package="balance_bot_wheel_teleop",
                executable="wheel_teleop",
                name="wheel_teleop",
                output="screen",
                parameters=[params],
                condition=IfCondition(PythonExpression(["'", mode, "' == 'wheel_test'"])),
            ),
        ]
    )
