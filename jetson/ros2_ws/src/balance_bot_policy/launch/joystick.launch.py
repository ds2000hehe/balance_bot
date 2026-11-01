"""PS4 controller: joy_node (gamepad -> /joy) and teleop_twist_joy (sticks -> /cmd_vel) with the teleop_twist_joy
package's standard settings (same as ``ros2 launch teleop_twist_joy teleop-launch.py``, ps3 config), plus joy_enable
(Circle -> start balancing, Cross -> stop). ``params_file`` (default config/joystick.yaml) configures joy_enable and
overrides individual teleop_twist_joy settings (enable button).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("balance_bot_policy")
    teleop_config = os.path.join(get_package_share_directory("teleop_twist_joy"), "config", "ps3.config.yaml")
    params = LaunchConfiguration("params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=os.path.join(share, "config", "joystick.yaml")),
            Node(
                package="joy",
                executable="joy_node",
                name="joy_node",
                output="screen",
                parameters=[{"device_id": 0, "deadzone": 0.3, "autorepeat_rate": 20.0}],
            ),
            Node(
                package="teleop_twist_joy",
                executable="teleop_node",
                name="teleop_twist_joy_node",
                output="screen",
                parameters=[teleop_config, params, {"publish_stamped_twist": False}],
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
