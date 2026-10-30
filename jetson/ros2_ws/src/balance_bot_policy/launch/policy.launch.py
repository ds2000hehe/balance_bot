import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory("balance_bot_policy"), "config", "policy.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            Node(
                package="balance_bot_policy",
                executable="policy_node",
                name="balance_bot_policy",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
