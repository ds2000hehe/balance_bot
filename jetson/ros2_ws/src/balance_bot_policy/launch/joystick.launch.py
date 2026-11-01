"""Gamepad: joy_node (gamepad -> /joy) and joy_teleop (sticks -> /cmd_vel while R2 is held, Circle / Cross start /
stop balancing). Parameters: ``params_file`` (default config/joystick.yaml), then ``map_file`` on top if it exists
(the axis / button numbers joy_calibrate measured).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def nodes(context):
    params = [LaunchConfiguration("params_file").perform(context)]
    map_file = LaunchConfiguration("map_file").perform(context)
    if map_file and os.path.exists(map_file):
        params.append(map_file)
    else:
        print(f"[joystick.launch] no {map_file or 'map file'}: default axis / button numbers "
              "(run joy_calibrate to measure them)")
    return [
        Node(package="joy", executable="joy_node", name="joy_node", output="screen", parameters=params),
        Node(package="balance_bot_policy", executable="joy_teleop", name="joy_teleop", output="screen",
             parameters=params),
    ]


def generate_launch_description():
    share = get_package_share_directory("balance_bot_policy")
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=os.path.join(share, "config", "joystick.yaml")),
            DeclareLaunchArgument("map_file", default_value="/config/joystick_map.yaml"),
            OpaqueFunction(function=nodes),
        ]
    )
