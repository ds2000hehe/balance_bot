from glob import glob

from setuptools import setup

package_name = "balance_bot_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="david",
    maintainer_email="davidssweetman@gmail.com",
    description="Runs the Isaac Lab trained balance bot policy on the real robot.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy_node = balance_bot_policy.policy_node:main",
            "teleop_key = balance_bot_policy.teleop_key:main",
            "wheel_serial_bridge = balance_bot_policy.wheel_serial_bridge:main",
            "wheel_check = balance_bot_policy.wheel_check:main",
            "joy_enable = balance_bot_policy.joy_enable:main",
            "latency_check = balance_bot_policy.latency_check:main",
            "brake_check = balance_bot_policy.brake_check:main",
        ],
    },
)
