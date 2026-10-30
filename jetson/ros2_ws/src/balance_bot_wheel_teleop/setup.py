from glob import glob

from setuptools import setup

package_name = "balance_bot_wheel_teleop"

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
    description="Wheel test without balancing: teleop commands (cmd_vel) straight to wheel speed targets.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["wheel_teleop = balance_bot_wheel_teleop.wheel_teleop:main"]},
)
