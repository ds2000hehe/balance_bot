# Balance bot robot runtime: ROS 2 Jazzy + the balance_bot_policy package (USB serial bridge to the ESP32 + policy)
# and the PS4 controller nodes (joy, teleop_twist_joy). One image for both services of docker-compose.yml.
# Multi-arch (x86_64 and arm64, e.g. Jetson / Raspberry Pi 5). Built from this folder (jetson/), see docker-compose.yml.
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-numpy python3-serial python3-yaml \
        ros-jazzy-joy ros-jazzy-teleop-twist-joy \
    && rm -rf /var/lib/apt/lists/*
# the policy runs as ONNX on the CPU (~15 MB instead of ~1 GB for PyTorch)
RUN pip3 install --no-cache-dir --break-system-packages onnxruntime

WORKDIR /ros2_ws
COPY ros2_ws/src/balance_bot_policy src/balance_bot_policy
COPY ros2_ws/src/balance_bot_wheel_teleop src/balance_bot_wheel_teleop
RUN . /opt/ros/jazzy/setup.sh && colcon build && rm -rf build log

COPY entrypoint.sh fastdds_udp.xml /
ENTRYPOINT ["/entrypoint.sh"]
CMD ["robot"]
