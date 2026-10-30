"""Wheel test without balancing: teleop commands (geometry_msgs/Twist) straight to wheel speed targets.

Differential drive kinematics, no policy: ``left = (v - w * track / 2) / r``, ``right = (v + w * track / 2) / r``
[rad/s, positive = forward], with acceleration limits, published on the wheel command topic of
``wheel_serial_bridge`` (the same one policy_node uses). Use it to check that both wheels turn the right way and at
the commanded speed: wheels off the ground (robot on a stand), or with the robot resting on a caster / support.

Do NOT run it together with policy_node: both would command the wheels.

Topics:
    cmd_vel_topic      geometry_msgs/Twist             in   linear.x [m/s], angular.z [rad/s] (teleop_twist_joy, ...)
    wheel_cmd_topic    std_msgs/Float32MultiArray      out  [left, right] wheel speed targets [rad/s]
    wheel_state_topic  std_msgs/Float32MultiArray      in   [left, right] measured [rad/s] (only logged)

Without cmd_vel for ``cmd_vel_timeout`` (e.g. the dead-man trigger released) the targets ramp to zero.
"""

import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray


class WheelTeleop(Node):
    def __init__(self):
        super().__init__("wheel_teleop")
        p = self.declare_parameter
        cmd_vel_topic = p("cmd_vel_topic", "/cmd_vel").value
        wheel_cmd_topic = p("wheel_cmd_topic", "/wheel_cmd").value
        wheel_state_topic = p("wheel_state_topic", "/wheel_vel").value
        self.radius = p("wheel_radius", 0.0825).value  # m
        self.track = p("track_width", 0.18).value  # m, wheel center to wheel center
        self.signs = np.array(p("wheel_signs", [1.0, 1.0]).value, dtype=np.float64)
        self.max_speed = p("max_wheel_speed", 10.0).value  # rad/s (the firmware allows 30)
        self.max_acc = p("max_wheel_acc", 30.0).value  # rad/s^2
        self.timeout = p("cmd_vel_timeout", 0.5).value  # s
        self.rate = p("rate", 50.0).value  # Hz
        self.log_period = p("log_period", 1.0).value  # s, 0 = quiet

        self.cmd = (0.0, 0.0)
        self.cmd_time = -1e9
        self.target = np.zeros(2)  # current (ramped) [left, right] forward rad/s
        self.measured = None
        self.last_log = 0.0
        self.pub = self.create_publisher(Float32MultiArray, wheel_cmd_topic, 10)
        self.create_subscription(Twist, cmd_vel_topic, self._on_cmd_vel, 10)
        self.create_subscription(Float32MultiArray, wheel_state_topic, self._on_wheels, qos_profile_sensor_data)
        self.create_timer(1.0 / self.rate, self._step)
        self.get_logger().info(
            f"{cmd_vel_topic} -> {wheel_cmd_topic}: wheel radius {self.radius} m, track {self.track} m, "
            f"max {self.max_speed} rad/s, signs {self.signs.tolist()}. Do not run with policy_node."
        )

    def _on_cmd_vel(self, msg: Twist):
        self.cmd = (float(msg.linear.x), float(msg.angular.z))
        self.cmd_time = time.monotonic()

    def _on_wheels(self, msg: Float32MultiArray):
        if len(msg.data) == 2:
            self.measured = np.array(msg.data, dtype=np.float64) * self.signs  # back to forward-positive

    def wheel_speeds(self, v: float, w: float) -> np.ndarray:
        """[left, right] forward wheel speeds [rad/s] for forward speed ``v`` [m/s] and turn rate ``w`` [rad/s]."""
        half = 0.5 * self.track
        speeds = np.array([v - w * half, v + w * half]) / self.radius
        # keep the turn / straight ratio when limiting: scale both instead of clipping one
        peak = np.abs(speeds).max()
        return speeds * (self.max_speed / peak) if peak > self.max_speed else speeds

    def _step(self):
        now = time.monotonic()
        v, w = self.cmd if now - self.cmd_time < self.timeout else (0.0, 0.0)
        goal = self.wheel_speeds(v, w)
        step = self.max_acc / self.rate
        self.target += np.clip(goal - self.target, -step, step)
        self.pub.publish(Float32MultiArray(data=(self.target * self.signs).astype(np.float32).tolist()))

        if self.log_period > 0 and now - self.last_log >= self.log_period:
            self.last_log = now
            meas = "no /wheel_vel" if self.measured is None else f"[{self.measured[0]:+6.2f}, {self.measured[1]:+6.2f}]"
            self.get_logger().info(
                f"cmd v {v:+.2f} m/s w {w:+.2f} rad/s -> target [{self.target[0]:+6.2f}, {self.target[1]:+6.2f}] "
                f"measured {meas} rad/s (left, right, forward +)"
            )


def main():
    rclpy.init()
    node = WheelTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # leave the wheels stopped (the firmware also switches off 200 ms after the last command)
        if rclpy.ok():
            node.pub.publish(Float32MultiArray(data=[0.0, 0.0]))
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
