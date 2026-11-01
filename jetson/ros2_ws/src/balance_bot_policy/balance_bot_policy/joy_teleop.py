"""Gamepad -> robot: sticks to /cmd_vel while the dead-man (R2) is held, Circle / Cross start / stop balancing.

Replaces teleop_twist_joy + joy_enable. Axis / button numbers differ between drivers (SDL, hid_playstation,
Bluetooth vs USB), so they are parameters: measure them with  ros2 run balance_bot_policy joy_calibrate  which writes
/config/joystick_map.yaml (loaded on top of joystick.yaml).

* The dead-man can be a button (deadman_button) or an analog trigger axis (deadman_axis with its resting and fully
  pressed values: triggers rest at +1, -1 or 0 depending on the driver). It counts as held beyond 70 % of its travel.
* While it is held, /cmd_vel is published at publish_rate (not on every joystick event); releasing it, or the
  controller going quiet for joy_timeout, publishes one zero command.
* Turning: stick right = turn right = negative angular z (ROS convention: counter-clockwise positive).

Topics:
    joy_topic      sensor_msgs/Joy       in
    cmd_vel_topic  geometry_msgs/Twist   out
    enable_topic   std_msgs/Bool         out (policy_node's enable topic)
"""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool


def stick(value: float, deadzone: float) -> float:
    """Stick value with a dead zone around centre, rescaled so full deflection is still 1."""
    if abs(value) <= deadzone:
        return 0.0
    return (abs(value) - deadzone) / (1.0 - deadzone) * (1.0 if value > 0 else -1.0)


class JoyTeleop(Node):
    def __init__(self):
        super().__init__("joy_teleop")
        p = self.declare_parameter
        self.forward_axis = p("forward_axis", 1).value
        self.forward_sign = p("forward_sign", -1.0).value  # value * sign > 0 = forward
        self.forward_scale = p("forward_scale", 0.5).value  # m/s at full stick
        self.turn_axis = p("turn_axis", 2).value
        self.turn_sign = p("turn_sign", -1.0).value  # value * sign > 0 = turn left (counter-clockwise)
        self.turn_scale = p("turn_scale", 1.5).value  # rad/s at full stick
        self.stick_deadzone = p("stick_deadzone", 0.15).value
        self.deadman_button = p("deadman_button", -1).value  # -1: use deadman_axis
        self.deadman_axis = p("deadman_axis", 5).value  # -1: no dead-man (always drives)
        self.deadman_rest = p("deadman_rest", 1.0).value
        self.deadman_pressed = p("deadman_pressed", -1.0).value
        self.start_button = p("start_button", 1).value
        self.stop_button = p("stop_button", 0).value
        self.publish_rate = p("publish_rate", 20.0).value
        self.joy_timeout = p("joy_timeout", 0.5).value
        joy_topic = p("joy_topic", "/joy").value
        cmd_vel_topic = p("cmd_vel_topic", "/cmd_vel").value
        enable_topic = p("enable_topic", "/balance_bot/enable").value

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.enable_pub = self.create_publisher(Bool, enable_topic, 10)
        self.create_subscription(Joy, joy_topic, self._on_joy, 10)
        self.create_timer(1.0 / self.publish_rate, self._tick)
        self.joy = None
        self.joy_time = 0.0
        self.prev_buttons = None
        self.driving = False
        deadman = (f"button {self.deadman_button}" if self.deadman_button >= 0 else
                   f"axis {self.deadman_axis} ({self.deadman_rest:+.0f} -> {self.deadman_pressed:+.0f})"
                   if self.deadman_axis >= 0 else "none")
        self.get_logger().info(
            f"forward: axis {self.forward_axis} x {self.forward_sign:+.0f} ({self.forward_scale} m/s), "
            f"turn: axis {self.turn_axis} x {self.turn_sign:+.0f} ({self.turn_scale} rad/s), dead-man: {deadman}, "
            f"start: button {self.start_button}, stop: button {self.stop_button}"
        )

    @staticmethod
    def _get(values, i: int, default: float = 0.0) -> float:
        return float(values[i]) if 0 <= i < len(values) else default

    def _deadman_held(self, joy: Joy) -> bool:
        if self.deadman_button >= 0:
            return self._get(joy.buttons, self.deadman_button) == 1
        if self.deadman_axis < 0:
            return True
        travel = self.deadman_pressed - self.deadman_rest
        if abs(travel) < 1e-3:
            return False
        return (self._get(joy.axes, self.deadman_axis, self.deadman_rest) - self.deadman_rest) / travel > 0.7

    def _pressed(self, buttons, i: int) -> bool:
        """Button ``i`` went down since the last message (edges only: holding it does not repeat)."""
        now = self._get(buttons, i) == 1
        before = self.prev_buttons is not None and self._get(self.prev_buttons, i) == 1
        return now and not before

    def _on_joy(self, msg: Joy):
        self.joy = msg
        self.joy_time = time.monotonic()
        # stop wins if both are pressed together
        if self._pressed(msg.buttons, self.stop_button):
            self.enable_pub.publish(Bool(data=False))
            self.get_logger().info("stop button: balancing off")
        elif self._pressed(msg.buttons, self.start_button):
            self.enable_pub.publish(Bool(data=True))
            self.get_logger().info("start button: balancing on")
        self.prev_buttons = list(msg.buttons)

    def _tick(self):
        joy = self.joy
        fresh = joy is not None and time.monotonic() - self.joy_time < self.joy_timeout
        if fresh and self._deadman_held(joy):
            cmd = Twist()
            fwd = stick(self._get(joy.axes, self.forward_axis), self.stick_deadzone)
            turn = stick(self._get(joy.axes, self.turn_axis), self.stick_deadzone)
            cmd.linear.x = self.forward_sign * fwd * self.forward_scale
            cmd.angular.z = self.turn_sign * turn * self.turn_scale
            self.cmd_pub.publish(cmd)
            self.driving = True
        elif self.driving:
            self.cmd_pub.publish(Twist())  # released: one zero command, the robot stops and stands
            self.driving = False


def main():
    rclpy.init()
    node = JoyTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
