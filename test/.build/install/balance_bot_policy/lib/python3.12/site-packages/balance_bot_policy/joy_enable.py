"""Start / stop balancing from a gamepad: a button press publishes true / false on the policy's enable topic.

Defaults are for a PS4 controller (DualShock 4) on Linux (hid_playstation driver) read by the ``joy`` package's
``joy_node``: Circle (button 1) starts balancing, Cross (button 0) stops it. Check the button numbers with
``ros2 topic echo /joy`` and set ``start_button`` / ``stop_button`` if your controller differs.

Stopping switches the motors off, so the robot falls over: hold it, or stop it while it is near the ground.
If the controller disconnects, the robot keeps balancing in place (cmd_vel goes quiet and the policy node ramps the
command to zero); reconnect it (PS button) and press Cross to stop.

Topics:
    joy_topic      sensor_msgs/Joy    in
    enable_topic   std_msgs/Bool      out (policy_node's enable topic)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool


class JoyEnable(Node):
    def __init__(self):
        super().__init__("joy_enable")
        p = self.declare_parameter
        self.start_button = p("start_button", 1).value  # Circle
        self.stop_button = p("stop_button", 0).value  # Cross
        joy_topic = p("joy_topic", "/joy").value
        enable_topic = p("enable_topic", "/balance_bot/enable").value
        self.pub = self.create_publisher(Bool, enable_topic, 10)
        self.create_subscription(Joy, joy_topic, self._on_joy, 10)
        self.prev = None
        self.get_logger().info(
            f"{joy_topic} button {self.start_button} -> start balancing, button {self.stop_button} -> stop "
            f"(on {enable_topic})"
        )

    def _pressed(self, buttons, i: int) -> bool:
        """Button ``i`` went down since the last message (edges only, so holding it does not repeat)."""
        now = i < len(buttons) and buttons[i] == 1
        before = self.prev is not None and i < len(self.prev) and self.prev[i] == 1
        return now and not before

    def _on_joy(self, msg: Joy):
        # stop wins if both are pressed together
        if self._pressed(msg.buttons, self.stop_button):
            self.pub.publish(Bool(data=False))
            self.get_logger().info("stop button: balancing off")
        elif self._pressed(msg.buttons, self.start_button):
            self.pub.publish(Bool(data=True))
            self.get_logger().info("start button: balancing on")
        self.prev = list(msg.buttons)


def main():
    rclpy.init()
    node = JoyEnable()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
