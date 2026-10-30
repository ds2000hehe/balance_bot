"""Keyboard teleop for the balance bot that publishes /cmd_vel continuously.

Keys set a target speed instead of having to be held down (a terminal cannot see key releases, so hold-to-drive
teleops stutter with the keyboard's auto-repeat delay):

    w / x : faster forward / backward (steps of ``lin_step``)
    a / d : turn left / right (steps of ``ang_step``)
    s / space : stop
    q / Ctrl-C : quit (publishes a stop first)
"""

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist

HELP = """
Balance bot teleop -- publishing on {topic} at {rate:.0f} Hz
  w / x : faster forward / backward     a / d : turn left / right
  s / space : stop                       q : quit
"""


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("balance_bot_teleop")
    topic = node.declare_parameter("topic", "/cmd_vel").value
    rate = node.declare_parameter("rate", 20.0).value
    lin_step = node.declare_parameter("lin_step", 0.25).value
    ang_step = node.declare_parameter("ang_step", 0.5).value
    max_lin = node.declare_parameter("max_lin_vel", 1.5).value
    max_ang = node.declare_parameter("max_ang_vel", 2.0).value
    pub = node.create_publisher(Twist, topic, 10)

    lin, ang = 0.0, 0.0
    settings = termios.tcgetattr(sys.stdin)
    print(HELP.format(topic=topic, rate=rate))
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            # wait one publish period for a key
            if select.select([sys.stdin], [], [], 1.0 / rate)[0]:
                key = sys.stdin.read(1).lower()
                if key == "w":
                    lin += lin_step
                elif key == "x":
                    lin -= lin_step
                elif key == "a":
                    ang += ang_step
                elif key == "d":
                    ang -= ang_step
                elif key in ("s", " "):
                    lin, ang = 0.0, 0.0
                elif key in ("q", "\x03"):
                    break
                lin = max(-max_lin, min(max_lin, round(lin, 3)))
                ang = max(-max_ang, min(max_ang, round(ang, 3)))
                print(f"\rforward {lin:+.2f} m/s   turn {ang:+.2f} rad/s    ", end="", flush=True)
            msg = Twist()
            msg.linear.x = lin
            msg.angular.z = ang
            pub.publish(msg)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        pub.publish(Twist())
        print()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
