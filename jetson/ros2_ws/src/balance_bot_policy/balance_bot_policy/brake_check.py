"""Measure how hard the drivers slow a spinning wheel when the throttle drops (does the ZS-X11H brake, or coast?).

    MODE=bridge docker compose up -d          # bridge only, no policy
    docker compose exec -it balance_bot /entrypoint.sh ros2 run balance_bot_policy brake_check

Hold the robot in the air by its body, wheels free. Both wheels spin up forward to a steady speed (duty ``spin``), then
the throttle drops to a lower duty (``low``) or to zero, and the wheel speed is recorded for a second.

The training model assumes the driver drives current both ways: dropping the throttle below the wheel's speed
brakes it hard (current limit), so the wheel speed falls to the new duty's speed within a few tens of ms. If the real
driver only coasts, the wheel slows slowly, and a policy trained on the model cannot slow the robot down in time.
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray


class BrakeCheck(Node):
    def __init__(self):
        super().__init__("brake_check")
        p = self.declare_parameter
        self.spin = p("spin", 0.4).value
        self.lows = p("lows", [0.2, 0.1, 0.0]).value
        self.lock = threading.Lock()
        self.wheels = []  # (time, left, right) |speed|
        self.duty_pub = self.create_publisher(Float32MultiArray, p("wheel_duty_topic", "/wheel_duty").value, 10)
        self.create_subscription(Float32MultiArray, p("wheel_state_topic", "/wheel_vel").value, self._on_wheels,
                                 qos_profile_sensor_data)

    def _on_wheels(self, msg: Float32MultiArray):
        if len(msg.data) >= 2:
            with self.lock:
                self.wheels.append((time.monotonic(), abs(msg.data[0]), abs(msg.data[1])))

    def hold(self, duty: float, seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.duty_pub.publish(Float32MultiArray(data=[duty, duty]))
            time.sleep(0.01)

    def series(self, t0, t1):
        with self.lock:
            d = [w for w in self.wheels if t0 <= w[0] <= t1]
        return np.array(d) if d else np.zeros((0, 3))


def speed_at(s: np.ndarray, t: float) -> float:
    """Mean of both wheels' speeds, interpolated at time t."""
    return float(np.interp(t, s[:, 0], 0.5 * (s[:, 1] + s[:, 2])))


def main():
    rclpy.init()
    node = BrakeCheck()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(__doc__)
    input("Robot held in the air by its body, wheels free? Press Enter to start ")
    node.hold(0.0, 1.0)
    with node.lock:
        if len(node.wheels) < 30:
            raise SystemExit("no wheel speeds: is the bridge running (MODE=bridge)?")
    rows = []
    for low in node.lows:
        node.hold(node.spin, 2.0)  # spin up (firmware ramps the throttle) and settle
        t0 = time.monotonic()
        node.hold(low, 1.2)
        s = node.series(t0 - 0.3, t0 + 1.2)
        v0 = float(np.mean(0.5 * (s[s[:, 0] < t0, 1] + s[s[:, 0] < t0, 2])))
        v = [speed_at(s, t0 + dt) for dt in (0.05, 0.1, 0.2, 0.5, 1.0)]
        decel = (v0 - v[2]) / 0.2
        rows.append((low, v0, v, decel))
        print(f"  {node.spin:.2f} -> {low:.2f}: {v0:5.1f} rad/s, after 50/100/200/500/1000 ms "
              + " ".join(f"{x:5.1f}" for x in v) + f"  | slows {decel:6.0f} rad/s^2 over the first 200 ms")
        node.hold(0.0, 2.5)  # coast to a stop
    node.hold(0.0, 0.3)
    full = node.spin / max(rows[0][1], 1e-3)
    print(f"\nsteady speed at duty {node.spin:.2f}: {rows[0][1]:.1f} rad/s (full throttle ~{1 / full:.0f} rad/s)")
    print("The training model brakes to the lower duty's speed in ~50 ms (hundreds of rad/s^2).")
    print("If the drops to 0.2 / 0.1 slow about as gently as the drop to 0, the driver coasts instead of braking.")
    print("Paste these lines.")
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
