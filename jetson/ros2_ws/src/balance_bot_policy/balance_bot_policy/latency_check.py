"""Measure the real control delay: throttle pulse on /wheel_duty -> first reaction in the IMU gyro and wheel speed.

    MODE=bridge docker compose up -d          # bridge only, no policy
    docker compose exec balance_bot ros2 run balance_bot_policy latency_check

Hold the robot in the air by its body (wheels free, not touching anything), still. Both wheels get short throttle
pulses; spinning a wheel up twists the body the other way, which the gyro sees as soon as the motor produces torque.
The time from publishing the pulse to that twist is the whole loop the policy lives with (ROS, USB serial, Arduino,
driver, motor, IMU filtering), minus the policy's own sampling (up to 20 ms at 50 Hz, the training covers that).
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray


class LatencyCheck(Node):
    def __init__(self):
        super().__init__("latency_check")
        p = self.declare_parameter
        self.duty = p("duty", 0.2).value
        self.pulse_s = p("pulse_s", 0.15).value
        self.pulses = p("pulses", 8).value
        self.lock = threading.Lock()
        self.imu = []  # (receive time, |gyro|)
        self.wheels = []  # (receive time, max |wheel speed|)
        self.duty_pub = self.create_publisher(Float32MultiArray, p("wheel_duty_topic", "/wheel_duty").value, 10)
        self.create_subscription(Imu, p("imu_topic", "/imu").value, self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, p("wheel_state_topic", "/wheel_vel").value, self._on_wheels,
                                 qos_profile_sensor_data)

    def _on_imu(self, msg: Imu):
        g = msg.angular_velocity
        with self.lock:
            self.imu.append((time.monotonic(), float(np.sqrt(g.x**2 + g.y**2 + g.z**2))))

    def _on_wheels(self, msg: Float32MultiArray):
        with self.lock:
            self.wheels.append((time.monotonic(), float(np.max(np.abs(msg.data))) if len(msg.data) else 0.0))

    def hold(self, duty: float, seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.duty_pub.publish(Float32MultiArray(data=[duty, duty]))
            time.sleep(0.01)

    def series(self, which, t0, t1):
        with self.lock:
            data = [(t, v) for t, v in (self.imu if which == "imu" else self.wheels) if t0 <= t <= t1]
        return np.array(data) if data else np.zeros((0, 2))


def first_rise(s: np.ndarray, t_start: float, baseline_until: float, k: float = 6.0, floor: float = 0.05):
    """Time after t_start when the signal first leaves its noise band (baseline mean + k std, at least +floor)."""
    base = s[s[:, 0] < baseline_until, 1]
    after = s[s[:, 0] >= t_start]
    if len(base) < 5 or len(after) == 0:
        return float("nan")
    level = base.mean() + max(k * base.std(), floor)
    hit = after[after[:, 1] > level]
    return float(hit[0, 0] - t_start) if len(hit) else float("nan")


def main():
    rclpy.init()
    node = LatencyCheck()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(__doc__)
    input("Robot held in the air by its body, wheels free and still? Press Enter to start ")
    node.hold(0.0, 1.5)
    with node.lock:
        n_imu, n_wheels = len(node.imu), len(node.wheels)
    if n_imu < 50:
        raise SystemExit(f"only {n_imu} IMU messages in 1.5 s: is the IMU driver running (imu_topic)?")
    if n_wheels < 50:
        raise SystemExit(f"only {n_wheels} wheel speed messages in 1.5 s: is the bridge running (MODE=bridge)?")
    gyro_lat, wheel_lat = [], []
    for i in range(node.pulses):
        sign = 1.0 if i % 2 == 0 else -1.0
        t_base = time.monotonic()
        node.hold(0.0, 0.6)  # quiet baseline
        t0 = time.monotonic()
        node.hold(sign * node.duty, node.pulse_s)
        node.hold(0.0, 0.8)  # coast down
        g = first_rise(node.series("imu", t_base, t0 + 0.5), t0, t0)
        w = first_rise(node.series("wheels", t_base, t0 + 0.5), t0, t0, k=0.0, floor=0.3)
        gyro_lat.append(g)
        wheel_lat.append(w)
        print(f"  pulse {i + 1}: gyro reacts after {g * 1000:5.0f} ms, wheel speed seen after {w * 1000:5.0f} ms")
    node.hold(0.0, 0.3)
    g = np.nanmedian(gyro_lat) * 1000
    w = np.nanmedian(wheel_lat) * 1000
    print(f"\nmedian: gyro {g:.0f} ms, wheel speed {w:.0f} ms (from publishing the throttle command)")
    print("Paste these lines to compare with the training's delay range (5-40 ms on top of the policy's sampling).")
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
