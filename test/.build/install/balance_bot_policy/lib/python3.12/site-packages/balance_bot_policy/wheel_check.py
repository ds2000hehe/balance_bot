"""Bench test of the wheels (and IMU) before the policy is allowed to drive the robot.

Checks that the wheels behave the way the policy was trained to expect: directions and left/right, hall counts,
the drive's response through the same topics and sign conventions as the policy node (config/policy.yaml), and that
the wheels stop when commands stop. It also measures the motor and driver characteristics that the training's drive
model (``mdp/hoverboard.py``) assumes, so the model can be matched to the real robot.

**Lift the wheels off the ground** (robot on a stand or held) and start only the serial bridge, not the policy:

    ros2 launch balance_bot_policy robot.launch.py mode:=bridge port:=/dev/ttyUSB0
    ros2 run balance_bot_policy wheel_check

Options: ``--yes`` skips the steps that need your hands or eyes, ``--params`` another policy.yaml, ``--csv`` where to
write the log of every command and measurement (default ~/wheel_check_<date>.csv). Ctrl-C stops the wheels.
"""

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

from .policy_core import quat_rotate_inverse

PASS, WARN, FAIL, INFO, SKIP = "PASS", "WARN", "FAIL", "INFO", "SKIP"
WHEELS = ("left", "right")

# what the training's drive model assumes (mdp/hoverboard.py, HoverboardDriveCfg) and the firmware's defaults
MODEL_COUNTS_PER_REV = 90
MODEL_FREE_SPEED = (33.0 / 0.65, 41.0 / 0.55)  # battery voltage / back-EMF constant ranges [rad/s at full duty]
MODEL_DEAD_ZONE_MAX = 0.06
DEAD_ZONE_TOLERANCE = 0.01  # the ramp measurement reads a little high (delay, 0.5 rad/s detection threshold)
FIRMWARE_MOTOR_MAX_RAD_S = 60.0
# the training model's own results in these in-air tests (sim_bridge.py --hold, randomized motors): a free wheel
# with the driver's dead zone, friction and coarse low-speed hall feedback is rough at low speed; on the ground
# the robot's weight smooths it out. Real wheels that do about as well as the model are what the policy expects.
MODEL_IN_AIR = {
    "peak": (23.0, 28.0),
    "steady_err": (0.6, 2.4),
    "ripple": (4.2, 5.5),
    "hold_mean_err": (0.6, 6.1),
    "hold_std": (1.1, 15.6),
    "sine_rms": (3.1, 4.7),
    "sine_lag": (0.025, 0.03),
    "stop": (0.087, 0.117),
}


@dataclass
class Result:
    name: str
    status: str
    measured: str
    expected: str
    note: str = ""


class Bench:
    def __init__(self, params: dict, csv_path: str):
        self.p = params
        self.signs = np.array(params.get("wheel_signs", [1.0, 1.0]), dtype=float)
        self.imu_to_body = np.array(params.get("imu_to_body", np.eye(3).flatten().tolist()), dtype=float).reshape(3, 3)
        self.node = rclpy.create_node("wheel_check")
        self.cmd_pub = self.node.create_publisher(Float32MultiArray, params.get("wheel_cmd_topic", "/wheel_cmd"), 10)
        self.duty_pub = self.node.create_publisher(Float32MultiArray, "/wheel_duty", 10)
        self.node.create_subscription(
            Float32MultiArray, params.get("wheel_state_topic", "/wheel_vel"), self._on_wheels, qos_profile_sensor_data
        )
        self.node.create_subscription(Imu, params.get("imu_topic", "/imu"), self._on_imu, qos_profile_sensor_data)
        self.lock = threading.Lock()
        self.wheel = []  # (t, left_fwd, right_fwd)
        self.imu = []  # (t, gravity_body xyz, gyro_body xyz)
        self.cmd = None  # (kind, fn(t) -> [left, right] forward, start time)
        self.label = ""
        self.running = True
        self.csv = open(csv_path, "w")
        self.csv.write("time,test,kind,cmd_left,cmd_right,meas_left,meas_right\n")
        self.t0 = time.monotonic()
        self.last_cmd = (None, 0.0, 0.0)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        self.pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.pub_thread.start()

    # -- ROS

    def _on_wheels(self, msg):
        if len(msg.data) != 2:
            return
        fwd = np.array(msg.data, dtype=float) * self.signs
        t = time.monotonic()
        with self.lock:
            self.wheel.append((t, fwd[0], fwd[1]))
            kind, cl, cr = self.last_cmd
        self.csv.write(f"{t - self.t0:.4f},{self.label},{kind or ''},{cl:.4f},{cr:.4f},{fwd[0]:.4f},{fwd[1]:.4f}\n")

    def _on_imu(self, msg):
        q = msg.orientation
        has_q = msg.orientation_covariance[0] >= 0.0 and (q.x, q.y, q.z, q.w) != (0.0, 0.0, 0.0, 0.0)
        if has_q:
            g = quat_rotate_inverse(np.array([q.x, q.y, q.z, q.w]), np.array([0.0, 0.0, -1.0]))
        else:
            a = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            g = -a / max(np.linalg.norm(a), 1e-6)
        w = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        with self.lock:
            self.imu.append((time.monotonic(), *(self.imu_to_body @ g), *(self.imu_to_body @ w)))

    def _publish_loop(self):
        # like the policy node: 50 Hz, forward speeds converted with wheel_signs
        while self.running:
            cmd = self.cmd
            if cmd is not None:
                kind, fn, start = cmd
                fwd = np.asarray(fn(time.monotonic() - start), dtype=float)
                msg = Float32MultiArray(data=[float(v) for v in fwd * self.signs])
                (self.duty_pub if kind == "duty" else self.cmd_pub).publish(msg)
                with self.lock:
                    self.last_cmd = (kind, float(fwd[0]), float(fwd[1]))
            else:
                with self.lock:
                    self.last_cmd = (None, 0.0, 0.0)
            time.sleep(0.02)

    # -- commands

    def run(self, kind, fn, seconds, label):
        """Command ``fn(t)`` (forward [left, right]) for ``seconds``; returns (t, cmd, meas) arrays, t from the start."""
        self.label = label
        start = time.monotonic()
        self.cmd = None if kind is None else (kind, fn, start)
        time.sleep(seconds)
        return self.samples_since(start, fn if kind is not None else (lambda t: (0.0, 0.0)))

    def stop(self, seconds=0.0):
        self.cmd = None
        time.sleep(seconds)

    def samples_since(self, start, fn=None):
        with self.lock:
            rows = [r for r in self.wheel if r[0] >= start]
        if not rows:
            return np.zeros(0), np.zeros((0, 2)), np.zeros((0, 2))
        a = np.array(rows)
        t = a[:, 0] - start
        cmd = np.array([fn(x) for x in t], dtype=float) if fn is not None else np.zeros((len(t), 2))
        return t, cmd, a[:, 1:3]

    def imu_since(self, start):
        with self.lock:
            rows = [r for r in self.imu if r[0] >= start]
        return np.array(rows) if rows else np.zeros((0, 7))

    def close(self):
        self.cmd = None
        # a few explicit zero speed commands, then silence (the ESP32 switches the motors off after its timeout)
        for _ in range(10):
            self.cmd_pub.publish(Float32MultiArray(data=[0.0, 0.0]))
            time.sleep(0.02)
        self.running = False
        self.pub_thread.join(timeout=1.0)
        self.executor.shutdown()
        self.spin_thread.join(timeout=2.0)
        self.node.destroy_node()
        self.csv.close()


# -- helpers


def first_time(t, cond):
    idx = np.nonzero(cond)[0]
    return float(t[idx[0]]) if len(idx) else math.nan


def ask(question, interactive):
    if not interactive:
        return None
    while True:
        a = input(f"\n>>> {question} [y/n] ").strip().lower()
        if a in ("y", "n"):
            return a == "y"


def wait_enter(text, interactive):
    if interactive:
        input(f"\n>>> {text} (press Enter) ")


def status(ok, warn=False):
    return PASS if ok else (WARN if warn else FAIL)


# -- tests


def test_health(b: Bench, results):
    start = time.monotonic()
    time.sleep(2.0)
    t, _, _ = b.samples_since(start)
    imu = b.imu_since(start)
    rate = len(t) / 2.0
    results.append(Result("wheel speed messages", status(rate >= 80, rate >= 40), f"{rate:.0f} Hz", ">= 80 Hz (ESP32: 100)"))
    imu_rate = len(imu) / 2.0
    results.append(Result("IMU messages", status(imu_rate >= 90, imu_rate >= 40), f"{imu_rate:.0f} Hz", ">= 100 Hz",
                          "the policy runs at 50 Hz on fresh gyro data" if imu_rate < 90 else ""))
    others = b.node.count_publishers(b.p.get("wheel_cmd_topic", "/wheel_cmd")) - 1
    if others > 0:
        results.append(Result("other /wheel_cmd publishers", FAIL, str(others), "0",
                              "stop the policy node (robot.launch.py mode:=bridge) and run again"))
        return False
    if rate == 0:
        results.append(Result("wheel speeds", FAIL, "none", "", "is the serial bridge running and the ESP32 connected?"))
        return False
    return True


def test_hand_turn(b: Bench, results, interactive):
    if not interactive:
        results.append(Result("hall counts per turn (by hand)", SKIP, "", "", "needs --yes off"))
        return
    for i, name in enumerate(WHEELS):
        b.cmd = ("duty", lambda t: (0.0, 0.0), time.monotonic())  # driver on, no push: the wheels turn freely
        b.label = f"hand_{name}"
        wait_enter(f"Mark the {name.upper()} wheel (left/right seen from BEHIND the robot). When ready, press Enter, "
                   f"then turn it exactly ONE turn FORWARD by hand", interactive)
        start = time.monotonic()
        wait_enter("Done turning", interactive)
        t, _, meas = b.samples_since(start)
        angle = float(np.sum(meas[1:, i] * np.diff(t))) if len(t) > 1 else 0.0
        other = float(np.sum(np.abs(meas[1:, 1 - i]) * np.diff(t))) if len(t) > 1 else 0.0
        turns = angle / (2 * math.pi)
        note = ""
        if turns < -0.5:
            note = f"counts backward: flip {name.upper()}_FEEDBACK_INVERT in the sketch (or wheel_signs)"
        elif abs(turns) < 0.5:
            note = "hardly any counts: check the hall wiring / dividers"
        elif abs(turns - 1.0) > 0.2:
            note = f"hall counts per turn are {MODEL_COUNTS_PER_REV * turns:.0f}, not {MODEL_COUNTS_PER_REV}: set POLE_PAIRS"
        if other > math.pi:
            note += " | the OTHER wheel counted too: left/right swapped?"
        results.append(Result(f"{name}: one turn by hand", status(abs(turns - 1.0) <= 0.2 and other <= math.pi),
                              f"{turns:+.2f} turns", "+1.00 +-0.2", note))
    b.stop()


def test_open_loop(b: Bench, results, interactive):
    """Dead zone, direction, left/right, speed per duty, spin-up, coasting."""
    info = {}
    for i, name in enumerate(WHEELS):
        def ramp(t, i=i):
            d = min(0.2, 0.05 * t)  # 0 -> 0.2 duty over 4 s
            return (d, 0.0) if i == 0 else (0.0, d)

        t, cmd, meas = b.run("duty", ramp, 4.5, f"deadzone_{name}")
        moving = np.abs(meas[:, i]) > 0.5
        dead_zone = float(cmd[moving, i][0]) if moving.any() else math.nan
        info[f"dead_zone_{name}"] = dead_zone
        if math.isnan(dead_zone):
            results.append(Result(f"{name}: starts turning", FAIL, "not moving at duty 0.2", "moves below 0.06",
                                  "check power and wiring, and that the drivers' BRAKE / STOP inputs are at run level"))
            b.stop(0.5)
            return info
        results.append(Result(f"{name}: dead zone", status(dead_zone <= MODEL_DEAD_ZONE_MAX + DEAD_ZONE_TOLERANCE, True),
                              f"duty {dead_zone:.3f}",
                              f"<= {MODEL_DEAD_ZONE_MAX} (training)",
                              "" if dead_zone <= MODEL_DEAD_ZONE_MAX + DEAD_ZONE_TOLERANCE
                              else "larger than trained: tell me, I'll widen the model"))
        # hold 0.2 duty while asking
        b.cmd = ("duty", lambda t, i=i: (0.2, 0.0) if i == 0 else (0.0, 0.2), time.monotonic())
        time.sleep(1.0)
        t, _, meas = b.samples_since(time.monotonic() - 0.8)
        sign = float(np.mean(meas[:, i])) if len(t) else 0.0
        crosstalk = float(np.mean(np.abs(meas[:, 1 - i]))) if len(t) else 0.0
        seen_forward = ask(f"Is the {name.upper()} wheel (seen from behind) spinning FORWARD (top of the wheel moving "
                           f"to the robot's front)?", interactive)
        measured_forward = sign > 0
        if seen_forward is None:
            ok = measured_forward
            note = "" if ok else f"measured backward: flip {name.upper()}_FEEDBACK_INVERT or _MOTOR_INVERT (check by eye)"
        elif seen_forward and measured_forward:
            ok, note = True, ""
        elif seen_forward and not measured_forward:
            ok, note = False, f"turns forward but measures backward: flip {name.upper()}_FEEDBACK_INVERT"
        elif not seen_forward and not measured_forward:
            ok, note = False, f"turns backward (and measures it): flip {name.upper()}_MOTOR_INVERT"
        else:
            ok, note = False, f"turns backward but measures forward: flip {name.upper()}_MOTOR_INVERT and _FEEDBACK_INVERT"
        if crosstalk > 0.5:
            ok, note = False, (note + " | the OTHER wheel moves: left/right swapped (pins or wheel_signs order)").strip(" |")
        results.append(Result(f"{name}: direction (duty +0.2)", status(ok), f"{sign:+.1f} rad/s"
                              + ("" if seen_forward is None else (", seen forward" if seen_forward else ", seen BACKWARD")),
                              "forward, positive", note))
        b.stop(1.5)

    # speed per duty and spin-up (both wheels), then slowing from 0.3 to 0.1 duty
    t, cmd, meas = b.run("duty", lambda t: (0.3, 0.3), 2.5, "duty_0.3")
    steady = meas[t > 1.8].mean(axis=0) if (t > 1.8).any() else np.full(2, math.nan)
    free = steady / 0.3
    for i, name in enumerate(WHEELS):
        tau_up = first_time(t, meas[:, i] > 0.63 * steady[i])
        info[f"free_speed_{name}"], info[f"tau_up_{name}"] = float(free[i]), tau_up
        lo, hi = MODEL_FREE_SPEED
        results.append(Result(f"{name}: speed at full duty", status(lo <= free[i] <= hi, True), f"{free[i]:.1f} rad/s",
                              f"{lo:.0f}-{hi:.0f} (training), firmware MOTOR_MAX_RAD_S {FIRMWARE_MOTOR_MAX_RAD_S:.0f}",
                              f"set MOTOR_MAX_RAD_S = {free[i]:.0f}" if abs(free[i] - FIRMWARE_MOTOR_MAX_RAD_S) > 6 else ""))
        results.append(Result(f"{name}: spin-up time constant (in air)", INFO, f"{tau_up * 1000:.0f} ms", "", ""))
    t2, cmd2, meas2 = b.run("duty", lambda t: (0.1, 0.1), 3.0, "duty_0.1")
    for i, name in enumerate(WHEELS):
        target = steady[i] / 3.0
        tau_down = first_time(t2, meas2[:, i] < steady[i] - 0.63 * (steady[i] - target))
        info[f"tau_down_{name}"] = tau_down
        tau_up = info[f"tau_up_{name}"]
        active = not math.isnan(tau_down) and tau_down < 3 * max(tau_up, 0.01)
        results.append(Result(f"{name}: lowering the duty", PASS if active else WARN,
                              f"slows in {tau_down * 1000:.0f} ms" if not math.isnan(tau_down) else "coasts (> 3 s)",
                              "driver slows the wheel (about as fast as it spins up)",
                              "" if active else "the wheel only coasts: the training model assumes the driver slows it "
                              "(drives current both ways); tell me"))
    b.stop(2.0)
    return info


def vs_model(value, key, margin=1.3, fail=None):
    """PASS when about as good as the training model (or better), WARN when clearly worse, FAIL beyond ``fail``."""
    lo, hi = MODEL_IN_AIR[key]
    if math.isnan(value) or (fail is not None and value > fail):
        return FAIL
    return PASS if value <= hi * margin else WARN


def model_range(key, unit="", scale=1.0, fmt="{:.1f}"):
    lo, hi = MODEL_IN_AIR[key]
    return f"model {fmt.format(lo * scale)}-{fmt.format(hi * scale)}{unit}"


def step_metrics(t, cmd, meas, t_step, value):
    after = t >= t_step
    dead = first_time(t[after], np.abs(meas[after]).max(axis=1) > 0.5) - t_step if value != 0 else math.nan
    rise = max(first_time(t[after], meas[after, 0] * np.sign(value) > 0.9 * abs(value)),
               first_time(t[after], meas[after, 1] * np.sign(value) > 0.9 * abs(value))) - t_step
    return dead, rise


def test_closed_loop(b: Bench, results):
    # step 0 -> 10 rad/s
    t, cmd, meas = b.run("speed", lambda t: (10.0, 10.0) if t >= 0.5 else (0.0, 0.0), 3.0, "step_10")
    dead, rise = step_metrics(t, cmd, meas, 0.5, 10.0)
    late = meas[t > 1.8]
    err = float(np.abs(late.mean(axis=0) - 10.0).max()) if len(late) else math.nan
    ripple = float(late.std(axis=0).max()) if len(late) else math.nan
    peak = float(meas[t > 0.5].max()) if (t > 0.5).any() else math.nan
    results.append(Result("speed step 0->10: dead time", status(dead <= 0.05, dead <= 0.08), f"{dead * 1000:.0f} ms",
                           "<= 50 ms (trained with 0-30 ms delay)"))
    results.append(Result("speed step 0->10: rise to 9", status(rise <= 0.15, rise <= 0.3), f"{rise * 1000:.0f} ms",
                           "<= 150 ms"))
    results.append(Result("speed step 0->10: peak (in air)", vs_model(peak, "peak", fail=45.0), f"{peak:.1f} rad/s",
                           model_range("peak")))
    worst = max(vs_model(err, "steady_err"), vs_model(ripple, "ripple"), key=[PASS, WARN, FAIL].index)
    results.append(Result("speed 10: error / ripple (in air)", worst, f"{err:.2f} / {ripple:.2f} rad/s",
                           f"{model_range('steady_err')} / {model_range('ripple')}"))
    # reversal 10 -> -10
    t, cmd, meas = b.run("speed", lambda t: (10.0, 10.0) if t < 0.8 else (-10.0, -10.0), 2.5, "reverse")
    rev = max(first_time(t[t > 0.8], meas[t > 0.8, 0] < -9), first_time(t[t > 0.8], meas[t > 0.8, 1] < -9)) - 0.8
    results.append(Result("reverse 10 -> -10", status(rev <= 0.4, rev <= 0.8), f"{rev * 1000:.0f} ms", "<= 400 ms",
                          "the driver reverses directly"))
    # slow speed hold
    t, cmd, meas = b.run("speed", lambda t: (1.0, 1.0), 3.0, "hold_1")
    late = meas[t > 1.0]
    mean, std = late.mean(axis=0), late.std(axis=0)
    worst = max(vs_model(float(np.abs(mean - 1).max()), "hold_mean_err"), vs_model(float(std.max()), "hold_std"),
                key=[PASS, WARN, FAIL].index)
    if np.any(mean < 0):
        worst = FAIL
    results.append(Result("hold 1 rad/s (in air)", worst, f"mean {mean[0]:.2f}/{mean[1]:.2f}  std {std[0]:.2f}/{std[1]:.2f}",
                          f"mean 1 +-{MODEL_IN_AIR['hold_mean_err'][1]:.1f}, std <= {MODEL_IN_AIR['hold_std'][1]:.0f} (model)"))
    # balancing-like back and forth: +-3 rad/s at 1.5 Hz
    fn = lambda t: (3.0 * math.sin(2 * math.pi * 1.5 * t),) * 2  # noqa: E731
    t, cmd, meas = b.run("speed", fn, 4.0, "sine")
    sel = t > 1.0
    rms = float(np.sqrt(np.mean((meas[sel] - cmd[sel]) ** 2))) if sel.any() else math.nan
    # lag: shift of the command that best matches the measurement
    grid = np.arange(1.0, t[-1], 0.005)
    m_i = np.interp(grid, t, meas.mean(axis=1))
    lags = np.arange(0, 0.15, 0.005)
    errs = [np.mean((m_i - np.array([fn(x - lag)[0] for x in grid])) ** 2) for lag in lags]
    lag = float(lags[int(np.argmin(errs))])
    worst = max(vs_model(rms, "sine_rms"), status(lag <= 0.06, lag <= 0.1), key=[PASS, WARN, FAIL].index)
    results.append(Result("back and forth +-3 rad/s 1.5 Hz (in air)", worst, f"rms {rms:.2f} rad/s, lag {lag * 1000:.0f} ms",
                          f"{model_range('sine_rms')}, lag <= 60 ms"))
    # top speed
    t, cmd, meas = b.run("speed", lambda t: (25.0, 25.0), 2.0, "max_25")
    reach = max(first_time(t, meas[:, 0] > 24), first_time(t, meas[:, 1] > 24))
    results.append(Result("reaches 25 rad/s", status(reach <= 1.0), f"{reach * 1000:.0f} ms" if not math.isnan(reach)
                          else f"max {meas.max():.1f} rad/s", "<= 1 s (policy may ask for 30)"))
    # stop: speed 0 (the speed loop slows the wheels), and command timeout
    t, cmd, meas = b.run("speed", lambda t: (0.0, 0.0), 1.5, "stop_cmd")
    stop = max(first_time(t, np.abs(meas[:, 0]) < 1.0), first_time(t, np.abs(meas[:, 1]) < 1.0))
    results.append(Result("speed 25 -> 0", status(stop <= 0.3, stop <= 0.6), f"{stop * 1000:.0f} ms",
                          f"<= 300 ms ({model_range('stop', ' ms', 1000, '{:.0f}')})"))
    # without commands the ESP32 switches the motors off after 200 ms: the wheels must stop being driven (coast)
    b.run("speed", lambda t: (10.0, 10.0), 1.5, "pre_timeout")
    t, _, meas = b.run(None, None, 1.5, "timeout")
    speed = np.abs(meas).max(axis=1)
    early = float(speed[t <= 0.25].max()) if (t <= 0.25).any() else math.nan
    late = float(speed[t >= 1.3].max()) if (t >= 1.3).any() else math.nan
    off = late < 1.0 or late < 0.7 * early
    results.append(Result("commands stop -> motors off", status(off), f"{early:.1f} -> {late:.1f} rad/s",
                          "slowing down (coasting) after the 200 ms timeout",
                          "" if off else "still driven: check CMD_TIMEOUT_MS in the sketch"))


def test_imu(b: Bench, results, interactive):
    wait_enter("Hold the robot UPRIGHT (balancing position) and still", interactive)
    start = time.monotonic()
    time.sleep(2.0)
    imu = b.imu_since(start)
    if len(imu) == 0:
        results.append(Result("IMU", FAIL, "no data", "", "check imu_topic and RMW_IMPLEMENTATION"))
        return
    g = imu[:, 1:4].mean(axis=0)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g[2] / max(np.linalg.norm(g), 1e-6)))))
    bias = float(np.abs(imu[:, 4:7].mean(axis=0)).max())
    results.append(Result("IMU upright", status(tilt <= 10, tilt <= 20), f"tilt {tilt:.1f} deg", "<= 10 deg",
                          "z axis not up? check imu_to_body" if tilt > 20 else ""))
    results.append(Result("gyro at rest", status(bias <= 0.05, bias <= 0.15), f"{bias:.3f} rad/s", "<= 0.05"))
    if not interactive:
        results.append(Result("IMU lean / pitch rate", SKIP, "", "", "needs --yes off"))
        return
    wait_enter("Lean the robot about 20 deg FORWARD (towards its front) and hold it", interactive)
    start = time.monotonic()
    time.sleep(1.5)
    g = b.imu_since(start)[:, 1:4].mean(axis=0)
    pitch = math.degrees(math.atan2(g[0], -g[2]))
    side = math.degrees(math.atan2(g[1], -g[2]))
    ok = pitch > 10 and abs(side) < 0.5 * pitch
    note = ""
    if not ok:
        note = ("leaning forward reads as BACKWARD: imu_to_body x axis reversed" if pitch < -10 else
                "leaning forward reads as SIDEWAYS: imu_to_body x/y swapped" if abs(side) > 10 else "lean further")
    results.append(Result("IMU: forward lean", status(ok), f"pitch {pitch:+.0f} deg, roll {side:+.0f} deg",
                          "pitch about +20, roll ~0", note))
    wait_enter("Rock the robot forward and back a few times for 3 s (start now)", interactive)
    start = time.monotonic()
    time.sleep(3.0)
    imu = b.imu_since(start)
    if len(imu) > 10:
        pitch = np.unwrap(np.arctan2(imu[:, 1], -imu[:, 3]))
        dpitch = np.gradient(pitch, imu[:, 0])
        if np.std(dpitch) < 0.05:
            results.append(Result("gyro pitch rate sign", FAIL, "no rocking seen", "> +0.7", "rock the robot harder"))
            return
        corr = float(np.corrcoef(dpitch, imu[:, 5])[0, 1])
        results.append(Result("gyro pitch rate sign", status(corr > 0.7), f"corr {corr:+.2f}", "> +0.7",
                              "gyro y axis reversed relative to the orientation" if corr < -0.5 else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--yes", action="store_true", help="Skip the steps that need your hands or eyes.")
    parser.add_argument("--params", default=os.path.join(get_package_share_directory("balance_bot_policy"), "config", "policy.yaml"))
    parser.add_argument("--csv", default=os.path.expanduser(time.strftime("~/wheel_check_%Y%m%d_%H%M%S.csv")))
    parser.add_argument("--skip", nargs="*", default=[], choices=["hand", "open_loop", "closed_loop", "imu"])
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    interactive = not args.yes
    with open(args.params) as f:
        params = next(iter(yaml.safe_load(f).values()))["ros__parameters"]

    print("\n\n".join(__doc__.split("\n\n")[2:4]))
    if interactive and not ask("Are the wheels OFF the ground and nothing near them?", True):
        print("Lift the wheels first.")
        return
    rclpy.init()
    bench = Bench(params, args.csv)
    results = []
    try:
        time.sleep(1.0)  # discovery
        if test_health(bench, results):
            if "hand" not in args.skip:
                test_hand_turn(bench, results, interactive)
            if "open_loop" not in args.skip:
                test_open_loop(bench, results, interactive)
            if "closed_loop" not in args.skip:
                test_closed_loop(bench, results)
        if "imu" not in args.skip:
            test_imu(bench, results, interactive)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        bench.close()
        rclpy.try_shutdown()

    print("\n" + "=" * 110)
    print(f"{'check':42s} {'result':6s}  {'measured':30s} expected")
    print("-" * 110)
    for r in results:
        print(f"{r.name:42s} {r.status:6s}  {r.measured:30s} {r.expected}")
        if r.note:
            print(f"{'':50s}-> {r.note}")
    print("=" * 110)
    fails = [r for r in results if r.status == FAIL]
    warns = [r for r in results if r.status == WARN]
    if fails:
        print(f"{len(fails)} FAILED: fix these before running the policy.")
    elif warns:
        print(f"No failures, {len(warns)} warnings: the policy may work, but the training model does not match the robot "
              "here. Send the log (CSV) to have the model matched and the policy retrained.")
    else:
        print("All checks passed: the wheels behave the way the policy expects.")
    print(f"log: {args.csv}")


if __name__ == "__main__":
    main()
