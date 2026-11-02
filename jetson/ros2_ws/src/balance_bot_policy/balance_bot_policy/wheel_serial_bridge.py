"""Bridges the wheel topics to the wheel controller over USB serial (arduino/balance_bot_atmega).

Subscribes:
    wheel_cmd_topic    std_msgs/Float32MultiArray   wheel speed targets [left, right] rad/s -> "V <l> <r>"
    wheel_duty_topic   std_msgs/Float32MultiArray   motor duty [left, right] in [-1, 1] -> "D <l> <r>" (no speed
                                                    loop; from the policy in duty mode, and wheel_check)
Publishes:
    wheel_state_topic  std_msgs/Float32MultiArray   measured wheel speeds [left, right] rad/s <- "W <l> <r>"

Live tuning of the firmware's PI speed loop (speed mode only; the raw throttle "D" bypasses it): parameters
speed_kp / speed_ki (firmware defaults) times kp_scale / ki_scale are sent as "G <kp> <ki>" whenever one changes and
again whenever the controller (re)starts. Sliders: ``ros2 run rqt_reconfigure rqt_reconfigure``; or
``ros2 param set /wheel_serial_bridge ki_scale 1.5``.

Lines starting with '#' from the wheel controller are logged. The serial port is reopened if the USB cable is
unplugged. On shutdown the wheels are told to stop ("S").

Opening the port resets most Arduinos, and an ATmega2560 (Mega) bootloader hangs if serial data arrives while it boots:
nothing is sent for boot_wait seconds after opening (the commands are dropped, the wheels are stopped anyway).
"""

import threading
import time

import rclpy
import serial
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult
from rclpy.exceptions import InvalidParameterTypeException, InvalidParameterValueException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray

from .serial_low_latency import set_low_latency_fd


class WheelSerialBridge(Node):
    def __init__(self):
        super().__init__("wheel_serial_bridge")
        p = self.declare_parameter
        self.port = p("port", "/dev/ttyUSB0").value
        self.baud = p("baud", 115200).value
        wheel_cmd_topic = p("wheel_cmd_topic", "/wheel_cmd").value
        wheel_state_topic = p("wheel_state_topic", "/wheel_vel").value
        wheel_duty_topic = p("wheel_duty_topic", "/wheel_duty").value
        self.boot_wait = p("boot_wait", 2.5).value

        def knob(name, default, hi, text):
            """Slider range 0..hi (rqt_reconfigure). A configured value outside it is used anyway (no slider then),
            never a reason to not start: the bridge is what keeps the wheels stoppable."""
            rng = FloatingPointRange(from_value=0.0, to_value=hi, step=0.0)
            try:
                return float(p(name, default, ParameterDescriptor(description=text, floating_point_range=[rng])).value)
            except (InvalidParameterValueException, InvalidParameterTypeException) as e:
                self.get_logger().warn(f"{name}: {e}; using it without a slider range")
            try:
                return float(p(name, default, ParameterDescriptor(description=text, dynamic_typing=True)).value)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"{name}: {e}; using the default {default}")
                return default

        self.gains = {
            "speed_kp": knob("speed_kp", 0.01, 1.0, "firmware KP [duty per rad/s] (sketch default 0.01)"),
            "speed_ki": knob("speed_ki", 0.2, 5.0, "firmware KI [duty per rad] (sketch default 0.2)"),
            "kp_scale": knob("kp_scale", 1.0, 10.0, "multiplies speed_kp"),
            "ki_scale": knob("ki_scale", 1.0, 10.0, "multiplies speed_ki"),
        }
        # firmware brake (ZS-X11H BRAKE inputs): automatic braking when the throttle asks for less speed than the
        # wheel turns, from brake_min_rad_s on. Sent as "B <0|1> <min>" with the gains
        self.auto_brake = bool(p("auto_brake", True).value)
        self.gains["brake_min_rad_s"] = knob("brake_min_rad_s", 3.0, 30.0, "wheel speed above which the firmware brakes")
        self.add_on_set_parameters_callback(self._on_params)

        self.send_after = 0.0
        self.last_report = 0.0
        self.reporting = False
        self.gains_pending = True
        self.serial = None
        self.write_lock = threading.Lock()
        self.running = True
        self.wheel_pub = self.create_publisher(Float32MultiArray, wheel_state_topic, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, wheel_cmd_topic, self._on_cmd, 10)
        self.create_subscription(Float32MultiArray, wheel_duty_topic, self._on_duty, 10)
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()
        self.get_logger().info(f"{wheel_cmd_topic} -> {self.port} @ {self.baud} -> {wheel_state_topic}")

    def _open(self):
        s = serial.Serial()
        s.port = self.port
        s.baudrate = self.baud
        s.timeout = 0.1
        # keep DTR/RTS low: on most ESP32 boards toggling them resets the chip
        s.dtr = False
        s.rts = False
        s.open()
        s.reset_input_buffer()
        # USB serial adapters can hold received bytes up to 16 ms: hand them over at once (lost on replug: set again)
        try:
            ok = set_low_latency_fd(s.fileno())
            self.get_logger().info(f"{self.port}: low latency {'on' if ok else 'not supported'}")
        except OSError as e:
            self.get_logger().info(f"{self.port}: low latency not supported ({e.strerror})")
        return s

    def _write(self, line: str):
        with self.write_lock:
            if self.serial is None or time.monotonic() < self.send_after:
                return
            try:
                self.serial.write(line.encode())
            except (serial.SerialException, OSError) as e:
                self.get_logger().warn(f"serial write failed: {e}")
                self._close()

    def _close(self):
        if self.serial is not None:
            try:
                self.serial.close()
            except (serial.SerialException, OSError):
                pass
        self.serial = None

    def _send_gains(self):
        g = self.gains
        kp, ki = g["speed_kp"] * g["kp_scale"], g["speed_ki"] * g["ki_scale"]
        self._write(f"G {kp:.5f} {ki:.5f}\n")
        self._write(f"B {int(self.auto_brake)} {g['brake_min_rad_s']:.2f}\n")
        self.get_logger().info(f"speed loop gains -> kp {kp:.4f} ki {ki:.4f} "
                               f"(scales {g['kp_scale']:.2f} / {g['ki_scale']:.2f}), auto brake "
                               f"{'on' if self.auto_brake else 'off'} above {g['brake_min_rad_s']:.1f} rad/s")

    def _on_params(self, params):
        changed = False
        for prm in params:
            if prm.name == "auto_brake":
                self.auto_brake = bool(prm.value)
                changed = True
            elif prm.name in self.gains:
                v = float(prm.value)
                if v < 0.0:
                    return SetParametersResult(successful=False, reason=f"{prm.name} must be >= 0")
                self.gains[prm.name] = v
                changed = True
        if changed:
            self._send_gains()
        return SetParametersResult(successful=True)

    def _on_cmd(self, msg: Float32MultiArray):
        if len(msg.data) != 2:
            self.get_logger().warn(f"wheel command needs 2 values [left, right], got {len(msg.data)}", once=True)
            return
        self._write(f"V {msg.data[0]:.3f} {msg.data[1]:.3f}\n")

    def _on_duty(self, msg: Float32MultiArray):
        if len(msg.data) != 2:
            return
        self._write(f"D {msg.data[0]:.4f} {msg.data[1]:.4f}\n")

    def _read_loop(self):
        while self.running:
            if self.serial is None:
                try:
                    s = self._open()
                    with self.write_lock:
                        self.serial = s
                        self.send_after = time.monotonic() + self.boot_wait
                    self.last_report = time.monotonic()
                    self.reporting = False
                    self.gains_pending = True  # the controller starts with the sketch's gains after every reset
                    self.get_logger().info(f"opened {self.port}, waiting {self.boot_wait:.1f} s for the controller to boot")
                except (serial.SerialException, OSError) as e:
                    self.get_logger().warn(f"cannot open {self.port}: {e}", throttle_duration_sec=5.0)
                    time.sleep(1.0)
                    continue
            try:
                raw = self.serial.readline()
            except (serial.SerialException, OSError, TypeError, AttributeError) as e:
                self.get_logger().warn(f"serial read failed ({e}), reconnecting")
                with self.write_lock:
                    self._close()
                time.sleep(0.5)
                continue
            line = raw.decode(errors="replace").strip()
            now = time.monotonic()
            if line.startswith("W "):
                try:
                    left, right = (float(v) for v in line[2:].split())
                except ValueError:
                    continue  # partial line (e.g. right after connecting)
                try:
                    self.wheel_pub.publish(Float32MultiArray(data=[left, right]))
                except Exception:  # shutting down
                    break
                self.last_report = now
                if not self.reporting:
                    self.reporting = True
                    self.get_logger().info("wheel controller is sending speed reports")
            elif line.startswith("#"):
                self.get_logger().info(f"controller: {line[1:].strip()}")
            elif line:
                self.get_logger().warn(f"unexpected line from the controller: {line[:60]!r}", throttle_duration_sec=5.0)
            if self.reporting and self.gains_pending and now >= self.send_after:
                self.gains_pending = False
                self._send_gains()
            if now - self.last_report > self.boot_wait + 1.0:
                self.reporting = False
                self.get_logger().warn(
                    f"no speed reports ('W ...') from {self.port} for {now - self.last_report:.0f} s: is "
                    "balance_bot_atmega flashed? baud right? another program on the port?",
                    throttle_duration_sec=5.0,
                )

    def shutdown(self):
        self._write("S\n")
        self.running = False
        self.reader.join(timeout=1.0)
        with self.write_lock:
            self._close()


def main(args=None):
    rclpy.init(args=args)
    node = WheelSerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
