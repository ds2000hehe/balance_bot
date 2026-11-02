"""Bridges the wheel topics to the wheel controller over USB serial (arduino/balance_bot_atmega).

Subscribes:
    wheel_cmd_topic    std_msgs/Float32MultiArray   wheel speed targets [left, right] rad/s -> "V <l> <r>"
    wheel_duty_topic   std_msgs/Float32MultiArray   motor duty [left, right] in [-1, 1] -> "D <l> <r>" (no speed
                                                    loop; from the policy in duty mode, and wheel_check)
Publishes:
    wheel_state_topic  std_msgs/Float32MultiArray   measured wheel speeds [left, right] rad/s <- "W <l> <r>"

Lines starting with '#' from the wheel controller are logged. The serial port is reopened if the USB cable is
unplugged. On shutdown the wheels are told to stop ("S").

Opening the port resets most Arduinos, and an ATmega2560 (Mega) bootloader hangs if serial data arrives while it boots:
nothing is sent for boot_wait seconds after opening (the commands are dropped, the wheels are stopped anyway).
"""

import threading
import time

import rclpy
import serial
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray


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

        self.send_after = 0.0
        self.last_report = 0.0
        self.reporting = False
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
