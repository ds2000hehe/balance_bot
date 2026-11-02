"""ROS 2 node that runs the trained balance bot policy at 50 Hz.

Subscribes:
    imu_topic          sensor_msgs/Imu                angular_velocity is required; orientation (if the IMU fuses it)
                                                      or linear_acceleration is used for the gravity direction
    cmd_vel_topic      geometry_msgs/Twist            linear.x = forward speed [m/s], angular.z = turn rate [rad/s]
    wheel_state_topic  std_msgs/Float32MultiArray     measured wheel speeds [left, right] in rad/s
    enable_topic       std_msgs/Bool                  false = stop the policy and command zero wheel speed

Publishes:
    wheel_duty_topic   std_msgs/Float32MultiArray     output_mode "duty": motor duty [left, right] in [-1, 1]
                                                      (= action target / motor_max_rad_s, the ESP32's "D" command)
    wheel_cmd_topic    std_msgs/Float32MultiArray     output_mode "speed": wheel velocity targets [left, right] rad/s;
                                                      in both modes zero speed while stopped / disabled

Wheel speeds on the topics are [left, right] as seen from behind the robot, in the Arduino's sign convention.
``wheel_signs`` is +1 for a wheel whose positive speed drives the robot forward, -1 otherwise.
The velocity command is ramped with ``max_lin_acc`` / ``max_ang_acc``, so steps from a keyboard become smooth.
Forward speed drift is removed by an integral loop on wheel odometry (``drift_ki``, see ``DriftCorrector``), and
turning drift by the same kind of loop on the gyro's yaw rate (``yaw_drift_ki``).
"""

import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32MultiArray

from .policy_core import (
    BODY_TO_POLICY,
    NUM_ACTIONS,
    NUM_OBS,
    CommandSmoother,
    DriftCorrector,
    GravityEstimator,
    Policy,
    PositionError,
    build_observation,
    quat_rotate_inverse,
)


class PolicyNode(Node):
    def __init__(self):
        super().__init__("balance_bot_policy")

        # -- parameters
        p = self.declare_parameter
        self.policy_path = p("policy_path", "").value
        imu_topic = p("imu_topic", "/imu").value
        cmd_vel_topic = p("cmd_vel_topic", "/cmd_vel").value
        wheel_state_topic = p("wheel_state_topic", "/wheel_vel").value
        wheel_cmd_topic = p("wheel_cmd_topic", "/wheel_cmd").value
        wheel_duty_topic = p("wheel_duty_topic", "/wheel_duty").value
        # must match the training task's action term (HoverboardWheelActionCfg.control)
        self.output_mode = p("output_mode", "speed").value
        if self.output_mode not in ("duty", "speed"):
            raise RuntimeError(f"output_mode must be 'duty' or 'speed', not '{self.output_mode}'")
        self.motor_max_rad_s = p("motor_max_rad_s", 60.0).value
        enable_topic = p("enable_topic", "/balance_bot/enable").value
        self.rate = p("control_rate", 50.0).value
        # rotation (row-major 3x3) taking IMU-frame vectors into the robot body frame (x forward, y left, z up);
        # the default (identity) is for an IMU mounted with its axes along the robot's
        self.imu_to_body = np.array(p("imu_to_body", np.eye(3).flatten().tolist()).value).reshape(3, 3)
        self.imu_to_policy = BODY_TO_POLICY @ self.imu_to_body
        self.use_imu_orientation = p("use_imu_orientation", True).value
        self.gravity_filter_tau = p("gravity_filter_tau", 0.5).value
        self.wheel_signs = np.array(p("wheel_signs", [1.0, 1.0]).value)
        # tilt cut-off: beyond max_tilt_deg the motors are switched off (zero duty, the wheels coast). The training ends
        # episodes at 30 deg, so between 30 deg and the cut-off the policy is outside what it learned. It resumes after
        # being held within resume_tilt_deg for resume_hold_s
        # (resume_tilt_deg 0: only after the enable topic / Circle again)
        self.max_tilt_deg = p("max_tilt_deg", 50.0).value
        self.resume_tilt_deg = p("resume_tilt_deg", 5.0).value
        self.resume_hold_s = p("resume_hold_s", 0.5).value
        self.tripped = False
        self.upright_since = None
        # while enabled, log what the policy sees and commands every debug_period seconds (0 = off)
        self.debug_period = p("debug_period", 1.0).value
        self.last_debug = 0.0
        self.action_scale = p("action_scale", 30.0).value
        self.max_wheel_speed = p("max_wheel_speed", 30.0).value
        self.max_lin_vel = p("max_lin_vel", 1.5).value
        self.max_ang_vel = p("max_ang_vel", 2.0).value
        self.cmd_smoother = CommandSmoother(p("max_lin_acc", 1.0).value, p("max_ang_acc", 4.0).value)
        self.cmd_timeout = p("cmd_vel_timeout", 1.0).value
        self.wheel_radius = p("wheel_radius", 0.0825).value
        self.drift = DriftCorrector(p("drift_ki", 0.5).value, p("drift_max_correction", 0.3).value)
        self.yaw_drift = DriftCorrector(p("yaw_drift_ki", 1.0).value, p("yaw_drift_max_correction", 0.5).value)
        self.sensor_timeout = p("sensor_timeout", 0.1).value
        self.enabled = p("start_enabled", True).value

        if not self.policy_path:
            raise RuntimeError("Parameter 'policy_path' is required (path to policy.pt or policy.onnx)")
        self.policy = Policy(self.policy_path)
        # position-holding policies have one more input: the distance behind / ahead of the forward command
        self.position_error = (PositionError(self.wheel_radius) if self.policy.input_size == NUM_OBS + 1 else None)
        if self.policy.input_size not in (NUM_OBS, NUM_OBS + 1):
            raise RuntimeError(f"policy takes {self.policy.input_size} inputs, expected {NUM_OBS} or {NUM_OBS + 1}")
        self.get_logger().info(
            f"Loaded policy from {self.policy_path}"
            + (" (position hold: forward drift_ki ignored)" if self.position_error is not None else "")
        )

        # -- state
        self.gravity = GravityEstimator(self.gravity_filter_tau)
        self.ang_vel_imu = None
        self.gravity_imu = None
        self.last_imu_stamp = None
        self.imu_time = 0.0
        self.wheel_vel = None
        self.wheel_time = 0.0
        self.cmd = (0.0, 0.0)
        self.cmd_time = 0.0
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # -- ROS interfaces
        self.create_subscription(Imu, imu_topic, self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Twist, cmd_vel_topic, self._on_cmd_vel, 10)
        self.create_subscription(Float32MultiArray, wheel_state_topic, self._on_wheels, qos_profile_sensor_data)
        self.create_subscription(Bool, enable_topic, self._on_enable, 10)
        self.wheel_pub = self.create_publisher(Float32MultiArray, wheel_cmd_topic, 10)
        self.duty_pub = self.create_publisher(Float32MultiArray, wheel_duty_topic, 10)
        self.create_timer(1.0 / self.rate, self._step)

        self.get_logger().info(
            f"imu: {imu_topic}, cmd_vel: {cmd_vel_topic}, wheel state: {wheel_state_topic}, "
            f"wheel cmd: {wheel_cmd_topic}, wheel duty: {wheel_duty_topic}, output: {self.output_mode}, "
            f"enable: {enable_topic}, enabled: {self.enabled}"
        )

    # -- callbacks

    def _on_imu(self, msg: Imu):
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        q = msg.orientation
        has_orientation = msg.orientation_covariance[0] >= 0.0 and (q.x, q.y, q.z, q.w) != (0.0, 0.0, 0.0, 0.0)
        if self.use_imu_orientation and has_orientation:
            self.gravity_imu = quat_rotate_inverse(np.array([q.x, q.y, q.z, q.w]), np.array([0.0, 0.0, -1.0]))
        else:
            stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            dt = stamp - self.last_imu_stamp if self.last_imu_stamp is not None else 0.0
            dt = dt if 0.0 < dt < 0.1 else 1.0 / self.rate
            self.last_imu_stamp = stamp
            accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            self.gravity_imu = self.gravity.update(gyro, accel, dt)
        self.ang_vel_imu = gyro
        self.imu_time = time.monotonic()

    def _on_cmd_vel(self, msg: Twist):
        self.cmd = (
            float(np.clip(msg.linear.x, -self.max_lin_vel, self.max_lin_vel)),
            float(np.clip(msg.angular.z, -self.max_ang_vel, self.max_ang_vel)),
        )
        self.cmd_time = time.monotonic()

    def _on_wheels(self, msg: Float32MultiArray):
        if len(msg.data) != 2:
            self.get_logger().warn(f"wheel state needs 2 values [left, right], got {len(msg.data)}", once=True)
            return
        self.wheel_vel = np.array(msg.data, dtype=np.float64)
        self.wheel_time = time.monotonic()

    def _on_enable(self, msg: Bool):
        if msg.data != self.enabled:
            self.get_logger().info("policy enabled" if msg.data else "policy disabled")
        self.enabled = msg.data
        self.tripped = False
        self._reset_state()

    def _reset_state(self):
        self.last_action[:] = 0.0
        self.cmd_smoother.reset()
        self.drift.reset()
        self.yaw_drift.reset()
        if self.position_error is not None:
            self.position_error.reset()

    def _tilt_cutoff(self, gravity_policy: np.ndarray, now: float) -> bool:
        """True while the wheels must stay stopped because the robot tilted beyond max_tilt_deg."""
        tilt = float(np.degrees(np.arccos(np.clip(-gravity_policy[2], -1.0, 1.0))))
        if not self.tripped:
            if tilt > self.max_tilt_deg:
                self.tripped = True
                self.upright_since = None
                self.get_logger().warn(
                    f"tilt {tilt:.0f} deg > {self.max_tilt_deg:.0f} deg: motors off"
                    + (" (hold it upright to resume)" if self.resume_tilt_deg > 0 else " (re-enable to resume)")
                )
            return self.tripped
        if self.resume_tilt_deg > 0 and tilt < self.resume_tilt_deg:
            if self.upright_since is None:
                self.upright_since = now
            elif now - self.upright_since >= self.resume_hold_s:
                self.tripped = False
                self._reset_state()
                self.get_logger().info("upright again: balancing resumed")
        else:
            self.upright_since = None
        return self.tripped

    # -- wheel conventions

    def _topic_to_sim(self, topic: np.ndarray) -> np.ndarray:
        """[left, right] (Arduino signs) -> sim joints [left_wheel_joint, right_wheel_joint].

        ``left_wheel_joint`` is the robot's right wheel and turns positive when driving forward; ``right_wheel_joint``
        is the robot's left wheel and turns negative when driving forward.
        """
        forward = topic * self.wheel_signs
        return np.array([forward[1], -forward[0]])

    def _sim_to_topic(self, sim: np.ndarray) -> np.ndarray:
        forward = np.array([-sim[1], sim[0]])
        return forward * self.wheel_signs

    def _forward_speed(self) -> float:
        """Ground speed from wheel odometry [m/s].

        The wheel encoders measure rotation relative to the body, so the body's pitch rate (gyro about the robot's
        left axis) is added back to get the wheels' rotation relative to the ground.
        """
        wheel_forward = float(np.mean(self.wheel_vel * self.wheel_signs))
        pitch_rate = float((self.imu_to_body @ self.ang_vel_imu)[1])
        return self.wheel_radius * (wheel_forward + pitch_rate)

    # -- control loop

    def _publish(self, wheel_cmd: np.ndarray):
        self.wheel_pub.publish(Float32MultiArray(data=[float(v) for v in wheel_cmd]))

    def _motors_off(self):
        """Zero motor duty: no throttle at all, the wheels coast (a zero speed target would still brake / correct)."""
        self.duty_pub.publish(Float32MultiArray(data=[0.0, 0.0]))

    def _step(self):
        now = time.monotonic()
        if not self.enabled:
            self._motors_off()
            return
        if self.ang_vel_imu is None or now - self.imu_time > self.sensor_timeout:
            self.get_logger().warn("no recent IMU data, commanding zero wheel speed", throttle_duration_sec=1.0)
            self._publish(np.zeros(2))
            return
        if self.wheel_vel is None or now - self.wheel_time > self.sensor_timeout:
            self.get_logger().warn("no recent wheel speed data, commanding zero wheel speed", throttle_duration_sec=1.0)
            self._publish(np.zeros(2))
            return
        # stop driving (but keep balancing) when cmd_vel goes quiet; ramp all command changes
        target = self.cmd if now - self.cmd_time < self.cmd_timeout else (0.0, 0.0)
        cmd_forward, cmd_yaw = self.cmd_smoother.update(target, 1.0 / self.rate)
        gravity_policy = self.imu_to_policy @ self.gravity_imu
        if self._tilt_cutoff(gravity_policy, now):
            self._motors_off()
            return
        upright = gravity_policy[2] < -0.9  # tilted less than ~25 deg
        if self.position_error is None:
            cmd_forward = self.drift.update(cmd_forward, self._forward_speed(), upright, 1.0 / self.rate)
        yaw_rate = float((self.imu_to_body @ self.ang_vel_imu)[2])
        cmd_yaw = self.yaw_drift.update(cmd_yaw, yaw_rate, upright, 1.0 / self.rate)

        cmd_lin_vel_policy = BODY_TO_POLICY @ np.array([cmd_forward, 0.0, 0.0])
        wheel_vel_sim = self._topic_to_sim(self.wheel_vel)
        position_error = None
        if self.position_error is not None:
            position_error = self.position_error.update(float(cmd_lin_vel_policy[1]), wheel_vel_sim, 1.0 / self.rate)
        obs = build_observation(
            ang_vel_policy=self.imu_to_policy @ self.ang_vel_imu,
            gravity_policy=gravity_policy,
            cmd_lin_vel_policy=cmd_lin_vel_policy,
            cmd_yaw=cmd_yaw,
            wheel_vel_sim=wheel_vel_sim,
            last_action=self.last_action,
            position_error=position_error,
        )
        action = self.policy(obs)
        self.last_action = action
        if self.debug_period > 0.0 and now - self.last_debug >= self.debug_period:
            self.last_debug = now
            g_body = self.imu_to_body @ self.gravity_imu  # unit vector pointing down, robot axes
            gyro_body = self.imu_to_body @ self.ang_vel_imu
            target = self._sim_to_topic(np.clip(action * self.action_scale, -self.max_wheel_speed, self.max_wheel_speed))
            self.get_logger().info(
                f"lean fwd {np.degrees(np.arcsin(np.clip(g_body[0], -1, 1))):+5.1f} deg, "
                f"left {np.degrees(np.arcsin(np.clip(g_body[1], -1, 1))):+5.1f} deg | "
                f"gyro roll/pitch/yaw {gyro_body[0]:+5.2f} {gyro_body[1]:+5.2f} {gyro_body[2]:+5.2f} rad/s | "
                f"cmd {cmd_forward:+.2f} m/s {cmd_yaw:+.2f} rad/s | "
                f"wheels meas [{self.wheel_vel[0]:+6.2f} {self.wheel_vel[1]:+6.2f}] "
                f"-> target [{target[0]:+6.2f} {target[1]:+6.2f}] rad/s (left, right)"
            )
        if self.output_mode == "duty":
            duty_sim = np.clip(action * self.action_scale / self.motor_max_rad_s, -1.0, 1.0)
            self.duty_pub.publish(Float32MultiArray(data=[float(v) for v in self._sim_to_topic(duty_sim)]))
        else:
            target_sim = np.clip(action * self.action_scale, -self.max_wheel_speed, self.max_wheel_speed)
            self._publish(self._sim_to_topic(target_sim))


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # leave the motors stopped
        if rclpy.ok():
            node._publish(np.zeros(2))
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
