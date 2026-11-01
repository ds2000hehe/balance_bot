"""ROS-independent parts of the policy runner: frame conversion, gravity estimation, observation layout.

Everything here must match the Isaac Lab training config (``balance_bot_env_cfg.py``):

* Policy ("base") frame: the robot's base_link frame in the USD. +X along the axle (towards ``right_wheel_joint``),
  +Y towards the robot's back, +Z up. The robot's front is -Y.
* Robot body frame (ROS REP-103): x forward, y left, z up. ``BODY_TO_POLICY`` converts between the two.
* Observation (13 values, in this order):
    base angular velocity [rad/s] * 0.25        (3)
    gravity direction, unit vector              (3)   upright = (0, 0, -1)
    velocity command (v_x, v_y, w_yaw), policy frame (3)
    wheel velocities [rad/s] * 0.05             (2)   sim joints [left_wheel_joint, right_wheel_joint]
    last action                                 (2)
* Action (2): wheel velocity target = action * 30 rad/s for [left_wheel_joint, right_wheel_joint].
* ``left_wheel_joint`` (at -X) is on the robot's right, ``right_wheel_joint`` on its left. Driving forward (-Y)
  turns ``left_wheel_joint`` positive and ``right_wheel_joint`` negative.
"""

import numpy as np

ANG_VEL_SCALE = 0.25
WHEEL_VEL_SCALE = 0.05
NUM_OBS = 13
NUM_ACTIONS = 2

# robot body frame (REP-103: x forward, y left, z up) -> policy frame (x along the axle, y backward, z up)
BODY_TO_POLICY = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the frame of orientation ``q`` (x, y, z, w)."""
    x, y, z, w = q_xyzw
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class GravityEstimator:
    """Gravity direction in the IMU frame from gyro + accelerometer (complementary filter).

    The gyro propagates the estimate between samples; the accelerometer (which measures -gravity at rest) slowly
    pulls it back so it does not drift, with time constant ``tau`` seconds (the same for any IMU rate). The training
    simulates exactly this filter (FilteredGravity in the Isaac Lab task): keep ``tau`` equal to it.
    """

    def __init__(self, tau: float = 0.5):
        self.tau = tau
        self.g = None

    def update(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
        norm = np.linalg.norm(accel)
        g_acc = -accel / norm if norm > 1e-3 else None
        if self.g is None:
            self.g = g_acc if g_acc is not None else np.array([0.0, 0.0, -1.0])
            return self.g
        # a fixed world vector seen from a frame rotating with w changes as dg/dt = -w x g
        g = self.g - dt * np.cross(gyro, self.g)
        # trust the accelerometer less while it measures more than gravity (driving, bumps)
        if g_acc is not None and abs(norm - 9.81) < 3.0:
            alpha = min(1.0, dt / self.tau)
            g = (1.0 - alpha) * g + alpha * g_acc
        self.g = g / np.linalg.norm(g)
        return self.g


class CommandSmoother:
    """Limits how fast the velocity command changes, so key presses and joystick jumps become smooth ramps."""

    def __init__(self, max_lin_acc: float, max_ang_acc: float):
        self.max_acc = np.array([max_lin_acc, max_ang_acc])
        self.value = np.zeros(2)

    def update(self, target: tuple[float, float], dt: float) -> tuple[float, float]:
        step = self.max_acc * dt
        self.value = self.value + np.clip(np.asarray(target) - self.value, -step, step)
        return float(self.value[0]), float(self.value[1])

    def reset(self):
        self.value[:] = 0.0


class DriftCorrector:
    """Outer integral loop around the policy that removes its steady-state speed error using wheel odometry.

    The policy alone settles a few cm/s off its command (e.g. -0.04 m/s at zero command): its tracking reward barely
    distinguishes small errors and it cannot observe position. This adds ``ki * integral(cmd - v) dt`` to the
    forward command. Any constant offset is integrated away; at zero command the integral equals minus the distance
    travelled, so the robot is also pulled back towards where it stopped.

    The correction is limited to ``max_correction`` and frozen while the robot is not upright (e.g. standing up),
    so it cannot wind up while the wheels spin to get up.
    """

    def __init__(self, ki: float, max_correction: float):
        self.ki = ki
        self.max_correction = max_correction
        self.integral = 0.0

    def update(self, cmd: float, v_measured: float, upright: bool, dt: float) -> float:
        """Returns the corrected forward command for a commanded speed ``cmd`` and measured speed ``v_measured``."""
        if upright:
            self.integral = float(
                np.clip(self.integral + self.ki * (cmd - v_measured) * dt, -self.max_correction, self.max_correction)
            )
        return cmd + self.integral

    def reset(self):
        self.integral = 0.0


class PositionError:
    """Distance behind (+) / ahead (-) of the forward command from wheel odometry, for position-holding policies.

    Same as ``PositionError`` in the Isaac Lab package: ``integral(cmd_forward - wheel_radius * mean(wheel speed)) dt``
    in the policy's frame and wheel convention, clipped to +-``limit``; the observation is ``error / limit``. At zero
    command it is minus the distance rolled since the robot stopped, so the policy can come back to its spot.
    """

    def __init__(self, wheel_radius: float, limit: float = 0.5):
        self.wheel_radius = wheel_radius
        self.limit = limit
        self.error = 0.0

    def update(self, cmd_forward_policy: float, wheel_vel_sim: np.ndarray, dt: float) -> float:
        v = self.wheel_radius * float(np.mean(wheel_vel_sim))
        self.error = float(np.clip(self.error + (cmd_forward_policy - v) * dt, -self.limit, self.limit))
        return self.error / self.limit

    def reset(self):
        self.error = 0.0


def build_observation(
    ang_vel_policy: np.ndarray,
    gravity_policy: np.ndarray,
    cmd_lin_vel_policy: np.ndarray,
    cmd_yaw: float,
    wheel_vel_sim: np.ndarray,
    last_action: np.ndarray,
    position_error: float | None = None,
) -> np.ndarray:
    """The policy's observation; ``position_error`` (scaled PositionError) only for policies trained with it."""
    parts = [
        ang_vel_policy * ANG_VEL_SCALE,
        gravity_policy,
        [cmd_lin_vel_policy[0], cmd_lin_vel_policy[1], cmd_yaw],
        wheel_vel_sim * WHEEL_VEL_SCALE,
        last_action,
    ]
    if position_error is not None:
        parts.append([position_error])
    obs = np.concatenate(parts).astype(np.float32)
    assert obs.shape == (NUM_OBS + (position_error is not None),)
    return obs


class Policy:
    """Loads ``policy.pt`` (TorchScript) or ``policy.onnx`` exported by Isaac Lab's play.py."""

    def __init__(self, path: str):
        if path.endswith(".onnx"):
            import onnxruntime as ort

            self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self._input = self._session.get_inputs()[0].name
            self.input_size = int(self._session.get_inputs()[0].shape[1])
            self._run = lambda obs: self._session.run(None, {self._input: obs[None]})[0][0]
        else:
            import torch

            torch.set_num_threads(1)
            self._model = torch.jit.load(path, map_location="cpu").eval()

            def run(obs):
                with torch.inference_mode():
                    return self._model(torch.from_numpy(obs)[None]).numpy()[0]

            self._run = run
            self.input_size = NUM_OBS
            try:
                run(np.zeros(NUM_OBS, dtype=np.float32))
            except Exception:
                self.input_size = NUM_OBS + 1

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._run(obs), dtype=np.float32)
