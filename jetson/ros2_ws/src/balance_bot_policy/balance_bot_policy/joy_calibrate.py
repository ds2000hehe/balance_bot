"""Find the gamepad's axis / button numbers interactively and write them for joy_teleop.

    docker compose exec -it joystick ros2 run balance_bot_policy joy_calibrate

Needs joy_node running (the joystick container) and the controller connected. Switch balancing off first (Cross),
or keep the robot on a stand. You are asked to move each control in turn; the axis or button that changes most is
taken, with its direction (and for an analog trigger, its resting and fully pressed values). The result is written
to /config/joystick_map.yaml (jetson/config/ on the robot computer); restart the joystick container to use it.
"""

import argparse
import threading
import time

import rclpy
import rclpy.executors
import yaml
from rclpy.node import Node
from sensor_msgs.msg import Joy

HEADER = """\
# Written by joy_calibrate (axis / button numbers of this controller with this driver). Loaded on top of
# joystick.yaml by the joystick container. Re-run joy_calibrate after changing controller or driver.
"""


class Watcher(Node):
    def __init__(self, topic: str):
        super().__init__("joy_calibrate")
        self.lock = threading.Lock()
        self.msg = None
        self.create_subscription(Joy, topic, self._on_joy, 10)

    def _on_joy(self, msg: Joy):
        with self.lock:
            self.msg = msg

    def latest(self):
        with self.lock:
            return None if self.msg is None else (list(self.msg.axes), list(self.msg.buttons))


def wait_for_joy(w: Watcher, timeout: float = 10.0):
    end = time.time() + timeout
    while time.time() < end:
        s = w.latest()
        if s is not None:
            return s
        time.sleep(0.05)
    raise SystemExit("no /joy messages: is the joystick container running and the controller connected (PS button)?")


def capture(w: Watcher, seconds: float):
    """Per axis: (min, max) seen; per button: pressed at any time."""
    axes, buttons = wait_for_joy(w)
    lo, hi = list(axes), list(axes)
    pressed = [b == 1 for b in buttons]
    end = time.time() + seconds
    while time.time() < end:
        a, b = w.latest()
        for i, v in enumerate(a[: len(lo)]):
            lo[i], hi[i] = min(lo[i], v), max(hi[i], v)
        for i, v in enumerate(b[: len(pressed)]):
            pressed[i] = pressed[i] or v == 1
        time.sleep(0.01)
    return lo, hi, pressed, w.latest()


def ask(text: str):
    input(f"\n{text}\n  then press Enter (keep holding while it measures, ~2 s) ")


def find_axis(w, rest, prompt, exclude):
    """The axis that moved most from rest while the user held the control: (index, value held)."""
    while True:
        ask(prompt)
        lo, hi, _, (axes, _) = capture(w, 2.0)
        best, best_dev = -1, 0.0
        for i in range(len(rest)):
            if i in exclude:
                continue
            dev = max(abs(lo[i] - rest[i]), abs(hi[i] - rest[i]))
            if dev > best_dev:
                best, best_dev = i, dev
        if best >= 0 and best_dev > 0.5:
            held = axes[best]
            print(f"  -> axis {best} (rest {rest[best]:+.2f}, held {held:+.2f})")
            return best, held
        print("  nothing moved far enough: push it all the way and keep holding it while it measures. Again:")


def find_button(w, rest_buttons, prompt, exclude=()):
    while True:
        ask(prompt)
        _, _, pressed, _ = capture(w, 2.0)
        hits = [i for i, p in enumerate(pressed) if p and not rest_buttons[i] and i not in exclude]
        if len(hits) == 1:
            print(f"  -> button {hits[0]}")
            return hits[0]
        print(f"  {'no button' if not hits else f'several buttons {hits}'} seen: press only that one. Again:")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/config/joystick_map.yaml")
    ap.add_argument("--topic", default="/joy")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    w = Watcher(args.topic)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(w)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    print("Gamepad calibration. Balancing should be OFF (or the robot on a stand).")
    wait_for_joy(w)
    # triggers on some drivers report 0 until first moved: squeeze both triggers once so they report their real rest
    input("\nSqueeze both triggers (L2, R2) fully once and release them, wiggle both sticks, then let go of "
          "everything and press Enter ")
    lo, hi, _, (rest, rest_buttons) = capture(w, 1.0)
    print(f"  {len(rest)} axes, {len(rest_buttons)} buttons; resting axes: "
          + " ".join(f"{i}:{v:+.2f}" for i, v in enumerate(rest)))

    fwd_axis, fwd_val = find_axis(w, rest, "Push the LEFT stick fully UP (forward) and hold it,", set())
    turn_axis, turn_val = find_axis(w, rest, "Push the RIGHT stick fully RIGHT and hold it,", {fwd_axis})

    params = {
        "forward_axis": fwd_axis,
        "forward_sign": 1.0 if fwd_val > rest[fwd_axis] else -1.0,
        # stick right = turn right = negative angular z
        "turn_axis": turn_axis,
        "turn_sign": -1.0 if turn_val > rest[turn_axis] else 1.0,
    }

    # dead-man: R2 as a button if the driver has one, else as an analog axis
    ask("Squeeze R2 (right trigger) fully and hold it,")
    lo, hi, pressed, (axes, _) = capture(w, 2.0)
    r2_buttons = [i for i, p in enumerate(pressed) if p and not rest_buttons[i]]
    moved = [(max(abs(lo[i] - rest[i]), abs(hi[i] - rest[i])), i) for i in range(len(rest))
             if i not in (fwd_axis, turn_axis)]
    dev, r2_axis = max(moved) if moved else (0.0, -1)
    if len(r2_buttons) == 1:
        params.update(deadman_button=r2_buttons[0], deadman_axis=-1)
        print(f"  -> R2 is button {r2_buttons[0]}" + (f" (also axis {r2_axis})" if dev > 0.5 else ""))
    elif dev > 0.5:
        params.update(deadman_button=-1, deadman_axis=r2_axis, deadman_rest=float(round(rest[r2_axis])),
                      deadman_pressed=float(round(axes[r2_axis])))
        print(f"  -> R2 is axis {r2_axis} (rest {rest[r2_axis]:+.2f}, pressed {axes[r2_axis]:+.2f})")
    else:
        raise SystemExit("R2 did not register as a button or an axis: run joy_calibrate again")
    used = {params["deadman_button"]} if params["deadman_button"] >= 0 else set()
    params["start_button"] = find_button(w, rest_buttons, "Press and hold CIRCLE (start balancing),", used)
    params["stop_button"] = find_button(w, rest_buttons, "Press and hold CROSS (stop balancing),",
                                        used | {params["start_button"]})

    print("\nResult:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    text = HEADER + yaml.safe_dump({"joy_teleop": {"ros__parameters": params}}, sort_keys=False)
    try:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"\nwritten to {args.out}. Restart the joystick container:  docker compose restart joystick")
    except OSError as e:
        print(f"\ncould not write {args.out} ({e}); put this into jetson/config/joystick_map.yaml:\n\n{text}")
    executor.shutdown()
    spinner.join(timeout=2.0)
    w.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
