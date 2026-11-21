"""Shared helpers: saved poses + a slow, interpolating controller for the reBot B601 follower.

Safety design:
  * every motion is interpolated at a capped joint speed (default 25 deg/s);
  * the follower's own `max_relative_target` clamp is also on (15 deg per command);
  * a background thread re-sends the current target at 30 Hz so the arm holds still
    while Claude is thinking;
  * on exit the arm returns to the pose named "rest" (if saved) before torque is turned off.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POSES_FILE = HERE / "poses.json"

FOLLOWER_PORT = "can0@1000000"
FOLLOWER_ID = "follower1"
LEADER_ID = "leader1"

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
GRIPPER = "gripper"
ALL_JOINTS = ARM_JOINTS + [GRIPPER]

DEFAULT_SPEED_DEG_S = 25.0
CONTROL_HZ = 30.0

# Per-command clamp relative to the present position. The gripper is force-limited and naturally
# lags its target, so it gets a wide limit (otherwise the clamp warns on every gripper move).
MAX_RELATIVE_TARGET = {**{j: 15.0 for j in ARM_JOINTS}, GRIPPER: 360.0}


# --------------------------------------------------------------------------- poses file
def load_poses() -> dict:
    if POSES_FILE.exists():
        data = json.loads(POSES_FILE.read_text())
    else:
        data = {}
    data.setdefault("poses", {})
    data.setdefault("gripper", {})
    return data


def save_poses(data: dict) -> None:
    POSES_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))


class _NoInterrupt:
    """While parking, Ctrl+C must not abort the slow return home (that would drop the arm)."""

    def __enter__(self):
        import signal

        self._signal = signal
        self._old = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: print("\n  ...still returning home - use the 48 V switch for an emergency stop"))
        return self

    def __exit__(self, *exc):
        self._signal.signal(self._signal.SIGINT, self._old)
        return False


def rest_joints() -> dict[str, float]:
    """The home pose for the 6 arm joints ('rest' if saved, else the calibrated zero pose)."""
    rest = load_poses()["poses"].get("rest")
    return {j: float(rest[j]) if rest else 0.0 for j in ARM_JOINTS}


def park_follower(follower, current: dict[str, float], speed_deg_s: float = DEFAULT_SPEED_DEG_S) -> None:
    """Drive a connected follower slowly from `current` (all joints, deg) to the home pose and hold
    it briefly, then DISCONNECT (motors off). Ctrl+C is ignored meanwhile. Used by every tool on exit."""
    goal = {**current, **rest_joints()}  # gripper stays as it is
    with _NoInterrupt():
        print("Returning the arm home before switching the motors off...")
        try:
            span = max(abs(goal[j] - current[j]) for j in ARM_JOINTS)
            steps = max(1, int(span / speed_deg_s * CONTROL_HZ))
            t0 = time.perf_counter()
            for i in range(1, steps + 1):
                a = i / steps
                follower.send_action({f"{j}.pos": current[j] + a * (goal[j] - current[j]) for j in ALL_JOINTS})
                time.sleep(max(0.0, t0 + i / CONTROL_HZ - time.perf_counter()))
            for _ in range(int(0.6 * CONTROL_HZ)):  # settle
                follower.send_action({f"{j}.pos": v for j, v in goal.items()})
                time.sleep(1.0 / CONTROL_HZ)
            print("Arm is home.")
        except Exception as e:
            print(f"Could not drive the arm home ({e}). Support the arm now!")
        try:
            follower.disconnect()
            print("Follower disconnected (motors off).")
        except Exception as e:
            print(f"Follower did not acknowledge switch-off ({e}). If its power is off, that's expected.")


def find_leader_port() -> str:
    import glob

    ports = sorted(glob.glob("/dev/tty.usbserial-*"))
    if not ports:
        raise SystemExit("Leader arm not found: no /dev/tty.usbserial-* port. Plug in its USB (and 12 V power).")
    return ports[0]


# --------------------------------------------------------------------------- follower
class StopMotion(RuntimeError):
    """Raised inside a motion when Stop was pressed; the arm holds where it is."""


class SlowArm:
    """Follower wrapper that only ever moves slowly toward saved targets."""

    def __init__(self, speed_deg_s: float = DEFAULT_SPEED_DEG_S, camera_index: int | None = None):
        from lerobot.robots.rebot_b601_follower import RebotB601Follower, RebotB601FollowerRobotConfig

        cameras = {}
        if camera_index is not None:
            from lerobot.cameras.opencv import OpenCVCameraConfig

            cameras["front"] = OpenCVCameraConfig(index_or_path=camera_index, fps=30, width=640, height=480)

        cfg = RebotB601FollowerRobotConfig(
            port=FOLLOWER_PORT,
            motor_family="rs",
            id=FOLLOWER_ID,
            max_relative_target=MAX_RELATIVE_TARGET,
            cameras=cameras,
        )
        self.robot = RebotB601Follower(cfg)
        self.speed = speed_deg_s
        self._target: dict[str, float] = {}
        self._lock = threading.Lock()  # guards self._target
        self._io = threading.Lock()  # one CAN transaction at a time (hold thread vs. main thread)
        self._stop = threading.Event()
        self.stop_requested = threading.Event()  # set by a Stop button: current motion halts and holds
        self._thread: threading.Thread | None = None

    # -- lifecycle
    def connect(self) -> None:
        self.robot.connect(calibrate=False)
        present = self.present()
        with self._lock:
            self._target = dict(present)  # hold exactly where it is
        self._send(self._target)
        self._thread = threading.Thread(target=self._hold_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Always return home first (Ctrl+C is ignored meanwhile), then switch the motors off."""
        if not self.robot.is_connected:
            return
        self.stop_requested.set()  # any motion still running in another thread (UI worker) halts now
        self.stop_follow()  # no more leader copying
        self._stop.set()  # stop the hold thread; park_follower takes over the bus
        if self._thread:
            self._thread.join(timeout=1.0)
        with self._lock:
            current = dict(self._target)
        with self._io:  # no other thread may talk to the motors while parking
            park_follower(self.robot, current, self.speed)

    # -- reading
    def present(self) -> dict[str, float]:
        with self._io:
            obs = self.robot.get_observation()
        return {j: float(obs[f"{j}.pos"]) for j in ALL_JOINTS}

    def camera_frame(self):
        """Latest RGB frame (H, W, 3 uint8) from the 'front' camera, or None."""
        cam = self.robot.cameras.get("front")
        return cam.read_latest() if cam is not None else None

    # -- motion
    def _send(self, target: dict[str, float]) -> None:
        with self._io:
            self.robot.send_action({f"{j}.pos": v for j, v in target.items()})

    def _hold_loop(self) -> None:
        period = 1.0 / CONTROL_HZ
        while not self._stop.is_set():
            with self._lock:
                target = dict(self._target)
            try:
                self._send(target)
            except Exception as e:  # keep holding; report once per failure
                print(f"[hold] send failed: {e}")
            time.sleep(period)

    def _interpolate_to(self, goal: dict[str, float], settle: float = 0.3) -> None:
        with self._lock:
            start = dict(self._target)
        joints = [j for j in goal if j in start]
        span = max((abs(goal[j] - start[j]) for j in joints), default=0.0)
        duration = span / self.speed if self.speed > 0 else 0.0
        steps = max(1, int(duration * CONTROL_HZ))
        t0 = time.perf_counter()
        for i in range(1, steps + 1):
            if self.stop_requested.is_set():
                raise StopMotion("stopped by the user - holding position")
            a = i / steps
            with self._lock:
                for j in joints:
                    self._target[j] = start[j] + a * (goal[j] - start[j])
            # hold thread sends the new target; pace ourselves to CONTROL_HZ
            sleep_until = t0 + i / CONTROL_HZ
            time.sleep(max(0.0, sleep_until - time.perf_counter()))
        time.sleep(settle)  # let it settle
        if self.stop_requested.is_set():
            raise StopMotion("stopped by the user - holding position")

    def move_to_joints(self, goal: dict[str, float], settle: float = 0.3) -> dict[str, float]:
        """Move the 6 arm joints (gripper unchanged) to computed angles, slowly (settle=0 for waypoints)."""
        self._interpolate_to({j: float(goal[j]) for j in ARM_JOINTS}, settle)
        return self.present()

    def move_to_pose(self, name: str) -> dict[str, float]:
        poses = load_poses()["poses"]
        if name not in poses:
            raise KeyError(f"Unknown pose '{name}'. Saved poses: {sorted(poses)}")
        goal = {j: float(poses[name][j]) for j in ARM_JOINTS}  # gripper unchanged
        self._interpolate_to(goal)
        return self.present()

    # -- leader follow (setup flows in the control window)
    FOLLOW_MAX_STEP_DEG = 2.0  # per 1/30 s tick -> at most 60 deg/s while copying the leader

    def start_follow(self, leader) -> None:
        """Copy the leader arm live (rate-limited) until stop_follow(), Stop (stop_requested) or close()."""
        self.stop_follow()
        self.leader_latest: dict[str, float] = {}
        self._follow_stop = threading.Event()
        stop_ev = self._follow_stop

        def loop():
            step = self.FOLLOW_MAX_STEP_DEG
            while not (stop_ev.is_set() or self._stop.is_set() or self.stop_requested.is_set()):
                t0 = time.perf_counter()
                try:
                    act = {k.removesuffix(".pos"): float(v) for k, v in leader.get_action().items()}
                except Exception as e:
                    print(f"[follow] leader read failed: {e}")
                    time.sleep(0.1)
                    continue
                self.leader_latest = act
                with self._lock:
                    for j in ALL_JOINTS:
                        if j in act and j in self._target:
                            d = act[j] - self._target[j]
                            self._target[j] += max(-step, min(step, d))
                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))

        self._follow_thread = threading.Thread(target=loop, daemon=True)
        self._follow_thread.start()

    def stop_follow(self) -> None:
        """Stop copying the leader; the follower holds where it is."""
        t = getattr(self, "_follow_thread", None)
        if t is not None:
            self._follow_stop.set()
            t.join(timeout=1.0)
            self._follow_thread = None

    @property
    def following(self) -> bool:
        t = getattr(self, "_follow_thread", None)
        return t is not None and t.is_alive()

    def set_gripper(self, state: str) -> dict[str, float]:
        grip = load_poses()["gripper"]
        if state not in grip:
            raise KeyError(f"Gripper value '{state}' not recorded yet. Recorded: {sorted(grip)}")
        self._interpolate_to({GRIPPER: float(grip[state])})
        return self.present()
