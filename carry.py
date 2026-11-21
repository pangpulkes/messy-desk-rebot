"""Carry a held pen to the box and drop it vertically, without dragging it over the table.

The direct joint move from the pick to `drop_vertical` turns the wrist down while the arm is still low,
so the pen's lower end hit the table. This does instead:

  1. lift STRAIGHT UP (pen keeps its angle) to a safe travel height,
  2. swing over to a point straight above `drop_vertical` (both ends high),
  3. go STRAIGHT DOWN into `drop_vertical`, open the gripper,
  4. straight back up, then home.

Straight up/down = move only the three parallel "pitch" joints (shoulder_lift, elbow_flex, wrist_flex)
so the gripper keeps its orientation; this relies only on those joints' geometry and signs (wrist_flex
sign checked against the fiducial poses), not on the (less reliable) model of the gripper's direction.

  python carry.py            # plan only, prints the waypoints and the lowest point of every segment (no motion)
"""

from __future__ import annotations

import math

import numpy as np

import kin
import picker
from arm import ARM_JOINTS, load_poses

DROP_POSE = "drop_vertical"
TRAVEL_HEIGHT_CM = 26.0  # tip height (above the recorded grab spot) for the sideways swing
STEP_CM = 3.0  # waypoint spacing for straight up/down moves
PITCH = ["shoulder_lift", "elbow_flex", "wrist_flex"]


def _tip(q: dict) -> np.ndarray:
    return kin.fk(q)[:3, 3]


def _ref() -> np.ndarray:
    return _tip(load_poses()["poses"]["down_highlighter"])


def height_cm(q: dict, ref: np.ndarray | None = None) -> float:
    ref = _ref() if ref is None else ref
    return float((_tip(q) - ref)[2] * 100)


def shift_vertical(q: dict, dz_cm: float, hold_wrist: bool = False) -> dict:
    """Joints that move the gripper straight up (+) / down (-) by dz_cm, same orientation. Raises kin.IKError.
    hold_wrist=True keeps wrist_flex fixed (position only; the gripper tilts) - used when wrist_flex is at its limit."""
    T0 = kin.fk(q)
    target_p = T0[:3, 3] + np.array([0.0, 0.0, dz_cm / 100.0])
    cur = {j: float(q[j]) for j in ARM_JOINTS}
    pitch = PITCH[:2] if hold_wrist else PITCH
    for _ in range(100):
        T = kin.fk(cur)
        rot = np.zeros(3) if hold_wrist else kin._rot_err(T[:3, :3], T0[:3, :3])
        err = np.concatenate([target_p - T[:3, 3], rot])
        if np.linalg.norm(err[:3]) < 0.001 and np.linalg.norm(err[3:]) < math.radians(0.5):
            break
        J = np.zeros((6, len(pitch)))
        for k, j in enumerate(pitch):
            d = dict(cur)
            d[j] += 0.1
            Td = kin.fk(d)
            J[:3, k] = (Td[:3, 3] - T[:3, 3]) / 0.1
            if not hold_wrist:
                J[3:, k] = kin._rot_err(T[:3, :3], Td[:3, :3]) / 0.1
        step = np.linalg.lstsq(J, err, rcond=None)[0]
        step = np.clip(step, -5.0, 5.0)
        for k, j in enumerate(pitch):
            cur[j] += float(step[k])
    else:
        raise kin.IKError(f"cannot move straight {'up' if dz_cm > 0 else 'down'} {abs(dz_cm):.0f} cm from here")
    T = kin.fk(cur)
    if np.linalg.norm(T[:3, 3] - target_p) > 0.003:
        raise kin.IKError("straight-line move did not converge")
    for j in PITCH:
        lo, hi = kin.LIMITS[j]
        if not lo <= cur[j] <= hi:
            raise kin.IKError(f"{j} would need {cur[j]:.0f} deg (limit {lo}..{hi})")
    return {j: round(cur[j], 2) for j in ARM_JOINTS}


def vertical_path(q: dict, dz_cm: float) -> list[dict]:
    """Waypoints (every STEP_CM) for a straight vertical move of dz_cm; the last one is the end pose.
    Keeps the gripper's angle while it can; once wrist_flex would pass its limit, wrist_flex stays put and the
    gripper tilts a little instead (going up from the pick spot this tilts the pen's front end UP)."""
    n = max(1, math.ceil(abs(dz_cm) / STEP_CM))
    step = dz_cm / n
    out, cur, hold = [], {j: float(q[j]) for j in ARM_JOINTS}, False
    for _ in range(n):
        if not hold:
            try:
                cur = shift_vertical(cur, step)
                out.append(cur)
                continue
            except kin.IKError:
                hold = True
        cur = shift_vertical(cur, step, hold_wrist=True)
        out.append(cur)
    return out


def lowest_on_segment(a: dict, b: dict, ref: np.ndarray, samples: int = 40) -> float:
    return min(height_cm({j: a[j] + t * (b[j] - a[j]) for j in ARM_JOINTS}, ref) for t in np.linspace(0, 1, samples))


def plan_drop(start: dict) -> dict:
    """Waypoint lists for the whole drop from `start` (the arm's current joints). Raises kin.IKError/KeyError."""
    picker._apply_config()  # wrist_yaw sign from kin_config.json
    poses = load_poses()["poses"]
    if DROP_POSE not in poses:
        raise KeyError(f"no '{DROP_POSE}' pose saved")
    ref = _ref()
    start = {j: float(start[j]) for j in ARM_JOINTS}
    drop = {j: float(poses[DROP_POSE][j]) for j in ARM_JOINTS}

    up = max(0.0, TRAVEL_HEIGHT_CM - height_cm(start, ref))
    lift = vertical_path(start, up) if up > 0.5 else []
    high_start = lift[-1] if lift else start

    over = max(0.0, TRAVEL_HEIGHT_CM - height_cm(drop, ref))
    high_drop = shift_vertical(drop, over) if over > 0.5 else drop
    down = list(reversed(vertical_path(drop, over)[:-1])) + [drop] if over > 0.5 else [drop]
    back_up = vertical_path(drop, over) if over > 0.5 else []

    swing_low = lowest_on_segment(high_start, high_drop, ref)
    return {"lift": lift, "swing": [high_drop], "down": down, "back_up": back_up,
            "heights": {"start": round(height_cm(start, ref), 1), "travel": TRAVEL_HEIGHT_CM,
                        "swing_lowest": round(swing_low, 1), "drop": round(height_cm(drop, ref), 1)}}


def run_drop(arm, log=print) -> dict:
    """Carry what the gripper holds into the box (connected SlowArm), drop it, go home."""
    plan = plan_drop(arm.present())
    log("  lifting straight up")
    for q in plan["lift"]:
        arm.move_to_joints(q, settle=0.0)
    log("  moving over the box")
    for q in plan["swing"]:
        arm.move_to_joints(q, settle=0.0)
    log("  lowering into the drop position")
    for q in plan["down"]:
        arm.move_to_joints(q, settle=0.0)
    arm.set_gripper("open")
    log("  dropped; lifting out")
    for q in plan["back_up"]:
        arm.move_to_joints(q, settle=0.0)
    arm.move_to_pose("rest")
    return plan


if __name__ == "__main__":
    poses = load_poses()["poses"]
    for start_name in ("above_highlighter", "rest"):
        p = plan_drop(poses[start_name])
        print(f"from {start_name}: heights {p['heights']}; waypoints lift {len(p['lift'])}, down {len(p['down'])}")
        for k in ("lift", "swing", "down"):
            for q in p[k]:
                print(f"   {k:5s} {height_cm(q):+5.1f} cm  ", {j: q[j] for j in ARM_JOINTS})
