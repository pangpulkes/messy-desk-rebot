"""Plan and run a pick at a table position given in cm relative to the MAIN tape mark.

The motion copies the recorded grab: it uses `down_highlighter` (grab pose over the tape) and
`above_highlighter` (hover) from poses.json, and shifts them to the requested spot with the arm's
geometry model (kin.py). Positions are limited to PICK_ZONE_CM around the tape.

Test without moving anything:
  python ~/rebot_lerobot/llm_arm/picker.py 3 -4      # plan a pick 3 cm forward, 4 cm right
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import kin
from arm import load_poses

HERE = Path(__file__).resolve().parent
KIN_CONFIG = HERE / "kin_config.json"
PICK_ZONE_CM = {"forward": (-10.0, 7.0), "left": (-10.0, 10.0)}


def _apply_config() -> None:
    cfg = json.loads(KIN_CONFIG.read_text()) if KIN_CONFIG.exists() else {}
    kin.set_wrist_yaw_sign(int(cfg.get("wrist_yaw_sign", 1)))


def _config() -> dict:
    return json.loads(KIN_CONFIG.read_text()) if KIN_CONFIG.exists() else {}


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _pointing_deg(T: np.ndarray) -> float:
    """Horizontal direction the gripper points (deg; 0 = forward, +90 = left)."""
    z = T[:3, 2]
    return math.degrees(math.atan2(z[1], z[0]))


def taught_correction(forward_cm: float, left_cm: float):
    """From the grabs taught with teach_grasps.py: where the TIP really was (arm geometry) versus where
    the camera saw the highlighter. Returns (d_forward_cm, d_left_cm, d_height_cm, n_samples) blended
    from nearby examples (inverse-distance weighting), or None if fewer than 2 grabs were taught."""
    path = HERE / "grasp_samples.json"
    if not path.exists():
        return None
    samples = json.loads(path.read_text())["samples"]
    if len(samples) < 2:
        return None
    _apply_config()
    ref = kin.fk(load_poses()["poses"]["down_highlighter"])[:3, 3]
    num, den = np.zeros(3), 0.0
    for s in samples:
        tip = (kin.fk(s["joints"])[:3, 3] - ref) * 100.0
        det = s["detected"]
        resid = np.array([tip[0] - det["forward_cm"], tip[1] - det["left_cm"], tip[2]])
        d2 = (forward_cm - det["forward_cm"]) ** 2 + (left_cm - det["left_cm"]) ** 2
        w = 1.0 / (d2 + 4.0)  # +4 cm^2 keeps it smooth right at a taught spot
        num += w * resid
        den += w
    c = num / den
    return float(c[0]), float(c[1]), float(c[2]), len(samples)


def plan_from_taught(forward_cm: float, left_cm: float, lift_m: float, hover_ref: dict, axis_deg: float | None = None):
    """Copy the NEAREST taught grab (teach_grasps.py): same gripper angle, same height and the same
    tip-to-object offset, carried over to the new spot by turning about the arm's base.
    Returns (hover, down, note) or None if no grabs were taught."""
    path = HERE / "grasp_samples.json"
    if not path.exists():
        return None
    samples = json.loads(path.read_text())["samples"]
    if not samples:
        return None
    ref = kin.fk(load_poses()["poses"]["down_highlighter"])[:3, 3]
    def axis_diff(a, b):  # difference between two undirected lines, -90..90 deg
        return (a - b + 90.0) % 180.0 - 90.0

    dist = [math.hypot(forward_cm - s["detected"]["forward_cm"], left_cm - s["detected"]["left_cm"]) for s in samples]
    # prefer grabs taught with the highlighter lying the same way: 6 deg of difference counts like 1 cm
    cost = [d + (abs(axis_diff(axis_deg, s["detected"].get("axis_deg", axis_deg))) / 6.0 if axis_deg is not None else 0.0)
            for d, s in zip(dist, samples)]
    i = int(np.argmin(cost))
    s = samples[i]
    delta_deg = axis_diff(axis_deg, s["detected"].get("axis_deg", axis_deg)) if axis_deg is not None else 0.0
    T_s = kin.fk(s["joints"])
    det_s = ref + np.array([s["detected"]["forward_cm"], s["detected"]["left_cm"], 0.0]) / 100.0  # where it was seen
    new = ref + np.array([forward_cm, left_cm, 0.0]) / 100.0  # where the new object is
    theta = math.atan2(new[1], new[0]) - math.atan2(det_s[1], det_s[0])  # turn about the base axis
    off = T_s[:3, 3] - det_s  # tip relative to the object in the taught grab

    def solve(extra_deg: float):
        Rz = _rz(theta + math.radians(extra_deg))  # also turn by the object's angle difference
        off_xy = Rz @ np.array([off[0], off[1], 0.0])
        down_T = np.eye(4)
        down_T[:3, :3] = Rz @ T_s[:3, :3]
        down_T[:3, 3] = [new[0] + off_xy[0], new[1] + off_xy[1], T_s[2, 3]]
        down = kin.ik(down_T, s["joints"])
        for frac in (1.0, 0.6, 0.3):  # hover lower if the full lift is out of reach at this angle
            hover_T = down_T.copy()
            hover_T[2, 3] += lift_m * frac
            try:
                return kin.ik(hover_T, down), down
            except kin.IKError:
                continue
        raise kin.IKError("cannot reach a hover point above this spot")

    turn_note = ""
    try:
        hover, down = solve(delta_deg)
        if abs(delta_deg) >= 3:
            turn_note = f", turned {delta_deg:+.0f} deg to match the object"
    except kin.IKError:
        if abs(delta_deg) < 3:
            raise
        hover, down = solve(0.0)
        turn_note = f", could NOT turn {delta_deg:+.0f} deg to match the object (out of reach)"
    note = f"copied taught grab #{i + 1} ({dist[i]:.0f} cm away{turn_note}; {len(samples)} taught)"
    if dist[i] > 15:
        note += " - far from any taught grab, consider teaching one near here"
    return hover, down, note


def plan_pick(forward_cm: float, left_cm: float, axis_deg: float | None = None) -> dict:
    """Plan a grab of an object CENTRED at (forward_cm, left_cm).

    The calibration was done by touching the gripper TIP to the fiducials, so positions are tip
    positions. To get the object BETWEEN the jaws, the tip is aimed `grasp_depth_cm` past the object
    along the gripper's pointing direction. If `axis_deg` (the object's long axis, from
    find_highlighters) is given, the gripper is first turned to point along it (so the jaws close
    across its width); if that is not reachable it keeps the recorded heading and says so.
    Returns {'hover','down', 'aligned', 'lift_cm', ...} or raises ValueError / kin.IKError.
    """
    _apply_config()
    import vision

    if vision.CALIB.exists():
        if not vision.inside_zone_cm(forward_cm, left_cm):
            raise ValueError(f"({forward_cm:+.1f}, {left_cm:+.1f}) cm is outside the fiducial square")
    else:
        (f0, f1), (l0, l1) = PICK_ZONE_CM["forward"], PICK_ZONE_CM["left"]
        if not (f0 <= forward_cm <= f1 and l0 <= left_cm <= l1):
            raise ValueError(f"({forward_cm:+.1f}, {left_cm:+.1f}) cm is outside the pick zone")
    poses = load_poses()["poses"]
    if "down_highlighter" not in poses or "above_highlighter" not in poses:
        raise ValueError("needs the recorded poses 'down_highlighter' and 'above_highlighter'")
    depth_m = float(_config().get("grasp_depth_cm", 2.0)) / 100.0
    down_ref, hover_ref = poses["down_highlighter"], poses["above_highlighter"]
    ref_T = kin.fk(down_ref)
    lift_m = float(kin.fk(hover_ref)[2, 3] - ref_T[2, 3])  # recorded hover height above the grab
    taught = plan_from_taught(forward_cm, left_cm, lift_m, hover_ref, axis_deg)
    if taught is not None:
        hover, down, note = taught
        return {"hover": hover, "down": down, "lift_cm": round(lift_m * 100, 1), "aligned": False,
                "note": note, "pointing_deg": round(_pointing_deg(kin.fk(down)), 0), "grasp_depth_cm": 0.0}
    corr = taught_correction(forward_cm, left_cm)
    dz_m = 0.0
    corr_note = "no taught grabs yet: using grasp_depth"
    if corr is not None:  # taught grabs already contain the tip offset and any calibration error
        forward_cm, left_cm = forward_cm + corr[0], left_cm + corr[1]
        dz_m, depth_m = corr[2] / 100.0, 0.0
        corr_note = f"corrected by ({corr[0]:+.1f}, {corr[1]:+.1f}) cm from {corr[3]} taught grabs"
    f, l = forward_cm / 100.0, left_cm / 100.0

    def attempt(turn_rad: float):
        base = kin.grasp_target(ref_T, f, l, 0.0)  # orientation follows the base turn
        R = _rz(turn_rad) @ base[:3, :3]
        u = R[:, 2].copy()
        u[2] = 0.0
        u /= np.linalg.norm(u)  # horizontal pointing direction
        down_T = np.eye(4)
        down_T[:3, :3] = R
        down_T[:3, 3] = base[:3, 3] + depth_m * u  # tip goes past the object by grasp_depth
        down_T[2, 3] += dz_m  # grab height as taught
        hover_T = down_T.copy()
        hover_T[2, 3] += lift_m
        hover = kin.ik(hover_T, hover_ref)
        down = kin.ik(down_T, hover)
        return hover, down, _pointing_deg(down_T)

    aligned, note = False, "kept the recorded gripper heading"
    if axis_deg is not None:
        heading = _pointing_deg(kin.grasp_target(ref_T, f, l, 0.0))
        # the object axis is a line: pick the direction (axis or axis+180) closest to the heading
        cand = [axis_deg, axis_deg + 180.0]
        target = min(cand, key=lambda a: abs((a - heading + 180) % 360 - 180))
        turn = (target - heading + 180) % 360 - 180
        try:
            hover, down, point = attempt(math.radians(turn))
            aligned, note = True, f"turned the gripper {turn:+.0f} deg to line up with the object"
        except kin.IKError:
            hover, down, point = attempt(0.0)
            note = f"could not turn {turn:+.0f} deg to line up with the object (out of reach); kept the recorded heading"
    else:
        hover, down, point = attempt(0.0)
    note = f"{note}; {corr_note}"
    return {"hover": hover, "down": down, "lift_cm": round(lift_m * 100, 1), "aligned": aligned,
            "note": note, "pointing_deg": round(point, 0), "grasp_depth_cm": round(depth_m * 100, 1)}


def run_pick(arm, forward_cm: float, left_cm: float, axis_deg: float | None = None, log=print) -> dict:
    """Full pick on a connected SlowArm: hover -> open -> down -> close -> lift -> back over the recorded spot."""
    plan = plan_pick(forward_cm, left_cm, axis_deg)
    log("  moving above the recorded spot")
    arm.move_to_pose("above_highlighter")
    log(f"  hovering over ({forward_cm:+.1f}, {left_cm:+.1f}) cm")
    arm.move_to_joints(plan["hover"])
    arm.set_gripper("open")
    log("  descending")
    arm.move_to_joints(plan["down"])
    arm.set_gripper("closed")
    log("  lifting")
    arm.move_to_joints(plan["hover"])
    arm.move_to_pose("above_highlighter")
    return plan


if __name__ == "__main__":
    import sys

    fwd, left = float(sys.argv[1]), float(sys.argv[2])
    axis = float(sys.argv[3]) if len(sys.argv) > 3 else None
    p = plan_pick(fwd, left, axis)
    print(p["note"], "| pointing", p["pointing_deg"], "deg | grasp depth", p["grasp_depth_cm"], "cm")
    ref = load_poses()["poses"]
    for k in ("hover", "down"):
        base = ref["above_highlighter" if k == "hover" else "down_highlighter"]
        print(f"{k:5s}:", {j: p[k][j] for j in kin.ARM})
        print("       change vs recorded:", {j: round(p[k][j] - base[j], 1) for j in kin.ARM})
    T = kin.fk(p["down"]); R = kin.fk(ref["down_highlighter"])
    print(f"grab point moves {100*(T[0,3]-R[0,3]):+.1f} cm x, {100*(T[1,3]-R[1,3]):+.1f} cm y, {100*(T[2,3]-R[2,3]):+.1f} cm z "
          f"(model frame); hover lift {p['lift_cm']} cm")
