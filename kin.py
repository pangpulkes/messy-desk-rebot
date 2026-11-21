"""Kinematics of the reBot B601-RS from Seeed's URDF (pure numpy).

  fk(pose)                      -> 4x4 transform of the gripper end in the arm's base frame (metres)
  ik(target_T, seed_pose)       -> public joint angles (deg) that reach target_T, or raise IKError

Public joint angles are the degrees LeRobot reports; SIGN maps them onto the URDF joints.
wrist_yaw's sign is not yet confirmed (see WRIST_YAW_SIGN); it is settled by a ruler check.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

def _find_urdf() -> Path:
    """Seeed's URDF ships inside the lerobot_robot_seeed_b601 package."""
    import importlib.util

    spec = importlib.util.find_spec("lerobot_robot_seeed_b601")
    if spec is None or not spec.origin:
        raise ImportError("lerobot_robot_seeed_b601 is not installed (pip install lerobot-robot-seeed-b601)")
    return Path(spec.origin).parent / "assets" / "00-arm-rs_asm-v3.urdf"


URDF = _find_urdf()
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
URDF_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

WRIST_YAW_SIGN = 1  # +1 = interpretation A (tape ~9 cm to the side), -1 = B (~20 cm)
SIGN = {"shoulder_pan": 1, "shoulder_lift": -1, "elbow_flex": -1, "wrist_flex": -1,
        "wrist_yaw": WRIST_YAW_SIGN, "wrist_roll": 1}

# Follower soft limits (public degrees), from the RS motor profile.
LIMITS = {"shoulder_pan": (-145, 145), "shoulder_lift": (-170, 0), "elbow_flex": (-200, 0),
          "wrist_flex": (-90, 80), "wrist_yaw": (-90, 90), "wrist_roll": (-90, 90)}


def set_wrist_yaw_sign(s: int) -> None:
    global WRIST_YAW_SIGN
    WRIST_YAW_SIGN = s
    SIGN["wrist_yaw"] = s


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_rot(axis, q):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


_CHAIN = []
for _j in ET.parse(URDF).getroot().findall("joint"):
    _o, _a = _j.find("origin"), _j.find("axis")
    _CHAIN.append((
        _j.get("name"), _j.get("type"),
        np.array([float(v) for v in (_o.get("xyz") or "0 0 0").split()]),
        _rpy(*[float(v) for v in (_o.get("rpy") or "0 0 0").split()]),
        [float(v) for v in _a.get("xyz").split()] if _a is not None else None,
    ))


def fk(pose_deg: dict) -> np.ndarray:
    """4x4 pose of the gripper end (between the jaws' tips) in the base frame, metres."""
    q = {URDF_JOINTS[i]: math.radians(SIGN[n] * float(pose_deg[n])) for i, n in enumerate(ARM)}
    T = np.eye(4)
    for name, typ, xyz, R, axis in _CHAIN:
        if name.startswith("joint_"):  # finger slides
            continue
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = xyz
        T = T @ A
        if typ == "revolute":
            J = np.eye(4)
            J[:3, :3] = _axis_rot(axis, q[name])
            T = T @ J
        if name == "j_gripper_end":
            break
    return T


def _rot_err(Rc, Rt):
    """Axis-angle vector rotating Rc onto Rt (in base frame)."""
    R = Rt @ Rc.T
    ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return v * ang


class IKError(RuntimeError):
    pass


def ik(target: np.ndarray, seed: dict, pos_tol_m=0.002, rot_tol_rad=math.radians(2.0), iters=300) -> dict:
    """Damped-least-squares IK for position + orientation, starting from `seed` (public deg)."""
    names = ARM
    q = np.array([float(seed[n]) for n in names])
    lam = 0.02
    for _ in range(iters):
        T = fk(dict(zip(names, q)))
        e = np.concatenate([target[:3, 3] - T[:3, 3], 0.3 * _rot_err(T[:3, :3], target[:3, :3])])
        if np.linalg.norm(e[:3]) < pos_tol_m and np.linalg.norm(e[3:]) / 0.3 < rot_tol_rad:
            break
        J = np.zeros((6, len(names)))  # per RADIAN, so the damping is well scaled
        h = 0.05  # deg
        hr = math.radians(h)
        for k in range(len(names)):
            qp = q.copy()
            qp[k] += h
            Tp = fk(dict(zip(names, qp)))
            J[:3, k] = (Tp[:3, 3] - T[:3, 3]) / hr
            J[3:, k] = 0.3 * _rot_err(T[:3, :3], Tp[:3, :3]) / hr
        dq = np.degrees(J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e))
        dq = np.clip(dq, -5.0, 5.0)  # deg per iteration
        q = q + dq
        for k, n in enumerate(names):
            lo, hi = LIMITS[n]
            q[k] = min(hi, max(lo, q[k]))
    T = fk(dict(zip(names, q)))
    pe = float(np.linalg.norm(target[:3, 3] - T[:3, 3]))
    re = float(np.linalg.norm(_rot_err(T[:3, :3], target[:3, :3])))
    if pe > pos_tol_m * 2 or re > rot_tol_rad * 2:
        raise IKError(f"cannot reach target (position error {pe*100:.1f} cm, angle error {math.degrees(re):.1f} deg)")
    return {n: round(float(v), 2) for n, v in zip(names, q)}


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def grasp_target(ref_T: np.ndarray, forward_m: float, left_m: float, up_m: float = 0.0) -> np.ndarray:
    """Target pose offset from the reference grasp (the taped spot).

    forward/left are in the table frame aligned with the arm's straight-ahead direction (+x) and
    its left (+y). The gripper keeps the reference orientation, turned about vertical so it still
    faces radially outward like at the reference spot.
    """
    p0 = ref_T[:3, 3]
    p = p0 + np.array([forward_m, left_m, up_m])
    dpan = math.atan2(p[1], p[0]) - math.atan2(p0[1], p0[0])
    T = np.eye(4)
    T[:3, :3] = _rz(dpan) @ ref_T[:3, :3]
    T[:3, 3] = p
    return T
