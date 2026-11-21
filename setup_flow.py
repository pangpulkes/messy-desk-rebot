"""Setup flows for the control window (ui_server.py): camera calibration, grab teaching, saving a pose.

Same logic as calibrate.py / teach_grasps.py / record_poses.py, but driven by buttons in the window while the
app keeps owning the follower (SlowArm). The follower copies the leader arm during the steps that need it.
Nothing is written to disk until the user keeps/saves; the file being replaced is backed up first
(backups/<file>.<time>). Leaving a flow (Done / Cancel) always drives the follower home slowly.

Modes:
  camera        4 fiducials with the leader -> arm home -> photo -> click 4 crosses -> check grid -> Keep
  camera_photo  fiducial poses already measured: photo -> click -> check -> Keep (use after the camera was bumped)
  grabs         A: lock the highlighter the camera sees, B: grab it with the leader, Save (repeat), Done
  pose          steer with the leader, type a name, Save
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from arm import ARM_JOINTS, LEADER_ID, StopMotion, find_leader_port, load_poses, save_poses

HERE = Path(__file__).resolve().parent
BACKUPS = HERE / "backups"
GREEN, RED, YELLOW, CYAN = (60, 220, 60), (40, 40, 230), (0, 220, 255), (255, 220, 0)


def backup(path: Path) -> None:
    if path.exists():
        BACKUPS.mkdir(exist_ok=True)
        shutil.copy2(path, BACKUPS / f"{path.name}.{time.strftime('%H%M%S')}")


class LeaderOnly:
    """Dry run (no follower): just read the leader so poses can still be recorded."""

    def __init__(self, leader):
        self.leader, self.leader_latest, self._stop = leader, {}, threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.leader_latest = {k.removesuffix(".pos"): float(v) for k, v in self.leader.get_action().items()}
            except Exception as e:
                print(f"[leader] {e}")
            time.sleep(1 / 30)

    def stop(self):
        self._stop.set()


class Setup:
    def __init__(self, brain):
        self.brain = brain
        self.lock = threading.RLock()
        self.mode: str | None = None
        self.step = ""
        self.flash = ""
        self.leader = None
        self.leader_only: LeaderOnly | None = None
        self._reset()

    # ------------------------------------------------------------------ helpers
    @property
    def arm(self):
        return self.brain.ex.arm

    def _reset(self):
        self.fid_joints: list[dict] = []
        self.photo = None  # BGR still for clicking
        self.clicks: list[tuple[float, float]] = []
        self.result = None  # (H, marks, check image)
        self.locked = None
        self.samples_added = 0

    def _frame_bgr(self):
        f = self.brain.ex._frame()
        return None if f is None else cv2.cvtColor(f, cv2.COLOR_RGB2BGR)

    def _leader_joints(self) -> dict:
        src = self.arm if self.arm is not None else self.leader_only
        latest = getattr(src, "leader_latest", {}) if src is not None else {}
        if not all(j in latest for j in ARM_JOINTS):
            raise RuntimeError("no reading from the leader arm yet (is its 12 V supply on?)")
        return {j: round(latest[j], 2) for j in ARM_JOINTS}

    def _connect_leader(self):
        if self.leader is not None:
            return
        from lerobot.teleoperators.rebot_102_leader import RebotArm102Leader, RebotArm102LeaderTeleopConfig

        leader = RebotArm102Leader(RebotArm102LeaderTeleopConfig(port=find_leader_port(), id=LEADER_ID))
        leader.connect(calibrate=False)
        self.leader = leader

    def _follow(self, on: bool):
        if self.arm is not None:
            if on:
                self.arm.stop_requested.clear()
                self.arm.start_follow(self.leader)
            else:
                self.arm.stop_follow()
        elif on and self.leader_only is None and self.leader is not None:
            self.leader_only = LeaderOnly(self.leader)

    def _go_home(self, then_step: str | None):
        """Drive the follower home in the background (slow), then switch to then_step (None = leave setup)."""
        self._follow(False)
        self.step = "homing"

        def run():
            ok = True
            try:
                if self.arm is not None:
                    self.arm.stop_requested.clear()
                    self.arm.move_to_pose("rest")
            except StopMotion:
                ok = False
            except Exception as e:
                ok = False
                self.brain.emit("error", text=f"Could not drive home: {e}")
            with self.lock:
                if then_step is None:
                    self._leave()
                else:
                    self.step = then_step if ok else "home_failed"
                    self._next_after_home = then_step

        threading.Thread(target=run, daemon=True).start()

    def _leave(self):
        if self.leader_only is not None:
            self.leader_only.stop()
            self.leader_only = None
        if self.leader is not None:
            try:
                self.leader.disconnect()
            except Exception:
                pass
            self.leader = None
        mode = self.mode
        self.mode, self.step = None, ""
        self._reset()
        self.brain.reload_tools()
        self.brain.busy = False
        self.brain.emit("status", text="Ready")
        self.brain.emit("idle")
        if mode:
            self.brain.emit("setup_done", text=f"Setup closed: {MODE_TITLES[mode]}.")

    # ------------------------------------------------------------------ API
    def start(self, mode: str) -> tuple[bool, str]:
        with self.lock:
            if mode not in MODE_TITLES:
                return False, "unknown setup"
            with self.brain.lock:
                if self.brain.busy:
                    return False, "The arm is busy - wait or press Stop first."
                self.brain.busy = True
            try:
                self._reset()
                self.flash = ""
                poses = load_poses()["poses"]
                if mode == "camera_photo":
                    missing = [n for n, _, _ in FIDS if n not in poses]
                    if missing:
                        raise RuntimeError(f"fiducial poses {missing} not measured yet - use the full camera calibration")
                    self.fid_joints = [poses[n] for n, _, _ in FIDS]
                if mode in ("camera", "camera_photo") and "down_highlighter" not in poses:
                    raise RuntimeError("record the 'down_highlighter' pose first (it is the reference point)")
                if mode == "grabs":
                    import vision

                    vision.load_calib()
                if self.brain.ex._frame() is None and mode != "pose":
                    raise RuntimeError("no camera")
                self.mode = mode
                if mode == "camera_photo":
                    self.step = "photo"
                else:
                    self._connect_leader()
                    self._follow(True)
                    self.step = {"camera": "fid", "grabs": "A", "pose": "pose"}[mode]
                self.brain.emit("status", text=f"Setup: {MODE_TITLES[mode]}")
                return True, ""
            except Exception as e:
                self.mode = None
                if self.leader is not None:
                    try:
                        self.leader.disconnect()
                    except Exception:
                        pass
                    self.leader = None
                self.brain.busy = False
                return False, str(e)

    def action(self, act: str, body: dict) -> tuple[bool, str]:
        with self.lock:
            if self.mode is None:
                return False, "no setup running"
            self.flash = ""
            try:
                msg = self._action(act, body)
                return True, msg or ""
            except Exception as e:
                self.flash = str(e)
                return False, str(e)

    def _action(self, act: str, body: dict) -> str | None:
        m, s = self.mode, self.step
        if act in ("cancel", "done"):
            if s == "homing":
                raise RuntimeError("wait until the arm is home")
            if self.arm is None:
                self._follow(False)
                self._leave()
            else:
                self._go_home(None)
            return None
        if act == "resume":
            self._follow(True)
            return None
        # ---- camera calibration
        if m in ("camera", "camera_photo"):
            if s == "fid" and act == "save":
                self.fid_joints.append(self._leader_joints())
                self.flash = f"Saved {FIDS[len(self.fid_joints) - 1][1]}"
                if len(self.fid_joints) == len(FIDS):
                    self._go_home("photo")
                return None
            if s == "fid" and act == "undo" and self.fid_joints:
                self.fid_joints.pop()
                return None
            if s == "photo" and act == "photo":
                time.sleep(0.3)
                photo = self._frame_bgr()
                if photo is None:
                    raise RuntimeError("no camera frame")
                self.photo, self.clicks, self.step = photo, [], "click"
                return None
            if s == "click" and act == "click":
                if len(self.clicks) < len(FIDS):
                    from calibrate import snap_to_cross

                    gray = cv2.cvtColor(self.photo, cv2.COLOR_BGR2GRAY)
                    self.clicks.append(snap_to_cross(gray, float(body["x"]), float(body["y"])))
                return None
            if s == "click" and act == "unclick" and self.clicks:
                self.clicks.pop()
                return None
            if s == "click" and act == "retake":
                self.step = "photo"
                return None
            if s == "click" and act == "compute" and len(self.clicks) == len(FIDS):
                from calibrate import draw_result

                marks = fid_marks_cm(self.fid_joints)
                H, _ = cv2.findHomography(np.array(self.clicks, np.float64), np.array(marks, np.float64), 0)
                if H is None:
                    raise RuntimeError("could not compute the mapping - redo the clicks")
                check = draw_result(self.photo, H, self.clicks, marks, [f for _, f, _ in FIDS])
                self.result = (H, marks, check)
                self.step = "check"
                return None
            if s == "check" and act == "redo":
                self.step, self.result = "click", None
                return None
            if s == "check" and act == "keep":
                self._save_camera()
                self.flash = "Saved the camera calibration"
                self._leave() if self.arm is None or m == "camera_photo" else self._go_home(None)
                return None
        # ---- grab teaching
        if m == "grabs":
            if s == "A" and act == "lock":
                import vision

                frame = self.brain.ex._frame()
                dets, _ = vision.find(frame)
                if not dets:
                    raise RuntimeError("no highlighter seen inside the square")
                self.locked = {k: dets[0][k] for k in ("colour", "forward_cm", "left_cm", "axis_deg", "length_cm")}
                self.step = "B"
                return None
            if s == "B" and act == "back":
                self.step, self.locked = "A", None
                return None
            if s == "B" and act == "save":
                from teach_grasps import load_samples, save_samples

                joints = self._leader_joints()
                samples = load_samples()
                if self.samples_added == 0:
                    backup(HERE / "grasp_samples.json")
                samples.append({"detected": self.locked, "joints": joints, "time": time.strftime("%H:%M:%S")})
                save_samples(samples)
                self.samples_added += 1
                self.flash = f"Saved grab {len(samples)} - now fold the arm away with the leader"
                self.step, self.locked = "A", None
                return None
            if s == "A" and act == "undo" and self.samples_added > 0:
                from teach_grasps import load_samples, save_samples

                samples = load_samples()
                samples.pop()
                save_samples(samples)
                self.samples_added -= 1
                self.flash = "Removed the last grab"
                return None
        # ---- save a pose
        if m == "pose" and act == "save":
            name = str(body.get("name", "")).strip()
            if not name or not name.replace("_", "").isalnum():
                raise RuntimeError("use a name made of letters, digits and _ (e.g. drop_vertical)")
            if name == "rest" and not body.get("confirm"):
                raise RuntimeError("'rest' is the home pose every program returns to - pick another name")
            joints = self._leader_joints()
            backup(HERE / "poses.json")
            data = load_poses()
            data["poses"][name] = joints
            save_poses(data)
            self.flash = f"Saved pose '{name}'"
            return None
        raise RuntimeError(f"'{act}' is not available now")

    def _save_camera(self):
        H, marks, check = self.result
        for f in ("camera_calib.json", "calib_photo.png", "calib_check.png"):
            backup(HERE / f)
        if self.mode == "camera":
            backup(HERE / "poses.json")
            data = load_poses()
            for (n, _, _), j in zip(FIDS, self.fid_joints):
                data["poses"][n] = j
            save_poses(data)
        cv2.imwrite(str(HERE / "calib_photo.png"), self.photo)
        cv2.imwrite(str(HERE / "calib_check.png"), check)
        (HERE / "camera_calib.json").write_text(json.dumps(
            {"H_pixel_to_cm": H.tolist(), "pixels": self.clicks, "marks_cm": marks,
             "fiducials": [n for n, _, _ in FIDS], "camera": 0, "size": [640, 480],
             "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))

    def close(self):
        """Called when the app quits (the arm itself is parked by ex.close())."""
        with self.lock:
            if self.leader_only is not None:
                self.leader_only.stop()
            if self.arm is not None:
                self.arm.stop_follow()
            if self.leader is not None:
                try:
                    self.leader.disconnect()
                except Exception:
                    pass
                self.leader = None

    # ------------------------------------------------------------------ what the window shows
    def state(self) -> dict:
        with self.lock:
            m, s = self.mode, self.step
            st = {"mode": m, "step": s, "flash": self.flash, "clickable": False, "buttons": [], "name_input": False,
                  "following": bool(self.arm is not None and self.arm.following)}
            if m is None:
                return st
            st["title"] = MODE_TITLES[m]
            b = st["buttons"]
            needs_follow = s in ("fid", "A", "B", "pose") and self.arm is not None
            if needs_follow and not st["following"]:
                st["warn"] = "The follower stopped copying the leader (Stop was pressed)."
                b.append(("resume", "Resume following", True))
            if s == "homing":
                st["text"] = "The arm is driving itself home slowly… please wait."
            elif s == "home_failed":
                st["text"] = "The arm stopped before reaching home. Press Done to try again."
            elif s == "fid":
                i = len(self.fid_joints)
                _, friendly, where = FIDS[i]
                st["heading"] = f"Fiducial {i + 1} of 4: {friendly}"
                st["text"] = (f"It is in the {where} (yellow box). Steer with the leader arm and touch the gripper TIP "
                              "to the centre of the cross, then press Save.")
                b.append(("save", "Save this fiducial", True))
                if i:
                    b.append(("undo", "Redo previous", False))
            elif s == "photo":
                st["heading"] = "Take a clean photo"
                st["text"] = ("Make sure both arms are out of the picture and all 4 fiducial crosses are visible, "
                              "then take the photo.")
                b.append(("photo", "Take photo", True))
            elif s == "click":
                n = len(self.clicks)
                st["clickable"] = n < len(FIDS)
                if n < len(FIDS):
                    _, friendly, where = FIDS[n]
                    st["heading"] = f"Click the centre of {friendly} ({n}/4 done)"
                    st["text"] = f"It is in the {where}. Clicks snap to the cross."
                else:
                    st["heading"] = "All 4 clicked"
                    st["text"] = "Compute the mapping and check it."
                    b.append(("compute", "Compute", True))
                if n:
                    b.append(("unclick", "Undo click", False))
                b.append(("retake", "Retake photo", False))
            elif s == "check":
                st["heading"] = "Check the result"
                st["text"] = ("The blue grid should be even 5 cm squares lying flat on the table, and the red crosses "
                              "on the fiducials. Keep it to save (the old calibration is backed up).")
                b.append(("keep", "Keep and save", True))
                b.append(("redo", "Redo the clicks", False))
            elif s == "A":
                from teach_grasps import load_samples

                st["heading"] = f"Grab {len(load_samples()) + 1}: place a highlighter"
                st["text"] = ("Fold the arm out of the view with the leader. Put ONE highlighter somewhere new inside the "
                              "square, lying forward or sideways. When its box is steady, lock it.")
                b.append(("lock", "Lock position", True))
                if self.samples_added:
                    b.append(("undo", "Remove last grab", False))
            elif s == "B":
                lk = self.locked
                st["heading"] = f"Grab it: {lk['colour']} at forward {lk['forward_cm']:+.1f}, left {lk['left_cm']:+.1f} cm"
                st["text"] = ("Steer with the leader so the OPEN jaws are around the highlighter exactly as it should be "
                              "grabbed (at grabbing height), then save. Afterwards fold the arm away.")
                b.append(("save", "Save grab", True))
                b.append(("back", "Back", False))
            elif s == "pose":
                st["heading"] = "Steer the arm with the leader, then save the pose"
                st["text"] = ("The follower copies the leader. Type a name (e.g. drop_vertical) and save. Saving an "
                              "existing name replaces it (poses.json is backed up).")
                st["name_input"] = True
                b.append(("save", "Save pose", True))
            if s != "homing":
                done_label = "Cancel" if m in ("camera", "camera_photo") and s != "check" else "Done"
                b.append(("done", done_label + (" (arm goes home)" if self.arm is not None and m != "camera_photo" else ""), False))
            st["buttons"] = [{"id": i, "label": lab, "primary": p} for i, lab, p in b]
            if m == "grabs":
                from teach_grasps import load_samples

                st["info"] = f"{len(load_samples())} grabs taught ({self.samples_added} added now)"
            return st

    def view_jpeg(self) -> bytes | None:
        with self.lock:
            m, s = self.mode, self.step
            if m is None:
                return None
            if s == "click":
                img = self.photo.copy()
                for n, (x, y) in enumerate(self.clicks):
                    cv2.drawMarker(img, (int(x), int(y)), RED, cv2.MARKER_CROSS, 18, 2)
                    cv2.putText(img, FIDS[n][1], (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)
                if len(self.clicks) < len(FIDS):
                    _hint(img, FIDS[len(self.clicks)][2])
            elif s == "check":
                img = self.result[2].copy()
            else:
                img = self._frame_bgr()
                if img is None:
                    return None
                if s == "fid":
                    _hint(img, FIDS[len(self.fid_joints)][2])
                elif s in ("A", "B"):
                    import vision

                    H = vision.load_calib()
                    if s == "A":
                        _, img = vision.find(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), H)
                    _draw_samples(img, H)
                    if s == "B" and self.locked:
                        _mark_cm(img, H, self.locked["forward_cm"], self.locked["left_cm"], RED, "locked")
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None


MODE_TITLES = {"camera": "Camera calibration", "camera_photo": "Camera: new photo only", "grabs": "Teach grabs",
               "pose": "Save a pose"}

from calibrate import FIDS  # noqa: E402  (fid pose names, friendly names, image corners)


def _hint(img, where: str) -> None:
    from calibrate import hint

    hint(img, where)


def fid_marks_cm(fid_joints: list[dict]) -> list[tuple[float, float]]:
    """Table positions (cm from the recorded grab spot) of the fiducials, from the arm's joints (forward kinematics)."""
    import kin
    from picker import _apply_config

    _apply_config()
    ref = kin.fk(load_poses()["poses"]["down_highlighter"])[:3, 3]
    out = []
    for j in fid_joints:
        d = (kin.fk(j)[:3, 3] - ref) * 100
        out.append((float(d[0]), float(d[1])))
    return out


def _mark_cm(img, H, f, l, colour, label):
    p = cv2.perspectiveTransform(np.array([[[f, l]]], np.float64), np.linalg.inv(H)).reshape(2)
    x, y = int(p[0]), int(p[1])
    cv2.drawMarker(img, (x, y), colour, cv2.MARKER_TILTED_CROSS, 18, 2)
    cv2.putText(img, label, (x + 9, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def _draw_samples(img, H):
    from teach_grasps import load_samples

    for i, s in enumerate(load_samples()):
        d = s["detected"]
        p = cv2.perspectiveTransform(np.array([[[d["forward_cm"], d["left_cm"]]]], np.float64), np.linalg.inv(H)).reshape(2)
        cv2.circle(img, (int(p[0]), int(p[1])), 7, GREEN, 2)
        cv2.putText(img, str(i + 1), (int(p[0]) + 8, int(p[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
