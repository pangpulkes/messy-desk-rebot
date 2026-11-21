"""Visual camera <-> arm calibration with 4 fiducials, all in one camera window.

  python ~/rebot_lerobot/llm_arm/calibrate.py

Before starting: both arms folded to rest, area clear, leader + follower powered, camera fixed overhead.
Click the camera window once so it receives your key presses.

  Step 1  The follower copies the leader. For the highlighted fiducial, touch the gripper TIP to
          its cross, then press SPACE. (4 times; U = redo the last one)
  Step 2  The follower drives itself home and switches off. Move the leader out of the view,
          press SPACE -> clean photo.
  Step 3  Click the centre of each fiducial in the order shown (clicks snap to the cross).
          Right-click = undo.
  Step 4  Check the result (5 cm grid drawn on the table), press any key. Saved.

Q / Esc / Ctrl+C at any time = stop. The follower ALWAYS drives itself home before its motors
switch off (a second Ctrl+C does not skip this; use the 48 V switch in an emergency).
"""

import _env  # noqa: F401  (re-launches inside the rebot env if needed)

import json
import time
from pathlib import Path

import cv2
import numpy as np

from arm import (ARM_JOINTS, FOLLOWER_ID, FOLLOWER_PORT, LEADER_ID, MAX_RELATIVE_TARGET, find_leader_port, load_poses,
                 park_follower, save_poses)

HERE = Path(__file__).resolve().parent
CALIB = HERE / "camera_calib.json"
SCALE = 1.5  # display size multiplier (640x480 -> 960x720)

# (pose name, friendly name, where it is in the overhead image)
FIDS = [
    ("fid_near_left", "NEAR-LEFT", "top-right of the image"),
    ("fid_near_right", "NEAR-RIGHT", "top-left of the image"),
    ("fid_far_right", "FAR-RIGHT", "bottom-left of the image"),
    ("fid_far_left", "FAR-LEFT", "bottom-right of the image"),
]
CORNER_HINT = {"top-right of the image": (0.85, 0.15), "top-left of the image": (0.15, 0.15),
               "bottom-left of the image": (0.15, 0.85), "bottom-right of the image": (0.85, 0.85)}

YELLOW, GREEN, RED, WHITE, CYAN = (0, 220, 255), (60, 220, 60), (40, 40, 230), (255, 255, 255), (255, 220, 0)


STRIP_H = 100  # instruction strip ABOVE the photo, so it never hides a fiducial


def banner(img, lines, colour=WHITE):
    """Return img with a black instruction strip stacked on top (clicks must subtract STRIP_H)."""
    strip = np.zeros((STRIP_H, img.shape[1], 3), np.uint8)
    for i, t in enumerate(lines[:3]):
        cv2.putText(strip, t, (12, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour if i == 0 else WHITE, 2, cv2.LINE_AA)
    return np.vstack([strip, img])


def hint(img, where: str) -> None:
    """Outline the quarter of the image where the requested fiducial is."""
    h, w = img.shape[:2]
    cx, cy = CORNER_HINT[where]
    x0, x1 = (w // 2, w - 4) if cx > 0.5 else (4, w // 2)
    y0, y1 = (h // 2, h - 4) if cy > 0.5 else (4, h // 2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), YELLOW, -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x1, y1), YELLOW, 4)


def big(img_bgr):
    return cv2.resize(img_bgr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)


def snap_to_cross(gray: np.ndarray, x: float, y: float, r: int = 22) -> tuple[float, float]:
    """Refine a click to the centroid of the dark fiducial ink near it (640x480 coordinates)."""
    h, w = gray.shape
    x0, x1, y0, y1 = max(0, int(x) - r), min(w, int(x) + r), max(0, int(y) - r), min(h, int(y) + r)
    win = gray[y0:y1, x0:x1]
    if win.size == 0:
        return x, y
    thr = min(int(win.mean()) - 40, 110)
    ys, xs = np.nonzero(win < thr)
    if len(xs) < 15:
        return x, y
    return float(xs.mean() + x0), float(ys.mean() + y0)


def fid_positions_cm(names: list[str]) -> list[tuple[float, float]]:
    """Arm-measured fiducial positions: cm relative to the recorded grab point, forward/left axes."""
    import kin
    from picker import _apply_config

    _apply_config()
    poses = load_poses()["poses"]
    ref = kin.fk(poses["down_highlighter"])[:3, 3]
    out = []
    for n in names:
        d = (kin.fk(poses[n])[:3, 3] - ref) * 100
        out.append((float(d[0]), float(d[1])))
    return out


def draw_result(photo_bgr, H, clicks, marks, labels) -> np.ndarray:
    """Photo with a 5 cm table grid, the fiducials and the recorded grab point."""
    vis = photo_bgr.copy()
    Hinv = np.linalg.inv(H)
    m = np.array(marks)
    f_lo, f_hi = np.floor(m[:, 0].min() / 5) * 5 - 5, np.ceil(m[:, 0].max() / 5) * 5 + 5
    l_lo, l_hi = np.floor(m[:, 1].min() / 5) * 5 - 5, np.ceil(m[:, 1].max() / 5) * 5 + 5

    def to_px(f, l):
        p = cv2.perspectiveTransform(np.array([[[f, l]]], np.float64), Hinv).reshape(2)
        return int(round(p[0])), int(round(p[1]))

    for f in np.arange(f_lo, f_hi + 0.1, 5):
        cv2.line(vis, to_px(f, l_lo), to_px(f, l_hi), CYAN, 1, cv2.LINE_AA)
    for l in np.arange(l_lo, l_hi + 0.1, 5):
        cv2.line(vis, to_px(f_lo, l), to_px(f_hi, l), CYAN, 1, cv2.LINE_AA)
    for (x, y), lab in zip(clicks, labels):
        cv2.drawMarker(vis, (int(x), int(y)), RED, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(vis, lab, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)
    gx, gy = to_px(0, 0)
    cv2.circle(vis, (gx, gy), 7, GREEN, 2)
    cv2.putText(vis, "recorded grab spot", (gx + 9, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
    # arrow showing the arm's forward direction
    cv2.arrowedLine(vis, to_px(0, 0), to_px(5, 0), GREEN, 2, tipLength=0.3)
    return vis


# ------------------------------------------------------------------------------------------ steps
def step1_record(cam, mirror) -> bool:
    """Returns True when all 4 fiducial poses are saved."""
    i = 0
    flash_until, flash_text = 0.0, ""
    while i < len(FIDS):
        frame = cv2.cvtColor(cam.read(), cv2.COLOR_RGB2BGR)
        disp = big(frame)
        name, friendly, where = FIDS[i]
        hint(disp, where)
        if time.time() < flash_until:
            cv2.putText(disp, flash_text, (12, disp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2, cv2.LINE_AA)
        disp = banner(disp, [f"STEP 1 of 3 - fiducial {i + 1}/4: {friendly}  ({where}, yellow box)",
                             "Steer with the leader: touch the gripper TIP to the cross.",
                             "SPACE = save    U = redo previous    Q = stop"], YELLOW)
        cv2.imshow("calibration", disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            return False
        if k == ord("u") and i > 0:
            i -= 1
            flash_text, flash_until = f"Redo {FIDS[i][1]}", time.time() + 1.5
        if k == ord(" "):
            cur = mirror.read()
            data = load_poses()
            data["poses"][name] = {j: cur[j] for j in ARM_JOINTS}
            save_poses(data)
            print(f"saved {name}: {data['poses'][name]}")
            flash_text, flash_until = f"Saved {friendly}", time.time() + 1.5
            i += 1
    return True


def show_message(cam, lines) -> None:
    frame = cv2.cvtColor(cam.read(), cv2.COLOR_RGB2BGR)
    cv2.imshow("calibration", banner(big(frame), lines, YELLOW))
    cv2.waitKey(1)


def step2_photo(cam) -> np.ndarray | None:
    while True:
        frame = cv2.cvtColor(cam.read(), cv2.COLOR_RGB2BGR)
        disp = big(frame)
        disp = banner(disp, ["STEP 2 of 3 - the arm is home. Move the LEADER out of the view too.",
                             "Make sure all 4 fiducials are visible, then press SPACE.", "Q = stop"], YELLOW)
        cv2.imshow("calibration", disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            return None
        if k == ord(" "):
            time.sleep(0.5)
            for _ in range(5):  # take a fresh, settled frame
                frame = cam.read()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def step3_click(photo) -> list[tuple[float, float]] | None:
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    clicks: list[tuple[float, float]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(FIDS) and y >= STRIP_H:
            clicks.append(snap_to_cross(gray, x / SCALE, (y - STRIP_H) / SCALE))
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    cv2.setMouseCallback("calibration", on_mouse)
    try:
        while True:
            disp = big(photo)
            for n, (x, y) in enumerate(clicks):
                p = (int(x * SCALE), int(y * SCALE))
                cv2.drawMarker(disp, p, RED, cv2.MARKER_CROSS, 24, 2)
                cv2.putText(disp, FIDS[n][1], (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
            if len(clicks) < len(FIDS):
                _, friendly, where = FIDS[len(clicks)]
                hint(disp, where)
                disp = banner(disp, [f"STEP 3 of 3 - click the centre of {friendly} ({where})",
                                     f"{len(clicks)}/4 clicked. Clicks snap to the cross. Right-click = undo.",
                                     "Q = stop"], YELLOW)
            else:
                disp = banner(disp, ["All 4 clicked - press SPACE to compute,", "or right-click to undo."], GREEN)
            cv2.imshow("calibration", disp)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("q"), 27):
                return None
            if k == ord(" ") and len(clicks) == len(FIDS):
                return clicks
    finally:
        cv2.setMouseCallback("calibration", lambda *a: None)


def main() -> None:
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.robots.rebot_b601_follower import RebotB601Follower, RebotB601FollowerRobotConfig
    from lerobot.teleoperators.rebot_102_leader import RebotArm102Leader, RebotArm102LeaderTeleopConfig
    from record_poses import Mirror

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--photo-only", action="store_true",
                    help="skip step 1 (fiducials already measured): just take the photo, click, compute; arms stay off")
    ap.add_argument("--recompute", action="store_true",
                    help="no camera/arms: rebuild the mapping from the saved clicks (after changing kin_config.json)")
    args = ap.parse_args()

    if args.recompute:
        cal = json.loads(CALIB.read_text())
        names, clicks = cal["fiducials"], cal["pixels"]
        marks = fid_positions_cm(names)
        H, _ = cv2.findHomography(np.array(clicks, np.float64), np.array(marks, np.float64), 0)
        cal.update(H_pixel_to_cm=H.tolist(), marks_cm=marks, recomputed=time.strftime("%Y-%m-%d %H:%M:%S"))
        CALIB.write_text(json.dumps(cal, indent=2))
        photo = cv2.imread(str(HERE / "calib_photo.png"))
        cv2.imwrite(str(HERE / "calib_check.png"), draw_result(photo, H, clicks, marks, [f for _, f, _ in FIDS]))
        for n, (f, l) in zip(names, marks):
            print(f"  {n:15s} forward {f:+6.1f}  left {l:+6.1f}")
        print("Recomputed camera_calib.json and calib_check.png from the saved clicks.")
        return

    poses = load_poses()["poses"]
    if "down_highlighter" not in poses:
        raise SystemExit("Record 'down_highlighter' first (the grab pose) - it is the reference point.")

    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480))
    cam.connect()

    if args.photo_only:
        missing = [n for n, _, _ in FIDS if n not in poses]
        if missing:
            cam.disconnect()
            raise SystemExit(f"Missing fiducial poses {missing}: run without --photo-only first.")
        cv2.namedWindow("calibration")
        try:
            photo = step2_photo(cam)
        finally:
            cam.disconnect()
        if photo is None:
            cv2.destroyAllWindows()
            raise SystemExit("Stopped.")
        return finish(photo)

    leader = RebotArm102Leader(RebotArm102LeaderTeleopConfig(port=find_leader_port(), id=LEADER_ID))
    leader.connect(calibrate=False)
    follower = RebotB601Follower(RebotB601FollowerRobotConfig(
        port=FOLLOWER_PORT, motor_family="rs", id=FOLLOWER_ID, max_relative_target=MAX_RELATIVE_TARGET))
    print("Connecting follower: it now copies the leader. Keep clear of it.")
    follower.connect(calibrate=False)
    mirror = Mirror(leader, follower)
    mirror.thread.start()
    cv2.namedWindow("calibration")
    photo = None
    ok = False
    parked = False

    def park():
        nonlocal parked
        if not parked:
            parked = True
            mirror.stop.set()  # stop copying the leader
            mirror.thread.join(timeout=1.0)
            park_follower(follower, dict(mirror.cmd))  # drives itself home, then motors off

    try:
        ok = step1_record(cam, mirror)
        if ok:
            # the follower drives itself home (out of the view), then the photo is taken
            show_message(cam, ["STEP 2 of 3 - the arm is driving itself home...", "Please wait."])
            park()
            photo = step2_photo(cam)
    finally:
        park()  # on Q, Ctrl+C or any error: always home first, never just drop
        try:
            leader.disconnect()
        except Exception:
            pass
        cam.disconnect()

    if not ok or photo is None:
        cv2.destroyAllWindows()
        if ok:
            raise SystemExit("Stopped at the photo step. The 4 fiducials ARE saved - finish with:\n"
                             "  python ~/rebot_lerobot/llm_arm/calibrate.py --photo-only")
        raise SystemExit("Calibration stopped before the end. Nothing changed in camera_calib.json.")
    finish(photo)


def finish(photo) -> None:
    cv2.imwrite(str(HERE / "calib_photo.png"), photo)
    clicks = step3_click(photo)
    if clicks is None:
        cv2.destroyAllWindows()
        raise SystemExit("Stopped. The fiducial poses are saved; run again to redo the clicking.")

    names = [n for n, _, _ in FIDS]
    marks = fid_positions_cm(names)
    H, _ = cv2.findHomography(np.array(clicks, np.float64), np.array(marks, np.float64), 0)
    result = draw_result(photo, H, clicks, marks, [f for _, f, _ in FIDS])
    cv2.imwrite(str(HERE / "calib_check.png"), result)
    CALIB.write_text(json.dumps({"H_pixel_to_cm": H.tolist(), "pixels": clicks, "marks_cm": marks,
                                 "fiducials": names, "camera": 0, "size": [640, 480],
                                 "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print("Fiducial positions measured by the arm (cm from the recorded grab spot):")
    for n, (f, l) in zip(names, marks):
        print(f"  {n:15s} forward {f:+6.1f}  left {l:+6.1f}")
    print(f"Saved {CALIB} and the check image {HERE / 'calib_check.png'}")

    disp = big(result)
    disp = banner(disp, ["DONE - the blue grid is 5 cm squares on the table.",
                         "It should look evenly spaced and match the table.", "Press any key to close."], GREEN)
    cv2.imshow("calibration", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
