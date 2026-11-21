"""Camera calibration: click the tape marks in a photo -> pixel-to-table mapping.

Table coordinates are centimetres relative to the MAIN tape mark (where `down_highlighter` grabs):
  forward = along the arm's straight-ahead direction (away from the base), left = to the arm's left
  (seen from behind the arm).

Usage (arm folded to rest so it does not hide the marks):
  python ~/rebot_lerobot/llm_arm/calib_camera.py
  python ~/rebot_lerobot/llm_arm/calib_camera.py --marks "0,0 10,0 -10,0 0,10 0,-10"   # custom offsets (fwd,left cm)

A window opens with the camera photo. Click each mark's centre in the order printed in the
terminal (right-click = undo the last click). Press any key when all are clicked.
Saves camera_calib.json next to this script.
"""

import _env  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = HERE / "camera_calib.json"
DEFAULT_MARKS = "0,0 10,10 10,-10 -10,-10 -10,10"
NAMES = {(0, 0): "MAIN tape mark (centre)",
         (10, 10): "fiducial FAR-LEFT (10 cm further from the arm, 10 cm to the arm's left)",
         (10, -10): "fiducial FAR-RIGHT (10 cm further, 10 cm to the arm's right)",
         (-10, -10): "fiducial NEAR-RIGHT (10 cm closer to the arm, 10 cm right)",
         (-10, 10): "fiducial NEAR-LEFT (10 cm closer to the arm, 10 cm left)",
         (10, 0): "10 cm FORWARD", (-10, 0): "10 cm BACK (toward the arm)",
         (0, 10): "10 cm LEFT", (0, -10): "10 cm RIGHT"}


def grab(index: int) -> np.ndarray:
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=index, fps=30, width=640, height=480))
    cam.connect()
    try:
        t0 = time.time()
        while time.time() - t0 < 1.0:  # let exposure settle
            frame = cam.read()
    finally:
        cam.disconnect()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


FIDUCIAL_POSES = ["fid_far_left", "fid_far_right", "fid_near_right", "fid_near_left"]


def marks_from_poses() -> tuple[list[tuple[float, float]], list[str]]:
    """Fiducial table positions measured by the arm: FK of each saved fid_ pose, in cm relative to
    the recorded grab point (down_highlighter), along the arm's forward (+x) and left (+y) axes."""
    import kin
    from arm import load_poses
    from picker import _apply_config

    _apply_config()
    poses = load_poses()["poses"]
    missing = [n for n in FIDUCIAL_POSES + ["down_highlighter"] if n not in poses]
    if missing:
        raise SystemExit(f"Missing saved poses: {missing}. Record them with record_poses.py --follow.")
    ref = kin.fk(poses["down_highlighter"])[:3, 3]
    marks, labels = [], []
    for n in FIDUCIAL_POSES:
        p = kin.fk(poses[n])[:3, 3]
        d = (p - ref) * 100
        marks.append((round(float(d[0]), 2), round(float(d[1]), 2)))
        labels.append(f"{n}  (forward {d[0]:+.1f} cm, left {d[1]:+.1f} cm; height {d[2]:+.1f} cm vs grab)")
    return marks, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--marks", default=None,
                    help='optional: space-separated "forward,left" cm offsets measured by ruler; '
                         "default = read the fid_* poses the arm recorded")
    args = ap.parse_args()
    if args.marks:
        marks = [tuple(float(v) for v in m.split(",")) for m in args.marks.split()]
        labels = None
    else:
        marks, labels = marks_from_poses()
    if len(marks) < 4:
        raise SystemExit("Need at least 4 marks.")

    img = grab(args.camera)
    cv2.imwrite(str(HERE / "calib_photo.png"), img)
    clicks: list[tuple[int, int]] = []

    def label(i):
        if labels is not None:
            return labels[i]
        f, l = marks[i]
        return NAMES.get((round(f), round(l)), f"forward {f:g} cm, left {l:g} cm")

    def redraw():
        vis = img.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(vis, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        msg = f"click {len(clicks) + 1}/{len(marks)}: {label(len(clicks))}" if len(clicks) < len(marks) else "done - press any key"
        cv2.putText(vis, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("calibration", vis)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(marks):
            clicks.append((x, y))
            print(f"  {len(clicks)}. {label(len(clicks) - 1)} -> pixel ({x}, {y})")
            if len(clicks) < len(marks):
                print(f"  next: {label(len(clicks))}")
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()
            print("  undone")
            redraw()

    print("Click the centre of each tape mark in this order:")
    for i in range(len(marks)):
        print(f"  {i + 1}. {label(i)}")
    cv2.namedWindow("calibration")
    cv2.setMouseCallback("calibration", on_mouse)
    redraw()
    while len(clicks) < len(marks):
        cv2.waitKey(50)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    px = np.array(clicks, dtype=np.float64)
    tb = np.array(marks, dtype=np.float64)
    H, _ = cv2.findHomography(px, tb, 0)
    pred = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(pred - tb, axis=1)
    print("Fit check (cm off for each mark):", np.round(err, 2).tolist())
    CALIB.write_text(json.dumps({"H_pixel_to_cm": H.tolist(), "pixels": clicks, "marks_cm": marks,
                                 "camera": args.camera, "size": [img.shape[1], img.shape[0]]}, indent=2))
    print(f"Saved {CALIB}")
    if err.max() > 1.0:
        print("WARNING: a mark is off by more than 1 cm - check the click order / mark positions and redo.")


if __name__ == "__main__":
    main()
