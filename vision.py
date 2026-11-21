"""Find highlighters in a camera frame by colour and convert them to table positions.

Table coordinates: centimetres relative to the MAIN tape mark (forward = away from the arm, left =
arm's left), via camera_calib.json from calib_camera.py.

Standalone test (arm folded to rest so it doesn't hide the table):
  python ~/rebot_lerobot/llm_arm/vision.py            # prints detections, saves vision_debug.png
"""

from __future__ import annotations

import _env  # noqa: F401  (re-launches inside the rebot env if needed)

import json
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = HERE / "camera_calib.json"

# OpenCV HSV: hue 0-179. Tuned for bright highlighter plastic; adjust after looking at vision_debug.png.
COLOURS = {
    "pink":   [((140, 70, 90), (179, 255, 255)), ((0, 90, 120), (6, 255, 255))],
    "yellow": [((18, 90, 120), (33, 255, 255))],
    "orange": [((7, 120, 120), (17, 255, 255))],
    "green":  [((45, 80, 70), (85, 255, 255))],   # NOTE: the arm's lime parts are green too
    "blue":   [((90, 100, 70), (125, 255, 255))],
}
MIN_AREA_PX = 120
ZONE_CM = {"forward": (-12.0, 9.0), "left": (-12.0, 12.0)}  # search area around the main mark


def load_calib() -> np.ndarray:
    if not CALIB.exists():
        raise FileNotFoundError("camera_calib.json missing - run calib_camera.py first")
    return np.array(json.loads(CALIB.read_text())["H_pixel_to_cm"], dtype=np.float64)


def px_to_cm(H: np.ndarray, pts) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def zone_polygon_px(H: np.ndarray) -> np.ndarray:
    """Search area = the quadrilateral of the 4 clicked fiducials (falls back to ZONE_CM)."""
    cal = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    if len(cal.get("pixels", [])) >= 4:
        return np.array(cal["pixels"], np.float64)
    (f0, f1), (l0, l1) = ZONE_CM["forward"], ZONE_CM["left"]
    corners_cm = np.array([[f0, l0], [f0, l1], [f1, l1], [f1, l0]], np.float64)
    return cv2.perspectiveTransform(corners_cm.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)


def zone_polygon_cm() -> np.ndarray:
    cal = json.loads(CALIB.read_text())
    return np.array(cal["marks_cm"], np.float64)


def inside_zone_cm(forward_cm: float, left_cm: float) -> bool:
    poly = zone_polygon_cm().astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(forward_cm), float(left_cm)), False) >= 0


def zone_mask(H: np.ndarray, shape) -> np.ndarray:
    corners_px = zone_polygon_px(H)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [corners_px.astype(np.int32)], 255)
    # stay ~10 px inside the fiducials so their tape isn't picked up
    m = cv2.erode(m, np.ones((21, 21), np.uint8))
    return m, corners_px


def find(frame_rgb: np.ndarray, H: np.ndarray | None = None, debug_path: Path | None = None) -> tuple[list[dict], np.ndarray]:
    """Returns (detections, annotated BGR image)."""
    H = load_calib() if H is None else H
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    zone, zone_px = zone_mask(H, bgr.shape)
    vis = bgr.copy()
    cv2.polylines(vis, [zone_px.astype(np.int32)], True, (255, 255, 255), 1)
    kernel = np.ones((5, 5), np.uint8)
    found = []
    for name, ranges in COLOURS.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask &= zone
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < MIN_AREA_PX:
                continue
            (cx, cy), (w, h), ang = cv2.minAreaRect(c)
            # long-axis end points in pixels -> table cm, to get length and direction on the table
            a = math.radians(ang if w >= h else ang + 90)
            half = max(w, h) / 2
            ends_px = [(cx - half * math.cos(a), cy - half * math.sin(a)), (cx + half * math.cos(a), cy + half * math.sin(a))]
            centre_cm = px_to_cm(H, [(cx, cy)])[0]
            ends_cm = px_to_cm(H, ends_px)
            d = ends_cm[1] - ends_cm[0]
            length_cm = float(np.hypot(*d))
            # direction of the long axis on the table, 0 deg = pointing forward (away from arm), +90 = left
            axis_deg = (math.degrees(math.atan2(d[1], d[0])) + 180) % 180
            if axis_deg > 90:
                axis_deg -= 180
            det = {"colour": name, "forward_cm": round(float(centre_cm[0]), 1), "left_cm": round(float(centre_cm[1]), 1),
                   "length_cm": round(length_cm, 1), "axis_deg": round(axis_deg, 0), "pixel": [int(cx), int(cy)],
                   "area_px": int(cv2.contourArea(c))}
            found.append(det)
            box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)
            cv2.putText(vis, f"{name} f{det['forward_cm']:+.0f} l{det['left_cm']:+.0f}", (int(cx) + 6, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    # mark the main tape point for reference
    o = cv2.perspectiveTransform(np.array([[[0.0, 0.0]]]), np.linalg.inv(H)).reshape(2)
    cv2.drawMarker(vis, (int(o[0]), int(o[1])), (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
    found.sort(key=lambda d: -d["area_px"])
    if debug_path is not None:
        cv2.imwrite(str(debug_path), vis)
    return found, vis


if __name__ == "__main__":
    import sys
    import time

    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480))
    cam.connect()
    try:
        if "--live" in sys.argv:
            # Live window: boxes + table positions update continuously. S = save snapshot, Q = quit.
            H = load_calib()
            print("Live detection - click the window, S = save vision_debug.png, Q = quit")
            while True:
                frame = cam.read()
                dets, vis = find(frame, H)
                disp = cv2.resize(vis, None, fx=1.5, fy=1.5)
                txt = "  |  ".join(f"{d['colour']} fwd {d['forward_cm']:+.0f} left {d['left_cm']:+.0f}" for d in dets[:3])
                cv2.rectangle(disp, (0, 0), (disp.shape[1], 34), (0, 0, 0), -1)
                cv2.putText(disp, f"{len(dets)} found: {txt}" if dets else "no highlighters found in the square",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("highlighters", disp)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("s"):
                    cv2.imwrite(str(HERE / "vision_debug.png"), vis)
                    print("saved vision_debug.png:", dets)
            cv2.destroyAllWindows()
            raise SystemExit
        t0 = time.time()
        while time.time() - t0 < 1.0:
            frame = cam.read()
    finally:
        cam.disconnect()
    dets, _ = find(frame, debug_path=HERE / "vision_debug.png")
    for d in dets:
        print(d)
    print(f"{len(dets)} object(s). Annotated image: {HERE / 'vision_debug.png'}")
