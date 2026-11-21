"""Teach the arm real grabs, so picks correct themselves from examples.

  python ~/rebot_lerobot/llm_arm/teach_grasps.py

Repeat for ~5 spots spread around the fiducial square (one highlighter on the table at a time):
  A. Arm folded out of the view. Put the highlighter somewhere new. When its red box is steady,
     press SPACE (the camera's position for it is locked in).
  B. Steer with the leader so the follower's OPEN jaws are around the highlighter exactly as you
     want it grabbed (at grabbing height), then press SPACE. Then fold the arm back up.
Keys: SPACE = next, U = undo last saved grab, D = done (needs 3+), Q = quit.
The follower always drives itself home before its motors switch off.
Saves grasp_samples.json (used automatically by pick_at).
"""

import _env  # noqa: F401  (re-launches inside the rebot env if needed)

import json
import time
from pathlib import Path

import cv2

from arm import (ARM_JOINTS, FOLLOWER_ID, FOLLOWER_PORT, LEADER_ID, MAX_RELATIVE_TARGET, find_leader_port,
                 park_follower)
from calibrate import GREEN, RED, YELLOW, banner, big

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "grasp_samples.json"


def load_samples() -> list[dict]:
    return json.loads(SAMPLES.read_text())["samples"] if SAMPLES.exists() else []


def save_samples(samples: list[dict]) -> None:
    SAMPLES.write_text(json.dumps({"samples": samples}, indent=2))


def draw_samples(disp, samples, H_inv) -> None:
    import numpy as np

    for i, s in enumerate(samples):
        d = s["detected"]
        p = cv2.perspectiveTransform(np.array([[[d["forward_cm"], d["left_cm"]]]], np.float64), H_inv).reshape(2)
        x, y = int(p[0] * 1.5), int(p[1] * 1.5)
        cv2.circle(disp, (x, y), 9, GREEN, 2)
        cv2.putText(disp, str(i + 1), (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)


def main() -> None:
    import numpy as np
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.robots.rebot_b601_follower import RebotB601Follower, RebotB601FollowerRobotConfig
    from lerobot.teleoperators.rebot_102_leader import RebotArm102Leader, RebotArm102LeaderTeleopConfig

    import vision
    from record_poses import Mirror

    H = vision.load_calib()
    H_inv = np.linalg.inv(H)
    samples = load_samples()
    if samples:
        print(f"{len(samples)} grab(s) already taught; new ones are added (U removes the last).")

    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480))
    cam.connect()
    leader = RebotArm102Leader(RebotArm102LeaderTeleopConfig(port=find_leader_port(), id=LEADER_ID))
    leader.connect(calibrate=False)
    follower = RebotB601Follower(RebotB601FollowerRobotConfig(
        port=FOLLOWER_PORT, motor_family="rs", id=FOLLOWER_ID, max_relative_target=MAX_RELATIVE_TARGET))
    print("Connecting follower: it now copies the leader. Keep clear of it.")
    follower.connect(calibrate=False)
    mirror = Mirror(leader, follower)
    mirror.thread.start()
    cv2.namedWindow("teach")

    phase, locked, flash, flash_until = "A", None, "", 0.0
    try:
        while True:
            frame = cam.read()
            if phase == "A":
                dets, vis = vision.find(frame, H)
                disp = big(vis)
            else:
                disp = big(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            draw_samples(disp, samples, H_inv)
            n = len(samples)
            if phase == "A":
                top = dets[0] if dets else None
                status = (f"seen: {top['colour']} at fwd {top['forward_cm']:+.1f}, left {top['left_cm']:+.1f} cm"
                          if top else "no highlighter seen in the square")
                lines = [f"GRAB {n + 1}  A: arm folded away, place ONE highlighter somewhere new. SPACE = lock",
                         status, f"{n} taught (green = taught spots).  U = undo  D = done (3+)  Q = quit"]
            else:
                lines = [f"GRAB {n + 1}  B: steer the OPEN jaws around the highlighter as you'd grab it. SPACE = save",
                         f"locked: {locked['colour']} at fwd {locked['forward_cm']:+.1f}, left {locked['left_cm']:+.1f} cm",
                         "Then fold the arm back up.   Q = quit"]
                p = cv2.perspectiveTransform(np.array([[[locked["forward_cm"], locked["left_cm"]]]], np.float64), H_inv)
                x, y = (p.reshape(2) * 1.5).astype(int)
                cv2.drawMarker(disp, (int(x), int(y)), RED, cv2.MARKER_TILTED_CROSS, 22, 2)
            if time.time() < flash_until:
                cv2.putText(disp, flash, (12, disp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2, cv2.LINE_AA)
            cv2.imshow("teach", banner(disp, lines, YELLOW if phase == "A" else GREEN))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("d") and phase == "A":
                if len(samples) >= 3:
                    break
                flash, flash_until = "Teach at least 3 grabs first", time.time() + 1.5
            if k == ord("u") and phase == "A" and samples:
                samples.pop()
                save_samples(samples)
                flash, flash_until = "Removed the last grab", time.time() + 1.5
            if k == ord(" "):
                if phase == "A":
                    if dets:
                        locked = {k2: dets[0][k2] for k2 in ("colour", "forward_cm", "left_cm", "axis_deg", "length_cm")}
                        phase = "B"
                    else:
                        flash, flash_until = "No highlighter seen - check it is inside the square", time.time() + 1.5
                else:
                    cur = mirror.read()
                    samples.append({"detected": locked, "joints": {j: cur[j] for j in ARM_JOINTS},
                                    "time": time.strftime("%H:%M:%S")})
                    save_samples(samples)
                    print(f"taught grab {len(samples)}: {locked['colour']} at ({locked['forward_cm']:+.1f}, "
                          f"{locked['left_cm']:+.1f}) cm")
                    flash, flash_until = f"Saved grab {len(samples)} - fold the arm away", time.time() + 2.0
                    phase, locked = "A", None
    finally:
        cv2.destroyAllWindows()
        mirror.stop.set()
        mirror.thread.join(timeout=1.0)
        park_follower(follower, dict(mirror.cmd))
        try:
            leader.disconnect()
        except Exception:
            pass
        cam.disconnect()
    print(f"{len(samples)} taught grab(s) saved in {SAMPLES}")


if __name__ == "__main__":
    main()
