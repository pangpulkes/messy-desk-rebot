"""Move the FOLLOWER slowly through saved poses, one step at a time, with a confirmation per step.
Use this to test every pose by hand before letting Claude drive.

Usage (in the `rebot` env):
  python ~/rebot_lerobot/llm_arm/run_pose.py rest above_highlighter grip:open down_highlighter grip:closed rest
  python ~/rebot_lerobot/llm_arm/run_pose.py --speed 15 rest

Each argument is a saved pose name, or grip:open / grip:closed.
Press Enter to run each step, 's' to skip it, 'q' to stop. Ctrl+C stops too.
On exit the arm returns to 'rest' (if saved) and the motors switch off.
"""

import _env  # noqa: F401  (re-launches inside the rebot env if needed)

import argparse

from arm import DEFAULT_SPEED_DEG_S, SlowArm, load_poses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED_DEG_S, help="max joint speed, deg/s")
    args = ap.parse_args()

    data = load_poses()
    for s in args.steps:
        if s.startswith("grip:"):
            if s[5:] not in data["gripper"]:
                raise SystemExit(f"Gripper value '{s[5:]}' not recorded yet (use go / gc in record_poses.py).")
        elif s not in data["poses"]:
            raise SystemExit(f"Unknown pose '{s}'. Saved: {sorted(data['poses'])}")

    arm = SlowArm(speed_deg_s=args.speed)
    print("Connecting follower (motors switch on and hold the current position)...")
    arm.connect()
    try:
        for s in args.steps:
            ans = input(f"Next: {s}  [Enter = go, s = skip, q = quit] ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue
            if s.startswith("grip:"):
                arm.set_gripper(s[5:])
            else:
                arm.move_to_pose(s)
            print("  done:", {k: round(v, 1) for k, v in arm.present().items()})
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
