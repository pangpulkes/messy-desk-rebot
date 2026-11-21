"""Talk to the reBot arm: Claude plans with saved poses, the arm executes slowly.

Usage (in the `rebot` env, with ANTHROPIC_API_KEY exported in that terminal):
  python ~/rebot_lerobot/llm_arm/agent.py --dry-run        # Claude plans, the arm is NOT connected
  python ~/rebot_lerobot/llm_arm/agent.py                  # real arm; you approve every motion
  python ~/rebot_lerobot/llm_arm/agent.py --auto           # real arm; no per-motion approval
  add --no-camera to run without the 'look' tool

Then type instructions such as:  move the highlighter to the box
Type 'q' (or Ctrl+C) to quit; the arm returns to 'rest' and switches off.
"""

from __future__ import annotations

import _env  # noqa: F401  (re-launches inside the rebot env if needed)

import argparse
import base64

import anthropic

from arm import DEFAULT_SPEED_DEG_S, SlowArm, load_poses

MODEL = "claude-opus-5-5"

SYSTEM = """You control a 6-axis robot arm with a parallel gripper (Seeed reBot B601) on a table.
You cannot command arbitrary coordinates. You can only move the arm to SAVED POSES (recorded by
the team by hand) and open or close the gripper. Pose names describe where they are,
e.g. "above_highlighter" (hovering over the highlighter's spot), "down_highlighter" (gripper
around it), "above_box", "rest" (folded, safe).

How to work:
- Plan a short sequence of tool calls, one motion at a time; each motion finishes before the next.
- To pick something up: go to the "above_..." pose, open the gripper, go to the "down_..." pose,
  close the gripper, return to the "above_..." pose. Always lift before moving sideways.
- To place: go above the target, go down if a "down_..." pose exists, open the gripper, lift.
- Only use pose names that exist (call list_poses if unsure). Never guess a pose.
- If a 'look' tool is available, use it to check the scene when it helps (for example, whether
  the highlighter is there before grabbing, or whether the pick worked).
- If the request cannot be done with the saved poses, say so and say which pose is missing.
- If a motion is declined by the human, stop and ask what to do instead.
- End at "rest" when the task is complete, if that pose exists.
- Keep replies to the human short: one or two sentences.

Picking highlighters that can be ANYWHERE in the work area (when find_highlighters / pick_at exist):
- The camera looks straight down at a calibrated square on the table (4 fiducial marks).
- Positions are in cm on the table: forward_cm (away from the arm's base) and left_cm (the arm's
  left), measured from the spot where the team recorded the original grab.
- Workflow: go to "rest" first (the folded arm can hide part of the table), call find_highlighters,
  choose the one the human asked for (by colour), then call pick_at with ITS forward_cm, left_cm and axis_deg
  exactly as reported. pick_at does the whole grab (hover, open, descend, close, lift) by itself.
- After a pick, if the drop_in_box tool exists, call it right away (do not move to "rest" or any other
  pose while holding the pen): it lifts straight up, carries the pen over the box, lowers it upright,
  drops it and returns to "rest" by itself. Without drop_in_box: move_to_pose "above_box", set_gripper
  "open", then "rest". You may call find_highlighters again to confirm it is gone from its old spot.
- If pick_at reports the spot is out of reach, say so; do not try saved poses as a substitute.
- If several highlighters of the requested colour exist, ask which one (or pick the closest to 0,0)."""


def build_tools(poses: dict, have_camera: bool) -> list[dict]:
    from carry import DROP_POSE

    have_drop = DROP_POSE in poses["poses"]
    # calibration poses are not destinations; the drop pose is only reached through drop_in_box (safe path)
    names = sorted(n for n in poses["poses"] if not n.startswith("fid_") and not (have_drop and n == DROP_POSE))
    grip_states = sorted(poses["gripper"]) or ["open", "closed"]
    tools = [
        {
            "name": "list_poses",
            "description": "List the saved arm poses and recorded gripper states.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False, "required": []},
            "strict": True,
        },
        {
            "name": "move_to_pose",
            "description": "Move the arm slowly to a saved pose. The gripper does not change. "
            "Returns the arm's joint angles after the move.",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": names or ["rest"], "description": "Saved pose name"}},
                "additionalProperties": False,
                "required": ["name"],
            },
            "strict": True,
        },
        {
            "name": "set_gripper",
            "description": "Open or close the gripper (the arm does not move). Closing is force-limited, "
            "so it is safe to close on an object.",
            "input_schema": {
                "type": "object",
                "properties": {"state": {"type": "string", "enum": grip_states}},
                "additionalProperties": False,
                "required": ["state"],
            },
            "strict": True,
        },
    ]
    if have_drop:
        tools.append(
            {
                "name": "drop_in_box",
                "description": "Carry the highlighter the gripper is holding into the box and drop it upright: lifts "
                "straight up, swings over the box high above the table, lowers, opens the gripper, lifts out and "
                "returns to 'rest'. Call it right after pick_at.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False, "required": []},
                "strict": True,
            }
        )
    if have_camera:
        tools.append(
            {
                "name": "look",
                "description": "Take a photo with the fixed overhead camera looking down at the table.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False, "required": []},
                "strict": True,
            }
        )
        from vision import CALIB

        if CALIB.exists():
            tools.append(
                {
                    "name": "find_highlighters",
                    "description": "Detect highlighters by colour in the calibrated work area. Returns each one's "
                    "colour, table position (forward_cm, left_cm), length and long-axis angle, plus an annotated "
                    "photo. Call it with the arm at 'rest' so the arm does not hide the table.",
                    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False, "required": []},
                    "strict": True,
                }
            )
            tools.append(
                {
                    "name": "pick_at",
                    "description": "Pick up the object at a table position reported by find_highlighters (pass its forward_cm, "
                    "left_cm and axis_deg): turn the gripper to line up with it if reachable, hover, open, descend, close, lift. Ends hovering over the original recorded spot, holding it. "
                    "Fails without moving if the position is outside the work area or out of reach.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "forward_cm": {"type": "number"},
                            "left_cm": {"type": "number"},
                            "axis_deg": {"type": ["number", "null"],
                                         "description": "the object's axis_deg from find_highlighters (null if unknown)"},
                        },
                        "additionalProperties": False,
                        "required": ["forward_cm", "left_cm", "axis_deg"],
                    },
                    "strict": True,
                }
            )
    return tools


class Executor:
    def __init__(self, dry_run: bool, auto: bool, camera_index: int | None, speed: float):
        self.dry_run = dry_run
        self.auto = auto
        self.arm: SlowArm | None = None
        self.cam = None
        if dry_run:
            if camera_index is not None:
                from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

                self.cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=camera_index, fps=30, width=640, height=480))
                self.cam.connect()
        else:
            self.arm = SlowArm(speed_deg_s=speed, camera_index=camera_index)
            print("Connecting follower (motors switch on and hold the current position)...")
            self.arm.connect()

    def close(self) -> None:
        if self.arm is not None:
            self.arm.close()
        if self.cam is not None:
            self.cam.disconnect()

    def _approve(self, what: str) -> bool:
        if self.dry_run:
            print(f"  [dry-run] would {what}")
            return True
        if self.auto:
            print(f"  -> {what}")
            return True
        ans = input(f"  Claude wants to {what}. Enter = do it, n = decline: ").strip().lower()
        return ans != "n"

    def run(self, name: str, inp: dict):
        """Returns (content, is_error)."""
        if name == "list_poses":
            from carry import DROP_POSE

            data = load_poses()
            names = sorted(n for n in data["poses"] if not n.startswith("fid_") and n != DROP_POSE)
            extra = " (to drop into the box use the drop_in_box tool)" if DROP_POSE in data["poses"] else ""
            return f"poses: {names}; gripper states: {sorted(data['gripper'])}{extra}", False
        if name == "move_to_pose":
            from carry import DROP_POSE

            if inp["name"] == DROP_POSE or inp["name"].startswith("fid_"):
                return (f"'{inp['name']}' cannot be reached directly (the direct path drags the pen on the table). "
                        "Use the drop_in_box tool to drop into the box. The arm did not move."), True
            if not self._approve(f"move to pose '{inp['name']}'"):
                return "The human declined this motion. The arm did not move.", False
            if self.dry_run:
                return f"(dry run) arm is now at '{inp['name']}'", False
            joints = self.arm.move_to_pose(inp["name"])
            return f"Reached '{inp['name']}'. Joints (deg): {_fmt(joints)}", False
        if name == "set_gripper":
            if not self._approve(f"set the gripper to '{inp['state']}'"):
                return "The human declined this motion. The gripper did not move.", False
            if self.dry_run:
                return f"(dry run) gripper is now '{inp['state']}'", False
            joints = self.arm.set_gripper(inp["state"])
            return f"Gripper '{inp['state']}'. Gripper angle: {joints['gripper']:.1f} deg", False
        if name == "look":
            frame = self._frame()
            if frame is None:
                return "No camera available.", True
            return [_jpeg_block(frame, rgb=True), {"type": "text", "text": "Current view from the overhead camera."}], False
        if name == "find_highlighters":
            import json as _json

            import vision

            frame = self._frame()
            if frame is None:
                return "No camera available.", True
            dets, vis = vision.find(frame, debug_path=vision.HERE / "vision_last.png")
            for d in dets:
                d.pop("pixel", None)
                d.pop("area_px", None)
            print(f"  [camera] {len(dets)} found: " + ", ".join(
                f"{d['colour']} ({d['forward_cm']:+.1f}, {d['left_cm']:+.1f})" for d in dets))
            return [_jpeg_block(vis, rgb=False),
                    {"type": "text", "text": "Detections (cm from the recorded grab spot): " + _json.dumps(dets)}], False
        if name == "pick_at":
            import kin
            import picker

            f, l = float(inp["forward_cm"]), float(inp["left_cm"])
            axis = inp.get("axis_deg")
            axis = None if axis is None else float(axis)
            try:
                plan = picker.plan_pick(f, l, axis)  # check reach before asking/moving
            except (ValueError, kin.IKError) as e:
                return f"Cannot pick there: {e}. The arm did not move.", False
            if not self._approve(f"pick up the object at forward {f:+.1f} cm, left {l:+.1f} cm ({plan['note']})"):
                return "The human declined this motion. The arm did not move.", False
            if self.dry_run:
                return f"(dry run) reachable; {plan['note']}; would pick at ({f:+.1f}, {l:+.1f}) and hold it above the recorded spot", False
            picker.run_pick(self.arm, f, l, axis)
            return f"Picked at ({f:+.1f}, {l:+.1f}) cm ({plan['note']}); now holding it above the recorded spot.", False
        if name == "drop_in_box":
            import carry
            import kin

            if not self._approve("carry the held object into the box and drop it"):
                return "The human declined this motion. The arm did not move.", False
            if self.dry_run:
                try:
                    p = carry.plan_drop(load_poses()["poses"]["above_highlighter"])
                except (KeyError, kin.IKError) as e:
                    return f"(dry run) cannot plan the drop: {e}", True
                return f"(dry run) would lift to {p['heights']['travel']} cm, swing over the box, drop, go to 'rest'", False
            try:
                carry.run_drop(self.arm)
            except (KeyError, kin.IKError) as e:
                return f"Cannot do the safe drop from here: {e}. The arm did not move.", True
            return "Dropped it into the box; the arm is back at 'rest'.", False
        return f"Unknown tool {name}", True

    def _frame(self):
        if self.arm is not None:
            return self.arm.camera_frame()
        return self.cam.read() if self.cam else None


def _jpeg_block(img, rgb: bool) -> dict:
    import cv2

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if rgb else img
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("could not encode the camera image")
    data = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def _fmt(joints: dict) -> str:
    return ", ".join(f"{k}={v:.1f}" for k, v in joints.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="plan only; do not connect the arm")
    ap.add_argument("--auto", action="store_true", help="do not ask before each motion")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED_DEG_S, help="max joint speed, deg/s")
    ap.add_argument("--model", default=MODEL, help="model ID (use the name your gateway, e.g. Parley, lists)")
    args = ap.parse_args()

    poses = load_poses()
    if not poses["poses"]:
        raise SystemExit("No saved poses yet. Run record_poses.py first.")
    camera_index = None if args.no_camera else args.camera

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL (e.g. Parley)
    tools = build_tools(poses, have_camera=camera_index is not None)

    # Request "tiers": full features first; if a gateway rejects an optional feature with a 400,
    # drop to the next tier and remember that for the rest of the session.
    tiers = [
        ("beta+fallbacks", lambda **kw: client.beta.messages.create(
            **kw, output_config={"effort": "low"}, betas=["server-side-fallback-2026-07-01"], fallbacks="default")),
        ("effort", lambda **kw: client.messages.create(**kw, output_config={"effort": "low"})),
        ("plain", lambda **kw: client.messages.create(**kw)),
    ]
    tier = 0

    def call_claude():
        nonlocal tier
        while True:
            name, fn = tiers[tier]
            try:
                return fn(model=args.model, max_tokens=16000, system=SYSTEM, tools=tools, messages=messages)
            except anthropic.BadRequestError as e:
                if tier + 1 < len(tiers):
                    print(f"  (gateway rejected '{name}' request features: {str(e)[:120]}; retrying simpler)")
                    tier += 1
                    continue
                raise

    ex = Executor(args.dry_run, args.auto, camera_index, args.speed)
    messages: list = []
    base = client.base_url
    print(f"Ready. API: {base}  model: {args.model}. Poses: {sorted(n for n in poses['poses'] if not n.startswith('fid_'))}. Type an instruction (q to quit).")

    try:
        while True:
            text = input("\nyou> ").strip()
            if text.lower() in ("q", "quit", "exit"):
                break
            if not text:
                continue
            messages.append({"role": "user", "content": text})

            while True:
                try:
                    response = call_claude()
                except anthropic.AuthenticationError:
                    raise SystemExit("API key missing or invalid for this endpoint (check ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL).")
                except anthropic.NotFoundError as e:
                    raise SystemExit(f"Model or endpoint not found ({e}). Run check_api.py to list the model names, "
                                     "then pass one with --model.")
                except anthropic.RateLimitError:
                    print("Rate limited by the API; wait a moment and try again.")
                    messages.pop()  # drop the unanswered user turn
                    break
                except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                    print(f"API error: {e}")
                    messages.pop()
                    break

                messages.append({"role": "assistant", "content": response.content})  # append-only history
                for block in response.content:
                    if block.type == "text" and block.text.strip():
                        print(f"claude> {block.text.strip()}")

                if response.stop_reason == "refusal":
                    print("claude> (declined this request)")
                    break
                if response.stop_reason == "max_tokens":
                    print("claude> (response cut off)")
                    break
                if response.stop_reason == "pause_turn":
                    continue
                if response.stop_reason != "tool_use":
                    break

                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    try:
                        content, is_err = ex.run(block.name, block.input)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:  # report the failure to Claude instead of crashing
                        content, is_err = f"Error: {e}", True
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_err}
                    )
                messages.append({"role": "user", "content": results})
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        ex.close()


if __name__ == "__main__":
    main()
