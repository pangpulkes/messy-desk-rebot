# Instructions for Claude Code — reBot arm project

Read `TECHNICAL.md` in this folder first (README.md is the public project page): it has the hardware, setup, architecture and file map. This file holds the
working rules and the current state.

## User & working style
- The user is building a hackathon demo (Revolute Thanksgivathon, Nov 21 2026, demos ~4 PM). Explain in plain language, short steps,
  give runnable commands in ```bash blocks. They often run commands from new terminal tabs — scripts here self-switch into
  the `rebot` conda env via `_env.py`; keep that pattern (import `_env` first) in any new script.
- **Ask before installing anything** (pip/conda/brew/system libraries).
- **Never handle the API key**: it lives in the macOS Keychain (service `anthropic-api-key`); launchers read it. Don't read or print
  it, don't put it in files. Endpoint: MIT Parley `ANTHROPIC_BASE_URL=https://parley.api.mit.edu`, model `claude-opus-5-5`.

## Hardware safety rules (important)
- The follower is a 48 V arm. **Any program that powers the follower must drive it home (`rest`) slowly before torque-off** on
  quit, Ctrl+C or error — never just disconnect (the arm drops onto the table). Use `arm.park_follower` / `SlowArm.close()`.
- Keep motion slow (≤ 25 °/s via `SlowArm`), keep the `max_relative_target` clamp, keep a visible Stop in UIs.
- Don't run motor-moving commands yourself; give the user the command to run. Read-only checks (ping/position reads) are OK:
  ```python
  from motorbridge import Controller   # follower read-only ping
  c = Controller(channel="can0@1000000"); m = c.add_robstride_motor(1, 0xFD, "rs-06"); m.robstride_ping(); c.close()
  ```
  Leader read-only: `motorbridge_smart_servo.FashionStarServo(port, 1_000_000).sync_monitor(list(range(7)))`.
- Only one program may hold the CAN adapter / camera at a time. Check with `pgrep -fl llm_arm` before telling the user to start something.
- Your own shell has NO camera permission on this Mac; camera tests must be run by the user in their terminal
  (`python ~/rebot_lerobot/cam_snapshot.py` writes `~/rebot_lerobot/cam_snapshot.png` you can then view).
- Never edit `grasp_samples.json` / `poses.json` while `teach_grasps.py` / `record_poses.py` is running (they rewrite the file). Back up before removing entries.

## Current state (end of 2026-11-21 afternoon)
- Working: teleop, calibration (leader1/follower1), camera calibration, colour detection, taught-grab picking (4 grabs) — the user
  confirmed picks "work really good"; terminal agent `claude_arm.sh`.
- Control window `claude_arm_ui.sh` (`ui_server.py` + `ui.html`): Claude-desktop-style prompt, voice input (Chrome
  webkitSpeechRecognition), no per-move approvals, action cards, Stop/Home/Quit. ~14:45 first REAL run worked end to end
  (find → pick green → box → open → rest; Ctrl+C parked home). Stop button mid-motion not yet tested on the real arm.
- Setup inside the window (~15:30, `setup_flow.py`, header button **Setup**): camera calibration (full / photo only), teach grabs,
  save pose. App keeps owning the follower; `SlowArm.start_follow(leader)` copies the leader (≤60 °/s, stops on Stop/close).
  Writes only on Keep/Save, backups in `backups/`. Tested offline, then the user confirmed it works on the real arm (~15:40).
- Vertical drop: `drop_vertical` pose recorded ~14:55 (user confirmed: over the box, gripper pointing down). The direct joint
  move pick→drop_vertical dragged the pen on the table, so ~15:15 `carry.py` was added and exposed as the agent tool
  `drop_in_box` (drop_vertical is hidden from move_to_pose): lift STRAIGHT up to 26 cm (pitch joints only, gripper angle kept;
  once wrist_flex hits its limit it holds and the nose tilts UP ≤27°), swing at 26 cm, straight down into drop_vertical, open,
  straight up, rest. `python carry.py` prints the plan (no motion). Offline-verified (monotone heights); user confirmed on the real arm ~15:45 (pen clears the table).
  NOTE: kin.fk's gripper DIRECTION is unreliable at big wrist_yaw/roll (says drop_vertical points 74° up); near the pick spot it
  matches reality. wrist_flex sign −1 re-confirmed (fiducial tips 4–8 cm vs 45–65 cm with +1). Box may have moved: drop_vertical
  is ~40 cm from `above_box`.
- Known quirks: USB camera occasionally stops sending frames after a crash → unplug/replug fixes it. Lime-green arm parts are close to
  the yellow/green colour ranges in `vision.py`. Highlighters at ~45° can't be aligned (gripper turns only ~±20–30°).

## Useful checks (safe, no motion)
```bash
cd ~/rebot_lerobot/llm_arm
python picker.py 4 -20 0          # plan a pick at forward 4, left -20 cm, axis 0° (prints which taught grab it copies)
python -c "import json;print(json.load(open('grasp_samples.json'))['samples'][0]['detected'])"
pgrep -fl llm_arm                 # what is holding the arm / camera
```
