# reBot Arm × Claude — talk to the robot, it picks up highlighters

Revolute Thanksgivathon 2026 (FabLab Kendall, Nov 20–21). A Seeed **reBot B601-RS** follower arm + **reBot Arm 102 (StarArm) leader**,
driven from a **Mac (Apple Silicon)**. You type or say *"pick up the pink highlighter and put it in the box"*; Claude looks through an
overhead camera, finds the highlighter by colour, plans the grab and the arm does it.

Everything lives in `~/rebot_lerobot/llm_arm/`. The LeRobot code (Seeed's fork) is in `~/rebot_lerobot/lerobot/`.

---

## 1. Quick start (demo)

```bash
~/rebot_lerobot/llm_arm/claude_arm_ui.sh            # control window (real arm)
~/rebot_lerobot/llm_arm/claude_arm_ui.sh --dry-run  # same, arm not connected
```
Before: follower 48 V on, follower folded at home, area clear, camera overhead, CAN adapter + camera plugged in.
The window: one prompt box, 🎤 voice input, every arm action shown as a card, **■ STOP** (or Esc) halts and holds,
**Home**, **Quit** (arm drives itself home before motors switch off). Only ONE program may use the arm/camera at a time.
**Setup** (header button) = calibration inside the window, with the leader arm plugged in (12 V on): *Camera calibration* (4 fiducials → photo → click → check grid → Keep), *Camera: new photo only* (camera bumped, fiducials not moved), *Teach grabs*, *Save a pose*. Nothing is saved until Keep/Save; replaced files are copied to `backups/` first. Leaving a flow (Done/Cancel) drives the arm home. Space = the highlighted button.

Terminal version (asks Enter before every move): `~/rebot_lerobot/llm_arm/claude_arm.sh` (`--dry-run`, `--auto`).

---

## 2. Hardware & software setup (already done on this Mac)

| Item | Detail |
|---|---|
| Python env | conda env **`rebot`** (Miniforge at `/opt/homebrew/Caskroom/miniforge/base`), Python 3.12, LeRobot 0.7.1 (Seeed fork, editable `.[all]`), plugins `lerobot-robot-seeed-b601`, `lerobot-teleoperator-rebot-arm-102`, `anthropic` 2.12. Activate: `eval "$(conda shell.zsh hook)" && conda activate rebot`. All scripts in `llm_arm/` re-launch themselves inside `rebot` (`_env.py`), so `python script.py` works from any tab. |
| ffmpeg | conda-forge 7.1.1 in `rebot` (needed `libintl<1 gettext<1` pins). |
| Follower (CAN) | PCAN-USB clone ("XCAN-USB", VID 0x0C72 PID 0x000C). On macOS it works through **MacCAN PCBUSB v0.13** (`~/.local/lib/libPCBUSB.dylib`, also in `/usr/local/lib`). Port string: **`can0@1000000`** (not `can0`; no `ip link` on Mac). RobStride motors CAN IDs 1–7. |
| Leader (serial) | CH340 USB-serial, `/dev/tty.usbserial-*` (number changes on replug; scripts auto-find it). Needs its **12 V supply** or the servos don't answer. FashionStar servos IDs 0–6. |
| Calibration | LeRobot calibration ids `leader1`, `follower1` (`~/.cache/huggingface/lerobot/calibration/...`). Zero pose = folded "sit-down" pose, gripper closed. |
| Camera | USB "iC700 720P HD", OpenCV index **0**, mounted **overhead looking straight down** (arm reaches toward the image bottom). 640×480 @ 30 fps. **Don't move it** – the camera calibration depends on it. If it stops delivering frames: unplug/replug it. |
| Claude API | Through **MIT Parley**: `ANTHROPIC_BASE_URL=https://parley.api.mit.edu`, model `claude-opus-5-5`. Key stored in macOS Keychain, service `anthropic-api-key` (the launchers read it; never paste it into files/chat). Test: `ANTHROPIC_BASE_URL=https://parley.api.mit.edu ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s anthropic-api-key -w)" python check_api.py` |

Tutorial commands from revolutehack.com work on the Mac with these substitutions: skip `sudo apt …`, `ip link …`, `chmod`, `dmesg`;
use `--robot.port=can0@1000000` and `--teleop.port=/dev/tty.usbserial-XXX`.
Teleop: `lerobot-teleoperate --robot.type=rebot_b601_follower --robot.motor_family=rs --robot.port=can0@1000000 --robot.id=follower1 --teleop.type=rebot_102_leader --teleop.port=/dev/tty.usbserial-XXX --teleop.id=leader1`

---

## 3. How it works

```
camera ──► vision.py (colour masks → boxes → table cm via homography) ──► Claude (agent tools)
                                                                              │
          poses.json + grasp_samples.json ──► picker.py (copy nearest taught grab, IK) ──► arm.py (slow, safe motion)
```

* **Table coordinates**: cm relative to the tip position of the recorded pose `down_highlighter`; `forward` = arm's +x (away from base), `left` = arm's +y. (In the camera image, "forward" is DOWN the image.)
* **Camera calibration** (`camera_calib.json`): 4 fiducials whose table positions were measured BY THE ARM (tip touched to each, forward kinematics) + clicked pixels → homography pixel→cm.
* **Kinematics** (`kin.py`): numpy FK + damped-least-squares IK from Seeed's URDF (`lerobot_robot_seeed_b601/assets/00-arm-rs_asm-v3.urdf`). Public→URDF sign mapping; `wrist_yaw` sign = −1 (verified by a ruler: fiducial spacing 44 × 29 cm). Settings in `kin_config.json`.
* **Picking** (`picker.py`): the geometry model alone missed, so picks **copy the nearest taught grab** (`grasp_samples.json`, recorded with `teach_grasps.py`): same gripper orientation, height and tip-to-object offset, rotated about the base to the new spot; prefers grabs whose highlighter axis matches (6° ≈ 1 cm), turns by the remaining angle if reachable, lowers the hover if needed. 4 good grabs are taught.
* **Claude agent** (`agent.py`, used by `ui_server.py`): Messages API manual tool loop, `claude-opus-5-5`, effort low, server-side refusal fallback (auto-drops unsupported features). Tools: `list_poses`, `move_to_pose`, `set_gripper`, `look`, `find_highlighters`, `pick_at`. System prompt in `agent.py`.
* **Safety** (`arm.py`): all motion interpolated at ≤ 25 °/s; per-command clamp 15° (gripper 360°); 30 Hz hold thread; **every tool drives the arm home (`rest`) before torque-off on quit/Ctrl+C/error** (`park_follower`, Ctrl+C ignored while parking); Stop button → `StopMotion` (holds position).

---

## 4. Files

| File | Purpose |
|---|---|
| `claude_arm_ui.sh` / `ui_server.py` / `ui.html` | Control window (FastAPI on 127.0.0.1:8765, opens Chrome `--app` window). |
| `setup_flow.py` | The window's Setup flows (camera calibration, photo-only, teach grabs, save pose); follower copies the leader via `SlowArm.start_follow`. |
| `claude_arm.sh` / `agent.py` | Terminal agent (approval per move unless `--auto`). |
| `arm.py` | `SlowArm` follower controller, `park_follower`, poses file helpers. |
| `kin.py` | URDF forward/inverse kinematics. |
| `carry.py` | Safe drop: straight up → swing high → straight down into `drop_vertical` → open → home (agent tool `drop_in_box`). `python carry.py` plans without moving. |
| `picker.py` | `plan_pick` / `run_pick` (taught-grab copying; geometry fallback). `python picker.py F L [AXIS]` plans without moving. |
| `vision.py` | Highlighter detection. `python vision.py --live` = live window with boxes. |
| `calibrate.py` | Visual fiducial calibration (`--photo-only`, `--recompute`). |
| `teach_grasps.py` | Visual grab teaching (camera locks position, you grab with the leader, Space). |
| `record_poses.py` | Save named poses with the leader (`--follow` = follower copies live). |
| `run_pose.py` | Play saved poses one step at a time (Enter per step). |
| `check_api.py` | Test the Parley/Claude connection. |
| `cam_snapshot.py`, `cam_fps_test.py` (in `~/rebot_lerobot/`) | Camera checks. |
| `poses.json` | Saved poses: `rest`, `above_highlighter`, `down_highlighter` (reference grab), `above_box`, `fid_*` (calibration); gripper open −130 / closed 0. |
| `camera_calib.json`, `calib_photo.png`, `calib_check.png` | Camera calibration + check image. |
| `grasp_samples.json` | 4 taught grabs (backups: `grasp_samples_backup_5.json`, `grasp_samples_old.json`). |
| `kin_config.json` | `wrist_yaw_sign: -1`, `grasp_depth_cm` (only used when no grabs are taught). |

---

## 5. Re-doing things

* **Camera moved** → window **Setup → Camera: new photo only** (fiducial tape not moved) or **Camera calibration** (tape moved); or the old scripts `python calibrate.py` (`--photo-only`). Re-teach grabs if picks are off.
* **Picks off in one area** → window **Setup → Teach grabs** (or `python teach_grasps.py`), add a grab there (keys: Space, U undo, D done, Q quit). Lay highlighters pointing forward or sideways like the taught ones (≈45° is out of reach).
* **Detection wrong colour/size** → tune `COLOURS` / `MIN_AREA_PX` in `vision.py`, check with `vision.py --live`.
* **Arm not answering** → follower 48 V on? CAN wires seated? Quick read-only check: ping motors via motorbridge (see `CLAUDE.md`). Leader not answering → its 12 V supply.
* **"PCAN initialize failed"** → another program still holds the CAN adapter (close other arm scripts).

## 6. Open ideas
* ~~Drop the pen vertically~~ done: pose `drop_vertical` (recorded with the leader, gripper pointing down over the box); the agent's `drop_in_box` tool (`carry.py`) lifts the pen straight up, swings over high, lowers it in. If the box moves, re-record it (`record_poses.py --follow`, `p drop_vertical`).
* Teach a grab near the middle of the square (currently ~10 cm from any taught grab).
