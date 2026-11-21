"""Desktop-style control window for the reBot arm: one prompt box, voice input, live action feed.

Started by claude_arm_ui.sh (which sets the Parley address + Keychain key). Options:
  --dry-run      plan only, the arm is not connected (camera still used unless --no-camera)
  --no-camera    no camera tools
  --no-window    don't open the app window automatically (open http://127.0.0.1:8765 yourself)

No per-move approvals: every action is shown in the window as it starts and finishes.
STOP halts the arm immediately and holds it. Quitting (button, Ctrl+C) drives the arm home first.
"""

import _env  # noqa: F401  (re-launches inside the rebot env if needed)

import argparse
import base64
import subprocess
import threading
import time
from pathlib import Path

import anthropic
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response

from agent import MODEL, SYSTEM, Executor, build_tools
from arm import DEFAULT_SPEED_DEG_S, StopMotion, load_poses

HERE = Path(__file__).resolve().parent
PORT = 8765

FRIENDLY_POSES = {"rest": "return home", "above_box": "move over the box",
                  "above_highlighter": "move over the recorded spot", "down_highlighter": "lower to the recorded spot"}


def describe(name: str, inp: dict) -> str:
    if name == "move_to_pose":
        return FRIENDLY_POSES.get(inp["name"], f"move to '{inp['name']}'").capitalize()
    if name == "set_gripper":
        return "Open the gripper" if inp["state"] == "open" else "Close the gripper"
    if name == "look":
        return "Take a photo"
    if name == "find_highlighters":
        return "Look for highlighters"
    if name == "pick_at":
        return f"Pick up the object at {inp['forward_cm']:+.1f} cm forward, {inp['left_cm']:+.1f} cm left"
    if name == "drop_in_box":
        return "Carry it over the box and drop it"
    if name == "list_poses":
        return "Check the saved positions"
    return name


class Brain:
    """Holds the arm, the conversation with Claude and the event feed shown in the window."""

    def __init__(self, args):
        self.args = args
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.busy = False
        self.messages: list = []
        self.client = anthropic.Anthropic()
        poses = load_poses()
        camera_index = None if args.no_camera else 0
        self.tools = build_tools(poses, have_camera=camera_index is not None)
        self.ex = Executor(args.dry_run, auto=True, camera_index=camera_index, speed=args.speed)
        self.tier = 0
        from setup_flow import Setup

        self.setup = Setup(self)
        self.emit("status", text="Ready" + (" (dry run - the arm will not move)" if args.dry_run else ""))

    def reload_tools(self) -> None:
        """Re-read poses / calibration (after a setup flow saved something new)."""
        self.tools = build_tools(load_poses(), have_camera=not self.args.no_camera)

    # ---------------------------------------------------------------- feed
    def emit(self, kind: str, **data) -> int:
        with self.lock:
            ev = {"id": len(self.events) + 1, "kind": kind, "t": time.time(), **data}
            self.events.append(ev)
            return ev["id"]

    # ---------------------------------------------------------------- Claude
    def _call(self):
        tiers = [
            lambda **kw: self.client.beta.messages.create(**kw, output_config={"effort": "low"},
                                                          betas=["server-side-fallback-2026-07-01"], fallbacks="default"),
            lambda **kw: self.client.messages.create(**kw, output_config={"effort": "low"}),
            lambda **kw: self.client.messages.create(**kw),
        ]
        while True:
            try:
                return tiers[self.tier](model=self.args.model, max_tokens=16000, system=SYSTEM,
                                        tools=self.tools, messages=self.messages)
            except anthropic.BadRequestError:
                if self.tier + 1 < len(tiers):
                    self.tier += 1
                    continue
                raise

    def run_command(self, text: str) -> None:
        arm = self.ex.arm
        if arm is not None:
            arm.stop_requested.clear()
        self.messages.append({"role": "user", "content": text})
        self.emit("user", text=text)
        try:
            while True:
                if arm is not None and arm.stop_requested.is_set():
                    self.emit("stopped", text="Stopped. The arm is holding its position.")
                    break
                self.emit("thinking")
                try:
                    response = self._call()
                except anthropic.AuthenticationError:
                    self.messages.pop()
                    self.emit("error", text="The API key was rejected (check the Keychain key / Parley).")
                    break
                except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                    self.messages.pop()
                    self.emit("error", text=f"Couldn't reach Claude: {e}")
                    break
                self.messages.append({"role": "assistant", "content": response.content})  # append-only
                for b in response.content:
                    if b.type == "text" and b.text.strip():
                        self.emit("claude", text=b.text.strip())
                if response.stop_reason == "pause_turn":
                    continue
                if response.stop_reason != "tool_use":
                    if response.stop_reason == "refusal":
                        self.emit("error", text="Claude declined this request.")
                    break
                results, stopped = [], False
                for b in response.content:
                    if b.type != "tool_use":
                        continue
                    if stopped:
                        results.append({"type": "tool_result", "tool_use_id": b.id,
                                        "content": "Not run: the user pressed Stop.", "is_error": True})
                        continue
                    aid = self.emit("action", tool=b.name, title=describe(b.name, b.input), state="running")
                    try:
                        content, is_err = self.ex.run(b.name, b.input)
                    except StopMotion as e:
                        content, is_err, stopped = f"Stopped by the user mid-motion: {e}", True, True
                    except Exception as e:
                        content, is_err = f"Error: {e}", True
                    image, summary = None, content
                    if isinstance(content, list):
                        image = next((c["source"]["data"] for c in content if c.get("type") == "image"), None)
                        summary = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
                    self.emit("action_done", action=aid, ok=not is_err, detail=str(summary)[:400], image=image,
                              state="stopped" if stopped else ("error" if is_err else "done"))
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": content, "is_error": is_err})
                self.messages.append({"role": "user", "content": results})
                if stopped:
                    self.emit("stopped", text="Stopped. The arm is holding its position.")
                    break
        finally:
            self.busy = False
            self.emit("idle")

    # ---------------------------------------------------------------- buttons
    def start(self, text: str) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        threading.Thread(target=self.run_command, args=(text,), daemon=True).start()
        return True

    def stop(self) -> None:
        if self.ex.arm is not None:
            self.ex.arm.stop_requested.set()
        self.emit("status", text="Stop pressed")

    def home(self) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.busy = True

        def go():
            aid = self.emit("action", tool="move_to_pose", title="Return home", state="running")
            try:
                if self.ex.arm is not None:
                    self.ex.arm.stop_requested.clear()
                    self.ex.arm.move_to_pose("rest")
                self.emit("action_done", action=aid, ok=True, detail="", state="done")
            except StopMotion as e:
                self.emit("action_done", action=aid, ok=False, detail=str(e), state="stopped")
            except Exception as e:
                self.emit("action_done", action=aid, ok=False, detail=str(e), state="error")
            finally:
                self.busy = False
                self.emit("idle")

        threading.Thread(target=go, daemon=True).start()
        return True

    def frame_jpeg(self) -> bytes | None:
        import cv2

        try:
            frame = self.ex._frame()
        except Exception:
            return None
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None


def make_app(brain: Brain, server_ref: dict) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (HERE / "ui.html").read_text()

    @app.get("/api/events")
    def events(after: int = 0):
        with brain.lock:
            evs = [e for e in brain.events if e["id"] > after]
        return {"events": evs, "busy": brain.busy, "dry_run": brain.args.dry_run}

    @app.post("/api/command")
    def command(body: dict):
        text = str(body.get("text", "")).strip()
        if not text:
            return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
        return {"ok": brain.start(text)}

    @app.post("/api/stop")
    def stop():
        brain.stop()
        return {"ok": True}

    @app.post("/api/home")
    def home():
        return {"ok": brain.home()}

    @app.post("/api/new")
    def new_chat():
        if brain.busy:
            return {"ok": False}
        brain.messages = []
        brain.emit("cleared")
        return {"ok": True}

    @app.post("/api/quit")
    def quit_():
        brain.stop()
        brain.emit("status", text="Quitting: the arm is driving itself home, then switching off.")
        server_ref["server"].should_exit = True
        return {"ok": True}

    @app.get("/api/setup")
    def setup_state():
        return brain.setup.state()

    @app.post("/api/setup/start")
    def setup_start(body: dict):
        ok, err = brain.setup.start(str(body.get("mode", "")))
        return {"ok": ok, "error": err}

    @app.post("/api/setup/action")
    def setup_action(body: dict):
        ok, msg = brain.setup.action(str(body.get("action", "")), body)
        return {"ok": ok, "error": "" if ok else msg}

    @app.get("/api/setup/view.jpg")
    def setup_view():
        jpg = brain.setup.view_jpeg()
        return Response(jpg, media_type="image/jpeg") if jpg else Response(status_code=204)

    @app.get("/api/camera.jpg")
    def camera():
        jpg = brain.frame_jpeg()
        return Response(jpg, media_type="image/jpeg") if jpg else Response(status_code=204)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED_DEG_S)
    args = ap.parse_args()

    brain = Brain(args)  # connects the arm (motors on, holding) unless --dry-run
    server_ref: dict = {}
    server = uvicorn.Server(uvicorn.Config(make_app(brain, server_ref), host="127.0.0.1", port=PORT, log_level="warning"))
    server_ref["server"] = server
    url = f"http://127.0.0.1:{PORT}"
    if not args.no_window:
        def open_window():
            time.sleep(1.5)
            subprocess.Popen(["open", "-na", "Google Chrome", "--args", f"--app={url}", "--window-size=1100,820"])
        threading.Thread(target=open_window, daemon=True).start()
    print(f"Arm control window: {url}   (Ctrl+C here or the Quit button to stop; the arm goes home first)")
    try:
        server.run()
    except KeyboardInterrupt:
        pass  # uvicorn re-raises Ctrl+C after its shutdown; the park below still runs
    finally:
        # Keep Stop set: a Claude turn still running in its worker thread must not move the arm while it parks.
        try:
            brain.setup.close()  # stop copying the leader, release it
        except Exception as e:
            print(f"(setup cleanup: {e})")
        if brain.ex.arm is not None:
            brain.ex.arm.stop_requested.set()
        brain.ex.close()  # drives the arm home, then motors off
    print("Bye.")


if __name__ == "__main__":
    main()
