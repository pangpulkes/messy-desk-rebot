"""Save named poses by posing the LEADER arm by hand.

Usage (in the `rebot` env):
  python ~/rebot_lerobot/llm_arm/record_poses.py            # leader only; the follower stays off
  python ~/rebot_lerobot/llm_arm/record_poses.py --follow   # follower copies the leader live (like teleop),
                                                            # so you can see where the real gripper is

Commands at the prompt:
  p <name>   save the current arm pose as <name>   (e.g. p rest, p above_highlighter)
  go         save the current gripper value as "open"
  gc         save the current gripper value as "closed"
  show       print the live joint values
  list       list saved poses
  del <name> delete a saved pose
  q          quit (with --follow the follower drives itself home first, then switches off)
Saved to ~/rebot_lerobot/llm_arm/poses.json
"""

import _env  # noqa: F401  (re-launches inside the rebot env if needed)

import argparse
import threading
import time

from arm import (
    ALL_JOINTS,
    ARM_JOINTS,
    FOLLOWER_ID,
    FOLLOWER_PORT,
    GRIPPER,
    LEADER_ID,
    MAX_RELATIVE_TARGET,
    find_leader_port,
    park_follower,
    load_poses,
    save_poses,
)

FOLLOW_HZ = 50.0
MAX_STEP_DEG = 3.0  # per tick -> at most 150 deg/s; also softens the first catch-up move


class Mirror:
    """Background loop: read the leader, send a rate-limited copy to the follower."""

    def __init__(self, leader, follower):
        self.leader, self.follower = leader, follower
        self.latest: dict[str, float] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        obs = follower.get_observation()
        self.cmd = {j: float(obs[f"{j}.pos"]) for j in ALL_JOINTS}  # start from where the follower is
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        period = 1.0 / FOLLOW_HZ
        while not self.stop.is_set():
            t0 = time.perf_counter()
            try:
                act = {k.removesuffix(".pos"): float(v) for k, v in self.leader.get_action().items()}
                with self.lock:
                    self.latest = act
                for j in ALL_JOINTS:
                    d = act[j] - self.cmd[j]
                    self.cmd[j] += max(-MAX_STEP_DEG, min(MAX_STEP_DEG, d))
                self.follower.send_action({f"{j}.pos": v for j, v in self.cmd.items()})
            except Exception as e:
                print(f"[follow] {e}")
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def read(self) -> dict[str, float]:
        with self.lock:
            return {k: round(v, 2) for k, v in self.latest.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow", action="store_true", help="follower copies the leader live while you record")
    args = ap.parse_args()

    from lerobot.teleoperators.rebot_102_leader import RebotArm102Leader, RebotArm102LeaderTeleopConfig

    port = find_leader_port()
    leader = RebotArm102Leader(RebotArm102LeaderTeleopConfig(port=port, id=LEADER_ID))
    leader.connect(calibrate=False)
    print(f"Leader connected on {port}.")

    follower = mirror = None
    if args.follow:
        from lerobot.robots.rebot_b601_follower import RebotB601Follower, RebotB601FollowerRobotConfig

        follower = RebotB601Follower(
            RebotB601FollowerRobotConfig(port=FOLLOWER_PORT, motor_family="rs", id=FOLLOWER_ID, max_relative_target=MAX_RELATIVE_TARGET)
        )
        print("Connecting follower: it will now copy the leader. Keep clear of it.")
        follower.connect(calibrate=False)
        mirror = Mirror(leader, follower)
        mirror.thread.start()
        time.sleep(0.3)

    def read() -> dict[str, float]:
        if mirror is not None:
            return mirror.read()
        return {k.removesuffix(".pos"): round(float(v), 2) for k, v in leader.get_action().items()}

    print("Move the leader by hand, then type a command (p <name> / go / gc / show / list / q).")
    try:
        while True:
            cmd = input("> ").strip()
            if not cmd:
                continue
            word, _, arg = cmd.partition(" ")
            arg = arg.strip()
            data = load_poses()
            if word == "q":
                break
            elif word == "show":
                print(read())
            elif word == "p":
                if not arg:
                    print("Give a name, e.g.  p above_highlighter")
                    continue
                cur = read()
                data["poses"][arg] = {j: cur[j] for j in ARM_JOINTS}
                save_poses(data)
                print(f"Saved pose '{arg}': {data['poses'][arg]}")
            elif word in ("go", "gc"):
                state = "open" if word == "go" else "closed"
                data["gripper"][state] = read()[GRIPPER]
                save_poses(data)
                print(f"Saved gripper '{state}' = {data['gripper'][state]}")
            elif word == "list":
                for name, p in sorted(data["poses"].items()):
                    print(f"  {name}: {p}")
                print(f"  gripper: {data['gripper']}")
            elif word == "del" and arg:
                print("Deleted." if data["poses"].pop(arg, None) is not None else "No such pose.")
                save_poses(data)
            else:
                print("Commands: p <name> | go | gc | show | list | del <name> | q")
    except KeyboardInterrupt:
        print()
    finally:
        if mirror is not None:
            mirror.stop.set()  # stop copying the leader; the follower now drives itself home
            mirror.thread.join(timeout=1.0)
            park_follower(follower, dict(mirror.cmd))
        try:
            leader.disconnect()
        except Exception as e:
            print(f"Leader disconnect: {e}")


if __name__ == "__main__":
    main()
