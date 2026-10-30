"""WitchPotion: choose blue-taped flasks and beakers, and a robot arm pours them into the red cup.

    .venv/bin/python pourbot/server.py             # real robot
    .venv/bin/python pourbot/server.py --dry-run   # real camera and planning, no motion or gripper commands

Motion starts only from the Run or Brew buttons, and only along rehearsed plans. Brew runs the
remaining steps on its own and ends at the first problem. Losing the browser connection stops the arm.
"""
import argparse
import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from viam.components.arm import Arm, JointPositions
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame

from geometry import pose_to_tf
from helpers import connect
from perception import SETTINGS, annotate, fuse, measure
from runlog import RunLog
from planner import Planner, PlanningError

MATCH_TOLERANCE_DEG = 1.5
RESUME_TOLERANCE_DEG = 3.0
RECONNECT_AFTER_FAILURES = 10

HOME_FILE = ROOT / "home.json"
HOME_STEPS = ("startpos", "endpos")

BEFORE = [
    ("check", "check", "Safety check", "Confirm the workspace is safe. Nothing moves."),
    ("startpos", "motion", "Go to home", "Moves to the saved home position, where every run starts."),
    ("survey", "perceive", "Find the potions", "Measures the table, every blue-taped container and the cup from the current view. Nothing moves."),
    ("overhead", "motion", "Look from above", "Moves the wrist camera straight above the table for a better measurement."),
    ("closeup", "perceive", "Measure again", "Measures again from above. Check the numbered rings sit on each container's mouth and the cup rim."),
    ("choose", "choose", "Choose your potions", "Pick the containers to mix, in the order you want them poured."),
    ("rehearse", "plan", "Rehearse the brew", "Plans every pour with collision checks, treating the other containers as obstacles. Nothing moves."),
]
POUR = [
    ("open", "gripper", "Open gripper", "Opens the fingers fully."),
    ("pregrasp", "motion", "Move behind it", "Moves behind the container in the orientation you taught. It should look centered between the fingertips. If not, correct its position before continuing."),
    ("approach", "motion", "Slide around it", "Moves straight forward until the container is between the fingers. Watch by eye and press STOP if a finger is about to touch it."),
    ("grab", "gripper", "Grip it", "Closes the fingers and checks the width matches this container."),
    ("lift", "motion", "Lift", "Lifts straight up, keeping it upright."),
    ("carry", "motion", "Carry over cup", "Carries it upright to above the red cup."),
    ("tilt1", "motion", "Tilt: start", "Starts tilting sideways and lowers toward the cup."),
    ("tilt2", "motion", "Tilt: pouring", "Liquid should begin to flow during this step."),
    ("tilt3", "motion", "Tilt: empty", "Tilts to the full angle to empty it."),
    ("untilt", "motion", "Tilt back upright", "Follows the same path back until it is upright."),
    ("return", "motion", "Carry back", "Carries it back above its original spot."),
    ("lower", "motion", "Set down", "Lowers it to the exact pose where it was gripped."),
    ("release", "gripper", "Release", "Opens the fingers."),
    ("retreat", "motion", "Back away", "Moves straight back out, clear of the container."),
    ("home", "motion", "Back to overhead view", "Returns to the overhead view."),
]
AFTER = [
    ("verify", "perceive", "Check final positions", "Measures how far each poured container ended up from where it started."),
    ("endpos", "motion", "Return to home", "Moves back to the saved home position."),
]
KIND = {s[0]: s[1] for s in BEFORE + POUR + AFTER}


def base_of(sid):
    return sid.split(":")[-1]


def kind_of(sid):
    return KIND.get(base_of(sid), "motion")


class NotStarted(RuntimeError):
    """A move was refused before any command reached the arm."""


def resume_path(now, waypoints):
    """Rest of a planned path from the arm's position, which may lie between two waypoints."""
    now = np.array(now)
    best_distance, best_next, best_point = np.inf, 0, waypoints[0]
    for i, (a, b) in enumerate(zip(waypoints, waypoints[1:])):
        a, b = np.array(a), np.array(b)
        span = b - a
        share = float(np.clip((now - a) @ span / max(span @ span, 1e-9), 0, 1))
        point = a + span * share
        distance = float(np.abs(now - point).max())
        if distance < best_distance - 1e-9:
            best_distance, best_next, best_point = distance, i + 1, point
    if best_distance > RESUME_TOLERANCE_DEG:
        return None, best_distance
    return [list(map(float, best_point))] + waypoints[best_next:], best_distance


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class App:
    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.machine = None
        self.lock = asyncio.Lock()
        self.log_lines = []
        self.status = dict(connected=False, moving=False, joints=None, gripper_pose=None,
                           gripper_position=None, holding=None, error=None)
        self.task = None
        self.overlays = {}
        self.poll_failures = 0
        self.last_sample = None
        self.camera_lock = asyncio.Lock()
        self.frames = {}
        self.last_view = 0.0
        self.reset_state()
        self.record = RunLog(on_error=self.log)
        self.begin_record()

    def reset_state(self):
        self.current = 0
        self.done = {}
        self.busy = None
        self.stopped = False
        self.failed = None
        self.observations = []
        self.nudges = {}
        self.scene = None
        self.selection = []
        self.first_positions = []
        self.plans = {}
        self.start_joints = None
        self.sim_joints = None
        self.view_joints = None
        self.progress = None
        self.checks = {}
        self.playing = False
        self.overlays = {}

    def begin_record(self):
        home = json.loads(HOME_FILE.read_text()) if HOME_FILE.exists() else None
        taught = ROOT / "approach.json"
        self.record.start_run("server", self.dry_run, SETTINGS, home, json.loads(taught.read_text()) if taught.exists() else None)

    def log(self, text):
        self.log_lines.append(dict(t=time.strftime("%H:%M:%S"), text=text))
        del self.log_lines[:-200]
        if getattr(self, "record", None):
            self.record.event(text)

    async def connect(self):
        self.machine = await asyncio.wait_for(connect(), 30)
        self.arm = Arm.from_robot(self.machine, SETTINGS["arm"])
        self.gripper = Gripper.from_robot(self.machine, SETTINGS["gripper"])
        self.planner = Planner(self.machine, self.log)
        self.status["connected"] = True
        self.log("Connected to the robot" + (" (dry run: no motion or gripper commands)" if self.dry_run else ""))

    async def poll(self):
        while True:
            try:
                if self.machine is None:
                    await self.connect()
                if self.camera_lock.locked() and not self.busy_moving():
                    await asyncio.sleep(0.2)  # a depth burst is large; don't compete with it for the connection
                    continue
                joints = list((await self.arm.get_joint_positions(timeout=5)).values)
                moving = await self.arm.is_moving(timeout=5)
                pose = (await self.machine.transform_pose(
                    PoseInFrame(reference_frame=SETTINGS["gripper"], pose=Pose(o_z=1)), "world")).pose
                held = await self.gripper.is_holding_something(timeout=5)
                self.status.update(
                    connected=True, moving=moving, error=None, joints=joints,
                    gripper_pose=dict(x=pose.x, y=pose.y, z=pose.z, o_x=pose.o_x, o_y=pose.o_y, o_z=pose.o_z, theta=pose.theta),
                    gripper_position=(held.meta or {}).get("position"), holding=held.is_holding_something)
                self.poll_failures = 0
                if moving or joints != self.last_sample:
                    self.last_sample = joints
                    self.record.sample(self.busy, self.status)
            except Exception as error:
                self.status.update(connected=False, error=str(error)[:200])
                self.poll_failures += 1
                if self.poll_failures == 1:
                    self.log("Lost contact with the robot; retrying")
                # The SDK reconnects by itself. Replacing the client would cut off a running move
                # and the only way to stop it, so that is a last resort for an idle app.
                if self.machine is not None and self.poll_failures >= RECONNECT_AFTER_FAILURES and not self.busy:
                    try:
                        await self.machine.close()
                    except Exception:
                        pass
                    self.machine = None
            await asyncio.sleep(0.5)

    async def camera_loop(self):
        """The only reader of the live feed: robot calls never depend on a browser, and never overlap."""
        while True:
            if self.machine is not None and time.time() - self.last_view < 3:
                for name in SETTINGS["cameras"]:
                    try:
                        async with self.camera_lock:
                            images, _ = await Camera.from_robot(self.machine, name).get_images(
                                filter_source_names=["color"], timeout=8)
                        self.frames[name] = images[0].data
                    except Exception:
                        pass
            await asyncio.sleep(0.35)

    def busy_moving(self):
        return self.busy is not None and kind_of(self.busy) in ("motion", "gripper")

    def steps(self):
        """The sequence so far: one full pour for each chosen container, in the chosen order."""
        pours = [(f"{vid}:{sid}", kind, f"Potion {n}: {title}", text)
                 for n, vid in enumerate(self.selection, 1) for sid, kind, title, text in POUR]
        return BEFORE + pours + AFTER

    def ids(self):
        return [s[0] for s in self.steps()]

    def vessel(self, vid):
        return next((v for v in (self.scene or {}).get("vessels", []) if v["id"] == vid), None)

    def fingers_around_container(self):
        return any(f"{v}:approach" in self.done and f"{v}:retreat" not in self.done for v in self.selection)

    def pouring_started(self):
        return any(f"{v}:approach" in self.done for v in self.selection)

    def forget_plans(self):
        """Measurements or choices changed: rehearse again before anything moves."""
        for sid in list(self.plans):
            if sid != "overhead":
                del self.plans[sid]
        for sid in list(self.done):
            if ":" in sid or sid == "rehearse":
                del self.done[sid]

    def joints_now(self):
        if self.dry_run and self.sim_joints is not None:
            return self.sim_joints
        return self.status["joints"]

    def duration(self, waypoints):
        hops = [max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(waypoints, waypoints[1:])]
        return sum(hops) / SETTINGS["arm_speed_deg_s"]

    def public_plan(self, plan):
        w = plan["waypoints"]
        hops = [max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(w, w[1:])] or [0]
        return dict(note=plan.get("note", ""), waypoints=len(w), seconds=round(self.duration(w), 1),
                    biggest_joint_change=round(max(abs(a - b) for a, b in zip(w[0], w[-1])), 1),
                    biggest_hop=round(max(hops), 1), targets=plan.get("targets", []))

    def snapshot(self):
        steps = []
        for i, (sid, kind, title, text) in enumerate(self.steps()):
            state = "done" if sid in self.done else "current" if i == self.current else "todo"
            if self.failed == sid:
                state = "failed"
            steps.append(dict(id=sid, kind=kind, title=title, text=text, state=state, result=self.done.get(sid),
                              plan=self.public_plan(self.plans[sid]) if sid in self.plans else None))
        ids = self.ids()
        return jsonable(dict(
            dry_run=self.dry_run, status=self.status, steps=steps, current=self.current, busy=self.busy,
            stopped=self.stopped, failed=self.failed, scene=self.scene, nudges=self.nudges, selection=self.selection,
            cup_half=SETTINGS["cup"]["outer_diameter_mm"] / 2,
            poured=sum(f"{v}:tilt3" in self.done for v in self.selection),
            pouring=bool(self.busy and base_of(self.busy) in ("tilt1", "tilt2", "tilt3")),
            can_change_positions=not self.pouring_started(),
            cameras=SETTINGS["cameras"], log=self.log_lines[-60:], progress=self.progress,
            playing=self.playing,
            home_saved=HOME_FILE.exists(), home_blocked=self.home_blocked(),
            fresh_measurement=bool(self.overlays) and not self.busy and self.current > 0
            and kind_of(ids[self.current - 1]) in ("perceive", "plan", "choose"),
            speed=SETTINGS["arm_speed_deg_s"], overlays=sorted(self.overlays),
            checks=self.checks))

    # ----- home -----

    def home_blocked(self):
        """Why the Home button can't be used right now, or None if it can."""
        if not HOME_FILE.exists():
            return "No home position saved"
        if self.playing:
            return "The brew is running"
        if self.busy:
            return "A step is running"
        if self.stopped:
            return "Clear the stop first"
        if not self.status["connected"]:
            return "Robot is not connected"
        if self.fingers_around_container():
            return "The fingers are around a container. Finish setting it down first"
        return None

    async def plan_home(self):
        home = json.loads(HOME_FILE.read_text())
        return await self.planner.to_home(home, self.scene, list(self.joints_now()))

    async def prepare_next(self):
        """Home moves depend on where the arm is, so they are planned as they come up."""
        ids = self.ids()
        sid = ids[self.current] if self.current < len(ids) else None
        if sid in HOME_STEPS:
            try:
                self.plans[sid] = await self.plan_home()
                self.record.plan(sid, self.plans[sid], self.duration(self.plans[sid]["waypoints"]))
            except Exception as error:
                self.log(f"Could not plan the move home yet: {error}")

    async def go_home(self):
        """Home button: any run in progress is abandoned, because its plans start elsewhere."""
        self.record.step_started("button", "Home button", "motion")
        try:
            result = await self.run_motion("button")
            self.record.step_finished("done", result)
            self.log(f"Home button: {result}")
            checks, sim = self.checks, self.sim_joints
            self.reset_state()
            self.checks, self.sim_joints = checks, sim
            self.done = {"check": "Operator confirmed the workspace.", "startpos": "At home."}
            self.current = self.ids().index("survey")
        except PlanningError as error:
            self.record.step_finished("failed", str(error))
            self.log(f"FAILED: Planning failed: {error}")
        except asyncio.CancelledError:
            self.record.step_finished("cancelled")
            self.log("Move home cancelled")
            raise
        except NotStarted as error:
            self.record.step_finished("refused", str(error))
            self.log(f"FAILED: {error}")
        except Exception as error:
            self.record.step_finished("failed", str(error))
            self.log(f"FAILED: {error}")
            await self.halt_after_failure()
        finally:
            self.busy = None
            self.progress = None

    # ----- steps -----

    async def perceive(self):
        async with self.camera_lock:
            self.observations = [o for name in SETTINGS["cameras"] for o in await measure(self.machine, name)]
        self.scene = fuse(self.observations, self.nudges)
        self.draw_overlays()
        self.record.scene(self.busy, self.scene)
        for name, jpeg in self.overlays.items():
            self.record.image(self.busy, name, jpeg)
        if self.scene["problems"] and not (self.scene.get("vessels") and self.scene.get("cup")):
            raise RuntimeError("; ".join(sorted(set(self.scene["problems"]))))
        return self.scene

    def draw_overlays(self):
        latest = {o.camera: o.frame for o in self.observations if o.frame is not None}
        self.overlays = {name: annotate(frame, self.scene) for name, frame in latest.items()}

    def describe_scene(self, scene):
        c = scene["cup"]
        return f"Found {len(scene['vessels'])} container{'' if len(scene['vessels']) == 1 else 's'}. Cup at ({c['x']:.0f}, {c['y']:.0f})."

    def largest_shift(self, scene, ids=None):
        """How far measured containers are from where they first were, matched by position."""
        shifts = []
        for v in scene["vessels"]:
            if ids is None or v["id"] in ids:
                now = v["measured"]
                shifts.append(min(np.hypot(now["x"] - p["x"], now["y"] - p["y"]) for p in self.first_positions))
        return float(max(shifts)) if shifts else 0.0

    async def run_step(self, sid):
        kind = kind_of(sid)
        if kind == "check":
            return "Operator confirmed the workspace."
        if kind == "choose":
            return f"Brewing containers {', then '.join(self.selection)}."
        if kind == "perceive":
            if sid == "survey":
                self.start_joints = list(self.joints_now())
                self.sim_joints = self.start_joints if self.dry_run else None
            previous = self.scene, self.observations, self.overlays
            try:
                scene = await self.perceive()
                if sid == "closeup" and len(scene["vessels"]) < len(previous[0]["vessels"]):
                    raise RuntimeError(f"only {len(scene['vessels'])} of {len(previous[0]['vessels'])} containers in view")
            except RuntimeError as error:
                if sid != "closeup":
                    raise
                # Nothing on the table has moved, so the first view is still valid.
                self.scene, self.observations, self.overlays = previous
                return f"Could not measure from above ({error}). Keeping the first measurement."
            result = self.describe_scene(scene)
            if sid == "survey":
                self.first_positions = [v["measured"] for v in scene["vessels"]]
                self.plans["overhead"] = await self.planner.overhead(scene, self.start_joints)
                self.record.plan("overhead", self.plans["overhead"], self.duration(self.plans["overhead"]["waypoints"]))
                result += " Overhead move planned."
            if sid == "closeup":
                result += f" Largest change from the first view: {self.largest_shift(scene):.1f} mm."
                self.first_positions = [v["measured"] for v in scene["vessels"]]
            if sid == "verify":
                result += f" Poured containers are within {self.largest_shift(scene, self.selection):.1f} mm of where they started."
            return result
        if kind == "plan":
            start = list(self.joints_now())
            view = self.view_joints or start
            plans = {}
            for n, vid in enumerate(self.selection, 1):
                # Each pour starts and ends at the overhead view, so the others stay valid.
                begin = start if n == 1 else view
                try:
                    pour = await self.planner.pour_sequence(self.scene, self.vessel(vid), begin)
                except PlanningError as error:
                    raise PlanningError(f"Container {vid}: {error}") from error
                if begin is start and max(abs(a - b) for a, b in zip(start, view)) > MATCH_TOLERANCE_DEG:
                    pour["home"]["waypoints"].append(view)
                plans.update({f"{vid}:{name}": plan for name, plan in pour.items()})
            self.view_joints = view
            self.plans.update(plans)
            for name, plan in plans.items():
                self.record.plan(name, plan, self.duration(plan["waypoints"]))
            total = sum(self.duration(p["waypoints"]) for p in plans.values())
            return f"All {len(self.selection)} pours planned with no collisions. About {total / 60:.1f} min of motion at {SETTINGS['arm_speed_deg_s']}°/s."
        if kind == "gripper":
            return await self.run_gripper(sid)
        return await self.run_motion(sid)

    async def run_gripper(self, sid):
        if self.dry_run:
            await asyncio.sleep(1)
            return "Dry run: gripper not moved."
        if base_of(sid) in ("open", "release"):
            await self.gripper.open(timeout=20)
            return "Gripper open."
        await self.gripper.grab(timeout=20)
        await asyncio.sleep(0.5)
        held = await self.gripper.is_holding_something(timeout=5)
        width = (held.meta or {}).get("position")
        if width is None:
            raise RuntimeError("Gripper did not report its width; check the container by eye before lifting")
        width_mm = width / 10
        expected = self.vessel(sid.split(":")[0])["expected_width"]
        if abs(width_mm - expected) > SETTINGS["grasp"]["width_tolerance_mm"]:
            raise RuntimeError(f"Fingers closed to {width_mm:.0f} mm but this container should be about "
                               f"{expected} mm. It may not be in the grip.")
        return f"Gripping at {width_mm:.0f} mm."

    async def run_motion(self, sid):
        if sid in HOME_STEPS and sid not in self.plans:
            self.plans[sid] = await self.plan_home()
        plan = self.plans.get(sid)
        if plan is None:
            raise RuntimeError("This step has no plan. Run the rehearsal first.")
        waypoints = plan["waypoints"]
        if self.dry_run:
            for i in range(len(waypoints)):
                self.progress = dict(step=sid, done=i + 1, total=len(waypoints))
                self.sim_joints = waypoints[i]
                await asyncio.sleep(min(0.15, 3 / len(waypoints)))
            self.progress = None
            return "Dry run: arm not moved."
        now = list((await self.arm.get_joint_positions(timeout=5)).values)
        remaining, distance = resume_path(now, waypoints)
        if remaining is None:
            raise NotStarted(f"The arm is {distance:.1f}° away from this plan, so it was not moved. "
                             "Use the Home button, or press Restart and measure again.")
        if max(abs(a - b) for a, b in zip(now, waypoints[0])) > MATCH_TOLERANCE_DEG:
            self.log(f"Resuming {sid} from where the arm stopped")
        if max(abs(a - b) for a, b in zip(now, waypoints[-1])) < MATCH_TOLERANCE_DEG:
            return "Already at the target."
        self.progress = dict(step=sid, done=0, total=len(remaining), seconds=round(self.duration(remaining), 1))
        await self.arm.move_through_joint_positions([JointPositions(values=w) for w in remaining], timeout=self.duration(remaining) + 90)
        self.progress = None
        end = list((await self.arm.get_joint_positions(timeout=5)).values)
        error = max(abs(a - b) for a, b in zip(end, waypoints[-1]))
        if error > MATCH_TOLERANCE_DEG:
            raise RuntimeError(f"Arm stopped {error:.1f}° short of the target.")
        return f"Reached target (within {error:.2f}°)."

    async def start(self, sid):
        async with self.lock:
            if self.task and not self.task.done():
                raise HTTPException(409, "A step is already running")
            if self.stopped:
                raise HTTPException(409, "Clear the stop first")
            if self.machine is None or not self.status["connected"]:
                raise HTTPException(409, "Robot is not connected")
            ids = self.ids()
            if self.current >= len(ids) or ids[self.current] != sid:
                raise HTTPException(409, "That is not the current step")
            if sid == "choose" and not self.selection:
                raise HTTPException(409, "Choose at least one container first")
            if sid == "check":
                if not all(self.checks.get(k) for k in CHECK_ITEMS):
                    raise HTTPException(409, "Tick every safety item first")
            self.failed = None
            self.busy = sid
            self.task = asyncio.create_task(self.execute(sid))

    async def play(self):
        """Runs every remaining step in order. Ends on the first problem instead of pressing on."""
        self.log("Brew started")
        try:
            while self.playing and self.current < len(self.ids()):
                sid = self.ids()[self.current]
                if sid == "choose" and not self.selection:
                    self.log("Brew paused: choose your potions, then press Brew again")
                    return
                await self.start(sid)
                await self.task
                if self.failed or self.stopped:
                    self.log("Brew ended: this step did not finish")
                    return
                await asyncio.sleep(0.2)
            self.log("The potion is brewed" if self.playing else "Brew paused")
        except HTTPException as error:
            self.log(f"Brew ended: {error.detail}")
        except asyncio.CancelledError:
            self.log("Brew ended: stopped")
        except Exception as error:
            self.log(f"Brew ended: {error}")
        finally:
            self.playing = False

    async def execute(self, sid):
        title = next(s[2] for s in self.steps() if s[0] == sid)
        self.log(f"Started: {title}")
        self.record.step_started(sid, title, kind_of(sid))
        try:
            result = await self.run_step(sid)
        except asyncio.CancelledError:
            self.log("Step cancelled")
            self.record.step_finished("cancelled")
            self.busy = None
            raise
        except PlanningError as error:
            self.record.step_finished("failed", str(error))
            self.fail(sid, f"Planning failed: {error}")
        except NotStarted as error:
            self.record.step_finished("refused", str(error))
            self.fail(sid, str(error))
        except Exception as error:
            self.record.step_finished("failed", str(error))
            self.fail(sid, str(error))
            if kind_of(sid) in ("motion", "gripper"):
                await self.halt_after_failure()
        else:
            self.record.step_finished("done", result)
            self.done[sid] = result
            self.log(f"Finished: {result}")
            self.current += 1
            await self.prepare_next()
        finally:
            self.busy = None
            self.progress = None

    def fail(self, sid, message):
        self.failed = sid
        self.log(f"FAILED: {message}")

    async def halt_after_failure(self):
        """A move command that errored may still be running on the robot, so never assume it ended."""
        self.stopped = True
        self.playing = False
        if self.dry_run or self.machine is None:
            return
        results = await asyncio.gather(self.arm.stop(timeout=5), self.gripper.stop(timeout=5), return_exceptions=True)
        if any(isinstance(r, Exception) for r in results):
            self.log("STOP: Could not confirm the arm stopped. It may still be moving. Use the hardware e-stop.")
        else:
            self.log("STOP: Arm told to stop after the failed move.")

    async def stop(self, reason):
        self.stopped = True
        self.playing = False
        self.log(f"STOP: {reason}")
        if self.task and not self.task.done():
            self.task.cancel()
        if self.dry_run or self.machine is None:
            return
        await asyncio.gather(self.arm.stop(timeout=5), self.gripper.stop(timeout=5), return_exceptions=True)


CHECK_ITEMS = {
    "people": "Everyone is clear of the arm's reach and someone is at the hardware e-stop",
    "electronics": "Laptops and electronics are moved away from the containers and cup",
    "water": "First run uses empty containers; liquid only after a clean empty run",
    "flask": "Every container is steady and its blue tape is visible to the camera",
    "cup": "The red cup can hold everything you plan to pour, and is taped down",
}

app_state = None


@asynccontextmanager
async def lifespan(_):
    tasks = [asyncio.create_task(app_state.poll()), asyncio.create_task(app_state.camera_loop())]
    yield
    for task in tasks:
        task.cancel()


api = FastAPI(lifespan=lifespan)


class RequireAppHeader:
    """Blocks other websites from driving the robot through this browser.

    Written as plain ASGI on purpose: FastAPI's @middleware cancels the request handler when a
    browser disconnects, and cancelling a robot call mid-flight can drop the robot connection,
    which cuts off any move in progress.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST" and dict(scope["headers"]).get(b"x-pourbot") != b"1":
            await Response("Forbidden", status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


api.add_middleware(RequireAppHeader)


@api.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-store"})


@api.get("/app.js")
async def script():
    return FileResponse(ROOT / "static" / "app.js", headers={"Cache-Control": "no-store"})


@api.get("/style.css")
async def style():
    return FileResponse(ROOT / "static" / "style.css", headers={"Cache-Control": "no-store"})


@api.get("/api/checks")
async def checks():
    return CHECK_ITEMS


@api.get("/api/camera/{name}")
async def camera(name: str):
    if name not in SETTINGS["cameras"]:
        raise HTTPException(404)
    app_state.last_view = time.time()
    for _ in range(30):
        if name in app_state.frames:
            return Response(app_state.frames[name], media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        await asyncio.sleep(0.1)
    raise HTTPException(503, "No camera frame yet")


@api.get("/api/overlay/{name}")
async def overlay(name: str):
    if name not in app_state.overlays:
        raise HTTPException(404)
    return Response(app_state.overlays[name], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@api.post("/api/run/{sid}")
async def run(sid: str):
    if app_state.playing:
        raise HTTPException(409, "The brew is running")
    await app_state.start(sid)
    return {"ok": True}


@api.post("/api/play")
async def play():
    state = app_state
    if state.playing:
        raise HTTPException(409, "The brew is already running")
    if state.busy:
        raise HTTPException(409, "A step is running")
    if state.stopped:
        raise HTTPException(409, "Clear the stop first")
    if not state.status["connected"]:
        raise HTTPException(409, "Robot is not connected")
    ids = state.ids()
    if state.current >= len(ids):
        raise HTTPException(409, "The sequence has finished. Press Restart first")
    if ids[state.current] == "check" and not all(state.checks.get(k) for k in CHECK_ITEMS):
        raise HTTPException(409, "Tick every safety item first")
    state.playing = True
    asyncio.create_task(state.play())
    return {"ok": True}


@api.post("/api/select")
async def select(request: Request):
    """The containers to pour, in order. Changing them means rehearsing again."""
    state = app_state
    if state.busy or state.playing:
        raise HTTPException(409, "A step is running")
    if state.pouring_started():
        raise HTTPException(409, "Pouring has started. Press Restart to choose again")
    if state.current < state.ids().index("choose"):
        raise HTTPException(409, "Measure the table first")
    ids = [str(i) for i in (await request.json())["ids"]]
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "A container can only be poured once")
    for vid in ids:
        vessel = state.vessel(vid)
        if vessel is None:
            raise HTTPException(400, f"There is no container {vid}")
        if not vessel["usable"]:
            raise HTTPException(409, f"Container {vid} doesn't match a type in settings.json yet")
    state.selection = ids
    state.forget_plans()
    state.done.pop("choose", None)
    state.current = state.ids().index("choose")
    return {"ok": True}


@api.post("/api/pause")
async def pause():
    if app_state.playing:
        app_state.playing = False
        app_state.log("The brew will pause after this step")
    return {"ok": True}


@api.post("/api/check/{key}/{value}")
async def set_check(key: str, value: int):
    if key not in CHECK_ITEMS:
        raise HTTPException(404)
    app_state.checks[key] = bool(value)
    return {"ok": True}


@api.post("/api/home/plan")
async def home_plan():
    """Plans the move home so the operator can see it; nothing moves."""
    blocked = app_state.home_blocked()
    if blocked:
        raise HTTPException(409, blocked)
    try:
        app_state.plans["button"] = await app_state.plan_home()
        app_state.record.plan("button", app_state.plans["button"], app_state.duration(app_state.plans["button"]["waypoints"]))
    except PlanningError as error:
        raise HTTPException(409, f"No safe path home: {error}")
    return app_state.public_plan(app_state.plans["button"])


@api.post("/api/home/run")
async def home_run():
    async with app_state.lock:
        blocked = app_state.home_blocked()
        if blocked:
            raise HTTPException(409, blocked)
        if "button" not in app_state.plans:
            raise HTTPException(409, "Plan the move home first")
        app_state.busy = "button"
        app_state.task = asyncio.create_task(app_state.go_home())
    return {"ok": True}


@api.post("/api/stop")
async def stop():
    await app_state.stop("Operator pressed STOP")
    return {"ok": True}


@api.post("/api/clear-stop")
async def clear_stop():
    if app_state.status["moving"] and not app_state.dry_run:
        raise HTTPException(409, "The arm is still moving")
    app_state.stopped = False
    if not app_state.dry_run and app_state.machine is not None:
        try:
            await app_state.arm.do_command({"clear_error": True}, timeout=10)
        except Exception as error:
            app_state.log(f"Could not clear arm error: {str(error)[:160]}")
    app_state.log("Stop cleared. Press Run to continue this step from where it stopped.")
    return {"ok": True}


@api.post("/api/restart")
async def restart():
    if app_state.busy or app_state.playing:
        raise HTTPException(409, "Stop the current step first")
    app_state.record.end_run(app_state.snapshot())
    app_state.reset_state()
    app_state.begin_record()
    app_state.log("Restarted from the first step. Nothing moved.")
    return {"ok": True}


@api.post("/api/nudge")
async def nudge(request: Request):
    """Operator correction to a measured position; the rehearsal must then be repeated."""
    if app_state.busy or app_state.playing:
        raise HTTPException(409, "A step is running")
    if app_state.pouring_started():
        raise HTTPException(409, "Positions can only be corrected before the first container is gripped")
    body = await request.json()
    item, axis, delta = str(body["item"]), body["axis"], float(body["mm"])
    if (item != "cup" and app_state.vessel(item) is None) or axis not in ("x", "y", "z") or abs(delta) > 20:
        raise HTTPException(400, "Bad correction")
    entry = app_state.nudges.setdefault(item, dict(x=0.0, y=0.0, z=0.0))
    entry[axis] += delta
    if app_state.observations:
        app_state.scene = fuse(app_state.observations, app_state.nudges)
        app_state.draw_overlays()
    rehearsed = "rehearse" in app_state.done
    app_state.forget_plans()
    if rehearsed:
        app_state.current = app_state.ids().index("rehearse")
    name = "cup" if item == "cup" else f"container {item}"
    app_state.log(f"Corrected {name} {axis} by {delta:+.0f} mm. Rehearse again.")
    return {"ok": True}


@api.websocket("/ws")
async def socket(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps(app_state.snapshot()))
            await asyncio.sleep(0.25)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if app_state.playing or (app_state.busy and kind_of(app_state.busy) in ("motion", "gripper")):
            await app_state.stop("Browser disconnected during motion")


def main():
    global app_state
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Use the real camera and planner but send no motion or gripper commands")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app_state = App(args.dry_run)
    print(f"WitchPotion{' (dry run)' if args.dry_run else ''}: http://127.0.0.1:{args.port}")
    uvicorn.run(api, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
