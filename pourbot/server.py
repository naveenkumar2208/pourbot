"""PourBot: step-by-step web GUI for pouring the flask into the red cup.

    .venv/bin/python pourbot/server.py             # real robot
    .venv/bin/python pourbot/server.py --dry-run   # real camera and planning, no motion or gripper commands

Motion only ever starts from the Run button, one step at a time, and only along a
plan the operator has seen. Losing the browser connection stops the arm.
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
from perception import SETTINGS, analyze, capture, fuse
from runlog import RunLog
from planner import Layout, Planner, PlanningError

MATCH_TOLERANCE_DEG = 1.5
RESUME_TOLERANCE_DEG = 3.0
RECONNECT_AFTER_FAILURES = 10

HOME_FILE = ROOT / "home.json"
HOME_STEPS = ("startpos", "endpos")

STEPS = [
    ("check", "check", "Safety check", "Confirm the workspace is safe. Nothing moves."),
    ("startpos", "motion", "Go to home", "Moves to the saved home position, where every run starts."),
    ("survey", "perceive", "Find flask and cup", "Measures the table, flask and cup from the current view. Nothing moves."),
    ("overhead", "motion", "Look from above", "Moves the wrist camera straight above the flask and cup for a better measurement."),
    ("closeup", "perceive", "Measure again", "Measures again from above. Check the rings sit on the flask mouth and cup rim."),
    ("rehearse", "plan", "Rehearse whole pour", "Plans every remaining step with collision checks. Nothing moves."),
    ("open", "gripper", "Open gripper", "Opens the fingers fully."),
    ("pregrasp", "motion", "Move behind flask", "Moves behind the flask in the orientation you taught. The flask should look centered between the fingertips, as it did when you taught this pose. If not, correct the flask position before continuing."),
    ("approach", "motion", "Slide around flask", "Moves straight forward until the flask is between the fingers. Watch by eye and press STOP if a finger is about to touch it."),
    ("grab", "gripper", "Grip the flask", "Closes the fingers and checks the width matches the flask."),
    ("lift", "motion", "Lift flask", "Lifts straight up, keeping the flask upright."),
    ("carry", "motion", "Carry over cup", "Carries the flask upright to above the red cup."),
    ("tilt1", "motion", "Tilt: start", "Starts tilting sideways, turning about the gripper's x axis and lowering the flask toward the cup."),
    ("tilt2", "motion", "Tilt: pouring", "Water should begin to flow during this step."),
    ("tilt3", "motion", "Tilt: empty", "Tilts to the full angle to empty the flask."),
    ("drain", "hold", "Let it drain", "Holds still while the last water drains. Nothing moves."),
    ("untilt", "motion", "Tilt back upright", "Follows the same path back until the flask is upright."),
    ("return", "motion", "Carry back", "Carries the flask back above its original spot."),
    ("lower", "motion", "Set flask down", "Lowers the flask to the exact pose where it was gripped."),
    ("release", "gripper", "Release flask", "Opens the fingers."),
    ("retreat", "motion", "Back away", "Moves straight back out, clear of the flask."),
    ("home", "motion", "Back to overhead view", "Returns to the overhead view, ready to check the result."),
    ("verify", "perceive", "Check final position", "Measures how far the flask ended up from where it started."),
    ("endpos", "motion", "Return to home", "Moves back to the saved home position."),
]
IDS = [s[0] for s in STEPS]
KIND = {s[0]: s[1] for s in STEPS}


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
        self.first_flask = None
        self.plans = {}
        self.start_joints = None
        self.sim_joints = None
        self.view_joints = None
        self.hold_until = None
        self.progress = None
        self.checks = {}
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
        for i, (sid, kind, title, text) in enumerate(STEPS):
            state = "done" if sid in self.done else "current" if i == self.current else "todo"
            if self.failed == sid:
                state = "failed"
            steps.append(dict(id=sid, kind=kind, title=title, text=text, state=state, result=self.done.get(sid),
                              plan=self.public_plan(self.plans[sid]) if sid in self.plans else None))
        layout = None
        if self.scene and self.scene.get("flask") and self.scene.get("cup"):
            lay = Layout(self.scene)
            layout = dict(neck_z=lay.neck_z, flask_height=lay.flask_height, base_r=lay.base_r,
                          cup_half=SETTINGS["cup"]["outer_diameter_mm"] / 2,
                          held=self.current > IDS.index("grab") and self.current <= IDS.index("release"))
        return jsonable(dict(
            dry_run=self.dry_run, status=self.status, steps=steps, current=self.current, busy=self.busy,
            stopped=self.stopped, failed=self.failed, scene=self.scene, layout=layout, nudges=self.nudges,
            cameras=SETTINGS["cameras"], log=self.log_lines[-60:], progress=self.progress,
            hold_left=max(0, round(self.hold_until - time.time(), 1)) if self.hold_until else None,
            home_saved=HOME_FILE.exists(), home_blocked=self.home_blocked(),
            fresh_measurement=bool(self.overlays) and not self.busy and self.current > 0
            and KIND[IDS[self.current - 1]] in ("perceive", "plan"),
            speed=SETTINGS["arm_speed_deg_s"], overlays=sorted(self.overlays), drain=SETTINGS["pour"]["drain_seconds"],
            checks=self.checks))

    # ----- home -----

    def home_blocked(self):
        """Why the Home button can't be used right now, or None if it can."""
        if not HOME_FILE.exists():
            return "No home position saved"
        if self.busy:
            return "A step is running"
        if self.stopped:
            return "Clear the stop first"
        if not self.status["connected"]:
            return "Robot is not connected"
        if "approach" in self.done and "retreat" not in self.done:
            return "The fingers are around the flask. Finish setting it down first"
        return None

    async def plan_home(self):
        home = json.loads(HOME_FILE.read_text())
        return await self.planner.to_home(home, self.scene, list(self.joints_now()))

    async def prepare_next(self):
        """Home moves depend on where the arm is, so they are planned as they come up."""
        sid = IDS[self.current] if self.current < len(IDS) else None
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
            self.current = IDS.index("survey")
        except PlanningError as error:
            self.record.step_finished("failed", str(error))
            self.log(f"FAILED: Planning failed: {error}")
        except asyncio.CancelledError:
            self.record.step_finished("cancelled")
            self.log("Move home cancelled")
            raise
        except Exception as error:
            self.record.step_finished("failed", str(error))
            self.log(f"FAILED: {error}")
            await self.halt_after_failure()
        finally:
            self.busy = None
            self.progress = None

    # ----- steps -----

    async def perceive(self):
        frames = [await capture(self.machine, name) for name in SETTINGS["cameras"]]
        self.observations = await asyncio.to_thread(lambda: [analyze(f) for f in frames])
        self.overlays = {o.camera: o.overlay for o in self.observations}
        self.scene = fuse(self.observations, self.nudges)
        self.record.scene(self.busy, self.scene)
        for o in self.observations:
            self.record.image(self.busy, o.camera, o.overlay)
        if self.scene["problems"]:
            raise RuntimeError("; ".join(self.scene["problems"]))
        return self.scene

    def describe_scene(self, scene):
        f, c = scene["flask"], scene["cup"]
        return (f"Flask ({f['x']:.0f}, {f['y']:.0f}), mouth {f['rim_z']:.0f} mm. "
                f"Cup ({c['x']:.0f}, {c['y']:.0f}), rim {c['rim_z']:.0f} mm.")

    async def run_step(self, sid):
        kind = KIND[sid]
        if kind == "check":
            return "Operator confirmed the workspace."
        if kind == "perceive":
            if sid == "survey":
                self.start_joints = list(self.joints_now())
                self.sim_joints = self.start_joints if self.dry_run else None
            previous = self.scene, self.observations, self.overlays
            try:
                scene = await self.perceive()
            except RuntimeError as error:
                if sid != "closeup":
                    raise
                # Nothing on the table has moved, so the first view is still valid.
                self.scene, self.observations, self.overlays = previous
                return f"Could not measure from above ({error}). Keeping the first measurement."
            result = self.describe_scene(scene)
            if sid == "survey":
                self.first_flask = scene["flask"]["measured"]
                self.plans["overhead"] = await self.planner.overhead(scene, self.start_joints)
                self.record.plan("overhead", self.plans["overhead"], self.duration(self.plans["overhead"]["waypoints"]))
                result += " Overhead move planned."
            if sid in ("closeup", "verify"):
                now, old = scene["flask"]["measured"], self.first_flask
                shift = float(np.hypot(now["x"] - old["x"], now["y"] - old["y"]))
                result += (f" Flask estimate moved {shift:.1f} mm from the first view." if sid == "closeup"
                           else f" Flask is {shift:.1f} mm from where it started.")
                if sid == "closeup":
                    self.first_flask = now
            return result
        if kind == "plan":
            start = list(self.joints_now())
            self.plans.update(await self.planner.pour_sequence(self.scene, start))
            # A correction made from behind the flask replans from there, but the way home stays the same.
            if self.view_joints is None:
                self.view_joints = start
            elif max(abs(a - b) for a, b in zip(start, self.view_joints)) > MATCH_TOLERANCE_DEG:
                self.plans["home"]["waypoints"].append(self.view_joints)
            for name, plan in self.plans.items():
                self.record.plan(name, plan, self.duration(plan["waypoints"]))
            total = sum(self.duration(self.plans[s]["waypoints"]) for s in self.plans if s != "overhead")
            return f"All steps planned with no collisions. About {total / 60:.1f} min of motion at {SETTINGS['arm_speed_deg_s']}°/s."
        if kind == "hold":
            self.hold_until = time.time() + SETTINGS["pour"]["drain_seconds"]
            while time.time() < self.hold_until:
                await asyncio.sleep(0.2)
            self.hold_until = None
            return "Drain time finished."
        if kind == "gripper":
            return await self.run_gripper(sid)
        return await self.run_motion(sid)

    async def run_gripper(self, sid):
        if self.dry_run:
            await asyncio.sleep(1)
            return "Dry run: gripper not moved."
        if sid in ("open", "release"):
            await self.gripper.open(timeout=20)
            return "Gripper open."
        await self.gripper.grab(timeout=20)
        await asyncio.sleep(0.5)
        held = await self.gripper.is_holding_something(timeout=5)
        width = (held.meta or {}).get("position")
        if width is None:
            raise RuntimeError("Gripper did not report its width; check the flask by eye before lifting")
        width_mm = width / 10
        grasp = SETTINGS["grasp"]
        if abs(width_mm - grasp["expected_neck_width_mm"]) > grasp["width_tolerance_mm"]:
            raise RuntimeError(f"Fingers closed to {width_mm:.0f} mm but the flask should be about "
                               f"{grasp['expected_neck_width_mm']} mm. The flask may not be in the grip.")
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
        distances = [max(abs(a - b) for a, b in zip(now, w)) for w in waypoints]
        nearest = int(np.argmin(distances))
        if distances[nearest] > RESUME_TOLERANCE_DEG:
            raise RuntimeError(f"The arm is {distances[nearest]:.1f}° away from this plan. "
                               "Press Restart and measure again.")
        if nearest != 0 and distances[0] > MATCH_TOLERANCE_DEG:
            self.log(f"Resuming {sid} from waypoint {nearest + 1} of {len(waypoints)}")
        remaining = waypoints[nearest:]
        if len(remaining) == 1:
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
            if self.current >= len(IDS) or IDS[self.current] != sid:
                raise HTTPException(409, "That is not the current step")
            if sid == "check":
                if not all(self.checks.get(k) for k in CHECK_ITEMS):
                    raise HTTPException(409, "Tick every safety item first")
            self.failed = None
            self.busy = sid
            self.task = asyncio.create_task(self.execute(sid))

    async def execute(self, sid):
        title = dict((s[0], s[2]) for s in STEPS)[sid]
        self.log(f"Started: {title}")
        self.record.step_started(sid, title, KIND[sid])
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
        except Exception as error:
            self.record.step_finished("failed", str(error))
            self.fail(sid, str(error))
            if KIND[sid] in ("motion", "gripper"):
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
            self.hold_until = None

    def fail(self, sid, message):
        self.failed = sid
        self.log(f"FAILED: {message}")

    async def halt_after_failure(self):
        """A move command that errored may still be running on the robot, so never assume it ended."""
        self.stopped = True
        if self.dry_run or self.machine is None:
            return
        results = await asyncio.gather(self.arm.stop(timeout=5), self.gripper.stop(timeout=5), return_exceptions=True)
        if any(isinstance(r, Exception) for r in results):
            self.log("STOP: Could not confirm the arm stopped. It may still be moving. Use the hardware e-stop.")
        else:
            self.log("STOP: Arm told to stop after the failed move.")

    async def stop(self, reason):
        self.stopped = True
        self.log(f"STOP: {reason}")
        if self.task and not self.task.done():
            self.task.cancel()
        if self.dry_run or self.machine is None:
            return
        await asyncio.gather(self.arm.stop(timeout=5), self.gripper.stop(timeout=5), return_exceptions=True)


CHECK_ITEMS = {
    "people": "Everyone is clear of the arm's reach and someone is at the hardware e-stop",
    "electronics": "Laptops and electronics are moved away from the flask and cup",
    "water": "First run uses an empty flask; water only after a clean empty run",
    "flask": "The flask is steady and its blue neck tape is visible to the camera",
    "cup": "The red cup is empty or can take all the water, and is taped down",
}

app_state = None


@asynccontextmanager
async def lifespan(_):
    poller = asyncio.create_task(app_state.poll())
    yield
    poller.cancel()


api = FastAPI(lifespan=lifespan)


@api.middleware("http")
async def require_app_header(request: Request, call_next):
    # Blocks other websites from driving the robot through this browser.
    if request.method == "POST" and request.headers.get("x-pourbot") != "1":
        return Response("Forbidden", status_code=403)
    return await call_next(request)


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
    if name not in SETTINGS["cameras"] or app_state.machine is None:
        raise HTTPException(404)
    try:
        images, _ = await Camera.from_robot(app_state.machine, name).get_images(filter_source_names=["color"], timeout=8)
    except Exception as error:
        raise HTTPException(503, str(error)[:200])
    return Response(images[0].data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@api.get("/api/overlay/{name}")
async def overlay(name: str):
    if name not in app_state.overlays:
        raise HTTPException(404)
    return Response(app_state.overlays[name], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@api.post("/api/run/{sid}")
async def run(sid: str):
    await app_state.start(sid)
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
    if app_state.busy:
        raise HTTPException(409, "Stop the current step first")
    app_state.record.end_run(app_state.snapshot())
    app_state.reset_state()
    app_state.begin_record()
    app_state.log("Restarted from the first step. Nothing moved.")
    return {"ok": True}


@api.post("/api/nudge")
async def nudge(request: Request):
    """Operator correction to a measured position; the rehearsal must then be repeated."""
    if app_state.busy:
        raise HTTPException(409, "A step is running")
    if "approach" in app_state.done:
        raise HTTPException(409, "Positions can only be corrected before the fingers go around the neck")
    body = await request.json()
    item, axis, delta = body["item"], body["axis"], float(body["mm"])
    if item not in ("flask", "cup") or axis not in ("x", "y", "z") or abs(delta) > 20:
        raise HTTPException(400, "Bad correction")
    entry = app_state.nudges.setdefault(item, dict(x=0.0, y=0.0, z=0.0))
    entry[axis] += delta
    if app_state.observations:
        app_state.scene = fuse(app_state.observations, app_state.nudges)
    for sid in list(app_state.plans):
        if sid != "overhead":
            del app_state.plans[sid]
    if "rehearse" in app_state.done:
        for sid in ("rehearse", "open", "pregrasp"):
            app_state.done.pop(sid, None)
        app_state.current = IDS.index("rehearse")
    app_state.log(f"Corrected {item} {axis} by {delta:+.0f} mm. Rehearse again.")
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
        if app_state.busy and KIND.get(app_state.busy, "motion") in ("motion", "gripper"):
            await app_state.stop("Browser disconnected during motion")


def main():
    global app_state
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Use the real camera and planner but send no motion or gripper commands")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app_state = App(args.dry_run)
    print(f"PourBot{' (dry run)' if args.dry_run else ''}: http://127.0.0.1:{args.port}")
    uvicorn.run(api, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
