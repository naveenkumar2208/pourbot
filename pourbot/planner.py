"""Turn a measured scene into joint-space plans. Planning never moves the arm."""
import json
import math
import time
from pathlib import Path

import numpy as np
from google.protobuf.json_format import MessageToJson
from viam.proto.common import (Capsule, Geometry, GeometriesInFrame, Pose, PoseInFrame, RectangularPrism,
                               Transform, Vector3, WorldState)
from viam.proto.service.motion import Constraints, LinearConstraint, MoveRequest, OrientationConstraint
from viam.services.motion import MotionClient
from viam.utils import dict_to_struct

from geometry import pose_to_tf, tf_to_pose
from perception import SETTINGS

UP = np.array([0.0, 0.0, 1.0])
TAUGHT = json.loads((Path(__file__).parent / "approach.json").read_text())
TAUGHT_R = pose_to_tf(TAUGHT["gripper_pose"])[0]


class PlanningError(RuntimeError):
    pass


def box(label, x0, x1, y0, y1, z0, z1):
    return Geometry(label=label, center=Pose(x=(x0 + x1) / 2, y=(y0 + y1) / 2, z=(z0 + z1) / 2, o_z=1),
                    box=RectangularPrism(dims_mm=Vector3(x=x1 - x0, y=y1 - y0, z=z1 - z0)))


class Layout:
    """Where everything goes, all derived from the scene and settings.json."""

    def __init__(self, scene):
        self.s = SETTINGS
        self.scene = scene
        self.flask, self.cup = scene["flask"], scene["cup"]
        self.neck_below = self.s["flask"]["grip_below_mouth_mm"]
        self.base_r = self.s["flask"]["base_diameter_mm"] / 2
        self.depth = self.s["grasp"]["neck_depth_in_gripper_mm"]
        self.table_z = scene["table_z"]
        self.flask_height = self.flask["rim_z"] - self.table_z
        self.mouth_r = self.flask["radius"]
        self.neck_z = self.flask["rim_z"] - self.neck_below

    @staticmethod
    def radial(x, y):
        u = np.array([x, y, 0.0])
        return u / np.linalg.norm(u)

    def upright(self, u):
        """The operator's taught orientation: fingers forward and nearly level, wrist camera on top."""
        return TAUGHT_R

    def grip_pose(self, center_xy, neck_z):
        R = self.upright(self.radial(*center_xy))
        return R, np.array([*center_xy, neck_z]) - self.depth * R[:, 2]

    def grasp(self):
        return self.grip_pose((self.flask["x"], self.flask["y"]), self.neck_z)

    def pregrasp(self):
        R, t = self.grasp()
        return R, t - R[:, 2] * self.s["grasp"]["approach_standoff_mm"]

    def lifted(self):
        R, t = self.grasp()
        return R, t + UP * self.s["grasp"]["lift_mm"]

    def lip_height(self, deg):
        """Lip height above the cup rim that keeps the flask's base clear of the rim."""
        p = self.s["pour"]
        a = math.radians(deg)
        lowest = min(-(side * self.base_r - self.mouth_r) * math.sin(a) - self.flask_height * math.cos(a)
                     for side in (-1, 1))
        return max(p["min_lip_above_rim_mm"], -lowest + p["flask_clearance_mm"])

    def pour_pose(self, deg):
        """Gripper pose with the flask tilted `deg` and the pouring lip held over the cup.

        The flask tips sideways, which is mostly a roll of the wrist about the fingers' axis.
        """
        cup_xy = np.array([self.cup["x"], self.cup["y"]])
        u = self.radial(*cup_xy)
        turn = math.radians(self.s["pour"]["mouth_turn_toward_base_deg"])
        mouth = math.cos(turn) * np.cross(UP, u) - math.sin(turn) * u
        lip_world = np.array([*(cup_xy - mouth[:2] * self.s["pour"]["lip_inset_from_cup_center_mm"]
                                - u[:2] * self.s["pour"]["lip_toward_base_mm"]),
                              self.cup["rim_z"] + self.lip_height(deg)])
        R0 = self.upright(u)
        axis = np.cross(UP, mouth)
        a = math.radians(deg)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotate = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
        lip_from_grip_point = self.neck_below * UP + self.mouth_r * mouth + self.depth * R0[:, 2]
        return rotate @ R0, lip_world - rotate @ lip_from_grip_point

    def overhead_camera(self):
        mid = np.array([(self.flask["x"] + self.cup["x"]) / 2, (self.flask["y"] + self.cup["y"]) / 2])
        return dict(x=float(mid[0]), y=float(mid[1]), z=self.table_z + self.s["overhead"]["camera_height_mm"])

    def world_state(self, holding, flask_on_table, include_neck=True):
        z0 = self.table_z
        obstacles = [box("table-measured", -1500, 1500, -1500, 1500, z0 - 60, z0)]
        c = self.cup
        half = self.s["cup"]["outer_diameter_mm"] / 2 + 3
        obstacles.append(box("cup", c["x"] - half, c["x"] + half, c["y"] - half, c["y"] + half, z0, c["rim_z"]))
        if flask_on_table:
            top = self.flask["rim_z"] if include_neck else z0 + 12  # fingers must reach the flask, so only its foot is modelled
            f, r = self.flask, self.base_r
            obstacles.append(box("flask", f["x"] - r, f["x"] + r, f["y"] - r, f["y"] + r, z0, top))
        for i, b in enumerate(self.scene.get("obstacles", [])):
            obstacles.append(box(f"seen-{i}", b["x0"], b["x1"], b["y0"], b["y1"], z0, b["top_z"]))
        transforms = [Transform(
            reference_frame="wrist-camera-body",
            pose_in_observer_frame=PoseInFrame(reference_frame="arm", pose=Pose(x=83, y=-27, z=18, o_z=1)),
            physical_object=box("wrist-camera-body", -13, 13, -46, 46, -13, 13))]  # D435 body
        if holding:
            length = self.flask_height - 10
            flask_up = self.upright(self.radial(self.flask["x"], self.flask["y"])).T @ UP
            center = np.array([0, 0, self.depth]) + flask_up * (self.neck_below - length / 2)
            ox, oy, oz = map(float, flask_up)
            transforms.append(Transform(
                reference_frame="held-flask",
                pose_in_observer_frame=PoseInFrame(reference_frame="gripper", pose=Pose(o_z=1)),
                physical_object=Geometry(
                    label="held-flask",
                    center=Pose(x=float(center[0]), y=float(center[1]), z=float(center[2]), o_x=ox, o_y=oy, o_z=oz),
                    capsule=Capsule(radius_mm=self.base_r, length_mm=max(length, 2 * self.base_r + 1)))))
        return WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=obstacles)],
                          transforms=transforms)


class Planner:
    def __init__(self, machine, log=print):
        self.motion = MotionClient.from_robot(machine, SETTINGS["motion"])
        self.log = log

    async def plan(self, component, pose, start_deg, world, constraints=None, via_deg=None):
        request = MoveRequest(name=SETTINGS["motion"], component_name=component,
                              destination=PoseInFrame(reference_frame="world", pose=Pose(**pose)), world_state=world)
        if constraints is not None:
            request.constraints.CopyFrom(constraints)
        start = [math.radians(v) for v in start_deg]
        extra = {"start_state": {"configuration": {SETTINGS["arm"]: start}}}
        if via_deg is not None:
            extra["waypoints"] = [{"configuration": {SETTINGS["arm"]: [math.radians(v) for v in via_deg]}}]
        request.extra.CopyFrom(dict_to_struct(extra))
        try:
            result = await self.motion.do_command({"plan": MessageToJson(request)}, timeout=90)
        except Exception as error:
            raise PlanningError(str(error)) from error
        return [[math.degrees(v) for v in step[SETTINGS["arm"]]] for step in result["plan"]]

    async def to_home(self, home, scene, start_deg):
        """Collision-checked path to the exact saved joints. Flask and cup are avoided once measured."""
        target = home["joints_deg"]
        if max(abs(a - b) for a, b in zip(start_deg, target)) < 0.5:
            return dict(waypoints=[list(start_deg), list(target)], targets=[], note="Already at home")
        if scene and scene.get("flask") and scene.get("cup"):
            world, note = Layout(scene).world_state(holding=False, flask_on_table=True), "Avoids the measured flask and cup"
        else:
            world, note = None, "Flask and cup are not measured yet, so watch the path"
        path = await self.plan(SETTINGS["gripper"], home["gripper_pose"], start_deg, world, via_deg=target)
        pose = home["gripper_pose"]
        return dict(waypoints=path, targets=[[pose["x"], pose["y"], pose["z"]]], note=note)

    async def overhead(self, scene, start_deg):
        """Camera straight down above the flask and cup; tries four camera rolls, keeps the shortest move."""
        layout = Layout(scene)
        spot = layout.overhead_camera()
        world = layout.world_state(holding=False, flask_on_table=True)
        best, last_error = None, None
        for theta in (0, 90, 180, -90):
            try:
                path = await self.plan("cam", dict(**spot, o_z=-1, theta=theta), start_deg, world)
            except PlanningError as error:
                last_error = error
                continue
            travel = max(abs(a - b) for a, b in zip(path[0], path[-1]))
            if best is None or travel < best[0]:
                best = (travel, path)
        if best is None:
            raise PlanningError(f"No overhead view reachable: {last_error}")
        return dict(waypoints=best[1], targets=[[spot["x"], spot["y"], spot["z"]]], note="Camera looks straight down")

    async def pour_sequence(self, scene, start_deg):
        """Plan every step in order, each one starting where the last ended."""
        layout = Layout(scene)
        steps = {}
        joint_now = list(start_deg)

        async def add(name, R, t, world, constraints=None, note="", via_deg=None):
            nonlocal joint_now
            started = time.time()
            try:
                path = await self.plan(SETTINGS["gripper"], tf_to_pose(R, t), joint_now, world, constraints, via_deg)
            except PlanningError as error:
                raise PlanningError(f"{name}: {error}") from error
            steps[name] = dict(waypoints=path, targets=[list(map(float, t))], note=note)
            joint_now = path[-1]
            self.log(f"Planned {name}: {len(path)} waypoints in {time.time() - started:.1f}s")

        line = Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=2, orientation_tolerance_degs=2)])
        level = Constraints(orientation_constraint=[OrientationConstraint(orientation_tolerance_degs=8)])

        free = layout.world_state(holding=False, flask_on_table=True)
        at_neck = layout.world_state(holding=False, flask_on_table=True, include_neck=False)
        held = layout.world_state(holding=True, flask_on_table=False)

        R, t = layout.pregrasp()
        # Solve from the taught joints so the wrist ends up in the configuration the operator showed.
        try:
            wrist = (await self.plan(SETTINGS["gripper"], tf_to_pose(R, t), TAUGHT["joints_deg"], free))[-1]
        except PlanningError as error:
            raise PlanningError(f"pregrasp: not reachable from the taught wrist position: {error}") from error
        await add("pregrasp", R, t, free, note="Same orientation you taught, fingers open behind the flask", via_deg=wrist)
        R, t = layout.grasp()
        await add("approach", R, t, at_neck, line, "Straight along the fingers until the flask is between them")
        R, t = layout.lifted()
        await add("lift", R, t, held, line, "Straight up, flask stays upright")
        R, t = layout.pour_pose(0)
        await add("carry", R, t, held, level, "Flask stays upright while it travels above the cup")

        angle = 0.0
        stage_names = ["tilt1", "tilt2", "tilt3"]
        step_deg = SETTINGS["pour"]["waypoint_every_deg"]
        for name, end in zip(stage_names, SETTINGS["pour"]["stages_deg"]):
            path, targets = [joint_now], []
            start_angle = angle
            while angle < end - 1e-6:
                angle = min(angle + step_deg, end)
                R, t = layout.pour_pose(angle)
                try:
                    segment = await self.plan(SETTINGS["gripper"], tf_to_pose(R, t), path[-1], held)
                except PlanningError as error:
                    raise PlanningError(f"{name}: tilt {angle:.0f}° not reachable: {error}") from error
                jump = max(abs(a - b) for a, b in zip(segment[0], segment[-1]))
                if jump > 45:
                    raise PlanningError(f"{name}: tilt {angle:.0f}° needs a {jump:.0f}° joint swing")
                path += segment[1:]
                targets.append(list(map(float, t)))
            steps[name] = dict(waypoints=path, targets=targets, angle_from=start_angle, angle_to=end,
                               note=f"Tilt {start_angle:.0f}° to {end:.0f}°, lip held over the cup")
            joint_now = path[-1]
            self.log(f"Planned {name}: {len(path)} waypoints")

        def reverse(*names):
            path = []
            for n in names:
                path += steps[n]["waypoints"][::-1]
            return [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]

        steps["untilt"] = dict(waypoints=reverse("tilt3", "tilt2", "tilt1"), targets=steps["carry"]["targets"],
                               note="Same path back to upright")
        steps["return"] = dict(waypoints=reverse("carry"), targets=steps["lift"]["targets"],
                               note="Same path back above the flask's spot")
        steps["lower"] = dict(waypoints=reverse("lift"), targets=steps["approach"]["targets"],
                              note="Straight down to the exact pose where the flask was gripped")
        steps["retreat"] = dict(waypoints=reverse("approach"), targets=steps["pregrasp"]["targets"],
                                note="Straight back out, away from the flask")
        steps["home"] = dict(waypoints=reverse("pregrasp"), targets=[], note="Back to the overhead view")
        return steps
