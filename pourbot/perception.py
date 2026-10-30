"""Find the table, flask and red cup in world coordinates. Commands no motion.

Every camera named in settings.json is measured on its own and the results are
averaged, so adding a second RealSense only needs its name and its Viam frame.
"""
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame

from geometry import pose_to_tf

SETTINGS = json.loads((Path(__file__).parent / "settings.json").read_text())

# Anything outside these real-world sizes (mm) is not the flask or cup, whatever its color.
FLASK_RIM_HEIGHT, FLASK_MOUTH_RADIUS = (70, 220), (14, 45)
CUP_RIM_HEIGHT, CUP_RADIUS = (80, 140), (35, 60)


@dataclass
class Frame:
    camera: str
    color: np.ndarray
    depth: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray
    t: np.ndarray


@dataclass
class Observation:
    camera: str
    table_z: float
    table_tilt_deg: float
    vessels: list = field(default_factory=list)
    cup: dict = None
    obstacles: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    frame: Frame = None


def parse_depth(data):
    if data[:8] != b"DEPTHMAP":
        raise ValueError("Unexpected depth format")
    w, h = int.from_bytes(data[8:16], "big"), int.from_bytes(data[16:24], "big")
    return np.frombuffer(data[24:], dtype=">u2").reshape(h, w).astype(float)


async def capture(machine, name):
    cam = Camera.from_robot(machine, name)
    images, _ = await cam.get_images(timeout=15)
    data = {i.name: i.data for i in images}
    k = (await cam.get_properties(timeout=10)).intrinsic_parameters
    world = await machine.transform_pose(PoseInFrame(reference_frame=name, pose=Pose(o_z=1)), "world")
    R, t = pose_to_tf(world.pose)
    # The fingertips in the image and the operator's taught pose both put the camera further to the side than its Viam frame says.
    t = t + R[:, 0] * SETTINGS["camera_sideways_correction_mm"]
    color = cv2.imdecode(np.frombuffer(data["color"], np.uint8), cv2.IMREAD_COLOR)
    return Frame(name, color, parse_depth(data["depth"]), k.focal_x_px, k.focal_y_px, k.center_x_px, k.center_y_px, R, t)


def fit_table(world, valid):
    x, y, z = world[..., 0], world[..., 1], world[..., 2]
    region = valid & (np.hypot(x, y) > 250) & (np.hypot(x, y) < 1000) & (z > -80) & (z < 60)
    pts = world[region][::7]
    if len(pts) < 500:
        raise ValueError("Not enough depth points on the table")
    rng = np.random.default_rng(3)
    best_count, best = 0, None
    for _ in range(300):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-6:
            continue
        n = n / length * (1 if n[2] > 0 else -1)
        count = (np.abs(pts @ n - n @ a) < 4).sum()
        if count > best_count:
            best_count, best = count, (n, -n @ a)
    n, c = best
    inliers = pts[np.abs(pts @ n + c) < 4]
    center = inliers.mean(0)
    n = np.linalg.svd(inliers - center, full_matrices=False)[2][-1]
    n = n if n[2] > 0 else -n
    return n, -n @ center


def fit_circle(xy):
    """Least-squares circle with one round of outlier rejection."""
    for _ in range(2):
        A = np.c_[2 * xy, np.ones(len(xy))]
        sol = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)[0]
        center, radius = sol[:2], float(np.sqrt(max(sol[2] + sol[:2] @ sol[:2], 0)))
        error = np.abs(np.hypot(*(xy - center).T) - radius)
        keep = error < max(4.0, 2 * np.median(error))
        if keep.sum() < 8 or keep.all():
            break
        xy = xy[keep]
    return center, radius, float(np.median(error))


def measure_rim(world, height, mask):
    """The rim is the highest ring of the object: take it and fit a circle."""
    h = height[mask]
    top = np.percentile(h, 99.5)
    ring = mask & (np.abs(height - top) < 6)
    xy = world[ring][:, :2]
    if len(xy) < 30:
        return None
    center, radius, error = fit_circle(xy)
    rim_z = float(np.median(world[ring][:, 2]))
    return dict(x=float(center[0]), y=float(center[1]), rim_z=rim_z, radius=radius,
                height=float(np.median(height[ring])), fit_error=error, points=int(ring.sum()))


def measure_mouth(frame, world, height, normal, offset, mask):
    """Flask mouth: its widest image row is the mouth's center line, from any viewing angle."""
    rows = np.where(mask.any(1))[0]
    widths = np.array([np.ptp(np.where(mask[r])[0]) + 1 for r in rows])
    # Only the upper half of the blue blob can be the mouth; the taped neck is below it.
    upper = rows <= rows[0] + (rows[-1] - rows[0]) * 0.6
    row = int(np.median(rows[upper][widths[upper] >= widths[upper].max() - 2]))
    columns = np.where(mask[row])[0]
    u, v = (columns[0] + columns[-1]) / 2, float(row)
    rim_height = float(np.percentile(height[mask], 97))
    ray = frame.R @ np.array([(u - frame.cx) / frame.fx, (v - frame.cy) / frame.fy, 1.0])
    distance = (rim_height - offset - normal @ frame.t) / (normal @ ray)
    center = frame.t + ray * distance
    depth_along_axis = distance
    radius = (columns[-1] - columns[0] + 1) / 2 * depth_along_axis / frame.fx
    return dict(x=float(center[0]), y=float(center[1]), rim_z=float(center[2]), radius=float(radius),
                height=rim_height, fit_error=0.0, points=int(mask.sum()))


def depth_edges(depth, jump_mm=20):
    """Pixels where depth jumps. Objects that overlap in the image separate along these."""
    d = depth.astype(np.float32)
    edges = np.zeros(d.shape, bool)
    edges[:, 1:] |= np.abs(d[:, 1:] - d[:, :-1]) > jump_mm
    edges[1:, :] |= np.abs(d[1:, :] - d[:-1, :]) > jump_mm
    return cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def blobs(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [(labels == i, stats[i]) for i in range(1, count) if stats[i, 4] >= min_area]


def find_depth_obstacles(world, height, keep_out):
    """Anything else standing on the table, as boxes for the motion planner."""
    cell = 40.0
    tall = (height > 30) & (height < 700)
    pts = world[tall]
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < 1000]
    for x, y, r in keep_out:
        pts = pts[np.hypot(pts[:, 0] - x, pts[:, 1] - y) > r]
    if not len(pts):
        return []
    origin = pts[:, :2].min(0)
    ij = ((pts[:, :2] - origin) / cell).astype(int)
    grid = np.zeros(ij.max(0) + 1, np.uint16)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    boxes = []
    for blob, stats in blobs(grid >= 12, 2):
        inside = blob[ij[:, 0], ij[:, 1]]
        p = pts[inside]
        low, high = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
        boxes.append(dict(x0=float(low[0]), x1=float(high[0]), y0=float(low[1]), y1=float(high[1]), top_z=float(high[2])))
    return boxes


def analyze(frame):
    d = frame.depth
    valid = d > 0
    y, x = np.indices(d.shape)
    cam_pts = np.stack(((x - frame.cx) * d / frame.fx, (y - frame.cy) * d / frame.fy, d), -1)
    world = cam_pts @ frame.R.T + frame.t
    n, c = fit_table(world, valid)
    height = world @ n + c
    table_z = float(-(n[0] * 550 + n[1] * 50 + c) / n[2])
    tilt = float(np.degrees(np.arccos(min(1.0, n[2]))))
    obs = Observation(frame.camera, table_z, tilt)

    hsv = cv2.cvtColor(frame.color, cv2.COLOR_BGR2HSV)
    standing = valid & (height > 15) & (height < 250)
    radial = np.hypot(world[..., 0], world[..., 1])
    workspace = standing & (radial > 300) & (radial < 900) & (np.abs(world[..., 1]) < 260) & ~depth_edges(d)
    red = cv2.inRange(hsv, (0, 90, 50), (12, 255, 255)) | cv2.inRange(hsv, (168, 90, 50), (180, 255, 255))
    blue = cv2.inRange(hsv, (85, 70, 30), (135, 255, 255))

    def good_cup(rim):
        return rim and CUP_RIM_HEIGHT[0] <= rim["height"] <= CUP_RIM_HEIGHT[1] and CUP_RADIUS[0] <= rim["radius"] <= CUP_RADIUS[1]

    def around(stats, pad):
        return stats[0] - pad, stats[1] - pad, stats[0] + stats[2] + pad, stats[1] + stats[3] + pad

    def inside(box):
        return workspace & (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])

    cup_box = None
    red_blobs = blobs((red > 0) & workspace, 300)  # a flask in front can hide most of the cup
    if red_blobs:
        red_all = (red > 0) & workspace
        for mask, stats in blobs(workspace & (height > 40), 300):
            if (mask & red_all).sum() >= 200:
                rim = measure_rim(world, height, mask)
                if good_cup(rim):
                    obs.cup, cup_box = rim, around(stats, 5)
                    break
    if obs.cup is None:
        # From straight above the red wall is hidden, so fall back to shape: a tall ring of cup size.
        shapes = [(measure_rim(world, height, m), s) for m, s in blobs(workspace & (height > 50), 800)]
        shapes = [(r, s) for r, s in shapes if good_cup(r)]
        if len(shapes) == 1:
            obs.cup, cup_box = shapes[0][0], around(shapes[0][1], 10)
    if obs.cup is None:
        obs.problems.append("Red cup not found")

    def outside_cup(st):
        if cup_box is None:
            return True
        cx, cy = st[0] + st[2] / 2, st[1] + st[3] / 2
        return not (cup_box[0] <= cx <= cup_box[2] and cup_box[1] <= cy <= cup_box[3])

    for mask, stats in blobs((blue > 0) & workspace, 800):
        if not outside_cup(stats):
            continue
        mouth = measure_mouth(frame, world, height, n, c, mask)
        if FLASK_RIM_HEIGHT[0] <= mouth["height"] <= FLASK_RIM_HEIGHT[1] and FLASK_MOUTH_RADIUS[0] <= mouth["radius"] <= FLASK_MOUTH_RADIUS[1]:
            obs.vessels.append(mouth)
    if not obs.vessels:
        obs.problems.append("No blue-taped flask or beaker found")

    keep_out = [(v["x"], v["y"], 90) for v in obs.vessels]
    if obs.cup:
        keep_out.append((obs.cup["x"], obs.cup["y"], 100))
    obs.obstacles = find_depth_obstacles(world, height, keep_out)
    obs.frame = frame
    return obs


def annotate(frame, scene):
    """The camera image with every container numbered, as they are listed for choosing."""
    image = frame.color.copy()

    def to_pixel(point):
        p = frame.R.T @ (np.asarray(point) - frame.t)
        return p[0] / p[2] * frame.fx + frame.cx, p[1] / p[2] * frame.fy + frame.cy

    items = [(v, str(v["id"]), (255, 220, 0)) for v in scene.get("vessels", [])]
    if scene.get("cup"):
        items.append((scene["cup"], "Cup", (0, 255, 255)))
    for item, label, color in items:
        ring = [to_pixel((item["x"] + item["radius"] * np.cos(a), item["y"] + item["radius"] * np.sin(a), item["rim_z"]))
                for a in np.linspace(0, 2 * np.pi, 48)]
        cv2.polylines(image, [np.array(ring, np.int32)], True, color, 2, cv2.LINE_AA)
        u, v = map(int, to_pixel((item["x"], item["y"], item["rim_z"])))
        cv2.putText(image, label, (u - 12, v + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, label, (u - 12, v + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


async def measure(machine, name, frames=5):
    """Depth flickers on shiny tape, so use every frame of a short burst that found both objects."""
    captured = [await capture(machine, name) for _ in range(frames)]
    observations = await asyncio.to_thread(lambda: [analyze(f) for f in captured])
    good = [o for o in observations if o.vessels and o.cup] or observations[-1:]
    for o in good[:-1]:
        o.obstacles = []
    return good


def fuse(observations, nudges=None):
    """Average per-camera results into a scene the planner can use."""
    nudges = nudges or {}
    scene = dict(problems=[], disagreement_mm={}, cameras=sorted({o.camera for o in observations}))
    for o in observations:
        scene["problems"] += [f"{o.camera}: {p}" for p in o.problems]
    scene["table_z"] = float(np.mean([o.table_z for o in observations]))
    scene["table_tilt_deg"] = float(np.mean([o.table_tilt_deg for o in observations]))
    scene["obstacles"] = [b for o in observations for b in o.obstacles]
    fields = ("x", "y", "rim_z", "radius", "height", "fit_error")

    def combine(group):
        # Median within a camera ignores a bad frame; cameras are then averaged by how much each saw.
        per_camera = {}
        for camera, item in group:
            per_camera.setdefault(camera, []).append(item)
        seen = [{**{k: float(np.median([s[k] for s in items])) for k in fields}, "points": sum(s["points"] for s in items)}
                for items in per_camera.values()]
        weights = np.array([s["points"] for s in seen], float)
        merged = {k: float(np.average([s[k] for s in seen], weights=weights)) for k in fields}
        spread = max(np.hypot(a["x"] - merged["x"], a["y"] - merged["y"]) for a in seen)
        return merged, float(spread) if len(seen) > 1 else None

    def corrected(item, key):
        nudge = nudges.get(key, {})
        item["measured"] = {k: item[k] for k in ("x", "y", "rim_z")}
        item["x"] += nudge.get("x", 0.0)
        item["y"] += nudge.get("y", 0.0)
        item["rim_z"] += nudge.get("z", 0.0)
        return item

    cups = [(o.camera, o.cup) for o in observations if o.cup]
    scene["cup"] = None
    if cups:
        merged, spread = combine(cups)
        scene["cup"] = corrected(merged, "cup")
        if spread is not None:
            scene["disagreement_mm"]["cup"] = spread

    # The same container seen in several frames or cameras: group by where it stands.
    clusters = []
    for o in observations:
        for v in o.vessels:
            for cluster in clusters:
                if np.hypot(v["x"] - cluster[0][1]["x"], v["y"] - cluster[0][1]["y"]) < 45:
                    cluster.append((o.camera, v))
                    break
            else:
                clusters.append([(o.camera, v)])
    vessels = []
    for cluster in clusters:
        if len(cluster) >= max(1, len(observations) * 0.4):  # ignore something only one odd frame saw
            merged, spread = combine(cluster)
            vessels.append((merged, spread))
    vessels.sort(key=lambda pair: -pair[0]["y"])  # numbered in a steady order, along the table
    scene["vessels"] = []
    for number, (merged, spread) in enumerate(vessels, 1):
        key = str(number)
        if spread is not None:
            scene["disagreement_mm"][f"container {key}"] = spread
        scene["vessels"].append({**corrected(merged, key), "id": key, **vessel_type(merged["height"])})
    return scene


def vessel_type(height):
    """Container settings by measured height. Grip and base can't be seen through clear plastic."""
    for kind in SETTINGS["vessels"]:
        low, high = kind["rim_height_mm"]
        if low <= height <= high:
            return dict(profile=kind["name"], grip_height=kind["grip_height_above_table_mm"],
                        base_diameter=kind["base_diameter_mm"], expected_width=kind["expected_width_mm"], usable=True)
    return dict(profile=None, usable=False)
