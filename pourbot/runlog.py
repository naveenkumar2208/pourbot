"""SQLite record of PourBot runs. A recording failure must never interfere with the robot."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "runs.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, started_at TEXT, ended_at TEXT, source TEXT, dry_run INTEGER,
    settings_json TEXT, home_json TEXT, approach_json TEXT, final_state_json TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES runs(id), at TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES runs(id), step_id TEXT, title TEXT, kind TEXT,
    started_at TEXT, finished_at TEXT, outcome TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES runs(id), at TEXT, step_id TEXT,
    flask_x REAL, flask_y REAL, flask_z REAL, cup_x REAL, cup_y REAL, cup_z REAL, table_z REAL, scene_json TEXT);
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES runs(id), at TEXT, step_id TEXT, note TEXT,
    seconds REAL, waypoint_count INTEGER, waypoints_json TEXT, targets_json TEXT);
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES runs(id), at TEXT, step_id TEXT, moving INTEGER,
    j1 REAL, j2 REAL, j3 REAL, j4 REAL, j5 REAL, j6 REAL,
    x REAL, y REAL, z REAL, o_x REAL, o_y REAL, o_z REAL, theta REAL, gripper_position REAL, holding INTEGER);
CREATE TABLE IF NOT EXISTS poses (
    id INTEGER PRIMARY KEY, label TEXT, recorded_at TEXT, source TEXT UNIQUE, joints_json TEXT, gripper_pose_json TEXT);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES runs(id), at TEXT, step_id TEXT, camera TEXT, jpeg BLOB);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY, path TEXT, modified_at TEXT, size INTEGER, sha256 TEXT, content BLOB, UNIQUE(path, sha256));
"""


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class RunLog:
    def __init__(self, path=DB_PATH, on_error=print):
        self.on_error = on_error
        self.run_id = None
        self.step_row = None
        self.db = None
        try:
            self.db = sqlite3.connect(path)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
        except sqlite3.Error as error:
            self._report(error)

    def _report(self, error):
        try:
            self.on_error(f"Run log problem, recording may be incomplete: {error}")
        except Exception:
            pass

    def _write(self, sql, values=()):
        if self.db is None:
            return None
        try:
            cursor = self.db.execute(sql, values)
            self.db.commit()
            return cursor.lastrowid
        except (sqlite3.Error, TypeError, ValueError) as error:
            self._report(error)
            return None

    def start_run(self, source, dry_run, settings=None, home=None, approach=None, started_at=None):
        self.end_run()
        self.run_id = self._write(
            "INSERT INTO runs (started_at, source, dry_run, settings_json, home_json, approach_json) VALUES (?,?,?,?,?,?)",
            (started_at or now(), source, int(dry_run), json.dumps(settings), json.dumps(home), json.dumps(approach)))
        return self.run_id

    def end_run(self, final_state=None):
        if self.run_id is not None:
            self._write("UPDATE runs SET ended_at=?, final_state_json=? WHERE id=?",
                        (now(), json.dumps(final_state), self.run_id))
            self.run_id = None

    def event(self, text, at=None):
        self._write("INSERT INTO events (run_id, at, text) VALUES (?,?,?)", (self.run_id, at or now(), text))

    def step_started(self, step_id, title, kind):
        self.step_row = self._write("INSERT INTO steps (run_id, step_id, title, kind, started_at) VALUES (?,?,?,?,?)",
                                    (self.run_id, step_id, title, kind, now()))

    def step_finished(self, outcome, result=None):
        if self.step_row is not None:
            self._write("UPDATE steps SET finished_at=?, outcome=?, result=? WHERE id=?",
                        (now(), outcome, result, self.step_row))
            self.step_row = None

    def scene(self, step_id, scene, at=None):
        flask, cup = (scene.get("vessels") or [{}])[0], scene.get("cup") or {}
        self._write("INSERT INTO scenes (run_id, at, step_id, flask_x, flask_y, flask_z, cup_x, cup_y, cup_z, table_z, scene_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (self.run_id, at or now(), step_id, flask.get("x"), flask.get("y"), flask.get("rim_z"),
                     cup.get("x"), cup.get("y"), cup.get("rim_z"), scene.get("table_z"), json.dumps(scene)))

    def plan(self, step_id, plan, seconds=None):
        waypoints = plan.get("waypoints") or []
        self._write("INSERT INTO plans (run_id, at, step_id, note, seconds, waypoint_count, waypoints_json, targets_json) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (self.run_id, now(), step_id, plan.get("note"), seconds, len(waypoints) or plan.get("waypoint_count"),
                     json.dumps(waypoints), json.dumps(plan.get("targets"))))

    def sample(self, step_id, status):
        joints, pose = status.get("joints"), status.get("gripper_pose")
        if not joints or not pose:
            return
        self._write("INSERT INTO samples (run_id, at, step_id, moving, j1, j2, j3, j4, j5, j6, x, y, z, o_x, o_y, o_z, theta, "
                    "gripper_position, holding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self.run_id, now(), step_id, int(bool(status.get("moving"))), *joints,
                     pose["x"], pose["y"], pose["z"], pose["o_x"], pose["o_y"], pose["o_z"], pose["theta"],
                     status.get("gripper_position"), int(bool(status.get("holding")))))

    def image(self, step_id, camera, jpeg):
        self._write("INSERT INTO images (run_id, at, step_id, camera, jpeg) VALUES (?,?,?,?,?)",
                    (self.run_id, now(), step_id, camera, jpeg))

    def pose(self, label, recorded_at, source, joints, gripper_pose):
        self._write("INSERT OR REPLACE INTO poses (label, recorded_at, source, joints_json, gripper_pose_json) VALUES (?,?,?,?,?)",
                    (label, recorded_at, source, json.dumps(joints), json.dumps(gripper_pose)))

    def file(self, path, root):
        content = Path(path).read_bytes()
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(Path(path).stat().st_mtime))
        self._write("INSERT OR IGNORE INTO files (path, modified_at, size, sha256, content) VALUES (?,?,?,?,?)",
                    (str(Path(path).resolve().relative_to(Path(root).resolve())), stamp, len(content), hashlib.sha256(content).hexdigest(), content))
