"""Records each pour to MongoDB as a timed episode for training an Action Chunking Transformer.

Collections:
    episodes   one document per pour: task, scene, settings, outcome, counts
    timesteps  one document per sample: joints, velocities, gripper, camera JPEG

Set MONGODB_URI in .env to turn it on. Writing happens on a background thread: if MongoDB is slow
or unreachable, samples are dropped and counted. Recording must never hold up the robot.
"""
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone

from bson.binary import Binary
from pymongo import ASCENDING, MongoClient

SCHEMA_VERSION = 1
QUEUE_SIZE = 200
RETRY_SECONDS = 30


def utc_now():
    return datetime.now(timezone.utc)


class ActRecorder:
    def __init__(self, log=print, uri=None, database=None):
        self.log = log
        self.uri = uri or os.environ.get("MONGODB_URI")
        self.database = database or os.environ.get("MONGODB_DB", "witchpotion")
        self.queue = queue.Queue(QUEUE_SIZE)
        self.episode_id = None
        self.started = 0.0
        self.count = 0
        self.images = 0
        self.dropped = 0
        self.error = None
        self.lost_writes = 0
        if self.uri:
            threading.Thread(target=self._writer, daemon=True).start()

    @property
    def enabled(self):
        return bool(self.uri)

    @property
    def active(self):
        return self.episode_id is not None

    def status(self):
        return dict(enabled=self.enabled, active=self.active, samples=self.count, dropped=self.dropped + self.lost_writes, error=self.error)

    def begin(self, task, scene, settings, dry_run, rate_hz, cameras, brew_id):
        if not self.enabled:
            return
        self.end("abandoned")
        self.episode_id = uuid.uuid4().hex
        self.started = time.time()
        self.count = self.images = self.dropped = 0
        self._send(("episode", dict(
            _id=self.episode_id, schema_version=SCHEMA_VERSION, brew_id=brew_id, dry_run=bool(dry_run),
            started_at=utc_now(), ended_at=None, outcome="recording", operator_label=None,
            task=task, scene=scene, settings=settings, rate_hz=rate_hz, camera_names=cameras,
            qpos_names=["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
            units=dict(joints="degrees", gripper_position="0 closed to 1 open", gripper_command="0 close, 1 open"),
            action_definition="At export: next sample's joints, plus the gripper command. See export_act.py.")))

    def add(self, step_id, joints, velocity, gripper, gripper_command, frames):
        """frames: {camera: (jpeg bytes, age in seconds)}"""
        if not self.active:
            return
        now = time.time()
        document = dict(
            episode_id=self.episode_id, i=self.count, t=now - self.started, step_id=step_id,
            qpos=list(joints) + [gripper], qvel=list(velocity), gripper_command=gripper_command,
            images={name: dict(jpeg=Binary(jpeg), age_s=age) for name, (jpeg, age) in frames.items()})
        if self._send(("timestep", document)):
            self.count += 1
            self.images += len(frames)
        else:
            self.dropped += 1

    def end(self, outcome):
        """Safe to call at any time; only the first call after begin() counts."""
        if not self.active:
            return
        self._send(("end", dict(episode_id=self.episode_id, outcome=outcome, ended_at=utc_now(),
                                n_timesteps=self.count, n_images=self.images, dropped_timesteps=self.dropped,
                                duration_s=time.time() - self.started)), block=True)
        self.log(f"Recorded episode: {outcome}, {self.count} samples, {self.dropped} dropped")
        self.episode_id = None

    def flush(self, seconds=5):
        end = time.time() + seconds
        while self.enabled and not self.queue.empty() and time.time() < end:
            time.sleep(0.05)

    def _send(self, item, block=False):
        try:
            self.queue.put(item, timeout=0.2 if block else None) if block else self.queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def _writer(self):
        client = None
        while True:
            kind, document = self.queue.get()
            give_up = time.time() + RETRY_SECONDS
            while True:
                try:
                    if client is None:
                        client = MongoClient(self.uri, serverSelectionTimeoutMS=3000, appname="witchpotion")
                        db = client[self.database]
                        db.timesteps.create_index([("episode_id", ASCENDING), ("i", ASCENDING)], unique=True)
                        db.episodes.create_index([("started_at", ASCENDING)])
                    if kind == "episode":
                        db.episodes.replace_one({"_id": document["_id"]}, document, upsert=True)
                    elif kind == "timestep":
                        db.timesteps.replace_one({"episode_id": document["episode_id"], "i": document["i"]}, document, upsert=True)
                    else:
                        db.episodes.update_one({"_id": document["episode_id"]},
                                               {"$set": {k: v for k, v in document.items() if k != "episode_id"}})
                    if self.error:
                        self.log("MongoDB recording is working again")
                    self.error = None
                    break
                except Exception as error:
                    # Keep the robot running; report once, not on every retry. The URI can hold a password.
                    if not self.error:
                        self.log(f"MongoDB recording problem, retrying: {type(error).__name__}")
                    self.error = type(error).__name__
                    if time.time() > give_up:
                        self.lost_writes += 1
                        break
                    time.sleep(1)
