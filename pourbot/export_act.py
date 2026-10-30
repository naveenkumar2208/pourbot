"""Export recorded pours from MongoDB to the HDF5 layout used by ACT's ALOHA training code.

    .venv/bin/python pourbot/export_act.py --out data/witchpotion [--width 640] [--include-dry-run]

Each file holds /observations/qpos, /observations/qvel, /observations/images/<camera> and /action.
Joints are converted to radians. The action at t is the joints at t+1 plus the gripper command at t.
Only completed real pours are exported unless asked otherwise.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

import cv2
import h5py
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient


def decode(jpeg, width):
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    height = round(image.shape[0] * width / image.shape[1])
    return cv2.cvtColor(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def to_arrays(steps, cameras, width):
    """Only samples with an image from every camera are usable for training."""
    usable = [s for s in steps if all(c in s["images"] for c in cameras)]
    if len(usable) < 2:
        return None
    qpos = np.array([s["qpos"] for s in usable], float)
    qpos[:, :6] = np.radians(qpos[:, :6])
    qvel = np.array([s["qvel"] + [0.0] for s in usable], float)
    qvel[:, :6] = np.radians(qvel[:, :6])
    commands = np.array([s["gripper_command"] for s in usable], float)
    action = np.column_stack((np.vstack((qpos[1:, :6], qpos[-1:, :6])), commands))
    images = {c: np.stack([decode(s["images"][c]["jpeg"], width) for s in usable]) for c in cameras}
    return qpos, qvel, action, images


def main():
    load_dotenv(ROOT.parent / ".env")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--include-dry-run", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("MONGODB_URI"):
        raise SystemExit("MONGODB_URI is not set. Add it to .env.")
    db = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=5000)[os.environ.get("MONGODB_DB", "witchpotion")]
    query = {"outcome": "completed", "operator_label": {"$ne": "bad"}}
    if not args.include_dry_run:
        query["dry_run"] = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for episode in db.episodes.find(query).sort("started_at", 1):
        steps = list(db.timesteps.find({"episode_id": episode["_id"]}).sort("i", 1))
        arrays = to_arrays(steps, episode["camera_names"], args.width)
        if arrays is None:
            skipped += 1
            continue
        qpos, qvel, action, images = arrays
        with h5py.File(out / f"episode_{written}.hdf5", "w") as f:
            f.attrs["sim"] = False
            f.attrs["compress"] = False
            f.attrs["episode_id"] = episode["_id"]
            f.attrs["rate_hz"] = episode["rate_hz"]
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=qpos)
            obs.create_dataset("qvel", data=qvel)
            group = obs.create_group("images")
            for name, frames in images.items():
                group.create_dataset(name, data=frames, dtype="uint8", chunks=(1, *frames.shape[1:]), compression="gzip", compression_opts=1)
            f.create_dataset("action", data=action)
        written += 1
    print(f"Wrote {written} episodes to {out}; skipped {skipped} without enough images.")


if __name__ == "__main__":
    main()
