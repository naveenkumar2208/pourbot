"""Save the arm's current position as PourBot's home. Commands no motion."""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

from viam.components.arm import Arm
from viam.proto.common import Pose, PoseInFrame

from helpers import connect
from perception import SETTINGS


async def main():
    async with await asyncio.wait_for(connect(), 30) as machine:
        arm = Arm.from_robot(machine, SETTINGS["arm"])
        if await arm.is_moving(timeout=10):
            raise SystemExit("The arm is moving. Set home only after it has stopped.")
        joints = list((await arm.get_joint_positions(timeout=10)).values)
        pose = (await machine.transform_pose(PoseInFrame(reference_frame=SETTINGS["gripper"], pose=Pose(o_z=1)), "world")).pose
        home = dict(recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), joints_deg=joints,
                    gripper_pose=dict(x=pose.x, y=pose.y, z=pose.z, o_x=pose.o_x, o_y=pose.o_y, o_z=pose.o_z, theta=pose.theta))
        (ROOT / "home.json").write_text(json.dumps(home, indent=2) + "\n")
        print("Home saved:", [round(v, 1) for v in joints])


if __name__ == "__main__":
    asyncio.run(main())
