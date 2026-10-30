"""Record the current robot state without commanding any movement."""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from viam.components.arm import Arm
from viam.proto.common import Pose, PoseInFrame

from helpers import connect


async def record(label):
    async with await asyncio.wait_for(connect(), 30) as machine:
        arm = Arm.from_robot(machine, "arm")
        if await arm.is_moving(timeout=10):
            raise RuntimeError("Arm is moving. Record only after it has stopped.")
        joints = await arm.get_joint_positions(timeout=10)
        end = await arm.get_end_position(timeout=10)
        gripper = await asyncio.wait_for(machine.transform_pose(
            PoseInFrame(reference_frame="gripper", pose=Pose(o_z=1)), "world"
        ), 15)
        if await arm.is_moving(timeout=10):
            raise RuntimeError("Arm moved during recording; snapshot discarded.")
        data = {
            "label": label,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "joint_positions": MessageToDict(joints, preserving_proto_field_name=True),
            "arm_end_pose": MessageToDict(end, preserving_proto_field_name=True),
            "gripper_world_pose": MessageToDict(gripper, preserving_proto_field_name=True),
            "note": "Sequential state readings; not a validated movement target or trajectory.",
        }
        folder = Path(__file__).parent / "poses"
        folder.mkdir(exist_ok=True)
        path = folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Recorded: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="Descriptive name, for example initial-state")
    asyncio.run(record(parser.parse_args().label))
