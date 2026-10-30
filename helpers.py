import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.gripper import Gripper
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.services.generic import Generic as GenericService

load_dotenv(Path(__file__).resolve().parent / ".env")

def _require(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=_require('VIAM_API_KEY'),
        api_key_id=_require('VIAM_API_KEY_ID')
    )
    # The SDK's check gives up after three 1-second timeouts and closes the connection, cancelling
    # anything running on it, including an arm move. Large depth frames can delay it that long.
    opts.check_connection_interval = 3600

    return await RobotClient.at_address(_require('VIAM_ADDRESS'), opts)

async def main():
    async with await connect() as machine:
        print('Resources:')
        print(machine.resource_names)
        
        # wall-back
        wall_back: Gripper = Gripper.from_robot(machine, "wall-back")
        wall_back_return_value = await wall_back.is_moving()
        print(f"wall-back is_moving return value: {wall_back_return_value}")

        # wall-left
        wall_left: Gripper = Gripper.from_robot(machine, "wall-left")
        wall_left_return_value = await wall_left.is_moving()
        print(f"wall-left is_moving return value: {wall_left_return_value}")

        # arm
        arm: Arm = Arm.from_robot(machine, "arm")
        arm_return_value = await arm.get_end_position()
        print(f"arm get_end_position return value: {arm_return_value}")

        # cam
        cam: Camera = Camera.from_robot(machine, "cam")
        cam_return_value = await cam.get_images()
        print(f"cam get_images return value: {cam_return_value}")

        # gripper
        gripper: Gripper = Gripper.from_robot(machine, "gripper")
        gripper_return_value = await gripper.is_moving()
        print(f"gripper is_moving return value: {gripper_return_value}")

        # table
        table: Gripper = Gripper.from_robot(machine, "table")
        table_return_value = await table.is_moving()
        print(f"table is_moving return value: {table_return_value}")

        # ceiling
        ceiling: Gripper = Gripper.from_robot(machine, "ceiling")
        ceiling_return_value = await ceiling.is_moving()
        print(f"ceiling is_moving return value: {ceiling_return_value}")

if __name__ == '__main__':
    asyncio.run(main())
