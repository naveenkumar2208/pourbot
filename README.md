# WitchPotion

You play the wizard. Choose the potions in a Halloween-themed web app, and a robot arm brews them: it picks up each blue-taped flask or beaker in the order you chose, pours it into a red cup, and puts it back.

Built on [Viam](https://www.viam.com/) with a UFactory xArm 850 and a wrist-mounted Intel RealSense D435, for the Viam Oct SF Hackathon.

## How it works

1. **See:** the camera looks at the table from the home and overhead positions. It finds the blue tape and the red cup, then measures position and height from depth.
2. **Choose:** each container is matched to a type in `pourbot/settings.json` by its measured height. You pick the ones to mix, in order.
3. **Rehearse:** every move is planned around the table, walls, camera and other containers before the arm moves.
4. **Brew:** press **Play** to run the whole brew, or run one step at a time. For each container the arm grips it, lifts it, carries it over the cup, tilts in stages, and returns it.
5. **Record:** each run is logged to SQLite. If MongoDB is set up, each pour is also saved as a training episode for an Action Chunking Transformer (ACT).

## Hardware and machine setup

- UFactory xArm 850 with gripper
- Intel RealSense D435 on the wrist
- A Viam machine with components named `arm`, `gripper` and `cam`, plus the `builtin` motion service
- Containers marked with blue tape, and a red cup

The hardware config used for this project is in `machine-config.json` and `hardware-fragment.json`.

## Setup

Requires Python 3.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with the values from your machine's **CONNECT** tab in the Viam app. Never commit this file.

| Variable | Required | Purpose |
| --- | --- | --- |
| `VIAM_API_KEY` | Yes | Machine API key |
| `VIAM_API_KEY_ID` | Yes | Machine API key ID |
| `VIAM_ADDRESS` | Yes | Machine address |
| `MONGODB_URI` | No | Turns on recording for training |
| `MONGODB_DB` | No | Database name, `witchpotion` by default |

## Run

Try a dry run first. It uses the real camera and planner but sends no motion or gripper commands:

```bash
.venv/bin/python pourbot/server.py --dry-run
```

To drive the real arm:

```bash
.venv/bin/python pourbot/server.py
```

Then open http://127.0.0.1:8765. Use `--port` to change the port.

## Safety

This app moves a real robot arm.

- Keep a hand on the hardware e-stop. The app's **Stop** button is not instant, and the arm can keep moving for about 2 seconds after a failure.
- Only one client should control the arm at a time. Close the Viam app's control tab and stop other scripts during a run.
- A collision is still possible. The saved positions and container sizes were tuned for one table setup, so check them against yours and watch the first runs closely.
- Closing the browser during motion stops the arm.

## Adapting it to your setup

Save the arm's current position as home (this sends no motion):

```bash
.venv/bin/python pourbot/set_home.py
```

Settings are stored in three files in `pourbot/`:

| File | Contents |
| --- | --- |
| `settings.json` | Container types, grip heights, and pour settings |
| `home.json` | Home joint angles |
| `approach.json` | Taught joint angles for approaching a container |

To add a container type, add an entry under the vessels in `settings.json` with its height range and grip, pour and width measurements.

Workflows save and restore all three files. Restart the server after `use`.

```bash
.venv/bin/python pourbot/workflow.py list
.venv/bin/python pourbot/workflow.py save NAME
.venv/bin/python pourbot/workflow.py use NAME
```

## Recording data for training

Run records are saved to `pourbot/runs.sqlite`, which git ignores: events, steps, scenes, plans, poses and images.

With `MONGODB_URI` set, each pour is recorded at 5 Hz:

- `episodes`: one document per pour, with its task, scene, settings and outcome
- `timesteps`: one document per sample, with joints, velocities, gripper and a camera JPEG

If MongoDB is slow or unreachable, samples are dropped and counted so the robot is never held up.

Export episodes to the HDF5 layout used by ACT's ALOHA training code:

```bash
.venv/bin/python pourbot/export_act.py --out data/witchpotion
```

Only completed real pours are exported unless you add `--include-dry-run`. Use `--width` to set the image width.

## Project layout

| Path | Purpose |
| --- | --- |
| `helpers.py` | Connects to the machine using `.env` |
| `pourbot/server.py` | Web server, brew steps and run control |
| `pourbot/perception.py` | Finds containers and the cup from color and depth |
| `pourbot/planner.py` | Grip poses, obstacles and motion planning |
| `pourbot/geometry.py` | Pose and orientation math |
| `pourbot/runlog.py` | SQLite run log |
| `pourbot/act_recorder.py` | MongoDB episode recorder |
| `pourbot/export_act.py` | Exports episodes to HDF5 |
| `pourbot/workflow.py` | Saves and restores setups |
| `pourbot/static/` | Web interface |
| `pourbot/workflows/` | Saved setups |
| `planning/`, `poses/`, `record_pose.py` | Early experiments and recorded poses |

## Known limitations

- Moves sometimes fail with `context canceled`. The cause isn't confirmed, but another client commanding the arm is the leading suspect. The app halts when this happens: press **Clear stop**, then **Home**.
- The gripper can still hit a beaker on approach, because of position error in the overhead measurement.
- Planning can fail for containers close to the arm's base. If that happens, move them out to around x = 585 mm.
- The ACT export has only been tested on the data format. No model has been trained on it yet.
