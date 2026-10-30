"""Save and restore named PourBot setups. Restoring only changes files; nothing moves.

    .venv/bin/python pourbot/workflow.py list
    .venv/bin/python pourbot/workflow.py save NAME [--episode snapshot.json]
    .venv/bin/python pourbot/workflow.py use NAME

Restart the server after `use` so it loads the restored setup.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "workflows"
SETUP_FILES = ("settings.json", "home.json", "approach.json")


def save(name, episode_file):
    folder = STORE / name
    if folder.exists():
        raise SystemExit(f"Workflow {name} already exists. Choose a new name so it isn't overwritten.")
    folder.mkdir(parents=True)
    for file in SETUP_FILES:
        shutil.copy2(ROOT / file, folder / file)
    info = dict(name=name, saved_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if episode_file:
        state = json.loads(Path(episode_file).read_text())
        info["episode"] = dict(
            dry_run=state["dry_run"], scene=state["scene"], corrections=state["nudges"], log=state["log"],
            steps=[dict(id=s["id"], title=s["title"], state=s["state"], result=s["result"], plan=s["plan"]) for s in state["steps"]])
    (folder / "workflow.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Saved {folder.relative_to(ROOT.parent)}")


def use(name):
    folder = STORE / name
    if not folder.exists():
        raise SystemExit(f"No workflow named {name}.")
    for file in SETUP_FILES:
        shutil.copy2(folder / file, ROOT / file)
    print(f"Restored {name}. Restart the server to load it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("list", "save", "use"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--episode", help="Server state to keep with the workflow")
    args = parser.parse_args()
    if args.action == "list":
        for folder in sorted(STORE.iterdir()) if STORE.exists() else []:
            print(folder.name)
    elif not args.name:
        parser.error("a workflow name is required")
    elif args.action == "save":
        save(args.name, args.episode)
    else:
        use(args.name)


if __name__ == "__main__":
    main()
