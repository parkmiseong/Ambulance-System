"""Seed packaged models once, retain all server results, then exec Uvicorn."""
import os
from pathlib import Path
import shutil
import sys


def seed_results(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for original in source.rglob("*"):
        if not original.is_file() or original.name == ".keep":
            continue
        target = destination / original.relative_to(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)


if __name__ == "__main__":
    import sumo
    os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
    seed_results("/app/seed-results", "/app/results")
    if len(sys.argv) < 2:
        raise SystemExit("A server command is required.")
    os.execvp(sys.argv[1], sys.argv[1:])
