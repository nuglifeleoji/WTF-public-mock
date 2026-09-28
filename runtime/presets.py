"""Short names for the published selections; manifests remain the source of truth."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def task_presets() -> list[dict]:
    presets = []
    for folder, collection in (("Set1_tasks", "set1"), ("scale_up_tasks", "scale-up")):
        directory = ROOT / "Task" / folder
        manifest = json.loads((directory / "MANIFEST.json").read_text())
        for entry in manifest["sets"]:
            benchmark = entry["benchmark"].replace("swebench_lite", "swebench")
            name = Path(entry["json"]).stem.removeprefix(benchmark + "_")
            name = re.sub(r"_\d+$", "", name).replace("_", "-")
            presets.append({
                "benchmark": benchmark,
                "tasks": "set1" if collection == "set1" else f"scale-up/{name}",
                "n_tasks": entry["n_tasks"],
                "selection_group": entry.get("selection_group", ""),
                "task_file": str(directory / entry["json"]),
            })
    return presets


def resolve_tasks(benchmark: str, tasks: str) -> str:
    for preset in task_presets():
        if (preset["benchmark"], preset["tasks"]) == (benchmark, tasks):
            return preset["task_file"]
    raise ValueError(f"Unknown task selection {tasks!r} for {benchmark}; use {benchmark} --list")


def config_path(benchmark: str) -> Path:
    directory = ROOT / "configs"
    if not directory.is_dir():  # The wheel bundles the same examples inside runtime/.
        directory = Path(__file__).parent / "configs"
    return directory / f"{benchmark}.yaml"


def automatic_run_id(benchmark: str, selection: dict, config: dict) -> str:
    def slug(value):
        return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip(".-")[:64]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return "_".join(slug(x) for x in (
        benchmark, selection["set_id"], config["method"], config["model"], stamp,
    ))
