"""One timestamped experiments tree for every generated run artifact."""

import datetime
import json
import os
from pathlib import Path
import re


TIMESTAMP_PATTERN = r"[0-9]{8}_[0-9]{6}(?:_[0-9]{6})?"


def resolve_output_dir(root, value=None):
    root = Path(root).resolve()
    experiments = (root / "experiments").resolve()
    if experiments != root / "experiments":
        raise ValueError("experiments must not redirect to a different directory")
    path = experiments if value is None else (root / Path(value)).resolve()
    if path == experiments:
        path = experiments / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not path.is_relative_to(experiments):
        raise ValueError("All outputs must stay in experiments/<timestamp>/")
    relative = path.relative_to(experiments)
    if not re.fullmatch(TIMESTAMP_PATTERN, relative.parts[0]):
        raise ValueError("Output directory must use experiments/YYYYMMDD_HHMMSS[_ffffff]/[task]")
    return path


def create_output_dir(root, value=None):
    path = resolve_output_dir(root, value)
    path.mkdir(parents=True, exist_ok=True)
    bootstrap_files = {"launcher.log", "launcher.pid", "console.log"}
    if any(item.name not in bootstrap_files or not item.is_file() or item.is_symlink()
           for item in path.iterdir()):
        raise FileExistsError("Output directory already contains results: " + str(path))
    # Exclusive claim permits shell-redirection logs but prevents accidental reruns.
    with (path / ".output_claim.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created_at": datetime.datetime.now().isoformat()}, handle)
    return path
