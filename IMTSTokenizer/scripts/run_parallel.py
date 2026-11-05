"""Run the four forecasting datasets with a dynamically replenished GPU queue."""

import argparse
from collections import deque
import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.run_paths import create_output_dir, resolve_output_dir
CONFIGS = {
    "physionet": dict(history=24, patch_size=12, num_var=41, hid_dim=24,
                     dropout=0.0, batch_size=32, num_layers=2, mile_stone="[27]"),
    "activity": dict(history=3000, patch_size=1500, num_var=12, hid_dim=12,
                    dropout=0.1, batch_size=32, num_layers=4, mile_stone="[42,50]"),
    "mimic": dict(history=24, patch_size=12, num_var=96, hid_dim=16,
                 dropout=0.2, batch_size=32, num_layers=2, mile_stone="[25]"),
    "ushcn": dict(history=24, patch_size=3, num_var=5, hid_dim=2,
                 dropout=0.0, batch_size=192, num_layers=5, mile_stone="[8]"),
}


def make_jobs(seeds):
    return [dict(dataset=dataset, seed=seed, status="queued")
            for seed in seeds for dataset in CONFIGS]


def command_for(job, gpu, output_dir):
    runner = job.get("runner", "train")
    if runner == "pipeline":
        return [sys.executable, "-u", str(ROOT / "IMTSTokenizer/run_pipeline.py"),
                "--recipe", str(ROOT / job["recipe"]), "--dataset", job["dataset"],
                "--seed", str(job["seed"]), "--gpu", str(gpu), "--device", "cuda",
                "--output-dir", str(output_dir)]
    if runner == "ridge":
        command = [sys.executable, "-u", str(ROOT / "IMTSTokenizer/fit_ridge.py"),
                   "--device", "cuda", "--gpu", str(gpu), "--evaluate-test",
                   "--expected-dataset", job["dataset"], "--expected-seed", str(job["seed"]),
                   "--output-dir", str(output_dir)]
        flags = {"init-checkpoint": "checkpoint", "ridge-mode": "modes", "ridge-grid": "ridge-grid",
                 "ridge-rank": "rank", "batch_size": "batch-size", "cpu-threads": "cpu-threads",
                 "uniform-ridge": "uniform-ridge"}
        for name, value in job.get("overrides", {}).items():
            flag = "--" + flags[name]
            if name == "init-checkpoint":
                value = value.format(seed=job["seed"], dataset=job["dataset"])
            if name == "ridge-grid":
                command.extend([flag] + [str(strength) for strength in value])
            elif isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.extend([flag, str(value)])
        return command
    if runner != "train":
        raise ValueError("Unsupported task runner: " + str(runner))
    command = [sys.executable, "-u", str(ROOT / "IMTSTokenizer/run_models.py"),
               "--dataset", job["dataset"], "--seed", str(job["seed"]), "--gpu", str(gpu),
               "--epoch", "1000", "--patience", "10", "--lr", "0.001",
               "--diagnostics", "on", "--output-dir", str(output_dir)]
    for name, value in CONFIGS[job["dataset"]].items():
        command.extend(["--" + name, str(value)])
    for name, value in job.get("overrides", {}).items():
        if name == "init-checkpoint":
            value = value.format(seed=job["seed"], dataset=job["dataset"])
        if name == "correction-variables" and isinstance(value, list):
            command.extend(["--" + name] + [str(index) for index in value])
        elif isinstance(value, bool):
            if value:
                command.append("--" + name)
        else:
            command.extend(["--" + name, str(value)])
    return command


def jobs_from_plan(path, seeds):
    from IMTSTokenizer.run_models import make_parser, validate_args
    profiles = json.loads(Path(path).read_text(encoding="utf-8"))["profiles"]
    allowed = {"variant", "scheduler", "patience", "min-epochs", "lr-patience", "lr-factor", "min-lr",
               "lr", "w_decay", "grad-clip", "loss-reduction", "num_layers", "hid_dim", "dropout", "batch_size",
               "qk-normalization", "source-reliability", "source-reliability-mode", "target-query-mode", "temporal-correction", "correction-only", "dataset-storage", "init-checkpoint",
               "seasonal-period", "seasonal-bandwidth", "correction-variables"}
    ridge_allowed = {"init-checkpoint", "ridge-mode", "ridge-grid", "ridge-rank", "batch_size",
                     "cpu-threads", "uniform-ridge"}
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Plan must contain a nonempty profiles array")
    names = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("Plan profiles must be objects")
        name = profile["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("Plan profile names must be unique filesystem-safe identifiers")
        names.add(name)
        runner = profile.get("runner", "train")
        if not isinstance(runner, str) or runner not in {"train", "ridge", "pipeline"}:
            raise ValueError("Plan runner must be train, ridge or pipeline")
        overrides = profile.get("overrides", {})
        permitted = ridge_allowed if runner == "ridge" else allowed
        if (profile["dataset"] not in CONFIGS or not isinstance(overrides, dict)
                or not set(overrides).issubset(permitted)):
            raise ValueError("Plan contains an unsupported dataset or option")
        if runner == "pipeline":
            value = profile.get("recipe")
            if not isinstance(value, str) or not value or overrides:
                raise ValueError("Pipeline profiles require a recipe path and no flat overrides")
            recipe_path = (ROOT / value).resolve()
            if not recipe_path.is_relative_to(ROOT.resolve()) or not recipe_path.is_file():
                raise ValueError("Pipeline recipe must exist inside this project: " + value)
        if runner == "ridge":
            if not {"init-checkpoint", "ridge-mode"}.issubset(overrides):
                raise ValueError("Ridge profiles require init-checkpoint and one ridge-mode")
            if not isinstance(overrides["ridge-mode"], str) or overrides["ridge-mode"] not in {"history", "readout", "hybrid"}:
                raise ValueError("ridge-mode must be one of history, readout or hybrid")
        if "seeds" in profile:
            selected = profile["seeds"]
            if (not isinstance(selected, list) or not selected or
                    any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in selected) or
                    len(set(selected)) != len(selected)):
                raise ValueError("Profile seeds must be a nonempty array of unique valid seed integers")
        boolean_flags = {"uniform-ridge"} if runner == "ridge" else {"source-reliability", "temporal-correction", "correction-only"}
        for option, value in overrides.items():
            if (option in boolean_flags) != isinstance(value, bool):
                raise ValueError("Boolean flags require JSON booleans; numeric/string options must not be booleans")
            if option == "init-checkpoint" and not isinstance(value, str):
                raise ValueError("init-checkpoint must be a path string")
            if runner == "ridge" and option in {"ridge-rank", "batch_size", "cpu-threads"} and type(value) is not int:
                raise ValueError(option + " must be an integer")
            if option == "ridge-grid" and (not isinstance(value, list) or not value or
                    any(type(strength) not in {int, float} or not math.isfinite(strength) or strength <= 0
                        for strength in value)):
                raise ValueError("ridge-grid must be a nonempty array of finite positive numbers")
            if option == "correction-variables" and (not isinstance(value, list) or not value or any(type(index) is not int for index in value)):
                raise ValueError("correction-variables must be a nonempty integer array")
    jobs = [dict(dataset=p["dataset"], seed=seed, profile=p["name"],
                 runner=p.get("runner", "train"), overrides=p.get("overrides", {}), status="queued",
                 **({"recipe": p["recipe"]} if p.get("runner") == "pipeline" else {})) for seed in seeds for p in profiles
            if "seeds" not in p or seed in p["seeds"]]
    if not jobs:
        raise ValueError("No tasks selected by the command and profile seeds")
    for job in jobs:
        arguments = command_for(job, 0, ROOT / "experiments/20000101_000000/plan_validation")
        if job["runner"] == "pipeline":
            from IMTSTokenizer.run_pipeline import validate_recipe
            recipe = json.loads((ROOT / job["recipe"]).read_text(encoding="utf-8"))
            validate_recipe(recipe, job["dataset"], job["seed"], "cuda")
        elif job["runner"] == "ridge":
            from IMTSTokenizer.fit_ridge import make_parser as ridge_parser, validate_options
            validate_options(ridge_parser().parse_args(arguments[3:]))
        else:
            validate_args(make_parser().parse_args(arguments[3:]))
    return jobs


def validate_job_inputs(jobs):
    """Check selected initialization assets before any worker or output starts."""
    missing = []
    for job in jobs:
        template = job.get("overrides", {}).get("init-checkpoint")
        if template is None:
            continue
        value = template.format(seed=job["seed"], dataset=job["dataset"])
        path = (ROOT / value).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Initialization checkpoint must stay inside this project: " + value)
        if not path.is_file():
            missing.append(value)
    if missing:
        raise ValueError("Missing initialization checkpoint(s):\n  " + "\n  ".join(missing) +
                         "\nExtract the complete checkpoint-inclusive update package before launching.")


def dataset_limits(values):
    """Parse optional per-dataset concurrency limits for each physical GPU."""
    limits = {}
    for value in values:
        name, separator, raw_limit = value.partition("=")
        if not separator or name not in CONFIGS or name in limits:
            raise ValueError("Dataset limits require unique DATASET=COUNT entries")
        try:
            limit = int(raw_limit)
        except ValueError as error:
            raise ValueError("Dataset concurrency counts must be positive integers") from error
        if limit < 1 or str(limit) != raw_limit:
            raise ValueError("Dataset concurrency counts must be positive integers")
        limits[name] = limit
    return limits


def dispatch(jobs, slots, launch, changed, stopping=lambda: False, poll_seconds=0.5,
             per_dataset_limits=None):
    """Each slot owns at most one child; completion frees exactly that slot."""
    waiting = deque(jobs)
    running = {}
    limits = per_dataset_limits or {}
    if any(name not in CONFIGS or type(limit) is not int or limit < 1
           for name, limit in limits.items()):
        raise ValueError("Invalid per-dataset GPU concurrency limits")
    try:
        while waiting or running:
            if stopping():
                break
            for slot, gpu in enumerate(slots):
                if not waiting or stopping():
                    break
                if slot in running:
                    continue
                # Skip incompatible jobs without blocking later lightweight work.
                eligible = next((index for index, candidate in enumerate(waiting)
                    if sum(active["dataset"] == candidate["dataset"] and active["gpu"] == gpu
                           for active, _ in running.values())
                    < limits.get(candidate["dataset"], len(slots) + 1)), None)
                if eligible is None:
                    continue
                job = waiting[eligible]
                del waiting[eligible]
                job.update(gpu=gpu, slot=slot, started_at=datetime.datetime.now().isoformat())
                try:
                    process = launch(job, gpu)
                except OSError as error:
                    job.update(status="failed", error=str(error), returncode=None)
                    changed(job)
                    continue
                job.update(status="running", pid=process.pid)
                running[slot] = (job, process)
                changed(job)
            completed = False
            for slot, (job, process) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                job.update(status="succeeded" if code == 0 else "failed", returncode=code,
                           finished_at=datetime.datetime.now().isoformat())
                del running[slot]
                changed(job)
                completed = True
            if running and not completed and not stopping():
                time.sleep(poll_seconds)
    finally:
        # Only stop children owned by this launcher, then reap every child.
        for job, process in running.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 10
        for job, process in running.values():
            try:
                code = process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            job.update(status="cancelled", returncode=code,
                       finished_at=datetime.datetime.now().isoformat())
            changed(job)
        for job in waiting:
            job["status"] = "cancelled"
            changed(job)
    return 0 if all(job["status"] == "succeeded" for job in jobs) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 4, 5, 6, 7])
    parser.add_argument("--per-gpu", type=int, default=2)
    parser.add_argument("--dataset-per-gpu", nargs="+", default=[], metavar="DATASET=COUNT",
                        help="Optional per-dataset limits inside each GPU's total slots")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--plan", type=Path, help="Optional JSON profiles for controlled candidate experiments")
    parser.add_argument("--output-dir", help="Unified experiments/<timestamp> queue directory")
    parser.add_argument("--dry-run", action="store_true", help="Print the task plan without starting training")
    args = parser.parse_args(argv)
    if args.per_gpu < 1 or any(gpu < 0 for gpu in args.gpus) or len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU IDs must be unique and nonnegative; per-gpu must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    if any(not 0 <= seed < 2 ** 32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32)")
    try:
        concurrency_limits = dataset_limits(args.dataset_per_gpu)
    except ValueError as error:
        parser.error(str(error))
    try:
        queue_dir = resolve_output_dir(ROOT, args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    if args.plan:
        plan_path = args.plan.resolve()
        if not plan_path.is_relative_to(ROOT):
            parser.error("Plan must stay inside this project")
        try:
            jobs = jobs_from_plan(plan_path, args.seeds)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            parser.error(str(error))
    else:
        jobs = make_jobs(args.seeds)
    try:
        validate_job_inputs(jobs)
    except (OSError, ValueError, KeyError, IndexError) as error:
        parser.error(str(error))
    slots = args.gpus * args.per_gpu
    if args.dry_run:
        print(json.dumps({"gpus": args.gpus, "slots_per_gpu": args.per_gpu,
                          "dataset_per_gpu": concurrency_limits,
                          "max_parallel": len(slots), "output_dir": str(queue_dir),
                          "tasks": jobs, "configs": CONFIGS}, indent=2))
        return 0

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
    # Fail before scheduling if this environment would silently fall back to CPU.
    probe = ("import json,torch; n=torch.cuda.device_count(); "
             f"assert torch.cuda.is_available() and n == {len(args.gpus)}, "
             f"'Requested GPUs unavailable: expected {len(args.gpus)}, visible=' + str(n); "
             f"print(json.dumps({{'physical_gpus': {args.gpus!r}, 'devices': ["
             "{'logical_index': i, 'name': torch.cuda.get_device_name(i), "
             "'free_bytes': torch.cuda.mem_get_info(i)[0], "
             "'total_bytes': torch.cuda.mem_get_info(i)[1]} for i in range(n)]}))")
    queue_dir = create_output_dir(ROOT, queue_dir)
    queue_id = queue_dir.name
    if args.plan:
        (queue_dir / "plan.json").write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def changed(job=None):
        status = dict(queue_id=queue_id, gpus=args.gpus, per_gpu=args.per_gpu,
                      dataset_per_gpu=concurrency_limits,
                      plan=str(args.plan) if args.plan else None, jobs=jobs)
        temporary = queue_dir / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(queue_dir / "status.json")
        if job is not None:
            label = job.get("profile", job["dataset"])
            print(f"{job['status']:10s} {label:9s} seed={job['seed']} "
                  f"gpu={job.get('gpu')} pid={job.get('pid')} code={job.get('returncode')}", flush=True)

    def launch(job, gpu):
        name = f"{job.get('profile', job['dataset'])}_seed{job['seed']}"
        output_dir = queue_dir / name
        output_dir.mkdir(parents=True, exist_ok=False)
        command = command_for(job, gpu, output_dir)
        log_path = output_dir / "console.log"
        job.update(command=command, log=str(log_path), diagnostics=str(output_dir),
                   checkpoints=str(output_dir), output_dir=str(output_dir))
        child_env = env.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open("wb") as output:
            return subprocess.Popen(command, cwd=ROOT, env=child_env,
                                    stdout=output, stderr=subprocess.STDOUT)

    print(f"Queue: {queue_dir}\nTasks: {len(jobs)}; slots: {len(slots)}", flush=True)
    changed()
    with (queue_dir / "gpu_probe.log").open("wb") as probe_log:
        try:
            subprocess.run([sys.executable, "-c", probe], env=env, check=True,
                           stdout=probe_log, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as error:
            for job in jobs:
                job.update(status="cancelled", error="GPU preflight failed: " + str(error))
            changed()
            raise
    code = dispatch(jobs, slots, launch, changed, stopping=lambda: stopped,
                    per_dataset_limits=concurrency_limits)
    print(f"Finished: {sum(job['status'] == 'succeeded' for job in jobs)}/{len(jobs)} succeeded", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
