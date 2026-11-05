"""Run backbone training, temporal correction and readout fitting."""

import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.checkpoint_names import load_checkpoint
from lib.run_paths import create_output_dir, resolve_output_dir


TRAIN_OPTIONS = {
    "variant", "epoch", "patience", "lr", "history", "patch_size", "w_decay",
    "batch_size", "num_var", "quantization", "hid_dim", "num_layers", "dropout",
    "mile_stone", "scheduler", "lr-patience", "lr-factor", "min-lr", "min-epochs",
    "cpu-threads", "qk-normalization", "source-reliability", "temporal-correction",
    "correction-only", "seasonal-period", "seasonal-bandwidth", "correction-variables",
    "dataset-storage", "loss-reduction", "grad-clip", "diagnostics", "diag-every",
    "diag-first-steps", "diag-max-elements", "diag-snapshots", "max-train-batches",
    "max-eval-batches", "detect-anomaly", "test-policy", "eval-rng-policy",
    "variable-head", "variable-head-rank", "variable-head-penalty",
    "artifact-storage", "source-reliability-mode", "target-query-mode",
}
CORRECT_OPTIONS = TRAIN_OPTIONS - {
    "variant", "history", "patch_size", "num_var", "quantization", "hid_dim",
    "num_layers", "dropout", "qk-normalization", "source-reliability", "source-reliability-mode", "target-query-mode",
    "variable-head", "variable-head-rank",
}
RIDGE_OPTIONS = {
    "ridge-mode", "ridge-modes", "ridge-grid", "ridge-rank", "batch_size", "cpu-threads",
    "uniform-ridge", "evaluate-test", "diagnostics", "max-train-batches",
    "max-val-batches",
}
READOUT_OPTIONS = {
    "readout-modes", "ridge-grid", "global-ridge-grid", "ridge-rank", "batch_size", "cpu-threads",
    "extra-rank", "nested-rank",
    "adaptive-ridge-grid", "support-power", "context-penalty-grid", "reliability-diagnostics",
    "shrink-grid", "shrink-min-count",
    "bias-penalty", "group-folds", "cv-gain-grid", "cv-min-count", "candidate-storage", "feature-cache",
    "artifact-storage",
    "chunk-rows", "uniform-ridge", "evaluate-test", "diagnostics", "max-train-batches", "max-val-batches",
}
BOOLEAN_OPTIONS = {
    "source-reliability", "temporal-correction", "correction-only", "detect-anomaly",
    "uniform-ridge", "evaluate-test",
}
IDENTITY_FIELDS = (
    "dataset", "seed", "variant", "history", "patch_size", "num_var", "hid_dim",
    "num_layers", "dropout", "qk_normalization", "source_reliability", "source_reliability_mode", "target_query_mode",
    "temporal_correction", "correction_only", "seasonal_period", "seasonal_bandwidth",
    "correction_variables", "ridge_mode", "ridge_rank",
    "variable_head", "variable_head_rank", "variable_head_penalty", "fitted_readout",
    "readout_basis", "readout_extra_rank", "readout_shrinkage",
)

CLI_CORE_OPTIONS = {
    "hid_dim": int, "num_layers": int, "dropout": float, "history": int,
    "patch_size": float, "num_var": int, "quantization": float,
}
CLI_TRAIN_OPTIONS = {
    "lr": float, "w_decay": float, "patience": int, "scheduler": str,
    "lr-patience": int, "lr-factor": float, "min-lr": float,
    "min-epochs": int, "grad-clip": float,
}


def apply_cli_overrides(recipe, options):
    """Apply explicit command-line values to the dataset's stage settings."""
    result = copy.deepcopy(recipe)
    if not isinstance(result, dict) or not isinstance(result.get("stages"), list):
        raise ValueError("Recipe requires a stages list")
    minimum_epochs = 0
    for stage in result["stages"]:
        if not isinstance(stage, dict):
            raise ValueError("Recipe stages must be objects")
        overrides = stage.setdefault("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("Stage overrides must be an object")
        kind = stage.get("kind")
        keys = CLI_CORE_OPTIONS if kind == "train" else ()
        for key in keys:
            value = getattr(options, key.replace("-", "_"))
            if value is not None:
                overrides[key] = value
        if kind in {"train", "correct"}:
            for key in (*CLI_TRAIN_OPTIONS, "mile_stone"):
                value = getattr(options, key.replace("-", "_"))
                if value is not None:
                    overrides[key] = value
            minimum_epochs = overrides.get("min-epochs", minimum_epochs)
            if options.epoch is not None:
                overrides["epoch"] = options.epoch
                if options.min_epochs is None:
                    minimum_epochs = min(minimum_epochs, options.epoch)
                    overrides["min-epochs"] = minimum_epochs
        if options.batch_size is not None:
            overrides["batch_size"] = options.batch_size
        if kind == "correct":
            if options.correct_lr is not None:
                overrides["lr"] = options.correct_lr
            if options.correct_batch_size is not None:
                overrides["batch_size"] = options.correct_batch_size
        elif kind in {"ridge", "readout"} and options.readout_batch_size is not None:
            overrides["batch_size"] = options.readout_batch_size
    kinds = {stage.get("kind") for stage in result["stages"]}
    if (options.correct_lr is not None or options.correct_batch_size is not None) and "correct" not in kinds:
        raise ValueError("Correction options require a correct stage")
    if options.readout_batch_size is not None and not kinds.intersection({"ridge", "readout"}):
        raise ValueError("Readout batch size requires a readout stage")
    return result


def _json(path):
    from lib.artifact_io import read_json
    return read_json(path)


def _write(path, payload):
    from IMTSTokenizer.run_models import json_safe
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _flags(options):
    result = []
    for key, value in options.items():
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif key == "mile_stone" and isinstance(value, list):
            result.extend([flag, ",".join(map(str, value))])
        elif isinstance(value, list):
            result.extend([flag] + list(map(str, value)))
        else:
            result.extend([flag, str(value)])
    return result


def _check_overrides(options, allowed):
    if not isinstance(options, dict) or not set(options).issubset(allowed):
        raise ValueError("Stage contains unsupported options or forbidden checkpoint/output paths")
    for key, value in options.items():
        if key in BOOLEAN_OPTIONS:
            if type(value) is not bool:
                raise ValueError(key + " must be a JSON boolean")
        elif isinstance(value, bool) or value is None or isinstance(value, dict):
            raise ValueError(key + " requires a non-boolean argument")
        elif isinstance(value, list):
            if key not in {"mile_stone", "correction-variables", "ridge-grid", "global-ridge-grid", "adaptive-ridge-grid", "context-penalty-grid", "shrink-grid", "cv-gain-grid", "ridge-modes", "readout-modes"} or not value:
                raise ValueError("Unsupported or empty list option: " + key)
            if key not in {"ridge-grid", "global-ridge-grid", "adaptive-ridge-grid", "context-penalty-grid", "shrink-grid", "cv-gain-grid", "ridge-modes", "readout-modes"} and any(type(item) is not int for item in value):
                raise ValueError(key + " must contain integers")
            if key in {"ridge-grid", "global-ridge-grid", "adaptive-ridge-grid", "context-penalty-grid", "shrink-grid", "cv-gain-grid"} and any(type(item) not in {int, float} for item in value):
                raise ValueError(key + " must contain numbers")
            if key in {"ridge-modes", "readout-modes"} and any(not isinstance(item, str) for item in value):
                raise ValueError(key + " must contain mode names")
        elif not isinstance(value, (str, int, float)):
            raise ValueError("Unsupported value type for " + key)


def build_stage_command(stage, dataset, seed, gpu, device, output_dir, previous_checkpoint=None):
    """Build argv from a normalized stage; only this function inserts inputs."""
    output_dir = Path(output_dir)
    options = copy.deepcopy(stage["options"])
    if stage["kind"] in {"ridge", "readout"}:
        if previous_checkpoint is None:
            raise ValueError("Ridge stage requires its preceding stage checkpoint")
        mapping = {"ridge-mode": "modes", "ridge-modes": "modes", "readout-modes": "modes",
                   "ridge-rank": "rank", "batch_size": "batch-size"}
        options = {mapping.get(key, key): value for key, value in options.items()}
        script = "fit_ridge.py" if stage["kind"] == "ridge" else "fit_readout.py"
        return [sys.executable, "-B", "-u", str(ROOT / "IMTSTokenizer" / script),
                "--checkpoint", str(previous_checkpoint), "--expected-dataset", dataset,
                "--expected-seed", str(seed), "--device", device, "--gpu", str(gpu),
                "--output-dir", str(output_dir)] + _flags(options)
    command = [sys.executable, "-B", "-u", str(ROOT / "IMTSTokenizer/run_models.py"),
               "--dataset", dataset, "--seed", str(seed), "--gpu", str(gpu),
               "--device", device, "--output-dir", str(output_dir)] + _flags(options)
    if stage["kind"] == "correct":
        if previous_checkpoint is None:
            raise ValueError("Correction requires its preceding stage checkpoint")
        command.extend(["--init-checkpoint", str(previous_checkpoint)])
    elif previous_checkpoint is not None:
        raise ValueError("Fresh train stage must not have a checkpoint input")
    return command


def validate_recipe(recipe, dataset, seed, device="cuda", *, max_epochs=None,
                    max_train_batches=None, max_eval_batches=None, cpu_threads=None):
    """Validate static config without creating outputs or requiring old assets."""
    from IMTSTokenizer.run_models import make_parser as train_parser, validate_args
    from IMTSTokenizer.fit_ridge import make_parser as ridge_parser, validate_options
    from IMTSTokenizer.fit_readout import make_parser as readout_parser, validate_options as validate_readout, READOUT_MODES, mode_identity
    from IMTSTokenizer.scripts.run_parallel import CONFIGS
    if device not in {"cpu", "cuda"} or dataset not in CONFIGS:
        raise ValueError("Unsupported pipeline dataset or device")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if (not isinstance(recipe, dict) or
            set(recipe) - {"schema_version", "dataset", "description", "stages"} or
            type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 1 or
            recipe.get("dataset") != dataset):
        raise ValueError("Recipe requires schema_version=1 and the scheduled dataset")
    stages = recipe.get("stages")
    if not isinstance(stages, list) or not 1 <= len(stages) <= 3:
        raise ValueError("Recipe requires one to three stages")
    for name, limit, minimum in (("max_epochs", max_epochs, 1),
                                 ("max_train_batches", max_train_batches, 0),
                                 ("max_eval_batches", max_eval_batches, 0),
                                 ("cpu_threads", cpu_threads, 1)):
        if limit is not None and (type(limit) is not int or limit < minimum):
            raise ValueError(name + " has an invalid limit")
    result, ids, kinds = [], set(), []
    inherited = {**CONFIGS[dataset], "variant": "legacy", "epoch": 1000,
                 "patience": 10, "lr": .001, "diagnostics": "on",
                 "test-policy": "best", "eval-rng-policy": "legacy"}
    previous_args = None
    for index, raw in enumerate(stages):
        if not isinstance(raw, dict) or set(raw) - {"id", "kind", "overrides"}:
            raise ValueError("Stages accept only id, kind and overrides")
        stage_id, kind = raw.get("id"), raw.get("kind")
        if (not isinstance(stage_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", stage_id)
                or stage_id in ids):
            raise ValueError("Stage ids must be unique filesystem-safe names")
        if not isinstance(kind, str) or kind not in {"train", "correct", "ridge", "readout"}:
            raise ValueError("Stage kind must be train, correct, ridge or readout")
        kinds.append(kind)
        if kinds not in (["train"], ["train", "correct"], ["train", "ridge"], ["train", "correct", "ridge"],
                         ["train", "readout"], ["train", "correct", "readout"]):
            raise ValueError("Stage order must be fresh train, optional correct, optional final ridge/readout")
        ids.add(stage_id)
        overrides = raw.get("overrides", {})
        allowed = (RIDGE_OPTIONS if kind == "ridge" else READOUT_OPTIONS if kind == "readout"
                   else CORRECT_OPTIONS if kind == "correct" else TRAIN_OPTIONS)
        _check_overrides(overrides, allowed)
        requested_epochs = None
        if kind == "readout":
            options = {"readout-modes": ["random4", "full", "hierarchical"], "ridge-rank": 4,
                       "ridge-grid": [.001, .01, .1, 1., 10.],
                       "global-ridge-grid": [.001, .01, .1, 1., 10.],
                       "batch_size": inherited["batch_size"], "cpu-threads": 4,
                       "evaluate-test": True, **overrides}
            selected = options["readout-modes"]
            if (not isinstance(selected, list) or not selected
                    or any(not isinstance(mode, str) for mode in selected)
                    or len(set(selected)) != len(selected)
                    or any(mode not in READOUT_MODES for mode in selected)):
                raise ValueError("Readout stage requires unique valid modes")
        elif kind == "ridge":
            if "ridge-mode" in overrides and "ridge-modes" in overrides:
                raise ValueError("Use ridge-mode or ridge-modes, not both")
            options = {"ridge-rank": 4,
                       "ridge-grid": [.001, .01, .1, 1., 10.],
                       "batch_size": inherited["batch_size"], "cpu-threads": 4,
                       "evaluate-test": True, **overrides}
            if "ridge-modes" not in options and "ridge-mode" not in options:
                options["ridge-mode"] = "hybrid"
            selected = options.get("ridge-modes", [options.get("ridge-mode")])
            if (not isinstance(selected, list) or not selected or
                    any(not isinstance(mode, str) for mode in selected) or len(set(selected)) != len(selected)
                    or any(mode not in {"history", "readout", "hybrid"} for mode in selected)):
                raise ValueError("Ridge stage requires unique valid mode names")
        else:
            options = {**inherited, **overrides}
            requested_epochs = options["epoch"]
            if kind == "correct":
                if previous_args.temporal_correction:
                    raise ValueError("Do not add another correction stage to an already corrected fresh model")
                if any(overrides.get(key) is False for key in ("temporal-correction", "correction-only")):
                    raise ValueError("Correct stage must enable temporal-correction and correction-only")
                options.update({"temporal-correction": True, "correction-only": True})
            elif options.get("correction-only", False):
                raise ValueError("Fresh training requires a trained core before holding its parameters fixed")
            inherited_requested = options.copy()
            if max_epochs is not None:
                options["epoch"] = min(options["epoch"], max_epochs)
                options["min-epochs"] = min(options.get("min-epochs", 0), options["epoch"])
        if max_train_batches is not None:
            options["max-train-batches"] = max_train_batches
        if max_eval_batches is not None:
            options["max-val-batches" if kind in {"ridge", "readout"} else "max-eval-batches"] = max_eval_batches
        if cpu_threads is not None:
            options["cpu-threads"] = cpu_threads
        stage = {"id": stage_id, "kind": kind, "options": options}
        stage["requested_epoch_budget"] = requested_epochs
        stage["epoch_budget_capped"] = requested_epochs is not None and options["epoch"] < requested_epochs
        stage_path = ROOT / "experiments/20000101_000000/validation" / "{:02d}".format(index)
        prior = stage_path.parent / (str(index - 1) + "_" + result[-1]["id"]) / "best.ckpt" if result else None
        argv = build_stage_command(stage, dataset, seed, 0, device, stage_path, prior)[4:]
        try:
            if kind == "readout":
                args = readout_parser().parse_args(argv)
                validate_readout(args)
                expected = vars(previous_args).copy()
                mode = args.modes[0]
                if any(mode_identity(name, nested_rank=args.nested_rank)["readout_basis"] == "spectral_nested" for name in args.modes):
                    width = previous_args.hid_dim * previous_args.npatch * previous_args.multi
                    if args.nested_rank > width:
                        raise ValueError("nested-rank exceeds the core readout feature width")
                    stage["original_readout_rank"] = args.rank
                    stage["nested_readout_rank"] = args.nested_rank
                expected.update(ridge_rank=args.rank, readout_extra_rank=args.extra_rank)
                expected.update(mode_identity(mode, nested_rank=args.nested_rank))
            elif kind == "ridge":
                args = ridge_parser().parse_args(argv)
                validate_options(args)
                expected = vars(previous_args).copy()
                expected.update(ridge_mode=args.modes[0], ridge_rank=args.rank)
            else:
                args = train_parser().parse_args(argv)
                validate_args(args)
                if args.min_epochs > args.epoch:
                    raise ValueError("min-epochs cannot exceed the stage epoch budget")
                expected = vars(args).copy()
                previous_args = args
                inherited = inherited_requested
        except SystemExit as error:
            raise ValueError("Invalid stage arguments: " + stage_id) from error
        stage["expected_identity"] = {key: expected.get(key) for key in IDENTITY_FIELDS}
        stage["requested_overrides"] = copy.deepcopy(overrides)
        result.append(stage)
    return result


def _verify_checkpoint(stage, stage_dir, dataset, seed, metadata):
    """Check a completed stage's model identity and selected validation result."""
    import torch
    stage_dir = Path(stage_dir).resolve()
    checkpoint = stage_dir / "best.ckpt"
    if not checkpoint.is_file() or checkpoint.resolve().parent != stage_dir:
        raise ValueError("Stage did not produce its own best.ckpt")
    summary = _json(stage_dir / "summary.json")
    if not isinstance(summary, dict):
        raise ValueError("Stage summary must be an object")
    allowed = {"completed", "validated"} if stage["kind"] in {"ridge", "readout"} else {"completed", "early_stopped"}
    if summary.get("status") not in allowed:
        raise ValueError("Stage did not report a successful terminal status")
    reported = summary.get("checkpoint")
    if not isinstance(reported, str) or (ROOT / reported).resolve() != checkpoint:
        raise ValueError("Stage summary references a foreign or absent checkpoint")
    stored = load_checkpoint(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(stored, dict) or not isinstance(stored.get("state_dict"), dict):
        raise ValueError("Stage checkpoint is missing model weights")
    saved = stored.get("args")
    if not isinstance(saved, dict) or saved.get("dataset") != dataset or saved.get("seed") != seed:
        raise ValueError("Stage checkpoint dataset/seed differs from the pipeline")
    for key, expected in stage["expected_identity"].items():
        defaults = {"source_reliability_mode": "gated", "target_query_mode": "none"}
        actual = saved.get(key, defaults[key]) if key in defaults else saved.get(key)
        if actual != expected:
            raise ValueError("Stage checkpoint differs in " + key)
    mse = summary.get("best_val_mse")
    if type(mse) not in {int, float} or not math.isfinite(mse) or mse < 0:
        raise ValueError("Stage has no finite selected validation error")
    if not math.isclose(float(stored.get("best_val_mse", math.nan)), mse, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Stage checkpoint and summary validation differ")
    if summary.get("best_epoch") != stored.get("epoch"):
        raise ValueError("Stage checkpoint and summary epoch differ")
    if stage["kind"] == "train" and (stored.get("epoch", -1) < 0 or summary.get("global_step", 0) <= 0):
        raise ValueError("Fresh core did not complete any training")
    if type(summary.get("partial_run")) is not bool:
        raise ValueError("Stage summary must declare partial_run")
    if stage["kind"] == "ridge" and stored.get("training_method") != "train_only_ridge":
        raise ValueError("Ridge checkpoint does not declare the expected fitting method")
    if stage["kind"] == "readout":
        from IMTSTokenizer.fit_readout import CV_SOURCES, GUARDED_SOURCES, mode_training_method
        expected_mode = stage.get("expected_readout_mode", stage["options"]["readout-modes"][0])
        expected_method = mode_training_method(expected_mode)
        if stored.get("training_method") != expected_method or summary.get("training_method") != expected_method:
            raise ValueError("Readout checkpoint/summary do not declare the expected fitting method")
        if stored.get("readout_mode") != expected_mode or summary.get("readout_mode") != expected_mode:
            raise ValueError("Readout checkpoint/summary mode differs from its selected stage")
        expected_splits = ({"parameter_selection_split": "train_group_cv", "gain_selection_split": "train_group_cv"}
                           if expected_mode in CV_SOURCES else
                           {"parameter_selection_split": "val", "gain_selection_split": "val_with_train_group_guard"}
                           if expected_mode in GUARDED_SOURCES else {})
        for field, expected in expected_splits.items():
            if stored.get(field) != expected or summary.get(field) != expected:
                raise ValueError("Grouped readout checkpoint/summary differs in " + field)
    actual_epochs = None
    if stage["kind"] not in {"ridge", "readout"}:
        batches = metadata.get("data", {}).get("n_train_batches")
        limit = stage["options"].get("max-train-batches", 0)
        if isinstance(batches, int) and batches > 0:
            actual_epochs = summary["global_step"] / min(batches, limit or batches)
            if actual_epochs > stage["options"]["epoch"] or actual_epochs < 1 or not actual_epochs.is_integer():
                raise ValueError("Stage exceeded its epoch budget")
    elif summary.get("initialization_selected") not in (None, True, False):
        raise ValueError("Ridge initialization selection is invalid")
    return {"checkpoint": str(checkpoint.relative_to(ROOT)),
            "summary": summary, "completed_epochs": actual_epochs,
            "training_method": stored["training_method"] if stage["kind"] == "readout" else "train_only_ridge" if stage["kind"] == "ridge" else
                               ("fixed_core_correction" if stage["kind"] == "correct" else "fresh_sgd"),
            "partial_run": summary["partial_run"]}


def verify_stage_result(stage, stage_dir, dataset, seed):
    """Lock the terminal mode by validation, retaining every independent result."""
    stage_dir = Path(stage_dir)
    metadata = _json(stage_dir / "run.json")
    if not isinstance(metadata, dict):
        raise ValueError("Stage run metadata must be an object")
    modes = (stage["options"].get("readout-modes") if stage["kind"] == "readout" else
             stage["options"].get("ridge-modes") if stage["kind"] == "ridge" else None)
    if modes is None:
        modes = [stage["options"]["ridge-mode"]] if stage["kind"] == "ridge" else []
    if len(modes) <= 1 and stage["kind"] != "readout":
        return _verify_checkpoint(stage, stage_dir, dataset, seed, metadata)
    results = _json(stage_dir / "results.json")
    if results.get("status") not in {"completed", "validated"}:
        raise ValueError("Multi-mode ridge stage did not complete all requested modes")
    if set(results.get("modes", {})) != set(modes):
        raise ValueError("Multi-mode ridge output does not match requested modes")
    verified = {}
    for mode in modes:
        child_stage = copy.deepcopy(stage)
        if stage["kind"] == "readout":
            from IMTSTokenizer.fit_readout import mode_identity
            child_stage["expected_readout_mode"] = mode
            # Old normalized/standalone stage contracts already carry their
            # actual rank; only mixed nested stages need an explicit reset.
            if "original_readout_rank" in stage:
                child_stage["expected_identity"]["ridge_rank"] = stage["original_readout_rank"]
            nested_rank = stage.get("nested_readout_rank", stage["options"].get("nested-rank", 16))
            child_stage["expected_identity"].update(mode_identity(mode, nested_rank=nested_rank))
        else:
            child_stage["expected_identity"]["ridge_mode"] = mode
        verified[mode] = _verify_checkpoint(child_stage, stage_dir / mode, dataset, seed, metadata)
        result = results["modes"][mode]
        if (not math.isclose(result["val"]["mse"], verified[mode]["summary"]["best_val_mse"], rel_tol=1e-9)
                or (ROOT / result["checkpoint"]).resolve() != (ROOT / verified[mode]["checkpoint"]).resolve()):
            raise ValueError("Multi-mode results differ from their checkpoint summaries")
    selected = min(modes, key=lambda mode: verified[mode]["summary"]["best_val_mse"])
    return {**verified[selected], "selected_mode": selected, "selection_split": "val",
            "terminal_modes": verified,
            "partial_run": any(row["partial_run"] for row in verified.values())}


def stop_child(process):
    """Reap the owned worker before the queue is allowed to reuse this slot."""
    if process.poll() is None:
        process.terminate()
    try:
        return process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=3)


def make_parser():
    from IMTSTokenizer.run_models import milestones

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", help="Defaults to configs/final/<dataset>.json")
    parser.add_argument("--dataset", required=True, choices=["activity", "physionet", "mimic", "ushcn"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--dry-run", action="store_true")
    for key, converter in {**CLI_CORE_OPTIONS, **CLI_TRAIN_OPTIONS}.items():
        names = list(dict.fromkeys(("--" + key, "--" + key.replace("_", "-"), "--" + key.replace("-", "_"))))
        kwargs = {"type": converter, "default": None}
        if key == "scheduler":
            kwargs["choices"] = ["milestone", "plateau"]
        parser.add_argument(*names, **kwargs)
    parser.add_argument("--mile_stone", "--mile-stone", type=milestones)
    parser.add_argument("--batch_size", "--batch-size", type=int, help="Batch size for all stages")
    parser.add_argument("--epoch", type=int, help="Epoch budget for each neural stage")
    parser.add_argument("--correct-lr", "--correct_lr", type=float)
    parser.add_argument("--correct-batch-size", "--correct_batch_size", type=int)
    parser.add_argument("--readout-batch-size", "--readout_batch_size", type=int)
    return parser


def main(argv=None):
    parser = make_parser()
    options = parser.parse_args(argv)
    try:
        if options.gpu < 0:
            raise ValueError("gpu must be nonnegative")
        recipe_path = (ROOT / (options.recipe or "IMTSTokenizer/configs/final/{}.json".format(options.dataset))).resolve()
        if not recipe_path.is_relative_to(ROOT) or not recipe_path.is_file():
            raise ValueError("recipe must be an existing config inside the project")
        recipe_bytes = recipe_path.read_bytes()
        recipe = apply_cli_overrides(json.loads(recipe_bytes.decode("utf-8")), options)
        stages = validate_recipe(recipe, options.dataset, options.seed, options.device,
                                 max_epochs=options.max_epochs, max_train_batches=options.max_train_batches,
                                 max_eval_batches=options.max_eval_batches, cpu_threads=options.cpu_threads)
        destination = resolve_output_dir(ROOT, options.output_dir)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    if options.dry_run:
        print(json.dumps({"dataset": options.dataset, "seed": options.seed,
                          "external_checkpoint_inputs": [], "stages": stages}, indent=2))
        return 0
    destination = create_output_dir(ROOT, destination)
    started = time.perf_counter()
    report = {"schema_version": 1, "status": "running", "dataset": options.dataset,
              "seed": options.seed, "device": options.device, "gpu": options.gpu,
              "recipe_path": str(recipe_path.relative_to(ROOT)),
              "recipe": recipe, "external_checkpoint_inputs": [],
              "run_id": destination.relative_to(ROOT / "experiments").as_posix(),
              "PID": os.getpid(), "partial_run": any(stage["epoch_budget_capped"] for stage in stages),
              "stages": [], "checkpoint": None,
              "requested_limits": {key: getattr(options, key) for key in
                                   ("max_epochs", "max_train_batches", "max_eval_batches", "cpu_threads")}}
    for index, stage in enumerate(stages):
        report["stages"].append({"id": stage["id"], "kind": stage["kind"], "status": "queued",
                                 "options": stage["options"], "index": index,
                                 "requested_epoch_budget": stage["requested_epoch_budget"],
                                 "epoch_budget_capped": stage["epoch_budget_capped"]})
    previous_handlers = {}
    interrupted = None
    process = None
    current = None

    def persist():
        report["seconds"] = time.perf_counter() - started
        _write(destination / "pipeline.json", report)
        _write(destination / "summary.json", {key: report.get(key) for key in
               ("status", "dataset", "seed", "run_id", "checkpoint", "best_val_mse",
                "test_at_best_validation", "seconds", "partial_run", "error", "error_type", "failed_stage",
                "selected_mode", "selection_split", "terminal_modes")})

    def event(kind, **values):
        record = {"time": datetime.datetime.now().isoformat(), "kind": kind, **values}
        with (destination / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = signum

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        persist()
        predecessor = None
        for index, stage in enumerate(stages):
            if interrupted is not None:
                raise KeyboardInterrupt("Pipeline cancellation requested")
            current = report["stages"][index]
            stage_dir = destination / "{:02d}".format(index)
            if stage_dir.resolve().parent != destination:
                raise ValueError("Stage output escaped its task")
            stage_dir.mkdir(exist_ok=False)
            if predecessor is not None:
                if not predecessor.is_file() or predecessor.resolve().parent.parent != destination:
                    raise ValueError("Predecessor checkpoint is missing or outside this run")
                current["input"] = {"path": str(predecessor.relative_to(ROOT))}
            else:
                current["input"] = None
            command = build_stage_command(stage, options.dataset, options.seed, options.gpu,
                                          options.device, stage_dir, predecessor)
            current.update(status="running", command=command, output_dir=str(stage_dir.relative_to(ROOT)),
                           started_at=datetime.datetime.now().isoformat())
            persist()
            event("stage_started", stage=stage["id"], command=command)
            environment = os.environ.copy()
            environment.update(PYTHONDONTWRITEBYTECODE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                               CUDA_VISIBLE_DEVICES=str(options.gpu) if options.device == "cuda" else "")
            with (stage_dir / "console.log").open("wb") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT)
                current["pid"] = process.pid
                persist()
                while True:
                    if interrupted is not None:
                        raise KeyboardInterrupt("Pipeline cancellation requested")
                    code = process.poll()
                    if code is not None:
                        process.wait()
                        current["returncode"] = code
                        process = None
                        break
                    time.sleep(.1)
            if current["returncode"] != 0:
                raise RuntimeError("Stage {} exited with code {}".format(stage["id"], current["returncode"]))
            verified = verify_stage_result(stage, stage_dir, options.dataset, options.seed)
            current.update(verified, status="succeeded", finished_at=datetime.datetime.now().isoformat())
            report["partial_run"] = report["partial_run"] or verified["partial_run"]
            predecessor = ROOT / verified["checkpoint"]
            event("stage_succeeded", stage=stage["id"], checkpoint=verified["checkpoint"])
            persist()
        if interrupted is not None:
            raise KeyboardInterrupt("Pipeline cancellation requested")
        final_path = destination / "best.ckpt"
        shutil.copyfile(predecessor, final_path)
        terminal = current["summary"]
        report.update(status="completed", checkpoint=str(final_path.relative_to(ROOT)),
                      best_val_mse=terminal["best_val_mse"],
                      test_at_best_validation=terminal.get("test_at_best_validation"),
                      selected_mode=current.get("selected_mode"), selection_split="val",
                      terminal_modes=current.get("terminal_modes"))
        event("pipeline_completed", checkpoint=report["checkpoint"])
    except BaseException as error:
        cancelled = isinstance(error, KeyboardInterrupt) or interrupted is not None
        report.update(status="cancelled" if cancelled else "failed", error=str(error),
                      error_type=type(error).__name__, failed_stage=current["id"] if current else None)
        if current is not None and current["status"] != "succeeded":
            current.update(status=report["status"], error=str(error), finished_at=datetime.datetime.now().isoformat())
        for row in report["stages"]:
            if row["status"] == "queued":
                row["status"] = "cancelled" if cancelled else "skipped_due_to_failure"
        event("pipeline_" + report["status"], error_type=type(error).__name__, error=str(error))
    finally:
        try:
            if process is not None:
                current["returncode"] = stop_child(process)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            persist()
    return 0 if report["status"] == "completed" else (130 if report["status"] == "cancelled" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
