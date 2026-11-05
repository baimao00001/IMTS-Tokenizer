"""Fit one fixed single-model residual readout from train-only sufficient statistics."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.checkpoint_names import load_checkpoint
from lib.run_paths import create_output_dir


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-dataset", choices=["activity", "physionet", "mimic", "ushcn"])
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--modes", nargs="+", choices=["history", "readout", "hybrid"],
                        default=["history", "readout", "hybrid"])
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=[.001, .01, .1, 1., 10.])
    parser.add_argument("--uniform-ridge", action="store_true", help="Disable train-support-dependent regularization")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--diagnostics", choices=["on", "off"], default="on",
                        help="Sampled tensor logs; aggregate fit and split statistics are always saved")
    parser.add_argument("--output-dir")
    return parser


def validate_options(options):
    expected_seed = getattr(options, "expected_seed", None)
    if expected_seed is not None and not 0 <= expected_seed < 2 ** 32:
        raise ValueError("expected-seed must be in [0, 2**32)")
    if len(set(options.modes)) != len(options.modes):
        raise ValueError("modes must be unique")
    if (not options.ridge_grid or len(set(options.ridge_grid)) != len(options.ridge_grid)
            or any(not math.isfinite(value) or value <= 0 for value in options.ridge_grid)):
        raise ValueError("ridge-grid must contain unique finite positive values")
    if min(options.rank, options.batch_size, options.cpu_threads) < 1:
        raise ValueError("rank, batch-size and cpu-threads must be positive")
    if min(options.max_train_batches, options.max_val_batches) < 0:
        raise ValueError("batch limits must be nonnegative")


def feature_indices(mode, rank):
    if mode == "history":
        return list(range(8))
    if mode == "readout":
        return [0] + list(range(8, 8 + rank))
    return list(range(8 + rank))


def collect_statistics(model, loader, batches, torch, utils, statistics_type, cache_batches=False,
                       diagnostics=None, split=None):
    stats = statistics_type(model.args.num_var, len(model.basis_names))
    cached = [] if cache_batches else None
    model.eval()
    with torch.no_grad():
        for index in range(batches):
            batch = utils.get_next_batch(loader)
            if cached is not None:
                cached.append(batch)
            core, basis = model.design(batch["tp_to_predict"], batch["observed_data"],
                                       batch["observed_tp"], batch["observed_mask"], batch.get("window_start"))
            stats.update(basis, core, batch["data_to_predict"], batch["mask_predicted_data"])
            if (index + 1) % 100 == 0:
                print("statistics batches", index + 1, "/", batches, flush=True)
                if diagnostics is not None:
                    from IMTSTokenizer.run_models import memory_status
                    diagnostics.event("ridge_statistics_progress", split=split, batches=index + 1,
                                      total_batches=batches, memory=memory_status(model.coefficients.device))
    return stats, batches, cached


def main(argv=None):
    parser = make_parser()
    options = parser.parse_args(argv)
    try:
        validate_options(options)
        checkpoint_path = (ROOT / options.checkpoint).resolve()
        if not checkpoint_path.is_relative_to(ROOT) or not checkpoint_path.is_file():
            raise ValueError("checkpoint must be an existing file inside the project")
        destination = create_output_dir(ROOT, options.output_dir)
    except (OSError, ValueError, FileExistsError) as error:
        parser.error(str(error))
    os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu
    import torch
    from IMTSTokenizer.model.factory import build_model
    from IMTSTokenizer.model.RidgeCorrector import RidgeCorrector
    from IMTSTokenizer.run_models import make_parser as core_parser, memory_status, validate_args, write_json
    from lib import utils
    from lib.branch_diagnostics import BranchDiagnostics
    from lib.diagnostic_metrics import ForecastDiagnostics
    from lib.diagnostics import Diagnostics
    from lib.evaluation import evaluation
    from lib.parse_datasets import parse_datasets
    from lib.ridge_stats import RidgeStatistics

    started = time.perf_counter()
    previous_dir, previous_threads = Path.cwd(), torch.get_num_threads()
    report = {"kind": "single_model_train_fitted_ridge", "status": "running", "options": vars(options),
              "checkpoint": str(checkpoint_path.relative_to(ROOT)),
              "modes": {},
              "command": list(sys.argv) if argv is None else [str(Path(__file__).resolve())] + list(argv),
              "PID": os.getpid(), "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
              "normalization_scope": "train+validation scaling except preprocessed USHCN",
              "protocol": {"fit_split": "train", "select_ridge_split": "val", "test_evaluated": False,
                           "data_rng_seed": 730017, "core_fixed": True, "core_calls_per_prediction": 1,
                           "optimizer": "full-statistics ridge",
                           "support_shrink": not options.uniform_ridge,
                           "solve_dtype": "float64", "deployed_coefficient_dtype": "float32",
                           "normal_equation_gradients": "evaluated at the float64 solve; deployed scores use float32 coefficients"}}
    write_json(destination / "run.json", report)
    write_json(destination / "results.json", report)
    diag = Diagnostics(destination, enabled=options.diagnostics == "on", every=50, first_steps=2,
                       max_elements=256, snapshots=0, metadata=report)
    device = None
    try:
        torch.set_num_threads(options.cpu_threads)
        os.chdir(ROOT / "IMTSTokenizer")
        if options.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA calibration, but CUDA is unavailable")
        device = torch.device("cuda:0" if options.device == "cuda" else "cpu")
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", weights_only=False)
        saved_args = checkpoint.get("args")
        if not isinstance(saved_args, dict) or "state_dict" not in checkpoint:
            raise ValueError("checkpoint must contain the trained model's args and state_dict")
        if options.expected_dataset is not None and saved_args.get("dataset") != options.expected_dataset:
            raise ValueError("Checkpoint dataset does not match the scheduled task")
        if options.expected_seed is not None and saved_args.get("seed") != options.expected_seed:
            raise ValueError("Checkpoint seed does not match the scheduled task")
        if saved_args.get("ridge_mode", "none") != "none" or saved_args.get("fitted_readout", "none") != "none":
            raise ValueError("Do not recursively calibrate a previously ridge-fitted checkpoint")
        args = core_parser().parse_args([])
        for name, value in saved_args.items():
            if name in vars(args):
                setattr(args, name, value)
        args.load = args.init_checkpoint = None
        args.device, args.gpu = device, options.gpu
        args.output_dir = args.save = args.diagnostics_dir = str(destination)
        args.batch_size, args.dataset_storage = options.batch_size, "cpu"
        args.collect_reference_prior, args.ridge_mode = True, "none"
        validate_args(args, require_checkpoint=False)
        if options.device == "cpu":
            utils.setup_seed(730017, cpu_only=True)
        else:
            utils.setup_seed(730017)
        diag.event("memory", stage="before_data", **memory_status(device))
        data = parse_datasets(args, patch_ts=True)
        args.num_var = args.ndim = data["input_dim"]
        args.time_max = float(data["time_max"])
        args.value_prior = data["training_value_prior"].tolist()
        args.training_target_counts = data["training_target_counts"].tolist()
        if saved_args.get("variable_head", "none") != "none":
            saved_counts = torch.as_tensor(saved_args.get("training_target_counts", []), dtype=torch.float64)
            current_counts = data["training_target_counts"].double().cpu()
            if saved_counts.shape != current_counts.shape or not torch.equal(saved_counts, current_counts):
                raise ValueError("Checkpoint variable-head training counts differ from the current training data")
        if "value_prior" in saved_args:
            saved_prior = torch.as_tensor(saved_args["value_prior"], dtype=torch.float64)
            current_prior = data["training_value_prior"].double().cpu()
            if saved_prior.shape != current_prior.shape or not torch.allclose(saved_prior, current_prior, rtol=1e-6, atol=1e-8):
                raise ValueError("Checkpoint training prior differs from the current training data")
        if "time_max" in saved_args and not math.isclose(float(saved_args["time_max"]), args.time_max,
                                                        rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("checkpoint and current data use different time normalization")
        core = build_model(args).to(device)
        core.load_state_dict(checkpoint["state_dict"], strict=True)
        core.requires_grad_(False).eval()
        hybrid_args = copy.deepcopy(args)
        hybrid_args.ridge_mode, hybrid_args.ridge_rank = "hybrid", options.rank
        hybrid = RidgeCorrector(core, hybrid_args).to(device).eval()
        train_batches = min(data["n_train_batches"], options.max_train_batches or data["n_train_batches"])
        val_batches = min(data["n_val_batches"], options.max_val_batches or data["n_val_batches"])
        if min(train_batches, val_batches) <= 0:
            raise ValueError("training and validation loaders must be nonempty")
        report.update(dataset=args.dataset, seed=args.seed, args=vars(args),
                      train_batches=train_batches, validation_batches=val_batches,
                      full_train_batches=data["n_train_batches"], full_validation_batches=data["n_val_batches"],
                      partial_run=(train_batches < data["n_train_batches"] or val_batches < data["n_val_batches"]),
                      data={key: value for key, value in data.items() if "dataloader" not in key})
        write_json(destination / "run.json", {key: value for key, value in report.items() if key != "modes"})
        diag.event("memory", stage="after_core_creation", **memory_status(device))
        if options.device == "cpu":
            utils.setup_seed(730017, cpu_only=True)
        else:
            utils.setup_seed(730017)
        print("Fitting", args.dataset, "seed", args.seed, "train/val batches", train_batches, val_batches, flush=True)
        train, _, _ = collect_statistics(hybrid, data["train_dataloader"], train_batches, torch, utils, RidgeStatistics,
                                                diagnostics=diag, split="train")
        validation, _, cached_validation = collect_statistics(
            hybrid, data["val_dataloader"], val_batches, torch, utils, RidgeStatistics,
            cache_batches=val_batches < data["n_val_batches"], diagnostics=diag, split="val")
        diag.event("memory", stage="after_statistics", **memory_status(device))
        if train_batches == data["n_train_batches"]:
            torch.testing.assert_close(train.counts, data["training_target_counts"].double().cpu(), rtol=1e-9, atol=1e-6)
        fields = ("counts", "gram", "cross", "reference_sse", "residual_sum")
        torch.save({"train": {name: getattr(train, name) for name in fields},
                    "validation": {name: getattr(validation, name) for name in fields},
                    "basis_names": hybrid.basis_names}, destination / "sufficient_statistics.pt")
        report["baseline_validation"] = validation.score(torch.zeros_like(hybrid.coefficients))
        chosen = {}
        for mode in options.modes:
            indices = feature_indices(mode, options.rank)
            train_mode, val_mode = train.select(indices), validation.select(indices)
            beta = torch.zeros(args.num_var, len(indices), dtype=torch.float32)
            best = {"ridge": None, "validation": val_mode.score(beta), "initialization_selected": True}
            candidates = []
            for strength in options.ridge_grid:
                fitted, details = train_mode.fit(strength, support_shrink=not options.uniform_ridge)
                deployed = fitted.float()
                score = val_mode.score(deployed)
                candidate = {"ridge": strength, "validation": score, "fit_diagnostics": details}
                candidates.append(candidate)
                if score["macro_mse"] < best["validation"]["macro_mse"]:
                    beta = deployed
                    best = {"ridge": strength, "validation": score, "initialization_selected": False}
            mode_args = copy.deepcopy(args)
            mode_args.ridge_mode, mode_args.ridge_rank = mode, options.rank
            model = RidgeCorrector(core, mode_args).to(device).eval()
            with torch.no_grad():
                model.coefficients.copy_(beta.to(device))
            chosen[mode] = model
            report["modes"][mode] = {**best, "candidates": candidates, "basis_names": list(model.basis_names),
                                     "coefficients": beta.tolist(), "train": train_mode.score(beta),
                                     "parameter_count": sum(p.numel() for p in model.parameters()),
                                     "trainable_parameter_count": model.coefficients.numel()}
            print(mode, "ridge", best["ridge"], "val MSE", best["validation"]["macro_mse"], flush=True)
        # All candidates and their strengths are locked before any test access.
        write_json(destination / "results.json", report)
        for mode, model in chosen.items():
            row = report["modes"][mode]
            child = destination if len(chosen) == 1 else destination / mode
            child.mkdir(exist_ok=True)
            model.diagnostics, model.collect_diagnostic_predictions = diag, True
            for split in (["val", "test"] if options.evaluate_test else ["val"]):
                count = val_batches if split == "val" else data["n_test_batches"]
                if split == "test":
                    report["protocol"]["test_evaluated"] = True
                model.branch_diagnostics = BranchDiagnostics(args.history / args.time_max)
                model.prediction_diagnostics = ForecastDiagnostics()
                loader = iter(cached_validation) if split == "val" and cached_validation is not None else data[split + "_dataloader"]
                with torch.no_grad():
                    metrics = evaluation(model, loader, count, data,
                                         diagnostics=diag, split=mode + "/" + split)
                if not all(math.isfinite(metrics[name]) for name in ("mse", "mae", "rmse")):
                    raise FloatingPointError("Nonfinite final ridge evaluation")
                row[split] = metrics
                row[split + "_diagnostics"] = model.branch_diagnostics.summary()
                diag.event("ridge_split", mode=mode, split=split, metrics=metrics,
                           branch_diagnostics=row[split + "_diagnostics"],
                           forecast_diagnostics=model.prediction_diagnostics.summary())
                if split == "val" and not math.isclose(metrics["mse"], row["validation"]["macro_mse"],
                                                       rel_tol=3e-5, abs_tol=2e-8):
                    raise ValueError("Direct validation differs from sufficient-statistic score")
            checkpoint_args = copy.deepcopy(vars(model.args))
            checkpoint_args.update(output_dir=str(child), save=str(child), diagnostics_dir=str(child),
                                   load=None, init_checkpoint=None)
            torch.save({"state_dict": model.state_dict(), "args": checkpoint_args,
                        "epoch": -1 if row["initialization_selected"] else 0, "step": train_batches,
                        "best_val_mse": row["val"]["mse"], "ridge": row["ridge"],
                        "training_method": "train_only_ridge"},
                       child / "best.ckpt")
            row["checkpoint"] = str((child / "best.ckpt").relative_to(ROOT))
            write_json(child / "summary.json", {"status": "completed" if options.evaluate_test else "validated",
                       "dataset": args.dataset, "seed": args.seed, "ridge_mode": mode,
                       "run_id": child.relative_to(ROOT / "experiments").as_posix(),
                       "best_epoch": -1 if row["initialization_selected"] else 0,
                       "best_val_mse": row["val"]["mse"], "test_at_best_validation": row.get("test"),
                       "initialization_selected": row["initialization_selected"], "ridge": row["ridge"],
                       "checkpoint": row["checkpoint"], "partial_run": report["partial_run"],
                       "global_step": train_batches, "training_method": "train_only_ridge"})
            write_json(destination / "results.json", report)
        report["protocol"]["test_evaluated"] = options.evaluate_test
        report["status"] = "completed" if options.evaluate_test else "validated"
        diag.event("memory", stage="finished", **memory_status(device))
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        diag.event("failure", error_type=type(error).__name__, message=str(error),
                   memory=memory_status(device) if device is not None else None)
        raise
    finally:
        diag.close()
        os.chdir(previous_dir)
        torch.set_num_threads(previous_threads)
        report["seconds"] = time.perf_counter() - started
        write_json(destination / "results.json", report)
        write_json(destination / "run.json", {key: value for key, value in report.items() if key != "modes"})
        if len(options.modes) != 1 or report["status"] == "failed":
            write_json(destination / "summary.json", {key: report.get(key) for key in
                       ("status", "dataset", "seed", "error", "seconds", "partial_run")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
