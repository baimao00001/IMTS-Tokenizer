import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.checkpoint_names import load_checkpoint
from lib.run_paths import create_output_dir, resolve_output_dir


def milestones(value):
    text = value.strip().strip("[]")
    try:
        result = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a list such as [27] or 42,50") from error
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("Milestones must be nonnegative")
    return result


def make_parser():
    parser = argparse.ArgumentParser("IMTS Forecasting")
    parser.add_argument("--state", default="def")
    parser.add_argument("-n", type=int, default=int(1e8))
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--logmode", choices=["a", "w"], default="a")
    parser.add_argument("--output-dir", help="Unified experiments/<timestamp>/[task] output directory")
    parser.add_argument("--save", default=None, help="Compatibility alias for --output-dir; cannot split outputs")
    parser.add_argument("--load", default=None, help="Checkpoint path or legacy experiment ID for evaluation")
    parser.add_argument("--init-checkpoint", help="Initialize single-model weights, then train with a fresh optimizer")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("-ps", "--patch_size", type=float, default=12)
    parser.add_argument("--w_decay", type=float, default=0.0)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", choices=["physionet", "mimic", "ushcn", "activity"], default="ushcn")
    parser.add_argument("--num_var", type=int, default=None, help="Defaults to the data's channel count")
    parser.add_argument("--quantization", type=float, default=0.0)
    parser.add_argument("--model", choices=["imts-tokenizer"], default="imts-tokenizer")
    parser.add_argument("--variant", choices=["legacy", "local", "local_residual", "local_gated", "local_adaptive"], default="legacy")
    parser.add_argument("--qk-normalization", choices=["none", "layer", "adaptive"], default="none")
    parser.add_argument("--source-reliability", action="store_true")
    parser.add_argument("--target-query-mode", choices=["none", "missing", "all", "conditional"], default="none",
                        help="Zero-initialized variable identity added only to first-layer attention Q")
    parser.add_argument("--source-reliability-mode", choices=["gated", "first_bias"], default="gated",
                        help="Historical local_residual gated sources, or zero-start first-layer source logits for legacy/local_residual")
    parser.add_argument("--temporal-correction", action="store_true")
    parser.add_argument("--correction-only", action="store_true", help="Train the temporal/prior correction with backbone parameters held fixed")
    parser.add_argument("--ridge-mode", choices=["none", "history", "readout", "hybrid"], default="none",
                        help="Restore a ridge-fitted single model with --load; fit it with fit_ridge.py")
    parser.add_argument("--ridge-rank", type=int, default=4)
    parser.add_argument("--readout-basis", choices=["full", "diagonal", "interaction", "context", "spectral", "spectral_nested"], default="full")
    parser.add_argument("--readout-extra-rank", type=int, default=4)
    parser.add_argument("--readout-shrinkage", choices=["none", "variable", "block"], default="none")
    parser.add_argument("--fitted-readout", choices=["none", "full", "hierarchical"], default="none",
                        help="Restore a full-feature fitted readout with --load; fit with fit_readout.py")
    parser.add_argument("--seasonal-period", type=float, default=0.0, help="Optional historical seasonal readout period, in raw time units")
    parser.add_argument("--seasonal-bandwidth", type=float, default=1.0)
    parser.add_argument("--correction-variables", nargs="+", type=int, help="Optional zero-based variable indices receiving correction; others stay unchanged")
    parser.add_argument("--dataset-storage", choices=["cpu", "model"], default="cpu",
                        help="MIMIC raw record storage; batches still use the model device")
    parser.add_argument("--loss-reduction", choices=["batch_macro", "global_macro"], default="batch_macro")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="0 disables gradient norm clipping")
    parser.add_argument("-hd", "--hid_dim", type=int, default=6)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--test-policy", choices=["best", "final", "never"], default="best")
    parser.add_argument("--eval-rng-policy", choices=["legacy", "isolated"], default="legacy")
    parser.add_argument("--variable-head", choices=["none", "fixed", "learned"], default="none")
    parser.add_argument("--variable-head-rank", type=int, default=4)
    parser.add_argument("--variable-head-penalty", type=float, default=0.01)
    parser.add_argument("--mile_stone", type=milestones, default=[42, 50])
    parser.add_argument("--scheduler", choices=["milestone", "plateau"], default="milestone")
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 keeps the runtime default")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--multi", type=int, default=1, help="Compatibility argument; inferred from history/patch_size")
    parser.add_argument("--diagnostics", choices=["on", "off"], default="on")
    parser.add_argument("--diagnostics-dir", default=None, help="Compatibility alias for --output-dir; cannot split outputs")
    parser.add_argument("--diag-every", type=int, default=50)
    parser.add_argument("--diag-first-steps", type=int, default=2)
    parser.add_argument("--diag-max-elements", type=int, default=4096)
    parser.add_argument("--diag-snapshots", type=int, default=0,
                        help="Raw tensor snapshots are opt-in; sampled summaries remain enabled")
    parser.add_argument("--artifact-storage", choices=["full", "compact"], default="full")
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 uses all batches; positive values are smoke runs")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 uses all batches; positive values are smoke runs")
    parser.add_argument("--detect-anomaly", action="store_true", help="Expensive autograd debugging")
    return parser


def validate_args(args, require_checkpoint=True):
    fitted_mode = getattr(args, "fitted_readout", "none")
    if fitted_mode not in {"none", "full", "hierarchical"}:
        raise ValueError("Unsupported fitted-readout mode")
    if fitted_mode != "none" and (not args.load or getattr(args, "ridge_mode", "none") != "none"):
        raise ValueError("Use --load to restore fitted readouts, without another ridge readout")
    if getattr(args, "ridge_rank", 4) < 1:
        raise ValueError("ridge-rank must be positive")
    if getattr(args, "readout_extra_rank", 4) < 1:
        raise ValueError("readout-extra-rank must be positive")
    basis = getattr(args, "readout_basis", "full")
    if basis not in {"full", "diagonal", "interaction", "context", "spectral", "spectral_nested"} or (basis != "full" and fitted_mode == "none"):
        raise ValueError("Non-full readout-basis requires a fitted readout")
    shrinkage = getattr(args, "readout_shrinkage", "none")
    if shrinkage not in {"none", "variable", "block"} or (shrinkage != "none" and fitted_mode == "none"):
        raise ValueError("Variable readout shrinkage requires a fitted readout")
    if shrinkage == "block" and basis not in {"spectral", "spectral_nested"}:
        raise ValueError("Block readout shrinkage requires a spectral basis")
    if basis == "spectral_nested" and args.ridge_rank < 4:
        raise ValueError("Nested spectral basis requires ridge-rank of at least four")
    if getattr(args, "ridge_mode", "none") != "none" and not args.load:
        raise ValueError("Fit ridge models with fit_ridge.py; run_models.py restores them with --load")
    selected = getattr(args, "correction_variables", None)
    if selected is not None:
        if (not selected or len(set(selected)) != len(selected) or
                any(type(index) is not int or index < 0 or (args.num_var is not None and index >= args.num_var) for index in selected)):
            raise ValueError("correction-variables must contain unique valid indices")
        if not getattr(args, "temporal_correction", False):
            raise ValueError("correction-variables requires temporal-correction")
    period = getattr(args, "seasonal_period", 0.0)
    bandwidth = getattr(args, "seasonal_bandwidth", 1.0)
    if (not math.isfinite(period) or not 0 <= period <= args.history
            or not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError("Seasonal period must be in [0,history] and bandwidth positive")
    if period and not getattr(args, "temporal_correction", False):
        raise ValueError("seasonal-period requires temporal-correction")
    if getattr(args, "qk_normalization", "none") not in {"none", "layer", "adaptive"}:
        raise ValueError("Unsupported qk-normalization mode")
    if getattr(args, "correction_only", False) and not getattr(args, "temporal_correction", False):
        raise ValueError("correction-only requires temporal-correction")
    if (require_checkpoint and getattr(args, "correction_only", False)
            and not (getattr(args, "init_checkpoint", None) or args.load)):
        raise ValueError("correction-only requires an initialized checkpoint; a fixed random core cannot train")
    if getattr(args, "init_checkpoint", None) and args.load:
        raise ValueError("--load evaluation and --init-checkpoint training are mutually exclusive")
    source_mode = getattr(args, "source_reliability_mode", "gated")
    if (getattr(args, "source_reliability", False) and args.variant != "local_residual"
            and not (args.variant == "legacy" and source_mode == "first_bias")):
        raise ValueError("source-reliability requires local_residual, or legacy with first_bias")
    if getattr(args, "source_reliability", False) and args.num_layers == 0:
        raise ValueError("source-reliability requires at least one attention layer")
    if source_mode not in {"gated", "first_bias"}:
        raise ValueError("Unsupported source-reliability-mode")
    if source_mode != "gated" and not getattr(args, "source_reliability", False):
        raise ValueError("first_bias requires source-reliability")
    query_mode = getattr(args, "target_query_mode", "none")
    if query_mode not in {"none", "missing", "all", "conditional"}:
        raise ValueError("Unsupported target-query-mode")
    if query_mode != "none" and (args.variant != "local_residual" or args.num_layers < 1):
        raise ValueError("target-query-mode requires local_residual with attention layers")
    positive = ["epoch", "patience", "batch_size", "n", "hid_dim", "history", "patch_size", "diag_every", "diag_max_elements"]
    for name in positive:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_layers < 0 or not 0 <= args.dropout < 1:
        raise ValueError("num_layers must be nonnegative and dropout must be in [0, 1)")
    if not math.isfinite(args.lr) or not math.isfinite(args.w_decay) or args.lr <= 0 or args.w_decay < 0:
        raise ValueError("lr must be positive and w_decay nonnegative")
    for name in ["diag_first_steps", "diag_snapshots", "max_train_batches", "max_eval_batches", "lr_patience", "min_epochs", "cpu_threads"]:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if (not 0 < args.lr_factor < 1 or not math.isfinite(args.min_lr) or args.min_lr < 0
            or (args.scheduler == "plateau" and args.min_lr > args.lr)):
        raise ValueError("lr-factor must be in (0,1); min-lr must be finite and in [0,lr]")
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        raise ValueError("grad-clip must be finite and nonnegative")
    ratio = args.history / args.patch_size
    power = round(math.log2(ratio))
    if power < 0 or not math.isclose(ratio, 2 ** power, rel_tol=1e-9):
        raise ValueError("history / patch_size must be a power of two for adjacent-patch merging")
    args.stride = args.patch_size
    args.npatch = 2 ** power
    args.multi = power + 1
    model_dim = args.hid_dim * args.npatch * args.multi
    if model_dim % 2 or (args.num_layers > 0 and model_dim % 8):
        raise ValueError(f"Attention dimension {model_dim} must be divisible by 8; adjust hid_dim or patch_size")
    head = getattr(args, "variable_head", "none")
    rank = getattr(args, "variable_head_rank", 4)
    penalty = getattr(args, "variable_head_penalty", 0.01)
    if head not in {"none", "fixed", "learned"}:
        raise ValueError("Unsupported variable-head mode")
    if rank < 1 or (head != "none" and rank > model_dim):
        raise ValueError("variable-head-rank must be positive and no greater than the model dimension")
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("variable-head-penalty must be finite and nonnegative")
    if head != "none" and args.variant not in {"legacy", "local", "local_residual"}:
        raise ValueError("variable-head supports legacy, local, and local_residual variants")


def project_path(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Outputs and checkpoints must stay inside {ROOT}")
    return path


def output_path(args):
    values = [value for value in (args.output_dir, args.save, args.diagnostics_dir) if value is not None]
    if values:
        paths = [project_path(value) for value in values]
        if any(path != paths[0] for path in paths):
            raise ValueError("Output, checkpoint and diagnostics paths must be the same directory")
        return resolve_output_dir(ROOT, paths[0])
    return resolve_output_dir(ROOT)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path, payload):
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def memory_status(device):
    import torch
    status = {"device": str(device), "cuda": device.type == "cuda"}
    if not status["cuda"]:
        return status
    try:
        # CUDA_VISIBLE_DEVICES remaps physical GPUs to process-local indices.
        index = device.index if device.index is not None else torch.cuda.current_device()
        status.update(device=str(torch.device("cuda", index)), device_index=index)
        free, total = torch.cuda.mem_get_info(index)
        status.update(device_free_bytes=free, device_total_bytes=total)
        status["process_allocated_bytes"] = torch.cuda.memory_allocated(index)
        status["process_reserved_bytes"] = torch.cuda.memory_reserved(index)
        status["process_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(index)
    except Exception as error:
        # Memory telemetry is optional.
        status.update(query_error=str(error), query_error_type=type(error).__name__)
    return status


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        run_dir = output_path(args)
    except ValueError as error:
        parser.error(str(error))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Dataset loaders resolve ../data relative to the original script directory.
    os.chdir(ROOT / "IMTSTokenizer")
    import torch
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    from lib import utils
    from lib.diagnostics import Diagnostics
    from lib.diagnostic_metrics import ForecastDiagnostics
    from lib.branch_diagnostics import BranchDiagnostics
    from lib.objective_diagnostics import ObjectiveAlignmentDiagnostics
    from lib.optimization_diagnostics import OptimizationDiagnostics
    from lib.evaluation import compute_all_losses, evaluation
    from lib.evaluation_rng import evaluation_rng
    from lib.parse_datasets import parse_datasets
    from IMTSTokenizer.model.factory import build_model, initialize_weights, check_source_reliability_identity, check_target_query_identity

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    args.device = torch.device("cuda" if requested_device == "cuda" or
                               (requested_device == "auto" and torch.cuda.is_available()) else "cpu")
    args.PID = os.getpid()
    if requested_device == "cpu":
        utils.setup_seed(args.seed, cpu_only=True)
    else:
        utils.setup_seed(args.seed)
    run_dir = create_output_dir(ROOT, run_dir)
    run_id = run_dir.relative_to((ROOT / "experiments").resolve()).as_posix()
    legacy_save_dir = project_path(args.save or "experiments")
    args.output_dir = args.save = args.diagnostics_dir = str(run_dir)
    ckpt_path = run_dir / "best.ckpt"
    checkpoint_used = None
    logger = utils.get_logger(str(run_dir / "train.log"), str(Path(__file__).resolve()), mode=args.logmode)
    metadata = {"run_id": run_id, "args": json_safe(vars(args)), "command": sys.argv,
                "python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "partial_run": bool(args.max_train_batches or args.max_eval_batches),
                "normalization_scope": "train+validation values and times (USHCN values remain preprocessed)",
                "split_protocol": "record shuffle seed=42, 60/20/20 before time chunking"}
    write_json(run_dir / "run.json", metadata)
    diag = Diagnostics(run_dir, enabled=args.diagnostics == "on", every=args.diag_every,
                       first_steps=args.diag_first_steps, max_elements=args.diag_max_elements,
                       snapshots=args.diag_snapshots, metadata=metadata, storage=args.artifact_storage)
    logger.info("Run %s | device %s | diagnostics %s", run_id, args.device, run_dir)
    best_val_mse, best_iter, global_step = float("inf"), -1, 0
    test_res = None
    status = "running"
    optimization_observer = None
    stage = "data_loading"
    started = time.perf_counter()
    try:
        diag.event("memory", stage="before_data_loading", **memory_status(args.device))
        args.collect_objective_counts = diag.enabled
        data_obj = parse_datasets(args, patch_ts=True)
        args.ndim = data_obj["input_dim"]
        if args.num_var is None:
            args.num_var = args.ndim
        if args.num_var != args.ndim:
            raise ValueError(f"num_var={args.num_var} disagrees with dataset input_dim={args.ndim}")
        for split in ["train", "val", "test"]:
            if data_obj[f"n_{split}_batches"] <= 0:
                raise ValueError(f"{split} loader is empty")
        args.time_max = float(data_obj["time_max"])
        if "training_value_prior" in data_obj:
            args.value_prior = data_obj["training_value_prior"].tolist()
        if "training_target_counts" in data_obj:
            args.training_target_counts = data_obj["training_target_counts"].tolist()
        model = build_model(args).to(args.device)
        if args.init_checkpoint:
            initial_path = project_path(args.init_checkpoint)
            initial_checkpoint = load_checkpoint(initial_path, map_location="cpu", weights_only=False)
            initialized = initialize_weights(model, initial_checkpoint)
            metadata["initialization"] = {"path": str(initial_path),
                "epoch": initial_checkpoint.get("epoch"), "validation_mse": initial_checkpoint.get("best_val_mse"),
                **initialized, "optimizer": "fresh"}
            diag.event("initialized_checkpoint", **metadata["initialization"])
        model.diagnostics = diag
        model.collect_diagnostic_predictions = diag.enabled
        model.objective_diagnostics = None
        model.optimization_diagnostics = None
        diag.event("memory", stage="after_model_creation", **memory_status(args.device))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
        if args.scheduler == "milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.mile_stone, gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
                threshold=0.0, min_lr=args.min_lr)
        metadata.update({"args": json_safe(vars(args)), "parameter_count": sum(p.numel() for p in model.parameters()),
                         "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                         "training_policy": "fixed_core_correction" if args.correction_only else "joint_model",
                         "data": {key: json_safe(value) for key, value in data_obj.items() if "dataloader" not in key}})
        data_min, data_max = data_obj["data_min"], data_obj["data_max"]
        metadata["normalization_audit"] = {
            "nonfinite_bound_variables": torch.where(~torch.isfinite(data_min) | ~torch.isfinite(data_max))[0].tolist(),
            "constant_range_variables": torch.where(data_min == data_max)[0].tolist(),
            "value_normalization_applied": args.dataset != "ushcn",
        }
        time_max = float(data_obj["time_max"])
        forecast_window = float(getattr(args, "pred_window", max(time_max - args.history, 1e-8)))
        normalized_window = forecast_window / max(time_max, 1e-8)
        horizon_edges = (0.0, normalized_window / 4, normalized_window / 2,
                         3 * normalized_window / 4, float("inf"))
        metadata["diagnostic_horizon_edges"] = json_safe(horizon_edges)
        metadata["diagnostic_horizon_definition"] = "query_time - last observed sample time, in normalized model units; last bin unbounded"
        write_json(run_dir / "run.json", metadata)
        diag.event("data_ready", **metadata["data"])
        diag.event("normalization_audit", **metadata["normalization_audit"])

        def evaluate(split, epoch):
            nonlocal stage
            stage = split
            model.eval()
            observer = ForecastDiagnostics(horizon_edges=horizon_edges) if diag.enabled else None
            model.prediction_diagnostics = observer
            branch_observer = BranchDiagnostics(args.history / args.time_max) if diag.enabled else None
            model.branch_diagnostics = branch_observer
            count = data_obj[f"n_{split}_batches"]
            if args.max_eval_batches:
                count = min(count, args.max_eval_batches)
            with evaluation_rng(args.eval_rng_policy, args.device), torch.no_grad():
                result = evaluation(model, data_obj[f"{split}_dataloader"], count, data_obj,
                                    diagnostics=diag, epoch=epoch, step=global_step, split=split)
            details = observer.summary() if observer is not None else None
            diag.event("split_summary", split=split, epoch=epoch, step=global_step,
                       metrics=result, diagnostics=details, n_batches=count,
                       branch_diagnostics=branch_observer.summary() if branch_observer else None)
            if any(not math.isfinite(result[name]) for name in ("loss", "mse", "mae", "rmse")):
                raise FloatingPointError(f"Nonfinite {split} error; inspect masks, values, and predictions")
            return result, details

        if args.load:
            candidate = project_path(args.load)
            if not candidate.is_file():
                candidate = project_path(legacy_save_dir / ("experiment_" + args.load + ".ckpt"))
            checkpoint = load_checkpoint(candidate, map_location=args.device, weights_only=False)
            check_source_reliability_identity(args, checkpoint.get("args", {}))
            check_target_query_identity(args, checkpoint.get("args", {}))
            model.load_state_dict(checkpoint["state_dict"])
            checkpoint_used = candidate
            best_iter = checkpoint.get("epoch", -1)
            global_step = checkpoint.get("step", 0)
            val_res, _ = evaluate("val", best_iter)
            if args.test_policy != "never":
                test_res, _ = evaluate("test", best_iter)
            best_val_mse = val_res["mse"]
            status = "evaluated"
            logger.info("Loaded %s | validation %s | test %s", candidate, val_res, test_res)
        else:
            if args.init_checkpoint:
                val_res, _ = evaluate("val", -1)
                best_val_mse = val_res["mse"]
                if args.scheduler == "plateau":
                    scheduler.step(best_val_mse)
                if args.test_policy == "best":
                    test_res, _ = evaluate("test", -1)
                torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "epoch": -1, "step": 0,
                            "args": json_safe(vars(args)), "best_val_mse": best_val_mse}, ckpt_path)
                checkpoint_used = ckpt_path
                diag.event("best_checkpoint", epoch=-1, path=str(ckpt_path), validation=val_res,
                           test=test_res, initialized=True)
            num_batches = data_obj["n_train_batches"]
            if args.max_train_batches:
                num_batches = min(num_batches, args.max_train_batches)
            for epoch in range(args.epoch):
                stage = "train"
                epoch_start = time.perf_counter()
                model.train()
                observer = ForecastDiagnostics(horizon_edges=horizon_edges) if diag.enabled else None
                model.prediction_diagnostics = observer
                branch_observer = BranchDiagnostics(args.history / args.time_max) if diag.enabled else None
                model.branch_diagnostics = branch_observer
                objective_observer = ObjectiveAlignmentDiagnostics(
                    args.num_var, training_counts=data_obj.get("training_target_counts"),
                    training_batches=data_obj["n_train_batches"], reduction=args.loss_reduction
                ) if diag.enabled else None
                model.objective_diagnostics = objective_observer
                optimization_observer = OptimizationDiagnostics(args.num_var) if diag.enabled else None
                model.optimization_diagnostics = optimization_observer
                loss_sum = 0.0
                for batch_index in range(num_batches):
                    batch = utils.get_next_batch(data_obj["train_dataloader"])
                    diag.begin_batch("train", epoch, batch_index, global_step)
                    if diag.active:
                        diag.scalar("memory/before_forward", memory_status(args.device))
                    optimizer.zero_grad()
                    with torch.autograd.set_detect_anomaly(args.detect_anomaly):
                        train_res = compute_all_losses(model, batch, data_obj, metrics=False)
                        if not torch.isfinite(train_res["loss"]):
                            diag.event("nonfinite_loss", split="train", epoch=epoch, step=global_step)
                            raise FloatingPointError("Nonfinite training loss; inspect diagnostics before changing the model")
                        train_res["loss"].backward()
                    if diag.active:
                        diag.scalar("memory/after_backward", memory_status(args.device))
                    diag.parameters(model, when="before")
                    finite_gradients = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                                                    if p.grad is not None]).all()
                    if not finite_gradients:
                        diag.event("nonfinite_gradient", epoch=epoch, step=global_step)
                        raise FloatingPointError("Nonfinite gradient; optimizer update cancelled")
                    preclip_norm, applied_clip_scale = None, 1.0
                    if args.grad_clip:
                        preclip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        if diag.active:
                            diag.scalar("optimization/preclip_gradient_norm", preclip_norm)
                            diag.scalar("optimization/gradient_clip_scale", min(1.0, args.grad_clip / (float(preclip_norm) + 1e-6)))
                        if optimization_observer is not None:
                            applied_clip_scale = torch.clamp(args.grad_clip / (preclip_norm.detach() + 1e-6), max=1.0)
                    optimizer.step()
                    if optimization_observer is not None:
                        optimization_observer.complete_step(preclip_norm, applied_clip_scale)
                    diag.parameters(model, when="after")
                    diag.scalar("optimization/loss", train_res["loss"].detach())
                    diag.scalar("optimization/lr", optimizer.param_groups[0]["lr"])
                    diag.end_batch()
                    loss_sum += train_res["loss"].item()
                    global_step += 1
                train_details = observer.summary() if observer is not None else None
                model.objective_diagnostics = None
                model.optimization_diagnostics = None
                if optimization_observer is not None:
                    optimization_observer.close()
                    diag.event("optimization_alignment", split="train", epoch=epoch, step=global_step,
                               n_batches=num_batches, full_training_batches=data_obj["n_train_batches"],
                               partial_epoch=num_batches < data_obj["n_train_batches"],
                               diagnostics=optimization_observer.summary())
                    optimization_observer = None
                if objective_observer is not None:
                    diag.event("objective_alignment", split="train", epoch=epoch, step=global_step,
                               diagnostics=objective_observer.summary())
                diag.event("split_summary", split="train", epoch=epoch, step=global_step,
                           mean_batch_loss=loss_sum / num_batches, diagnostics=train_details, n_batches=num_batches,
                           branch_diagnostics=branch_observer.summary() if branch_observer else None)
                previous_lr = optimizer.param_groups[0]["lr"]
                if args.scheduler == "milestone":
                    scheduler.step()
                val_res, _ = evaluate("val", epoch)
                if args.scheduler == "plateau":
                    scheduler.step(val_res["mse"])
                if optimizer.param_groups[0]["lr"] != previous_lr:
                    diag.event("learning_rate_changed", epoch=epoch, previous_lr=previous_lr,
                               next_lr=optimizer.param_groups[0]["lr"], scheduler=args.scheduler)
                improved = val_res["mse"] < best_val_mse
                if improved:
                    best_val_mse, best_iter = val_res["mse"], epoch
                    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(), "epoch": epoch, "step": global_step,
                                "args": json_safe(vars(args)), "best_val_mse": best_val_mse}, ckpt_path)
                    checkpoint_used = ckpt_path
                    if args.test_policy == "best":
                        test_res, _ = evaluate("test", epoch)
                    diag.event("best_checkpoint", epoch=epoch, path=str(ckpt_path), validation=val_res, test=test_res)
                    if test_res is not None:
                        logger.info("Test at best validation epoch %d | MSE %.6f MAE %.6f", epoch, test_res["mse"], test_res["mae"])
                elapsed = time.perf_counter() - epoch_start
                epoch_memory = memory_status(args.device)
                diag.event("epoch_end", epoch=epoch, seconds=elapsed, mean_batch_loss=loss_sum / num_batches,
                           validation=val_res, test_at_best_validation=test_res, best_epoch=best_iter,
                           next_lr=optimizer.param_groups[0]["lr"],
                           cuda_peak_bytes=epoch_memory.get("process_peak_allocated_bytes"), memory=epoch_memory)
                logger.info("Epoch %03d | train mean batch loss %.6f | val MSE %.6f MAE %.6f | best %d | %.2fs",
                            epoch, loss_sum / num_batches, val_res["mse"], val_res["mae"], best_iter, elapsed)
                if epoch - best_iter >= args.patience and epoch + 1 >= args.min_epochs:
                    status = "early_stopped"
                    break
            if status == "running":
                status = "completed"
            if args.test_policy == "final":
                selected_checkpoint = load_checkpoint(ckpt_path, map_location=args.device, weights_only=False)
                model.load_state_dict(selected_checkpoint["state_dict"])
                test_res, _ = evaluate("test", best_iter)
                diag.event("final_test", epoch=best_iter, path=str(ckpt_path), test=test_res)
                logger.info("Final test at best validation epoch %d | MSE %.6f MAE %.6f",
                            best_iter, test_res["mse"], test_res["mae"])
    except BaseException as error:
        status = "failed"
        diag.event("failure", error_type=type(error).__name__, message=str(error), stage=stage,
                   step=global_step, memory=memory_status(args.device))
        logger.exception("Run failed")
        raise
    finally:
        if optimization_observer is not None:
            optimization_observer.close()
        diag.close()
        write_json(run_dir / "summary.json", {"status": status, "run_id": run_id,
                   "best_epoch": best_iter, "best_val_mse": best_val_mse, "test_at_best_validation": test_res,
                   "global_step": global_step, "checkpoint": str(checkpoint_used) if checkpoint_used is not None else None,
                   "initialization_selected": bool(args.init_checkpoint and best_iter == -1 and checkpoint_used is not None),
                   "test_policy": args.test_policy, "eval_rng_policy": args.eval_rng_policy,
                   "last_stage": stage,
                   "seconds": time.perf_counter() - started, "partial_run": metadata["partial_run"]})
    logger.info("Finished: %s | outputs %s", status, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
