"""Fit random-subspace, full, and hierarchical single-model residual readouts."""
import argparse
from collections import Counter
from contextlib import ExitStack
import copy
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

JOINT_SHRINK_SOURCES = {"joint_shrink_local": "full_local", "joint_shrink_hierarchical": "hierarchical",
                        "joint_shrink_spectral": "spectral", "joint_shrink_spectral_hierarchical": "spectral_hierarchical",
                        "joint_shrink_nested_spectral": "nested_spectral",
                        "joint_shrink_nested_spectral_hierarchical": "nested_spectral_hierarchical"}
BLOCK_SHRINK_SOURCES = {"block_shrink_nested_spectral": "nested_spectral",
                        "block_shrink_nested_spectral_hierarchical": "nested_spectral_hierarchical"}
SHRINK_SOURCES = {"shrink_local": "full_local", "shrink_hierarchical": "hierarchical",
                 "shrink_support": "support", "shrink_centered": "centered",
                 "shrink_spectral": "spectral", "shrink_spectral_hierarchical": "spectral_hierarchical",
                 **JOINT_SHRINK_SOURCES}
SPECTRAL_SOURCES = {"spectral", "spectral_hierarchical"}
NESTED_SPECTRAL_SOURCES = {"nested_spectral", "nested_spectral_hierarchical"}
CV_SOURCES = {"cv_local": "full_local", "cv_centered": "centered", "cv_hierarchical": "hierarchical"}
GUARDED_SOURCES = {"guard_local": "full_local", "guard_centered": "centered", "guard_hierarchical": "hierarchical"}
READOUT_MODES = ("random4", "full_local", "full", "hierarchical", "diagonal", "interaction",
                "support", "context", "spectral", "spectral_hierarchical", "nested_spectral", "nested_spectral_hierarchical",
                *SHRINK_SOURCES, *BLOCK_SHRINK_SOURCES,
                "full_local_fine", "centered", *CV_SOURCES, *GUARDED_SOURCES)


def mode_identity(mode, *, nested_rank=16):
    if mode not in READOUT_MODES:
        raise ValueError("Unsupported readout mode")
    source = BLOCK_SHRINK_SOURCES.get(mode, GUARDED_SOURCES.get(mode, CV_SOURCES.get(mode, SHRINK_SOURCES.get(mode, mode))))
    result = {"ridge_mode": "readout" if source == "random4" else "none",
              "fitted_readout": "none" if source == "random4" else "hierarchical" if source in {"hierarchical", "spectral_hierarchical", "nested_spectral_hierarchical"} else "full",
              "readout_basis": "spectral_nested" if source in NESTED_SPECTRAL_SOURCES else "spectral" if source in SPECTRAL_SOURCES else source if source in {"diagonal", "interaction", "context"} else "full",
              "readout_shrinkage": "block" if mode in BLOCK_SHRINK_SOURCES else "variable" if mode in SHRINK_SOURCES or mode in CV_SOURCES or mode in GUARDED_SOURCES else "none"}
    if source in NESTED_SPECTRAL_SOURCES:
        result["ridge_rank"] = nested_rank
    return result


def mode_training_method(mode):
    if mode in BLOCK_SHRINK_SOURCES:
        return "train_fit_joint_block_validation_selection"
    if mode in JOINT_SHRINK_SOURCES:
        return "train_fit_joint_validation_selection"
    if mode in CV_SOURCES:
        return "train_group_cv_readout"
    if mode in GUARDED_SOURCES:
        return "train_fit_group_guarded_validation_shrinkage"
    return "train_fit_validation_shrinkage" if mode in SHRINK_SOURCES else "train_only_full_feature_readout"


def _fit_source(mode):
    source = BLOCK_SHRINK_SOURCES.get(mode, GUARDED_SOURCES.get(mode, CV_SOURCES.get(mode, SHRINK_SOURCES.get(mode, mode))))
    return "full_local" if source == "full_local_fine" else source


def _candidate_key(stats, validation, source_mode, local, global_, context_penalty, options):
    """Only share identical immutable statistics and the exact fitting protocol."""
    return (id(stats), id(validation), _fit_source(source_mode), local, global_,
            context_penalty, options.uniform_ridge, options.support_power, options.bias_penalty)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-dataset", choices=["activity", "physionet", "mimic", "ushcn"])
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--modes", nargs="+", choices=READOUT_MODES,
                        default=["random4", "full", "hierarchical"])
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=[.001, .01, .1, 1., 10.])
    parser.add_argument("--global-ridge-grid", nargs="+", type=float, default=[.001, .01, .1, 1., 10.])
    parser.add_argument("--adaptive-ridge-grid", nargs="+", type=float,
                        default=[10 ** (index / 2) for index in range(-6, 3)])
    parser.add_argument("--support-power", type=float, default=1.5)
    parser.add_argument("--context-penalty-grid", nargs="+", type=float, default=[1., 10., 100.])
    parser.add_argument("--reliability-diagnostics", choices=["on", "off"], default="off")
    parser.add_argument("--shrink-grid", nargs="+", type=float, default=[1., .75, .5, .25, 0.])
    parser.add_argument("--shrink-min-count", type=int, default=64)
    parser.add_argument("--bias-penalty", type=float, default=.1)
    parser.add_argument("--group-folds", type=int, default=3)
    parser.add_argument("--cv-gain-grid", nargs="+", type=float, default=[1., .75, .5, .25, 0.])
    parser.add_argument("--cv-min-count", type=int, default=64)
    parser.add_argument("--candidate-storage", choices=["inline", "gzip"], default="inline")
    parser.add_argument("--feature-cache", choices=["off", "trainval"], default="off")
    parser.add_argument("--artifact-storage", choices=["full", "compact"], default="full")
    parser.add_argument("--uniform-ridge", action="store_true")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--extra-rank", type=int, default=4)
    parser.add_argument("--nested-rank", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--diagnostics", choices=["on", "off"], default="on")
    parser.add_argument("--output-dir")
    return parser


def validate_options(options):
    if options.expected_seed is not None and not 0 <= options.expected_seed < 2 ** 32:
        raise ValueError("expected-seed must be in [0,2**32)")
    if not options.modes or len(set(options.modes)) != len(options.modes):
        raise ValueError("modes must be nonempty and unique")
    for name in ("ridge_grid", "global_ridge_grid", "adaptive_ridge_grid", "context_penalty_grid"):
        values = getattr(options, name)
        if (not values or len(set(values)) != len(values)
                or any(not math.isfinite(value) or value <= 0 for value in values)):
            raise ValueError(name + " must contain unique finite positive numbers")
    if not math.isfinite(options.support_power) or options.support_power < 0:
        raise ValueError("support-power must be finite and nonnegative")
    if (not options.shrink_grid or 1.0 not in options.shrink_grid
            or len(set(options.shrink_grid)) != len(options.shrink_grid)
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in options.shrink_grid)):
        raise ValueError("shrink-grid requires unique finite values in [0,1], including1")
    if type(options.shrink_min_count) is not int or options.shrink_min_count < 1:
        raise ValueError("shrink-min-count must be a positive integer")
    if not math.isfinite(options.bias_penalty) or options.bias_penalty < 0:
        raise ValueError("bias-penalty must be finite and nonnegative")
    if type(options.group_folds) is not int or options.group_folds < 2:
        raise ValueError("group-folds must be an integer of at least two")
    if (not options.cv_gain_grid or 1.0 not in options.cv_gain_grid
            or len(set(options.cv_gain_grid)) != len(options.cv_gain_grid)
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in options.cv_gain_grid)):
        raise ValueError("cv-gain-grid requires unique finite values in [0,1], including1")
    if type(options.cv_min_count) is not int or options.cv_min_count < 1:
        raise ValueError("cv-min-count must be a positive integer")
    if options.uniform_ridge and set(options.modes) & ({"centered", "shrink_centered"} | set(CV_SOURCES) | set(GUARDED_SOURCES)):
        raise ValueError("centered/group-CV modes use a fixed support protocol; uniform-ridge is unsupported")
    if min(options.rank, options.extra_rank, options.batch_size, options.cpu_threads, options.chunk_rows) < 1:
        raise ValueError("rank, batch-size, cpu-threads and chunk-rows must be positive")
    if type(options.nested_rank) is not int or options.nested_rank < 4:
        raise ValueError("nested-rank must be an integer of at least four")
    if min(options.max_train_batches, options.max_val_batches) < 0:
        raise ValueError("batch limits must be nonnegative")


def collect(model, loader, batches, options, diagnostics, split, cache=False, extra_statistics=None,
            context_statistics=None, reliability_output=None, group_statistics=None, feature_cache=None,
            spectral_statistics=None, nested_spectral_statistics=None):
    import torch
    from lib import utils
    from lib.hierarchical_readout_stats import FullReadoutStatistics
    from lib.ridge_stats import RidgeStatistics
    from IMTSTokenizer.run_models import memory_status
    full = FullReadoutStatistics(model.args.num_var, model.projection.shape[0])
    if group_statistics is not None:
        if split != "train" or group_statistics:
            raise ValueError("Group statistics require a fresh training-only list")
        group_statistics.extend(FullReadoutStatistics(model.args.num_var, model.projection.shape[0])
                                for _ in range(options.group_folds))
    projected = RidgeStatistics(model.args.num_var, len(model.basis_names))
    transform = None
    if extra_statistics is not None:
        from IMTSTokenizer.model.InteractionBasis import ReadoutInteractionBasis
        transform = ReadoutInteractionBasis(model.projection.shape[0], options.rank, options.extra_rank).to(model.projection)
        for mode in ("diagonal", "interaction"):
            extra_statistics[mode] = FullReadoutStatistics(model.args.num_var, options.rank + options.extra_rank)
    spectral_transform = None
    if spectral_statistics is not None:
        from IMTSTokenizer.model.SpectralContextBasis import SpectralContextBasis
        spectral_transform = SpectralContextBasis(model.projection.shape[0], options.rank).to(model.projection)
        spectral_statistics["spectral"] = FullReadoutStatistics(model.args.num_var, spectral_transform.feature_dim)
        spectral_statistics["query_range"] = {"min": None, "max": None, "observed_query_slots": 0,
                                              "definition": "Query slots having at least one observed target; padding excluded"}
    nested_transform = None
    if nested_spectral_statistics is not None:
        from IMTSTokenizer.model.SpectralContextBasis import SpectralContextBasis
        nested_transform = SpectralContextBasis(model.projection.shape[0], options.nested_rank, nested=True).to(model.projection)
        nested_spectral_statistics["spectral_nested"] = FullReadoutStatistics(model.args.num_var, nested_transform.feature_dim)
        nested_spectral_statistics["projection"] = nested_transform.projection.detach().cpu().tolist()
        nested_spectral_statistics["query_range"] = {"min": None, "max": None, "observed_query_slots": 0,
                                                     "definition": "Query slots having at least one observed target; padding excluded"}
    context_transform = reliability = None
    if context_statistics is not None or reliability_output is not None or feature_cache is not None:
        from IMTSTokenizer.model.ContextReadoutBasis import ContextReadoutBasis, context_state_features
        width = model.args.history / model.args.time_max
        if context_statistics is not None:
            context_transform = ContextReadoutBasis(model.projection.shape[0], width, options.rank).to(model.projection)
            context_statistics["context"] = FullReadoutStatistics(model.args.num_var, context_transform.output_dim)
        if reliability_output is not None:
            from lib.reliability_diagnostics import ReliabilityDiagnostics
            reliability = ReliabilityDiagnostics(width)
    cached = [] if cache else None
    with torch.no_grad():
        for index in range(batches):
            batch = utils.get_next_batch(loader)
            if cached is not None:
                cached.append(batch)
            captured = model.design(
                batch["tp_to_predict"], batch["observed_data"], batch["observed_tp"],
                batch["observed_mask"], batch.get("window_start"), return_readout=True,
                **({"return_components": True} if transform is not None or spectral_transform is not None or nested_transform is not None else {}))
            core, basis, features = captured[:3]
            state = None
            if context_transform is not None:
                expanded, state = context_transform.compute(
                    features, batch["tp_to_predict"], batch["observed_data"], batch["observed_tp"],
                    batch["observed_mask"], batch.get("window_start"))
                context_statistics["context"].update_features(
                    expanded, core, batch["data_to_predict"], batch["mask_predicted_data"], chunk_rows=options.chunk_rows)
            if reliability is not None:
                if state is None:
                    state = context_state_features(features, batch["tp_to_predict"], batch["observed_data"],
                                                   batch["observed_tp"], batch["observed_mask"], width,
                                                   batch.get("window_start"))
                reliability.record(batch, core, components={"fitted_core": core}, target_tensors={
                    "model/fitted_readout": {"context": state, "context_names": ContextReadoutBasis.context_names}})
            if feature_cache is not None:
                if state is None:
                    state = context_state_features(features, batch["tp_to_predict"], batch["observed_data"],
                                                   batch["observed_tp"], batch["observed_mask"], width,
                                                   batch.get("window_start"))
                feature_cache.update(features, core, batch["data_to_predict"], batch["mask_predicted_data"],
                                     batch["tp_to_predict"], fold_labels=batch.get("readout_group_fold"), states=state)
            if transform is not None:
                query, context = captured[3:]
                for mode, stats in extra_statistics.items():
                    expanded = transform(features, query, context, kind=mode)
                    stats.update_features(expanded, core, batch["data_to_predict"],
                                          batch["mask_predicted_data"], chunk_rows=options.chunk_rows)
            if spectral_transform is not None:
                query, context = captured[3:]
                expanded = spectral_transform(features, query, context, batch["tp_to_predict"])
                spectral_statistics["spectral"].update_features(
                    expanded, core, batch["data_to_predict"], batch["mask_predicted_data"],
                    chunk_rows=options.chunk_rows)
                observed_query = (batch["mask_predicted_data"] > 0).any(dim=-1)
                query_values = batch["tp_to_predict"][observed_query]
                if query_values.numel():
                    bounds = spectral_statistics["query_range"]
                    low, high = float(query_values.min()), float(query_values.max())
                    bounds["min"] = low if bounds["min"] is None else min(bounds["min"], low)
                    bounds["max"] = high if bounds["max"] is None else max(bounds["max"], high)
                    bounds["observed_query_slots"] += query_values.numel()
            if nested_transform is not None:
                query, context = captured[3:]
                expanded = nested_transform(features, query, context, batch["tp_to_predict"])
                nested_spectral_statistics["spectral_nested"].update_features(
                    expanded, core, batch["data_to_predict"], batch["mask_predicted_data"],
                    chunk_rows=options.chunk_rows)
                observed_query = (batch["mask_predicted_data"] > 0).any(dim=-1)
                query_values = batch["tp_to_predict"][observed_query]
                if query_values.numel():
                    bounds = nested_spectral_statistics["query_range"]
                    low, high = float(query_values.min()), float(query_values.max())
                    bounds["min"] = low if bounds["min"] is None else min(bounds["min"], low)
                    bounds["max"] = high if bounds["max"] is None else max(bounds["max"], high)
                    bounds["observed_query_slots"] += query_values.numel()
            full.update_features(features, core, batch["data_to_predict"],
                                 batch["mask_predicted_data"], chunk_rows=options.chunk_rows)
            if group_statistics is not None:
                labels = batch.get("readout_group_fold")
                if (not torch.is_tensor(labels) or labels.dtype != torch.long or labels.device.type != "cpu"
                        or labels.shape != (features.shape[0],)
                        or bool(((labels < 0) | (labels >= options.group_folds)).any())):
                    raise ValueError("Training batch requires valid CPU record-group fold labels")
                target_mask = batch["mask_predicted_data"]
                for fold, stats in enumerate(group_statistics):
                    supported = (labels == fold).to(target_mask)[:, None, None]
                    stats.update_features(features, core, batch["data_to_predict"], target_mask * supported,
                                          chunk_rows=options.chunk_rows)
            # Preserve the old hybrid-statistic reduction before selecting its readout columns.
            projected.update(basis, core, batch["data_to_predict"], batch["mask_predicted_data"])
            if (index + 1) % 100 == 0 or index + 1 == batches:
                print(split, "statistics", index + 1, "/", batches, flush=True)
                diagnostics.event("readout_statistics_progress", split=split, batches=index + 1,
                                  total_batches=batches, memory=memory_status(core.device))
    if reliability is not None:
        reliability_output.update(stage="pre_readout_core", summary=reliability.summary())
    return full, projected.select([0] + list(range(8, 8 + options.rank))), cached, batches


def conditional_novelty(stats, core_rank):
    """Train-only residual energy of new columns given the old columns and bias."""
    import torch
    rows, start = [], core_rank + 1
    for variable in range(stats.variables):
        count = stats.counts[variable]
        if count <= 0:
            rows.append({"variable": variable, "train_count": 0., "new_energy_fraction": None})
            continue
        gram = stats.gram[variable] / count
        scale = gram.diagonal().clamp_min(1e-16).sqrt()
        gram = gram / (scale[:, None] * scale[None, :])
        old, cross, new = gram[:start, :start], gram[:start, start:], gram[start:, start:]
        conditional = new - cross.T @ torch.linalg.pinv(old, hermitian=True, rtol=1e-10) @ cross
        conditional = (conditional + conditional.T) / 2
        centered_energy = (new.diagonal() - gram[0, start:].square()).sum()
        rows.append({"variable": variable, "train_count": float(count),
                     "new_energy_fraction": float((conditional.trace() / centered_energy).clamp(0, 1)) if centered_energy > 1e-14 else None,
                     "conditional_gram_eigenvalues": torch.linalg.eigvalsh(conditional).tolist()})
    return {"definition": "Train feature covariance: residual energy given the intercept and original projection", "per_variable": rows}


def feature_shift(train, validation):
    import torch
    positive = train.counts > 0
    count = train.counts.clamp_min(1)
    val_count = validation.counts.clamp_min(1)
    common = (train.gram.diagonal(dim1=-2, dim2=-1)[positive] / count[positive, None]).mean(0).sqrt().clamp_min(1e-8)
    mean = train.gram[:, 0, :] / count[:, None]
    val_mean = validation.gram[:, 0, :] / val_count[:, None]
    covariance = train.gram / count[:, None, None] - mean[:, :, None] * mean[:, None, :]
    val_covariance = validation.gram / val_count[:, None, None] - val_mean[:, :, None] * val_mean[:, None, :]
    scaling = common[:, None] * common[None, :]
    covariance, val_covariance = covariance / scaling, val_covariance / scaling
    train_cross = train.cross / (count[:, None] * common)
    val_cross = validation.cross / (val_count[:, None] * common)
    rows = []
    for variable in range(train.variables):
        active = bool(positive[variable] and validation.counts[variable] > 0)
        norm = covariance[variable].norm()
        denominator = train_cross[variable].norm() * val_cross[variable].norm()
        rows.append({"variable": variable, "train_count": float(train.counts[variable]),
                     "validation_count": float(validation.counts[variable]),
                     "normalized_mean_shift_l2": float(((val_mean[variable] - mean[variable]) / common).norm()) if active else None,
                     "normalized_covariance_relative_shift": float((val_covariance[variable] - covariance[variable]).norm() / norm) if active and norm > 0 else None,
                     "residual_feature_moment_cosine": float((train_cross[variable] @ val_cross[variable] / denominator).clamp(-1, 1)) if active and denominator > 0 else None})
    return {"definition": "Observed-target moments in training macro-RMS coordinates",
            "train_common_feature_rms": common.tolist(), "per_variable": rows}


def main(argv=None):
    parser = make_parser()
    options = parser.parse_args(argv)
    try:
        validate_options(options)
        checkpoint_path = (ROOT / options.checkpoint).resolve()
        if not checkpoint_path.is_relative_to(ROOT) or not checkpoint_path.is_file():
            raise ValueError("checkpoint must be an existing file inside the project")
        destination = create_output_dir(ROOT, options.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu
    import torch
    from lib import utils
    from lib.parse_datasets import parse_datasets
    from lib.diagnostics import Diagnostics
    from lib.branch_diagnostics import BranchDiagnostics
    from lib.diagnostic_metrics import ForecastDiagnostics
    from lib.evaluation import evaluation
    from IMTSTokenizer.run_models import make_parser as core_parser, validate_args, write_json, json_safe, memory_status
    from IMTSTokenizer.model.factory import build_model
    from IMTSTokenizer.model.RidgeCorrector import RidgeCorrector
    from IMTSTokenizer.model.FittedReadout import FittedReadout

    started = time.perf_counter()
    previous_dir, previous_threads = Path.cwd(), torch.get_num_threads()
    report = {"kind": "single_model_full_feature_readout", "status": "running", "options": vars(options),
              "checkpoint": str(checkpoint_path.relative_to(ROOT)),
              "modes": {}, "PID": os.getpid(),
              "command": list(sys.argv) if argv is None else [str(Path(__file__).resolve())] + list(argv),
              "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
              "protocol": {"fit_split": "train", "selection_split": "val", "test_evaluated": False,
                           "core_fixed": True, "core_calls_per_prediction": 1, "data_rng_seed": 730017,
                           "training_method": "train_only_full_feature_readout", "solve_dtype": "float64",
                           "deployed_dtype": "float32", "normal_gradients_scope": "float64 fit and float32 deployed coefficients"}}
    compact_storage = options.artifact_storage == "compact"
    if compact_storage:
        report["artifact_layout"] = {
            "schema_version": 1, "results": "results.json.gz",
            "diagnostic_records": "split_records/*.json.gz",
            "definition": "Shared fit/split records, loaded with lib.artifact_io.resolve_reference",
        }

    def save_results():
        if compact_storage:
            from lib.artifact_io import write_json as write_artifact
            write_artifact(destination / "results.json", json_safe(report), compressed=True, compact=True)
        else:
            write_json(destination / "results.json", report)
    if set(options.modes) & set(SHRINK_SOURCES):
        report["protocol"].update(training_method="train_fit_validation_shrinkage",
                                 coefficients_fit_split="train", gain_selection_split="val",
                                 gain_selection="Per-variable validation gains after ridge selection",
                                 normal_gradients_scope="Unshrunk training-fitted coefficients")
    use_group_cv = bool(set(options.modes) & set(CV_SOURCES))
    use_guard = bool(set(options.modes) & set(GUARDED_SOURCES))
    use_group_statistics = use_group_cv or use_guard or options.feature_cache == "trainval"
    if use_group_cv or use_guard:
        report["protocol"].update(
            training_method="per_mode", selection_split="per_mode",
            coefficients_fit_split="train", gain_selection_split="per_mode",
            mode_parameter_selection={mode: "train_group_cv" if mode in CV_SOURCES else "val"
                                      for mode in options.modes},
            mode_gain_selection={mode: "train_group_cv" if mode in CV_SOURCES else
                                 "val_with_train_group_guard" if mode in GUARDED_SOURCES else
                                 "val" if mode in SHRINK_SOURCES else None for mode in options.modes},
            gain_selection="Per-mode discrete gains after source regularization is locked; see mode_gain_selection.",
            normal_gradients_scope="Unshrunk fitted coefficients; separate post-gain derivatives",
            group_cv_definition="Readout cross-validation with the neural core fitted on the full training set",
            group_assignment_uses="Training record identities")
    if set(options.modes) & (set(JOINT_SHRINK_SOURCES) | set(BLOCK_SHRINK_SOURCES)):
        report["protocol"].update(
            training_method="per_mode",
            gain_selection="Per-mode policy: joint_shrink/block_shrink modes calibrate each fixed candidate before comparing penalties; other modes preserve their original ordering.",
            mode_selection_order={mode: "candidate_fit_then_two_block_gains_then_effective_validation_ranking"
                                  if mode in BLOCK_SHRINK_SOURCES else "candidate_fit_then_gain_then_effective_validation_ranking"
                                  if mode in JOINT_SHRINK_SOURCES else "original_mode_protocol" for mode in options.modes},
            joint_selection_grid="Original source penalty grid and gain grid; no additional coefficient fits or neural forwards for feature collection")
    write_json(destination / "run.json", report)
    save_results()
    diagnostics = Diagnostics(destination, enabled=options.diagnostics == "on", every=50,
                              first_steps=2, max_elements=256, snapshots=0, metadata=report,
                              storage=options.artifact_storage)
    device = None
    fitting_scopes = ExitStack()
    try:
        torch.set_num_threads(options.cpu_threads)
        os.chdir(ROOT / "IMTSTokenizer")
        if options.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA fitting but CUDA is unavailable")
        device = torch.device("cuda:0" if options.device == "cuda" else "cpu")
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", weights_only=False)
        saved = checkpoint.get("args")
        if not isinstance(saved, dict) or "state_dict" not in checkpoint:
            raise ValueError("Input checkpoint must include model arguments and weights")
        if ((options.expected_dataset is not None and saved.get("dataset") != options.expected_dataset)
                or (options.expected_seed is not None and saved.get("seed") != options.expected_seed)):
            raise ValueError("Checkpoint dataset/seed differs from the scheduled task")
        if saved.get("ridge_mode", "none") != "none" or saved.get("fitted_readout", "none") != "none":
            raise ValueError("Use the trained pre-calibration core, not a previously fitted readout")
        args = core_parser().parse_args([])
        for key, value in saved.items():
            if key in vars(args):
                setattr(args, key, value)
        args.load = args.init_checkpoint = None
        args.device, args.gpu = device, options.gpu
        args.output_dir = args.save = args.diagnostics_dir = str(destination)
        args.batch_size, args.dataset_storage = options.batch_size, "cpu"
        args.collect_reference_prior, args.ridge_mode, args.fitted_readout = True, "none", "none"
        if use_group_statistics:
            args.collect_group_folds = options.group_folds
        validate_args(args, require_checkpoint=False)
        if options.device == "cpu":
            utils.setup_seed(730017, cpu_only=True)
        else:
            utils.setup_seed(730017)
        diagnostics.event("memory", stage="before_data", **memory_status(device))
        data = parse_datasets(args, patch_ts=True)
        args.num_var = args.ndim = data["input_dim"]
        args.time_max = float(data["time_max"])
        args.value_prior = data["training_value_prior"].tolist()
        args.training_target_counts = data["training_target_counts"].tolist()
        if saved.get("num_var", args.num_var) != args.num_var:
            raise ValueError("Checkpoint variable count differs from data")
        if "time_max" in saved and not math.isclose(float(saved["time_max"]), args.time_max, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Checkpoint time normalization differs from data")
        for key, value in (("value_prior", data["training_value_prior"]), ("training_target_counts", data["training_target_counts"])):
            if key not in saved:
                continue
            original = torch.as_tensor(saved[key], dtype=torch.float64)
            current = value.double().cpu()
            if original.shape != current.shape or not torch.allclose(original, current, rtol=1e-6 if key == "value_prior" else 0., atol=1e-8 if key == "value_prior" else 0.):
                raise ValueError("Checkpoint " + key + " differs from the training data")
        core = build_model(args).to(device)
        core.load_state_dict(checkpoint["state_dict"], strict=True)
        core.requires_grad_(False).eval()
        hybrid_args = copy.deepcopy(args)
        hybrid_args.ridge_mode, hybrid_args.ridge_rank = "hybrid", options.rank
        hybrid = RidgeCorrector(core, hybrid_args).to(device).eval()
        train_batches = min(data["n_train_batches"], options.max_train_batches or data["n_train_batches"])
        val_batches = min(data["n_val_batches"], options.max_val_batches or data["n_val_batches"])
        if min(train_batches, val_batches) < 1:
            raise ValueError("Training/validation loaders must be nonempty")
        report.update(dataset=args.dataset, seed=args.seed, args=vars(args), train_batches=train_batches,
                      validation_batches=val_batches, full_train_batches=data["n_train_batches"],
                      full_validation_batches=data["n_val_batches"],
                      partial_run=train_batches < data["n_train_batches"] or val_batches < data["n_val_batches"],
                      data={key: value for key, value in data.items() if "dataloader" not in key})
        write_json(destination / "run.json", {key: value for key, value in report.items() if key != "modes"})
        if options.device == "cpu":
            utils.setup_seed(730017, cpu_only=True)
        else:
            utils.setup_seed(730017)
        use_extra = bool(set(options.modes) & {"diagonal", "interaction"})
        extra_train, extra_validation = ({} if use_extra else None), ({} if use_extra else None)
        use_context = "context" in options.modes
        context_train, context_validation = ({} if use_context else None), ({} if use_context else None)
        use_spectral = any(_fit_source(mode) in SPECTRAL_SOURCES for mode in options.modes)
        spectral_train, spectral_validation = ({} if use_spectral else None), ({} if use_spectral else None)
        use_nested = any(_fit_source(mode) in NESTED_SPECTRAL_SOURCES for mode in options.modes)
        nested_train, nested_validation = ({} if use_nested else None), ({} if use_nested else None)
        use_reliability = options.reliability_diagnostics == "on" or bool(set(options.modes) & ({"support", "context", "centered"} | set(SHRINK_SOURCES) | set(CV_SOURCES) | set(GUARDED_SOURCES) | set(BLOCK_SHRINK_SOURCES)))
        train_reliability, val_reliability = ({} if use_reliability else None), ({} if use_reliability else None)
        group_train = [] if use_group_statistics else None
        train_cache = val_cache = None
        if options.feature_cache == "trainval":
            from lib.readout_feature_cache import ObservedReadoutCache
            from IMTSTokenizer.model.ContextReadoutBasis import ContextReadoutBasis
            train_cache = ObservedReadoutCache(destination / "feature_cache", "train", args.num_var,
                                              hybrid.projection.shape[0], options.chunk_rows)
            val_cache = ObservedReadoutCache(destination / "feature_cache", "val", args.num_var,
                                            hybrid.projection.shape[0], options.chunk_rows)
            report["feature_cache"] = {"root": "feature_cache", "state_names": list(ContextReadoutBasis.context_names),
                                       "scope": "Observed training and validation rows"}
        train, projected_train, _, _ = collect(
            hybrid, data["train_dataloader"], train_batches, options, diagnostics, "train", extra_statistics=extra_train,
            context_statistics=context_train, reliability_output=train_reliability, group_statistics=group_train,
            feature_cache=train_cache, **({"spectral_statistics": spectral_train} if use_spectral else {}),
            **({"nested_spectral_statistics": nested_train} if use_nested else {}))
        validation, projected_validation, cached_validation, _ = collect(
            hybrid, data["val_dataloader"], val_batches, options, diagnostics, "val", cache=val_batches < data["n_val_batches"],
            extra_statistics=extra_validation, context_statistics=context_validation, reliability_output=val_reliability,
            feature_cache=val_cache, **({"spectral_statistics": spectral_validation} if use_spectral else {}),
            **({"nested_spectral_statistics": nested_validation} if use_nested else {}))
        if train_cache is not None:
            report["feature_cache"].update(train=train_cache.finalize(), val=val_cache.finalize())
        if not (train.counts > 0).any() or not (validation.counts > 0).any():
            raise ValueError("Fit and validation each require observed targets")
        if train_batches == data["n_train_batches"]:
            torch.testing.assert_close(train.counts, data["training_target_counts"].double().cpu(), rtol=1e-9, atol=1e-6)
        torch.testing.assert_close(train.counts, projected_train.counts, rtol=1e-12, atol=1e-9)
        torch.testing.assert_close(validation.counts, projected_validation.counts, rtol=1e-12, atol=1e-9)
        fields = ("counts", "gram", "cross", "reference_sse", "residual_sum")
        statistics = {"train": train, "validation": validation, "random_train": projected_train, "random_validation": projected_validation}
        if use_group_statistics:
            if "readout_group_folds" not in data:
                raise ValueError("Training data did not expose record-group provenance")
            for field in fields:
                total = sum(getattr(stats, field) for stats in group_train)
                torch.testing.assert_close(total, getattr(train, field), rtol=1e-9, atol=1e-7)
            statistics.update({"train_fold_" + str(fold): stats for fold, stats in enumerate(group_train)})
            report["group_cv"] = {
                "assignment": data["readout_group_folds"],
                "fold_target_counts": [stats.counts.tolist() for stats in group_train],
                "statistics_scope": "Every requested training batch; saved under train_fold_0... in sufficient_statistics.pt"}
        if use_extra:
            for mode in extra_train:
                statistics[mode + "_train"], statistics[mode + "_validation"] = extra_train[mode], extra_validation[mode]
                torch.testing.assert_close(extra_train[mode].counts, train.counts, rtol=1e-12, atol=1e-9)
                torch.testing.assert_close(extra_validation[mode].counts, validation.counts, rtol=1e-12, atol=1e-9)
        if use_context:
            statistics["context_train"], statistics["context_validation"] = context_train["context"], context_validation["context"]
            torch.testing.assert_close(context_train["context"].counts, train.counts, rtol=1e-12, atol=1e-9)
            torch.testing.assert_close(context_validation["context"].counts, validation.counts, rtol=1e-12, atol=1e-9)
        if use_spectral:
            statistics["spectral_train"], statistics["spectral_validation"] = spectral_train["spectral"], spectral_validation["spectral"]
            torch.testing.assert_close(spectral_train["spectral"].counts, train.counts, rtol=1e-12, atol=1e-9)
            torch.testing.assert_close(spectral_validation["spectral"].counts, validation.counts, rtol=1e-12, atol=1e-9)
        if use_nested:
            statistics["nested_spectral_train"] = nested_train["spectral_nested"]
            statistics["nested_spectral_validation"] = nested_validation["spectral_nested"]
            torch.testing.assert_close(nested_train["spectral_nested"].counts, train.counts, rtol=1e-12, atol=1e-9)
            torch.testing.assert_close(nested_validation["spectral_nested"].counts, validation.counts, rtol=1e-12, atol=1e-9)
            if nested_train["projection"] != nested_validation["projection"]:
                raise ValueError("Nested spectral projection changed between splits")
        torch.save({label: {field: getattr(stats, field) for field in fields} for label, stats in statistics.items()},
                   destination / "sufficient_statistics.pt")
        report["feature_shift"] = feature_shift(train, validation)
        if use_extra:
            report["extra_features"] = {mode: {"shift": feature_shift(extra_train[mode], extra_validation[mode]),
                                                "train_novelty": conditional_novelty(extra_train[mode], options.rank)}
                                        for mode in extra_train}
        if use_context:
            context_novelty = conditional_novelty(context_train["context"], train.feature_dim)
            context_novelty["definition"] = "Train feature covariance: residual energy given the intercept and original readout features"
            report["context_features"] = {"shift": feature_shift(context_train["context"], context_validation["context"]),
                                          "train_novelty": context_novelty}
        if use_spectral:
            spectral_novelty = conditional_novelty(spectral_train["spectral"], train.feature_dim)
            spectral_novelty["definition"] = "Train spectral-feature covariance: residual energy given the intercept and original product features"
            report["spectral_features"] = {
                "definition": "Original product plus fixed projected history context times generic sine/cosine bands; one fixed core call.",
                "cycles_per_normalized_time_unit": [1., 2., 4., 8.],
                "phase_definition": "2*pi*cycle*normalized_absolute_query_time",
                "context_projection": hybrid.projection.detach().cpu().tolist(),
                "context_rank": options.rank, "extra_feature_dimension": 8 * options.rank,
                "coefficient_solver": "cholesky", "selection_split": "val", "fit_split": "train",
                "query_range": {"train": spectral_train["query_range"], "val": spectral_validation["query_range"]},
                "shift": feature_shift(spectral_train["spectral"], spectral_validation["spectral"]),
                "train_novelty": spectral_novelty}
            band_diagnostics = {}
            for band, cycles in enumerate((1, 2, 4, 8)):
                columns = list(range(train.features)) + [train.features + rank * 8 + offset
                    for rank in range(options.rank) for offset in (band, band + 4)]
                band_train = spectral_train["spectral"].select(columns)
                band_val = spectral_validation["spectral"].select(columns)
                band_diagnostics[str(cycles)] = {
                    "definition": "Diagnostic slice of the product features and this sine/cosine band",
                    "train_novelty": conditional_novelty(band_train, train.feature_dim),
                    "shift": feature_shift(band_train, band_val)}
            report["spectral_features"]["bands"] = band_diagnostics
        if use_nested:
            current, held_out = nested_train["spectral_nested"], nested_validation["spectral_nested"]
            band_diagnostics = {}
            for band, cycle in enumerate((1, 2, 4, 8)):
                columns = list(range(train.features)) + [train.features + rank * 8 + offset
                    for rank in range(options.nested_rank) for offset in (band, band + 4)]
                band_train, band_val = current.select(columns), held_out.select(columns)
                band_diagnostics[str(cycle)] = {
                    "definition": "Diagnostic slice of the product features and this nested-context frequency band",
                    "train_novelty": conditional_novelty(band_train, train.feature_dim),
                    "shift": feature_shift(band_train, band_val)}
            report["nested_spectral_features"] = {
                "definition": "Rank4 context projection with fixed orthogonal extension directions",
                "context_rank": options.nested_rank, "preserved_context_rank": 4,
                "context_projection": nested_train["projection"],
                "cycles_per_normalized_time_unit": [1., 2., 4., 8.],
                "extra_feature_dimension": 8 * options.nested_rank,
                "extension_beyond_rank4": 8 * (options.nested_rank - 4),
                "coefficient_solver": "cholesky", "fit_split": "train", "selection_split": "val",
                "query_range": {"train": nested_train["query_range"], "val": nested_validation["query_range"]},
                "train_novelty_given_product": conditional_novelty(current, train.feature_dim),
                "train_novelty_given_rank4": conditional_novelty(current, train.feature_dim + 32),
                "shift": feature_shift(current, held_out), "bands": band_diagnostics}
        if use_reliability:
            report["input_reliability"] = {"train": train_reliability, "val": val_reliability}
        report["baseline_validation"] = validation.score(torch.zeros(train.variables, train.features))
        chosen = {}
        fit_cache = {}
        remaining_sources = Counter(_fit_source(mode) for mode in options.modes if mode not in CV_SOURCES)
        for mode in options.modes:
            source_mode = BLOCK_SHRINK_SOURCES.get(mode, GUARDED_SOURCES.get(mode, CV_SOURCES.get(mode, SHRINK_SOURCES.get(mode, mode))))
            stats, val_stats = (projected_train, projected_validation) if source_mode == "random4" else (train, validation)
            if source_mode in {"diagonal", "interaction"}:
                stats, val_stats = extra_train[source_mode], extra_validation[source_mode]
            if source_mode == "context":
                stats, val_stats = context_train[source_mode], context_validation[source_mode]
            if source_mode in SPECTRAL_SOURCES:
                stats, val_stats = spectral_train["spectral"], spectral_validation["spectral"]
            if source_mode in NESTED_SPECTRAL_SOURCES:
                stats, val_stats = nested_train["spectral_nested"], nested_validation["spectral_nested"]
            adaptive = None
            if source_mode in {"support", "context"}:
                from lib.adaptive_readout_stats import AdaptiveReadoutStatistics
                adaptive = AdaptiveReadoutStatistics(stats.variables, stats.features - 1)
                for field in fields:
                    setattr(adaptive, field, getattr(stats, field))
                adaptive.batches = stats.batches
            beta = torch.zeros(stats.variables, stats.features)
            shared = torch.zeros(stats.features)
            selected = {"local_ridge": None, "global_ridge": None, "initialization_selected": True,
                        "validation": val_stats.score(beta), "fit_diagnostics": None}
            selected_archive = None
            if adaptive is not None:
                selected.update(support_power=0.0 if options.uniform_ridge else options.support_power,
                                context_penalty=None)
            centered = None
            if source_mode == "centered":
                from lib.centered_readout_stats import CenteredReadoutStatistics
                centered = CenteredReadoutStatistics(stats.variables, stats.features - 1)
                for field in fields:
                    setattr(centered, field, getattr(stats, field))
                centered.batches = stats.batches
                selected.update(support_power=1.0, bias_penalty=options.bias_penalty)
            for fit_statistics in (stats, adaptive, centered):
                if fit_statistics is not None and hasattr(fit_statistics, "fitting_cache"):
                    fitting_scopes.enter_context(fit_statistics.fitting_cache())
            candidates = []
            gains = None
            block_gains = None
            if mode in JOINT_SHRINK_SOURCES:
                from lib.deployed_readout_selection import calibrate_deployed_readout_candidate
                gains, effective_score, gain_report = calibrate_deployed_readout_candidate(
                    val_stats, beta, grid=options.shrink_grid, minimum_count=options.shrink_min_count,
                    active_variables=stats.counts > 0)
                selected.update(source_mode=source_mode, source_validation=selected["validation"],
                                validation=effective_score, gain_selection=gain_report, readout_gain=gains.tolist(),
                                parameter_selection_split="val", gain_selection_split="val",
                                fit_diagnostics_scope="Original train-fitted coefficients; penalty ranked after validation gain calibration")
            if mode in BLOCK_SHRINK_SOURCES:
                from lib.block_readout_selection import calibrate_block_readout_candidate
                _, block_gains, effective_score, block_report = calibrate_block_readout_candidate(
                    val_stats, beta, block_boundary=train.features, grid=options.shrink_grid,
                    minimum_count=options.shrink_min_count, active_variables=stats.counts > 0)
                selected.update(source_mode=source_mode, source_validation=block_report["unshrunk_validation"],
                                validation=effective_score, gain_selection=block_report,
                                readout_block_gain=block_gains.tolist(), parameter_selection_split="val",
                                gain_selection_split="val",
                                fit_diagnostics_scope="Original train-fitted coefficients; final two-block gains and penalty selected on validation")
            if mode in CV_SOURCES:
                from lib.grouped_readout_cv import fit_grouped_readout
                cv_grid = options.ridge_grid if source_mode == "hierarchical" else options.adaptive_ridge_grid
                fitted, gains, cv_report = fit_grouped_readout(
                    group_train, cv_grid, mode=source_mode, gain_grid=options.cv_gain_grid,
                    minimum_count=options.cv_min_count, bias_penalty=options.bias_penalty,
                    support_power=1.0, global_ridge_grid=options.global_ridge_grid)
                beta = fitted.float()
                details = cv_report["full_fit_diagnostics"]
                if details is not None:
                    shared = torch.as_tensor(details.get("shared_coefficients", [0.] * stats.features)).float()
                selected = {
                    "local_ridge": cv_report["selected_ridge"],
                    "global_ridge": cv_report["selected_global_ridge"],
                    "initialization_selected": cv_report["identity_selected"],
                    "source_mode": source_mode, "source_validation": val_stats.score(beta),
                    "validation": val_stats.score(beta * gains[:, None]),
                    "fit_diagnostics": details, "group_cv": cv_report,
                    "readout_gain": gains.tolist(),
                    "parameter_selection_split": "train_group_cv",
                    "gain_selection_split": "train_group_cv",
                    "fit_diagnostics_scope": "Original full-training refit before grouped-CV gain",
                }
            grid = (options.adaptive_ridge_grid if adaptive is not None or source_mode in {"centered", "full_local_fine", "spectral", "nested_spectral"}
                    else options.ridge_grid)
            for local in ([] if mode in CV_SOURCES else grid):
                globals_ = options.global_ridge_grid if source_mode in {"hierarchical", "spectral_hierarchical", "nested_spectral_hierarchical"} else [None]
                penalties = options.context_penalty_grid if source_mode == "context" else [None]
                for global_, context_penalty in ((global_, penalty) for global_ in globals_ for penalty in penalties):
                    cache_key = _candidate_key(stats, val_stats, source_mode, local, global_, context_penalty, options)
                    cached_fit = fit_cache.get(cache_key)
                    if cached_fit is None:
                        if source_mode == "random4":
                            fitted, details = stats.fit(local, support_shrink=not options.uniform_ridge)
                        elif source_mode == "hierarchical":
                            fitted, details = stats.fit_hierarchical(local, global_, support_shrink=not options.uniform_ridge)
                        elif source_mode in {"spectral_hierarchical", "nested_spectral_hierarchical"}:
                            fitted, details = stats.fit_hierarchical(local, global_, support_shrink=not options.uniform_ridge,
                                                                     solver="cholesky")
                        elif source_mode in {"spectral", "nested_spectral"}:
                            fitted, details = stats.fit_independent(local, support_shrink=not options.uniform_ridge,
                                                                    common_rms=False, solver="cholesky")
                        elif adaptive is not None:
                            penalty = torch.ones(stats.features, dtype=torch.float64)
                            if source_mode == "context":
                                penalty[train.features:] = context_penalty
                            fitted, details = adaptive.fit_adaptive(
                                local, support_power=0.0 if options.uniform_ridge else options.support_power,
                                feature_penalty=penalty)
                        elif centered is not None:
                            fitted, details = centered.fit_centered(
                                local, bias_penalty=options.bias_penalty, support_power=1.0)
                        else:
                            fitted, details = stats.fit_independent(local, support_shrink=not options.uniform_ridge, common_rms=source_mode == "full")
                        deployed = fitted.float()
                        score = val_stats.score(deployed)
                        cached_fit = {"deployed": deployed, "details": details, "score": score}
                        fit_cache[cache_key] = cached_fit
                    else:
                        deployed, details, score = (cached_fit[name] for name in ("deployed", "details", "score"))
                    candidate = {"local_ridge": local, "global_ridge": global_, "validation": score, "fit_diagnostics": details}
                    candidate_gains = None
                    candidate_block_gains = None
                    selection_score = score
                    if mode in JOINT_SHRINK_SOURCES:
                        if "deployed_calibration" not in cached_fit:
                            cached_fit["deployed_calibration"] = calibrate_deployed_readout_candidate(
                                val_stats, deployed, grid=options.shrink_grid, minimum_count=options.shrink_min_count,
                                active_variables=stats.counts > 0)
                        candidate_gains, selection_score, candidate_gain_report = cached_fit["deployed_calibration"]
                        candidate.update(source_mode=source_mode, source_validation=candidate_gain_report["unshrunk_validation"],
                                         validation=selection_score, readout_gain=candidate_gains.tolist(),
                                         gain_selection=candidate_gain_report,
                                         parameter_selection_split="val", gain_selection_split="val",
                                         fit_diagnostics_scope="Original train-fitted coefficients; penalty ranked after validation gain calibration")
                    if mode in BLOCK_SHRINK_SOURCES:
                        if "block_calibration" not in cached_fit:
                            cached_fit["block_calibration"] = calibrate_block_readout_candidate(
                                val_stats, deployed, block_boundary=train.features, grid=options.shrink_grid,
                                minimum_count=options.shrink_min_count, active_variables=stats.counts > 0)
                        _, candidate_block_gains, selection_score, block_report = cached_fit["block_calibration"]
                        candidate.update(source_mode=source_mode, source_validation=block_report["unshrunk_validation"],
                                         validation=selection_score, readout_block_gain=candidate_block_gains.tolist(),
                                         gain_selection=block_report, parameter_selection_split="val", gain_selection_split="val",
                                         fit_diagnostics_scope="Original train-fitted coefficients; final two-block gains and penalty selected on validation")
                    if adaptive is not None:
                        candidate.update(support_power=0.0 if options.uniform_ridge else options.support_power,
                                         context_penalty=context_penalty)
                    if centered is not None:
                        candidate.update(support_power=1.0, bias_penalty=options.bias_penalty)
                    candidate_record = candidate
                    if mode in JOINT_SHRINK_SOURCES or mode in BLOCK_SHRINK_SOURCES:
                        # Per-candidate gains and scores suffice to audit ranking;
                        # retain the full gain curve only for the selected model.
                        candidate_record = {key: value for key, value in candidate.items() if key != "gain_selection"}
                    if options.candidate_storage == "gzip":
                        from lib.readout_artifacts import archive_candidate_diagnostics
                        if "archive" not in cached_fit:
                            cached_fit["archive"] = archive_candidate_diagnostics(destination, details)
                        candidates.append({**candidate_record, "fit_diagnostics": cached_fit["archive"]})
                    else:
                        candidates.append(candidate_record)
                    if selection_score["macro_mse"] < selected["validation"]["macro_mse"]:
                        beta = deployed
                        shared = torch.as_tensor(details.get("shared_coefficients", [0.] * stats.features)).float()
                        selected = {**candidate, "initialization_selected": False}
                        selected_archive = cached_fit.get("archive")
                        if mode in JOINT_SHRINK_SOURCES:
                            gains = candidate_gains
                        if mode in BLOCK_SHRINK_SOURCES:
                            block_gains = candidate_block_gains
            prediction_beta = beta if gains is None else beta * gains[:, None]
            if block_gains is not None:
                expanded_block_gain = torch.cat((block_gains[:, :1].expand(-1, train.features),
                                                block_gains[:, 1:].expand(-1, stats.features - train.features)), dim=-1)
                prediction_beta = (beta * expanded_block_gain).float()
                selected["effective_coefficients"] = prediction_beta.tolist()
            if mode in SHRINK_SOURCES and mode not in JOINT_SHRINK_SOURCES:
                from lib.readout_shrinkage import select_readout_gains
                gains, gain_report = select_readout_gains(val_stats, beta, options.shrink_grid, options.shrink_min_count)
                prediction_beta = beta * gains[:, None]
                selected = {**selected, "source_mode": source_mode, "source_validation": selected["validation"],
                            "validation": val_stats.score(prediction_beta), "gain_selection": gain_report,
                            "readout_gain": gains.tolist(), "fit_diagnostics_scope": "Original train-fitted unshrunk coefficients"}
            if mode in GUARDED_SOURCES:
                from lib.group_guarded_readout import select_group_guarded_gains
                gains, guard_report = select_group_guarded_gains(
                    group_train, val_stats, beta, source_mode, selected["local_ridge"],
                    global_ridge=selected["global_ridge"], gain_grid=options.shrink_grid,
                    minimum_count=options.shrink_min_count, bias_penalty=options.bias_penalty)
                prediction_beta = beta * gains[:, None]
                selected = {
                    **selected, "source_mode": source_mode, "source_validation": selected["validation"],
                    "validation": val_stats.score(prediction_beta), "gain_selection": guard_report,
                    "readout_gain": gains.tolist(), "parameter_selection_split": "val",
                    "gain_selection_split": "val_with_train_group_guard",
                    "fit_diagnostics_scope": "Original train-fitted unshrunk coefficients; gain additionally constrained by conditional training-fold risk",
                }
            if mode in CV_SOURCES:
                selected.update(support_power=1.0,
                                bias_penalty=options.bias_penalty if source_mode == "centered" else None)
            model_args = copy.deepcopy(args)
            model_args.ridge_rank = options.rank
            for key, value in mode_identity(mode, nested_rank=options.nested_rank).items():
                setattr(model_args, key, value)
            model_args.readout_extra_rank = options.extra_rank
            if mode == "random4":
                model_args.ridge_mode, model_args.ridge_rank = "readout", options.rank
                model = RidgeCorrector(core, model_args).to(device).eval()
            else:
                model = FittedReadout(core, model_args).to(device).eval()
            with torch.no_grad():
                model.coefficients.copy_(beta.to(device))
                if mode != "random4":
                    model.shared_coefficients.copy_(shared.to(device))
                    model.active_variables.copy_((train.counts > 0).to(device))
                    if gains is not None:
                        model.readout_gain.copy_(gains.to(device))
                    if block_gains is not None:
                        model.readout_block_gain.copy_(block_gains.to(device))
            chosen[mode] = model
            report["modes"][mode] = {**selected, "candidates": candidates, "coefficients": beta.tolist(),
                                     "shared_coefficients": shared.tolist(), "train": stats.score(prediction_beta),
                                     "coefficient_count": beta.numel(), "core_fixed": True,
                                     "training_method": mode_training_method(mode)}
            if block_gains is not None:
                active = stats.counts > 0
                counts = torch.where(active, stats.counts, 1.)
                data_gradient = 2 * (torch.einsum("vij,vj->vi", stats.gram, prediction_beta.double()) - stats.cross)
                data_gradient = torch.where(active[:, None], data_gradient / counts[:, None] / active.sum().clamp_min(1), 0.)
                raw_gradient = data_gradient * expanded_block_gain.double()
                block_gradient = torch.stack(((data_gradient[:, :train.features] * beta[:, :train.features].double()).sum(-1),
                                              (data_gradient[:, train.features:] * beta[:, train.features:].double()).sum(-1)), dim=-1)
                report["modes"][mode].update(
                    calibration_parameter_count=block_gains.numel(),
                    post_block_train_diagnostics={
                        "definition": "Training macro-MSE derivatives at deployed beta32, with validation-selected block gains",
                        "gradient_wrt_block_gain": block_gradient.tolist(),
                        "gradient_wrt_raw_coefficients_norm_by_variable": raw_gradient.norm(dim=-1).tolist(),
                        "gradient_wrt_effective_coefficients_norm_by_variable": data_gradient.norm(dim=-1).tolist()})
            if gains is not None:
                active = stats.counts > 0
                counts = torch.where(active, stats.counts, 1.0)
                active_count = max(1, int(active.sum()))
                beta64, gain64 = beta.double(), gains.double()
                linear = (beta64 * stats.cross).sum(-1)
                energy = torch.einsum("vf,vfg,vg->v", beta64, stats.gram, beta64)
                gain_gradient = torch.where(active, 2 * (gain64 * energy - linear) / counts / active_count, 0.)
                effective = beta64 * gain64[:, None]
                data_gradient = 2 * ((stats.gram @ effective.unsqueeze(-1)).squeeze(-1) - stats.cross) / counts[:, None] / active_count
                raw_gradient = torch.where(active[:, None], gain64[:, None] * data_gradient, 0.)
                report["modes"][mode].update(effective_coefficients=prediction_beta.tolist(), source_train=stats.score(beta),
                                            calibration_parameter_count=gains.numel(),
                                            post_gain_train_diagnostics={
                                                "definition": "Analytical training macro-MSE derivatives of float64 gain-times-beta, using deployed raw beta32",
                                                "gradient_wrt_gain": gain_gradient.tolist(),
                                                "gradient_wrt_raw_beta": raw_gradient.tolist(),
                                                "gradient_wrt_raw_beta_norm": raw_gradient.norm(dim=-1).tolist()})
            if compact_storage and report["modes"][mode]["fit_diagnostics"] is not None:
                from lib.readout_artifacts import archive_candidate_diagnostics
                reference = selected_archive or archive_candidate_diagnostics(
                    destination, report["modes"][mode]["fit_diagnostics"])
                report["modes"][mode]["fit_diagnostics"] = reference
                if "group_cv" in report["modes"][mode]:
                    report["modes"][mode]["group_cv"] = {
                        **report["modes"][mode]["group_cv"], "full_fit_diagnostics": reference}
            print(mode, "selected", selected["local_ridge"], selected["global_ridge"],
                  "val", selected["validation"]["macro_mse"], flush=True)
            fitting_scopes.close()
            if mode not in CV_SOURCES:
                source_key = _fit_source(mode)
                remaining_sources[source_key] -= 1
                if not remaining_sources[source_key]:
                    fit_cache = {key: value for key, value in fit_cache.items() if key[2] != source_key}
        # Lock every hyperparameter and zero-correction fallback before test evaluation.
        save_results()
        for mode, model in chosen.items():
            row = report["modes"][mode]
            child = destination / mode
            child.mkdir(exist_ok=False)
            model.diagnostics, model.collect_diagnostic_predictions = diagnostics, True
            for split in (["val", "test"] if options.evaluate_test else ["val"]):
                count = val_batches if split == "val" else data["n_test_batches"]
                if split == "test":
                    report["protocol"]["test_evaluated"] = True
                model.branch_diagnostics = BranchDiagnostics(args.history / args.time_max)
                model.prediction_diagnostics = ForecastDiagnostics()
                loader = iter(cached_validation) if split == "val" and cached_validation is not None else data[split + "_dataloader"]
                with torch.no_grad():
                    metrics = evaluation(model, loader, count, data, diagnostics=diagnostics, split=mode + "/" + split)
                if not all(math.isfinite(metrics[key]) for key in ("mse", "mae", "rmse")):
                    raise FloatingPointError("Nonfinite final readout evaluation")
                if split == "val" and not math.isclose(metrics["mse"], row["validation"]["macro_mse"], rel_tol=5e-5, abs_tol=2e-8):
                    raise ValueError("Direct validation differs from sufficient-statistic prediction")
                row[split], row[split + "_diagnostics"] = metrics, model.branch_diagnostics.summary()
                forecast_details = model.prediction_diagnostics.summary()
                if compact_storage:
                    from lib.artifact_io import write_reference
                    payload = {"mode": mode, "split": split, "metrics": metrics,
                               "branch_diagnostics": row[split + "_diagnostics"]}
                    if diagnostics.enabled:
                        payload["forecast_diagnostics"] = forecast_details
                    record = diagnostics.prepare_event("readout_split", **payload)
                    reference = write_reference(destination, "split_records/" + mode + "_" + split + ".json", record)
                    row[split + "_diagnostics"] = {**reference, "member": ["payload", "branch_diagnostics"]}
                    diagnostics.event_reference("readout_split", reference, mode=mode, split=split, metrics=metrics)
                else:
                    diagnostics.event("readout_split", mode=mode, split=split, metrics=metrics,
                                      branch_diagnostics=row[split + "_diagnostics"],
                                      forecast_diagnostics=forecast_details)
            saved_output_args = copy.deepcopy(vars(model.args))
            saved_output_args.update(output_dir=str(child), save=str(child), diagnostics_dir=str(child), load=None, init_checkpoint=None)
            epoch = -1 if row["initialization_selected"] else 0
            torch.save({"state_dict": model.state_dict(), "args": saved_output_args,
                        "epoch": epoch, "step": train_batches, "best_val_mse": row["val"]["mse"],
                        "training_method": mode_training_method(mode), "readout_mode": mode,
                        "local_ridge": row["local_ridge"], "global_ridge": row["global_ridge"],
                        "support_power": row.get("support_power"), "context_penalty": row.get("context_penalty"),
                        "bias_penalty": row.get("bias_penalty"),
                        "parameter_selection_split": row.get("parameter_selection_split", "val"),
                        "gain_selection_split": row.get("gain_selection_split", "val" if mode in SHRINK_SOURCES else None)}, child / "best.ckpt")
            row["checkpoint"] = str((child / "best.ckpt").relative_to(ROOT))
            write_json(child / "summary.json", {"status": "completed" if options.evaluate_test else "validated",
                       "dataset": args.dataset, "seed": args.seed, "readout_mode": mode,
                       "run_id": child.relative_to(ROOT / "experiments").as_posix(), "best_epoch": epoch,
                       "best_val_mse": row["val"]["mse"], "test_at_best_validation": row.get("test"),
                       "checkpoint": row["checkpoint"], "partial_run": report["partial_run"],
                       "global_step": train_batches, "initialization_selected": row["initialization_selected"],
                       "training_method": mode_training_method(mode),
                       **({"parameter_selection_split": "train_group_cv", "gain_selection_split": "train_group_cv"}
                          if mode in CV_SOURCES else
                          {"parameter_selection_split": "val", "gain_selection_split": "val_with_train_group_guard"}
                          if mode in GUARDED_SOURCES else {})})
            save_results()
        report["status"] = "completed" if options.evaluate_test else "validated"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        diagnostics.event("failure", error_type=type(error).__name__, message=str(error),
                          memory=memory_status(device) if device is not None else None)
        raise
    finally:
        fitting_scopes.close()
        report["seconds"] = time.perf_counter() - started
        diagnostics.close()
        save_results()
        if compact_storage:
            from lib.readout_artifacts import compact_readout_run_metadata
            final_metadata = compact_readout_run_metadata(destination, report)
        else:
            final_metadata = {key: value for key, value in report.items() if key != "modes"}
        write_json(destination / "run.json", final_metadata)
        write_json(destination / "summary.json", {"status": report["status"],
                   "dataset": report.get("dataset"), "seed": report.get("seed"),
                   "seconds": report["seconds"], "partial_run": report.get("partial_run"),
                   "error": report.get("error"), "modes": list(report["modes"]),
                   "protocol": report["protocol"]})
        os.chdir(previous_dir)
        torch.set_num_threads(previous_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
