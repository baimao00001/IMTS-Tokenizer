"""Model construction and checkpoint initialization."""

from collections.abc import Mapping
import math

from .IMTS_Tokenizer import IMTS_Tokenizer
from .LocalTokenizer import LocalTokenizer
from .GatedTokenizer import GatedTokenizer


def build_model(args):
    query_mode = getattr(args, "target_query_mode", "none")
    if query_mode not in {"none", "missing", "all", "conditional"}:
        raise ValueError("Unsupported target_query_mode")
    if query_mode != "none" and (args.variant != "local_residual" or args.num_layers < 1):
        raise ValueError("Target query identity requires local_residual with attention layers")
    if args.variant == "legacy":
        model = IMTS_Tokenizer(args)
    elif args.variant in {"local_gated", "local_adaptive"}:
        model = GatedTokenizer(args)
    else:
        model = LocalTokenizer(args)
    if getattr(args, "temporal_correction", False):
        from .TemporalCorrector import TemporalCorrector
        model = TemporalCorrector(model, args)
    if getattr(args, "ridge_mode", "none") != "none":
        from .RidgeCorrector import RidgeCorrector
        model = RidgeCorrector(model, args)
    if getattr(args, "fitted_readout", "none") != "none":
        if getattr(args, "ridge_mode", "none") != "none":
            raise ValueError("Ridge and full-feature readouts cannot be stacked")
        from .FittedReadout import FittedReadout
        model = FittedReadout(model, args)
    return model


def check_source_reliability_identity(args, saved_args):
    if not isinstance(saved_args, Mapping):
        raise ValueError("Checkpoint args must be a mapping")
    if saved_args.get("source_reliability", False) and getattr(args, "source_reliability", False):
        if saved_args.get("source_reliability_mode", "gated") != getattr(args, "source_reliability_mode", "gated"):
            raise ValueError("Checkpoint differs in source_reliability_mode")


def check_target_query_identity(args, saved_args, *, allow_new=False):
    if not isinstance(saved_args, Mapping):
        raise ValueError("Checkpoint args must be a mapping")
    previous = saved_args.get("target_query_mode", "none")
    current = getattr(args, "target_query_mode", "none")
    if previous not in {"none", "missing", "all", "conditional"} or current not in {"none", "missing", "all", "conditional"}:
        raise ValueError("Unsupported target_query_mode in model or checkpoint")
    if previous != current and not (allow_new and previous == "none" and current in {"missing", "all", "conditional"}):
        raise ValueError("Checkpoint differs in target_query_mode")


def initialize_weights(model, checkpoint):
    """Load checkpoint weights, allowing newly enabled zero-start modules."""
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError("Initialization checkpoint must contain a state_dict mapping")
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, Mapping):
        raise ValueError("Initialization checkpoint args must be a mapping")
    for name in ("dataset", "variant", "history", "patch_size"):
        if name in saved_args and saved_args[name] != getattr(model.args, name):
            raise ValueError("Initialization checkpoint differs in " + name)
    if "time_max" in saved_args and not math.isclose(float(saved_args["time_max"]), float(model.args.time_max), rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Initialization checkpoint differs in time_max")
    check_source_reliability_identity(model.args, saved_args)
    check_target_query_identity(model.args, saved_args, allow_new=True)
    state = checkpoint["state_dict"]
    if any(name.startswith("core.") for name in state):
        if saved_args.get("correction_variables") != getattr(model.args, "correction_variables", None):
            raise ValueError("Initialization checkpoint differs in correction_variables")
        for name, default in (("seasonal_period", 0.0), ("seasonal_bandwidth", 1.0)):
            if not math.isclose(float(saved_args.get(name, default)), float(getattr(model.args, name, default)),
                                rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError("Initialization checkpoint differs in " + name)
    target = model
    if hasattr(model, "core") and not any(name.startswith("core.") for name in state):
        target = model.core
    incompatible = target.load_state_dict(state, strict=False)
    allowed = {"source_reliability_weight", "source_reliability_logits"} if getattr(target, "source_reliability_enabled", False) else set()
    if hasattr(target, "core") and getattr(target.core, "source_reliability_enabled", False):
        allowed.update({"core.source_reliability_weight", "core.source_reliability_logits"})
    backbone = target.core if hasattr(target, "core") else target
    prefix = "core." if hasattr(target, "core") else ""
    if getattr(backbone, "target_query_mode", "none") != "none" and saved_args.get("target_query_mode", "none") == "none":
        allowed.add(prefix + "target_query_embedding")
    for index, layer in enumerate(getattr(backbone, "layer", [])):
        if getattr(layer.attn, "qk_normalization", "none") == "adaptive":
            allowed.add(prefix + "layer.{}.attn.qk_mix".format(index))
    if incompatible.unexpected_keys or set(incompatible.missing_keys) - allowed:
        raise ValueError("Incompatible initialization checkpoint: missing={}, unexpected={}".format(
            incompatible.missing_keys, incompatible.unexpected_keys))
    return {"missing_keys": list(incompatible.missing_keys), "unexpected_keys": list(incompatible.unexpected_keys)}
