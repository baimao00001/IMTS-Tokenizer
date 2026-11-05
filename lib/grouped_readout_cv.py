"""Cross-fit a fixed model's affine head using grouped training statistics."""

import math
from itertools import product

import torch

from lib.centered_readout_stats import CenteredReadoutStatistics
from lib.hierarchical_readout_stats import FullReadoutStatistics


FIELDS = ("counts", "gram", "cross", "reference_sse", "residual_sum")


def _grid(values, name, allow_zero=False):
    values = list(values)
    if not values or any(isinstance(value, bool) for value in values):
        raise ValueError(name + " must contain real finite distinct values")
    values = [float(value) for value in values]
    if (len(set(values)) != len(values) or any(not math.isfinite(value) for value in values)
            or any(value < 0 if allow_zero else value <= 0 for value in values)):
        raise ValueError(name + " must contain real finite distinct values in its allowed range")
    return values


def _copy_fold(source, variables, features):
    if not isinstance(source, FullReadoutStatistics) or (source.variables, source.features) != (variables, features):
        raise ValueError("folds must be FullReadoutStatistics with matching dimensions")
    output = FullReadoutStatistics(variables, features - 1)
    for field in FIELDS:
        value = getattr(source, field)
        expected = getattr(output, field).shape
        if not torch.is_tensor(value) or value.shape != expected or value.is_complex() or not value.is_floating_point():
            raise ValueError("fold " + field + " has an invalid shape or dtype")
        value = value.detach().to(device="cpu", dtype=torch.float64).clone()
        if not torch.isfinite(value).all():
            raise ValueError("fold sufficient statistics must be finite")
        setattr(output, field, value)
    if (output.counts < 0).any() or (output.reference_sse < 0).any():
        raise ValueError("fold counts and residual SSE must be nonnegative")
    if not torch.allclose(output.gram[:, 0, 0], output.counts, rtol=1e-10, atol=1e-12):
        raise ValueError("fold bias Gram must equal observed counts")
    if not torch.allclose(output.cross[:, 0], output.residual_sum, rtol=1e-10, atol=1e-12):
        raise ValueError("fold bias cross moment must equal residual sum")
    empty = output.counts == 0
    if any((getattr(output, field)[empty] != 0).any() for field in FIELDS):
        raise ValueError("zero-support fold variables must have zero statistics")
    if type(source.batches) is not int or source.batches < 0:
        raise ValueError("fold statistic update count must be a nonnegative integer")
    output.batches = source.batches
    return output


def _combine(folds):
    output = FullReadoutStatistics(folds[0].variables, folds[0].features - 1)
    for field in FIELDS:
        value = sum((getattr(fold, field) for fold in folds), torch.zeros_like(getattr(output, field)))
        if not torch.isfinite(value).all():
            raise ValueError("combined fold sufficient statistics overflowed")
        setattr(output, field, value)
    output.batches = sum(fold.batches for fold in folds)
    return output


def _score(sse, counts):
    active = counts > 0
    per_variable = sse / torch.where(active, counts, 1.)
    return {
        "macro_mse": per_variable[active].mean().item() if active.any() else None,
        "micro_mse": (sse.sum() / counts.sum()).item() if active.any() else None,
        "per_variable_mse": [float(value) if used else None for value, used in zip(per_variable, active)],
        "per_variable_sse": sse.tolist(), "counts": counts.tolist(),
        "count": counts.sum().item(), "observed_variables": int(active.sum()),
    }


def _fit(stats, ridge, mode, bias_penalty, support_power, global_ridge=None):
    if mode == "full_local":
        return stats.fit_independent(ridge, support_shrink=True, common_rms=False)
    if mode == "hierarchical":
        return stats.fit_hierarchical(ridge, global_ridge, support_shrink=True)
    centered = CenteredReadoutStatistics(stats.variables, stats.features - 1)
    for field in FIELDS:
        setattr(centered, field, getattr(stats, field))
    centered.batches = stats.batches
    return centered.fit_centered(ridge, bias_penalty=bias_penalty, support_power=support_power)


def _compact_fit(report):
    fields = ("kind", "counts", "active_variable_indices", "zero_support_variables",
              "effective_degrees_of_freedom", "bias_effective_degrees_of_freedom",
              "slope_effective_degrees_of_freedom", "active_regularized_condition",
              "local_normal_gradient_norm", "float32_deployed_local_normal_gradient_norm",
              "lambda_per_variable", "ridge", "local_ridge", "global_ridge", "support_power", "bias_penalty",
              "center_features", "coefficient_norm")
    return {field: report[field] for field in fields if field in report}


@torch.no_grad()
def fit_grouped_readout(folds, ridge_grid, mode="full_local",
                       gain_grid=(1., .75, .5, .25, 0.), minimum_count=64,
                       bias_penalty=.1, support_power=1., global_ridge_grid=None):
    """Return full-training coefficients, gains and out-of-fold diagnostics.

    Folds must be disjoint training-record groups supplied by the caller. The
    neural core is already trained and fixed; cross-fitting applies to the head.
    """
    folds = list(folds)
    if len(folds) < 2 or not isinstance(folds[0], FullReadoutStatistics):
        raise ValueError("at least two matching grouped training-statistic folds are required")
    if mode not in {"full_local", "centered", "hierarchical"}:
        raise ValueError("mode must be full_local, centered or hierarchical")
    if type(minimum_count) is not int or minimum_count < 1:
        raise ValueError("minimum_count must be a positive integer")
    # Validate centered options even when identity wins and no final solve runs.
    bias_penalty = CenteredReadoutStatistics._nonnegative(bias_penalty, "bias_penalty")
    if isinstance(support_power, bool):
        raise ValueError("support_power must be finite and nonnegative")
    support_power = float(support_power)
    if not math.isfinite(support_power) or support_power < 0:
        raise ValueError("support_power must be finite and nonnegative")
    if mode != "centered" and support_power != 1.:
        raise ValueError("original full_local and hierarchical controls fix support_power at1")
    ridge_grid = _grid(ridge_grid, "ridge_grid")
    global_ridge_grid = (_grid(global_ridge_grid, "global_ridge_grid")
                         if global_ridge_grid is not None else [.001, .01, .1, 1., 10.])
    global_candidates = global_ridge_grid if mode == "hierarchical" else [None]
    if gain_grid is not None:
        gain_grid = _grid(gain_grid, "gain_grid", allow_zero=True)
        if 1. not in gain_grid or any(value > 1 for value in gain_grid):
            raise ValueError("gain_grid must lie in [0,1] and include1")
        if len(set(torch.tensor(gain_grid, dtype=torch.float32).tolist())) != len(gain_grid):
            raise ValueError("gain_grid values must remain distinct in float32")
    variables, features = folds[0].variables, folds[0].features
    folds = [_copy_fold(fold, variables, features) for fold in folds]
    full = _combine(folds)
    if not (full.counts > 0).any():
        raise ValueError("grouped readout CV requires observed training targets")
    complementary = [_combine([fold for index, fold in enumerate(folds) if index != excluded])
                     for excluded in range(len(folds))]
    baseline = _score(full.reference_sse, full.counts)
    selected_ridge, selected_global_ridge, source_score = None, None, baseline
    selected_betas = [torch.zeros(variables, features) for _ in folds]
    selected_details = [None for _ in folds]
    candidates = []
    for ridge, global_ridge in product(ridge_grid, global_candidates):
        betas, details, scores = [], [], []
        for training, held_out in zip(complementary, folds):
            beta, diagnostics = _fit(training, ridge, mode, bias_penalty, support_power, global_ridge)
            beta = beta.float()
            betas.append(beta)
            details.append(_compact_fit(diagnostics))
            scores.append(held_out.score(beta))
        total_sse = sum((torch.tensor(score["per_variable_sse"], dtype=torch.float64) for score in scores),
                        torch.zeros_like(full.counts))
        score = _score(total_sse, full.counts)
        candidates.append({"ridge": ridge, "global_ridge": global_ridge,
                           "oof": score, "fold_scores": scores,
                           "fold_fit_diagnostics": details})
        if score["macro_mse"] < source_score["macro_mse"]:
            selected_ridge, selected_global_ridge = ridge, global_ridge
            source_score, selected_betas, selected_details = score, betas, details
    gains = torch.ones(variables, dtype=torch.float32)
    effective_sse = torch.tensor(source_score["per_variable_sse"], dtype=torch.float64)
    gain_candidates = []
    eligible = full.counts >= minimum_count
    if gain_grid is not None:
        for gain in gain_grid:
            gain32 = torch.tensor(gain, dtype=torch.float32)
            fold_scores = [fold.score((beta * gain32).float()) for fold, beta in zip(folds, selected_betas)]
            sse = sum((torch.tensor(score["per_variable_sse"], dtype=torch.float64) for score in fold_scores),
                      torch.zeros_like(full.counts))
            gain_candidates.append({"gain": float(gain32), "oof": _score(sse, full.counts)})
            improved = eligible & (sse < effective_sse)
            effective_sse = torch.where(improved, sse, effective_sse)
            gains = torch.where(improved, gain32, gains)
    final_details = None
    if selected_ridge is None:
        raw_beta = torch.zeros(variables, features, dtype=torch.float64)
    else:
        raw_beta, final_details = _fit(full, selected_ridge, mode, bias_penalty, support_power, selected_global_ridge)
    deployed_beta = raw_beta.float()
    effective_beta = (deployed_beta * gains[:, None]).float()
    energy = torch.zeros_like(full.counts)
    cross = torch.zeros_like(full.counts)
    for fold, beta in zip(folds, selected_betas):
        beta64 = beta.double()
        energy += torch.einsum("vf,vfg,vg->v", beta64, fold.gram, beta64)
        cross += (beta64 * fold.cross).sum(-1)
    train_count = torch.where(full.counts > 0, full.counts, 1.)
    prediction_distances = []
    for beta in selected_betas:
        difference = beta.double() - deployed_beta.double()
        squared = torch.einsum("vf,vfg,vg->v", difference, full.gram, difference) / train_count
        prediction_distances.append(squared.clamp_min(0).tolist())
    pair_alignment = []
    for first in range(len(folds)):
        for second in range(first + 1, len(folds)):
            a, b = selected_betas[first].double(), selected_betas[second].double()
            numerator = torch.einsum("vf,vfg,vg->v", a, full.gram, b)
            aa = torch.einsum("vf,vfg,vg->v", a, full.gram, a).clamp_min(0)
            bb = torch.einsum("vf,vfg,vg->v", b, full.gram, b).clamp_min(0)
            denominator = aa.sqrt() * bb.sqrt()
            cosine = numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny)
            pair_alignment.append({"folds": [first, second],
                                   "cosine": [float(value.clamp(-1, 1)) if usable else None
                                              for value, usable in zip(cosine, denominator > 0)]})
    report = {
        "kind": "conditional_fixed_core_grouped_head_cv",
        "mode": mode, "selection_split": "train_group_cv", "coefficients_fit_split": "train",
        "gain_selection_split": "train_group_cv" if gain_grid is not None else None,
        "definition": ("Only affine head coefficients are fitted on complementary grouped training folds. "
                       "The neural core is fixed and was trained on the complete training set. "
                       "Select one ridge or hierarchical ridge pair by aggregated observed-row OOF macro MSE, then gains "
                       "with ridge locked. Refit one head on all training data; no prediction ensemble."),
        "fold_count": len(folds), "counts": full.counts.tolist(),
        "statistic_update_calls_per_fold": [fold.batches for fold in folds],
        "refit_statistics_update_calls": full.batches,
        "statistics_batch_definition": "Grouped sufficient-statistic update calls, not neural training minibatches or independent records.",
        "fold_counts": [fold.counts.tolist() for fold in folds],
        "complementary_train_counts": [fold.counts.tolist() for fold in complementary],
        "positive_support_fold_count": torch.stack([fold.counts > 0 for fold in folds]).sum(0).tolist(),
        "ridge_grid": ridge_grid, "selected_ridge": selected_ridge,
        "global_ridge_grid": global_ridge_grid if mode == "hierarchical" else None,
        "selected_global_ridge": selected_global_ridge,
        "identity_selected": selected_ridge is None, "baseline_oof": baseline,
        "source_oof": source_score, "effective_oof": _score(effective_sse, full.counts),
        "source_lambda_locked_before_gain": True,
        "identity_policy": "Identity starts as winner and is replaced only by strictly lower OOF macro MSE.",
        "candidates": candidates, "gain_grid": gain_grid, "gains": gains.tolist(),
        "gain_eligible_variables": eligible.tolist(), "minimum_count": minimum_count,
        "gain_candidates": gain_candidates,
        "gain_tie_policy": "Start at gain1; strictly lower deployed SSE only; exact ties preserve earlier winner.",
        "fold_coefficients": [beta.tolist() for beta in selected_betas],
        "fold_fit_diagnostics": selected_details,
        "oof_correction_energy": energy.tolist(), "oof_residual_cross": cross.tolist(),
        "oof_moment_definition": "Sum of held-out fold quadratic/linear moments for each fold's float32 head; deployed gain candidates are rescored separately after float32 multiplication.",
        "refit_coefficients": deployed_beta.tolist(), "effective_coefficients": effective_beta.tolist(),
        "fold_to_refit_coefficient_distance": [(beta.double() - deployed_beta.double()).norm(dim=-1).tolist()
                                             for beta in selected_betas],
        "fold_to_refit_prediction_mse": prediction_distances,
        "fold_pair_prediction_alignment": pair_alignment,
        "prediction_stability_definition": "Unshrunk float32 fold/refit corrections compared in the complete training feature Gram. These are head stability diagnostics, not held-out generalization estimates.",
        "full_train_source_score": full.score(deployed_beta),
        "full_train_effective_score": full.score(effective_beta),
        "full_fit_diagnostics": final_details,
        "fit_diagnostics_scope": "Normal-gradient diagnostics use the train-fitted coefficients before OOF gain calibration.",
        "generalization_scope": "Readout-only cross-validation on a fixed core trained on all training records; counts refer to observed rows.",
    }
    return raw_beta, gains, report
