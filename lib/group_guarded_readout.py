"""Validation-selected gains constrained by fixed-backbone training-group behavior."""

import torch

from lib.centered_readout_stats import CenteredReadoutStatistics
from lib.grouped_readout_cv import _combine, _compact_fit, _copy_fold, _fit, _grid, _score
from lib.hierarchical_readout_stats import FullReadoutStatistics


@torch.no_grad()
def select_group_guarded_gains(folds, validation_stats, raw_beta, source_mode,
                              local_ridge, global_ridge=None,
                              gain_grid=(1., .75, .5, .25, 0.), minimum_count=64,
                              bias_penalty=.1):
    """Keep the caller's source head fixed; select a train-guarded validation gain."""
    folds = list(folds)
    if len(folds) < 2 or not isinstance(folds[0], FullReadoutStatistics):
        raise ValueError("at least two matching grouped training folds are required")
    if source_mode not in {"full_local", "centered", "hierarchical"}:
        raise ValueError("source_mode must be full_local, centered or hierarchical")
    if type(minimum_count) is not int or minimum_count < 1:
        raise ValueError("minimum_count must be a positive integer")
    bias_penalty = CenteredReadoutStatistics._nonnegative(bias_penalty, "bias_penalty")
    if local_ridge is None:
        if global_ridge is not None:
            raise ValueError("identity source cannot have a global ridge")
    else:
        local_ridge = FullReadoutStatistics._positive(local_ridge, "local_ridge")
        if source_mode == "hierarchical":
            if global_ridge is None:
                raise ValueError("a non-identity hierarchical source requires global_ridge")
            global_ridge = FullReadoutStatistics._positive(global_ridge, "global_ridge")
        elif global_ridge is not None:
            raise ValueError("only a hierarchical source accepts global_ridge")
    gain_grid = _grid(gain_grid, "gain_grid", allow_zero=True)
    if 1. not in gain_grid or any(value > 1. for value in gain_grid):
        raise ValueError("gain_grid must lie in [0,1] and include1")
    deployed_grid = torch.tensor(gain_grid, dtype=torch.float32)
    if len(set(deployed_grid.tolist())) != len(gain_grid):
        raise ValueError("gain_grid values must remain distinct in float32")
    variables, features = folds[0].variables, folds[0].features
    folds = [_copy_fold(fold, variables, features) for fold in folds]
    validation = _copy_fold(validation_stats, variables, features)
    full = _combine(folds)
    if not (full.counts > 0).any() or not (validation.counts > 0).any():
        raise ValueError("training and validation both require observed targets")
    beta_input = torch.as_tensor(raw_beta)
    if beta_input.is_complex() or beta_input.dtype == torch.bool or beta_input.shape != (variables, features):
        raise ValueError("raw_beta must be real with shape [variables, features]")
    beta = beta_input.detach().to(device="cpu", dtype=torch.float32).clone()
    if not torch.isfinite(beta_input).all() or not torch.isfinite(beta).all():
        raise ValueError("raw_beta must be finite and float32 deployable")
    if local_ridge is None and bool((beta_input != 0).any()):
        raise ValueError("identity source requires zero coefficients")
    complementary = [_combine([fold for index, fold in enumerate(folds) if index != excluded])
                     for excluded in range(len(folds))]
    fold_beta, fold_diagnostics = [], []
    for training in complementary:
        if local_ridge is None:
            fitted, details = torch.zeros_like(beta), None
        else:
            fitted, details = _fit(training, local_ridge, source_mode, bias_penalty, 1., global_ridge)
            fitted = fitted.float()
        fold_beta.append(fitted)
        fold_diagnostics.append(None if details is None else _compact_fit(details))
    source_validation = validation.score(beta)
    source_oof_sse = sum((torch.tensor(fold.score(fitted)["per_variable_sse"], dtype=torch.float64)
                          for fold, fitted in zip(folds, fold_beta)), torch.zeros_like(full.counts))
    eligible = validation.counts >= minimum_count
    gains = torch.ones(variables, dtype=torch.float32)
    ordinary_gains = torch.ones_like(gains)
    chosen_val = torch.tensor(source_validation["per_variable_sse"], dtype=torch.float64)
    ordinary_val = chosen_val.clone()
    chosen_oof, ordinary_oof = source_oof_sse.clone(), source_oof_sse.clone()
    candidates, masks = [], []
    for requested_gain, gain in zip(gain_grid, deployed_grid):
        candidate_val = validation.score((beta * gain).float())
        val_sse = torch.tensor(candidate_val["per_variable_sse"], dtype=torch.float64)
        oof_sse = sum((torch.tensor(fold.score((fitted * gain).float())["per_variable_sse"],
                                     dtype=torch.float64) for fold, fitted in zip(folds, fold_beta)),
                      torch.zeros_like(full.counts))
        tolerance = 64 * torch.finfo(torch.float64).eps * torch.maximum(
            oof_sse.abs(), source_oof_sse.abs()).clamp_min(1.)
        allowed = oof_sse <= source_oof_sse + tolerance
        if float(gain) == 1.:
            allowed = torch.ones_like(allowed)
        improves = eligible & allowed & (val_sse < chosen_val)
        gains = torch.where(improves, gain, gains)
        chosen_val = torch.where(improves, val_sse, chosen_val)
        chosen_oof = torch.where(improves, oof_sse, chosen_oof)
        ordinary_improves = eligible & (val_sse < ordinary_val)
        ordinary_gains = torch.where(ordinary_improves, gain, ordinary_gains)
        ordinary_val = torch.where(ordinary_improves, val_sse, ordinary_val)
        ordinary_oof = torch.where(ordinary_improves, oof_sse, ordinary_oof)
        masks.append(allowed)
        candidates.append({"requested_gain": requested_gain, "gain": float(gain),
                           "admissible_variables": allowed.tolist(), "oof_tolerance": tolerance.tolist(),
                           "oof": _score(oof_sse, full.counts), "validation": candidate_val})
    energy, cross = torch.zeros_like(full.counts), torch.zeros_like(full.counts)
    fold_energy = []
    for fold, fitted in zip(folds, fold_beta):
        coefficient = fitted.double()
        local_energy = torch.einsum("vf,vfg,vg->v", coefficient, fold.gram, coefficient)
        fold_energy.append(local_energy)
        energy += local_energy
        cross += (coefficient * fold.cross).sum(-1)
    effective = (beta * gains[:, None]).float()
    ordinary_effective = (beta * ordinary_gains[:, None]).float()
    allowed_mask = torch.stack(masks)
    report = {
        "kind": "train_group_guarded_validation_shrinkage",
        "source_mode": source_mode, "local_ridge": local_ridge, "global_ridge": global_ridge,
        "bias_penalty": bias_penalty, "support_power": 1.,
        "source_lambda_locked_before_gain": True, "source_parameters_unchanged": True,
        "source_fit_contract": "Caller fixes the ungated source coefficients and regularization using training fit and validation selection before calling this helper. This helper does not reselect regularization or replace coefficients.",
        "parameter_selection_split": "val", "gain_selection_split": "val_with_train_group_guard",
        "definition": ("At the locked source regularization, fit each head using complementary grouped training statistics. "
                       "A discrete gain is allowed when its aggregated held-out-head SSE is no worse than gain1 within "
                       "float64 roundoff. Validation chooses among allowed gains. Gain1 is always allowed; sparse "
                       "validation variables keep gain1. Fold heads are not deployed or averaged."),
        "fold_count": len(folds), "fold_counts": [fold.counts.tolist() for fold in folds],
        "complementary_train_counts": [fold.counts.tolist() for fold in complementary],
        "counts": full.counts.tolist(), "validation_counts": validation.counts.tolist(),
        "statistic_update_calls_per_fold": [fold.batches for fold in folds],
        "positive_support_fold_count": torch.stack([fold.counts > 0 for fold in folds]).sum(0).tolist(),
        "positive_correction_energy_fold_count": torch.stack([value > 0 for value in fold_energy]).sum(0).tolist(),
        "fold_coefficients": [fitted.tolist() for fitted in fold_beta],
        "fold_fit_diagnostics": fold_diagnostics,
        "source_coefficients": beta.tolist(), "effective_coefficients": effective.tolist(),
        "gain_grid": gain_grid, "minimum_count": minimum_count, "eligible_variables": eligible.tolist(),
        "gains": gains.tolist(), "ordinary_validation_gains": ordinary_gains.tolist(),
        "changed_variables": (gains != ordinary_gains).nonzero(as_tuple=True)[0].tolist(),
        "admissible_gain_mask": allowed_mask.tolist(),
        "excluded_gain_count": (~allowed_mask).sum(0).tolist(),
        "candidates": candidates,
        "source_validation": source_validation,
        "guarded_validation": validation.score(effective),
        "ordinary_validation": validation.score(ordinary_effective),
        "source_oof": _score(source_oof_sse, full.counts),
        "guarded_oof": _score(chosen_oof, full.counts),
        "ordinary_oof": _score(ordinary_oof, full.counts),
        "oof_correction_energy": energy.tolist(), "oof_residual_cross": cross.tolist(),
        "oof_gain1_gradient_unfolded": (2 * (energy - cross) / torch.where(full.counts > 0, full.counts, 1.)).tolist(),
        "oof_gradient_definition": "Per-variable continuous gamma derivative at gain1 for ideal float64 scaling of the float32 fold heads. Discrete deployed candidates are scored independently, including float32 multiplication.",
        "oof_admissibility_tolerance": "64*float64_eps*max(1,abs(candidate_SSE),abs(gain1_SSE))",
        "gain_tie_policy": "Start at gain1, replace only for strictly smaller validation SSE among allowed gains; exact ties retain the current winner.",
        "zero_energy_policy": "Equal-SSE candidates can pass the guard even without informative OOF correction energy; positive-energy fold counts are reported.",
        "generalization_scope": "Readout-only cross-fitting on a fixed core trained on all training records, with an out-of-fold non-increase constraint.",
        "fit_diagnostics_scope": "Fold fits describe ungated training coefficients; not stationarity of the final validation-selected gained predictor.",
    }
    return gains, report
