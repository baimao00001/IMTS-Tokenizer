"""Validation-selected discrete gains for an already fixed train-fitted readout."""

import math

import torch


@torch.no_grad()
def select_readout_gains(validation_stats, coefficients, grid=(1., .75, .5, .25, 0.), minimum_count=64):
    if type(minimum_count) is not int or minimum_count <= 0:
        raise ValueError("minimum_count must be a positive integer")
    supplied = list(grid)
    if not supplied or any(isinstance(value, bool) for value in supplied):
        raise ValueError("gain grid must contain unique finite values in [0,1], including1")
    values = [float(value) for value in supplied]
    if (any(not math.isfinite(value) or not 0 <= value <= 1 for value in values)
            or 1. not in values or len(set(values)) != len(values)):
        raise ValueError("gain grid must contain unique finite values in [0,1], including1")
    values = torch.tensor(values, dtype=torch.float32).tolist()
    if len(set(values)) != len(values):
        raise ValueError("gain grid values must remain unique after float32 deployment")
    ordered = [1.] + [value for value in values if value != 1.]
    original = torch.as_tensor(coefficients, device="cpu").detach()
    if original.is_complex() or original.dtype == torch.bool:
        raise ValueError("coefficients must be real non-boolean values")
    beta = original.float()
    expected = (validation_stats.variables, validation_stats.features)
    if beta.shape != expected or not torch.isfinite(beta).all():
        raise ValueError("coefficients must be finite float32 with shape [variables,features]")
    counts = validation_stats.counts.detach().cpu().double()
    if counts.shape != (expected[0],) or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("validation counts must be finite and nonnegative")
    initial = validation_stats.score(beta)
    best_sse = torch.tensor(initial["per_variable_sse"], dtype=torch.float64)
    gains = torch.ones(expected[0], dtype=torch.float32)
    eligible = counts >= minimum_count
    candidates = [{"gain": 1., "validation": initial}]
    for gain in ordered[1:]:
        score = validation_stats.score((beta * gain).float())
        candidate = torch.tensor(score["per_variable_sse"], dtype=torch.float64)
        better = eligible & (candidate < best_sse)
        gains = torch.where(better, gain, gains)
        best_sse = torch.where(better, candidate, best_sse)
        candidates.append({"gain": gain, "validation": score})
    folded = (beta * gains[:, None]).float()
    effective = validation_stats.score(folded)
    torch.testing.assert_close(torch.tensor(effective["per_variable_sse"], dtype=torch.float64), best_sse,
                               rtol=1e-12, atol=1e-12)
    beta64 = beta.double()
    linear = (beta64 * validation_stats.cross).sum(-1)
    quadratic = torch.einsum("vf,vfg,vg->v", beta64, validation_stats.gram, beta64)
    rows = []
    for variable in range(expected[0]):
        if counts[variable] < minimum_count:
            reason = "below_minimum_validation_observations"
        elif gains[variable] != 1:
            reason = "strictly_lower_validation_sse"
        else:
            reason = "original_gain_is_best_or_tied"
        before, after = initial["per_variable_mse"][variable], effective["per_variable_mse"][variable]
        rows.append({"variable": variable, "validation_count": counts[variable].item(),
                     "eligible": bool(eligible[variable]), "gain": gains[variable].item(), "selection_reason": reason,
                     "unshrunk_validation_mse": before, "effective_validation_mse": after,
                     "validation_mse_gain": before - after if before is not None else None,
                     "unshrunk_correction_energy_sum": quadratic[variable].item(),
                     "unshrunk_residual_cross_sum": linear[variable].item()})
    report = {
        "kind": "validation_selected_variable_readout_shrinkage", "selection_split": "validation",
        "definition": "Original coefficients are train-fitted and fixed; original lambda selection is already locked. Each eligible variable selects a discrete correction gain on validation without refitting beta or retuning lambda.",
        "grid": ordered, "minimum_count": minimum_count,
        "count_definition": "Observed validation target values, not independent patients or windows",
        "tie_policy": "Start with gain1 and replace only for strictly lower SSE; remaining exact ties preserve grid order",
        "score_definition": "Sufficient-statistic score of float32(original_beta32*gain32). Deployment evaluates raw correction first and then multiplies gain, so direct forward must verify rounding agreement.",
        "gradient_definition": "Normal-gradient diagnostics use the train-fitted coefficients before validation gain calibration.",
        "coefficients_refitted": False, "lambda_reselected": False,
        "gains": gains.tolist(), "eligible_count": int(eligible.sum()),
        "changed_count": int((gains != 1).sum()), "closed_variables": (gains == 0).nonzero(as_tuple=True)[0].tolist(),
        "unshrunk_validation": initial, "effective_validation": effective,
        "candidates": candidates, "per_variable": rows,
    }
    return gains, report
