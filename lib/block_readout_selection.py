"""Validation calibration of two fixed train-fitted readout blocks."""
import math

import torch


@torch.no_grad()
def calibrate_block_readout_candidate(validation_stats, coefficients, *, block_boundary,
                                      grid=(1., .75, .5, .25, 0.), minimum_count=64,
                                      active_variables=None):
    """Return effective beta32, two gains, its score, and selection diagnostics.

    The first block includes the intercept. Coefficients never get refitted here;
    the caller compares these effective scores with its zero-correction fallback.
    """
    if type(minimum_count) is not int or minimum_count <= 0:
        raise ValueError("minimum_count must be a positive integer")
    if type(block_boundary) is not int or not 1 <= block_boundary < validation_stats.features:
        raise ValueError("block_boundary must leave two nonempty blocks and include the intercept in block0")
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
    if active_variables is None:
        active = torch.ones(expected[0], dtype=torch.bool)
    else:
        active = torch.as_tensor(active_variables, device="cpu").detach()
        if active.dtype != torch.bool or active.shape != (expected[0],):
            raise ValueError("active_variables must be a boolean vector with one entry per variable")
    counts = validation_stats.counts.detach().cpu().double()
    if counts.shape != (expected[0],) or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("validation counts must be finite and nonnegative")
    supported = torch.where(active[:, None], beta, 0.)
    initial = validation_stats.score(supported)
    if initial["macro_mse"] is None or not math.isfinite(initial["macro_mse"]):
        raise ValueError("Candidate selection requires observed validation targets and finite macro MSE")
    best_sse = torch.tensor(initial["per_variable_sse"], dtype=torch.float64)
    gains = active[:, None].expand(-1, 2).float().clone()
    eligible = active & (counts >= minimum_count)
    for raw_gain in ordered:
        for extra_gain in ordered:
            if raw_gain == extra_gain == 1.:
                continue
            # Fold in float32 before the full quadratic score, retaining the
            # raw/extra cross term and exactly matching coefficient deployment.
            candidate_beta = torch.cat((supported[:, :block_boundary] * raw_gain,
                                        supported[:, block_boundary:] * extra_gain), dim=-1)
            score = validation_stats.score(candidate_beta)
            candidate_sse = torch.tensor(score["per_variable_sse"], dtype=torch.float64)
            better = eligible & (candidate_sse < best_sse)
            pair = torch.tensor([raw_gain, extra_gain], dtype=torch.float32)
            gains = torch.where(better[:, None], pair, gains)
            best_sse = torch.where(better, candidate_sse, best_sse)
    effective_beta = torch.cat((beta[:, :block_boundary] * gains[:, :1],
                                beta[:, block_boundary:] * gains[:, 1:]), dim=-1)
    effective = validation_stats.score(effective_beta)
    torch.testing.assert_close(torch.tensor(effective["per_variable_sse"], dtype=torch.float64),
                               best_sse, rtol=1e-12, atol=1e-12)

    beta64 = supported.double()
    first, second = beta64[:, :block_boundary], beta64[:, block_boundary:]
    gram = validation_stats.gram.detach().cpu().double()
    cross = validation_stats.cross.detach().cpu().double()
    first_energy = torch.einsum("vi,vij,vj->v", first, gram[:, :block_boundary, :block_boundary], first)
    second_energy = torch.einsum("vi,vij,vj->v", second, gram[:, block_boundary:, block_boundary:], second)
    interaction = torch.einsum("vi,vij,vj->v", first, gram[:, :block_boundary, block_boundary:], second)
    first_cross = (first * cross[:, :block_boundary]).sum(-1)
    second_cross = (second * cross[:, block_boundary:]).sum(-1)
    norm1, norm2 = first_energy.clamp_min(0).sqrt(), second_energy.clamp_min(0).sqrt()
    norm_sum = norm1 + norm2
    cosine_defined = (norm1 > 0) & (norm2 > 0)
    cosine = torch.where(cosine_defined, interaction / (norm1 * norm2).clamp_min(1e-300), 0.).clamp(-1, 1)
    combined_norm = (first_energy + second_energy + 2 * interaction).clamp_min(0).sqrt()
    cancellation = torch.where(norm_sum > 0, 1 - combined_norm / norm_sum.clamp_min(1e-300), 0.).clamp(0, 1)
    rows = []
    for variable in range(expected[0]):
        if not active[variable]:
            reason = "inactive_training_variable"
        elif not eligible[variable]:
            reason = "below_minimum_validation_observations"
        elif (gains[variable] != 1).any():
            reason = "strictly_lower_validation_sse"
        else:
            reason = "original_pair_is_best_or_tied"
        before = initial["per_variable_mse"][variable]
        after = effective["per_variable_mse"][variable]
        rows.append({"variable": variable, "training_active": bool(active[variable]),
            "validation_count": counts[variable].item(), "eligible": bool(eligible[variable]),
            "gains": gains[variable].tolist(), "selection_reason": reason,
            "unshrunk_validation_mse": before, "effective_validation_mse": after,
            "validation_mse_gain": before - after if before is not None else None,
            "unshrunk_block_energy_sums": [first_energy[variable].item(), second_energy[variable].item()],
            "unshrunk_block_residual_cross_sums": [first_cross[variable].item(), second_cross[variable].item()],
            "unshrunk_cross_block_inner_product": interaction[variable].item(),
            "unshrunk_cross_block_sse_term": 2 * interaction[variable].item(),
            "unshrunk_block_cosine": cosine[variable].item(),
            "unshrunk_block_cosine_defined": bool(cosine_defined[variable]),
            "unshrunk_block_cancellation_fraction": cancellation[variable].item()})
    report = {"kind": "validation_calibrated_two_block_readout_candidate",
        "definition": "Train-fitted coefficients stay fixed. Each eligible variable jointly selects two discrete block gains on validation; the outer caller selects its penalty by effective macro MSE.",
        "selection_split": "validation", "block_boundary": block_boundary,
        "block_definition": ["intercept_and_raw_features", "additional_features"],
        "grid": ordered, "gain_pairs_per_eligible_variable": len(ordered) ** 2,
        "validation_selected_scalars_per_eligible_variable": 2,
        "minimum_count": minimum_count,
        "count_definition": "Observed validation target weight, not independent patients or windows",
        "tie_policy": "Start with (1,1); enumerate raw gain outer and extra gain inner; replace only for strictly lower per-variable SSE",
        "candidate_tie_policy": "Caller retains zero correction first and replaces only for strictly lower effective macro MSE in original candidate order",
        "score_definition": "Float64 sufficient-statistic score of blockwise float32(beta32*gain32), including the full cross-block quadratic term",
        "deployment_definition": "Store fixed raw coefficients and two gains; multiply coefficient blocks in float32 before the readout matmul. Shared/local diagnostics apply the same per-variable gains to the shared coefficient vector.",
        "gradient_definition": "Unshrunk training normal-equation stationarity does not describe the validation-calibrated deployed prediction",
        "coefficients_refitted": False, "candidate_coefficients_fixed_during_gain_selection": True,
        "lambda_selection_locked_before_gain": False, "outer_selection_objective": "effective_validation_macro_mse",
        "active_mask_supplied": active_variables is not None, "active_variables": active.tolist(),
        "inactive_variables": (~active).nonzero(as_tuple=True)[0].tolist(),
        "inactive_gain_definition": "Both zero gains encode training inactivity, not validation fitting, including when0 is absent from grid",
        "eligible_count": int(eligible.sum()), "gains": gains.tolist(),
        "changed_count": int((gains != 1).any(-1).sum()),
        "closed_variables": (gains == 0).all(-1).nonzero(as_tuple=True)[0].tolist(),
        "unshrunk_validation": initial, "effective_validation": effective, "per_variable": rows}
    return effective_beta, gains, effective, report
