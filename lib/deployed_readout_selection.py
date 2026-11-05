"""Score one train-fitted candidate after its final validation gain calibration."""
import math

import torch

from lib.readout_shrinkage import select_readout_gains


@torch.no_grad()
def calibrate_deployed_readout_candidate(validation_stats, coefficients, *,
                                         grid=(1., .75, .5, .25, 0.),
                                         minimum_count=64, active_variables=None):
    """Return calibrated gains, validation score and candidate diagnostics.

    The caller retains the original candidate order and zero-correction fallback,
    and selects its penalty only after comparing these effective macro-MSEs.
    """
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
    supported_beta = torch.where(active[:, None], beta, 0.)
    gains, original_report = select_readout_gains(
        validation_stats, supported_beta, grid=grid, minimum_count=minimum_count)
    # Encode training inactivity in the returned gain too, so beta*gains is the
    # exact coefficient vector scored here even for a nonzero inactive input row.
    gains = torch.where(active, gains, 0.)
    effective = validation_stats.score((beta * gains[:, None]).float())
    if effective["macro_mse"] is None or not math.isfinite(effective["macro_mse"]):
        raise ValueError("Candidate selection requires observed validation targets and finite macro MSE")
    report = dict(original_report)
    report.pop("lambda_reselected", None)
    per_variable = []
    for variable, row in enumerate(original_report["per_variable"]):
        details = {**row, "training_active": bool(active[variable])}
        if not active[variable]:
            details.update(gain=0., eligible=False,
                           selection_reason="inactive_training_variable")
        per_variable.append(details)
    report.update({
        "kind": "validation_calibrated_readout_candidate",
        "definition": ("Each candidate's train-fitted coefficients stay fixed while eligible variable gains "
                       "are calibrated on validation. The caller compares effective validation macro MSE "
                       "across the unchanged coefficient/penalty candidate grid, including zero correction."),
        "outer_selection_objective": "effective_validation_macro_mse",
        "lambda_selection_locked_before_gain": False,
        "candidate_coefficients_fixed_during_gain_selection": True,
        "candidate_tie_policy": "Zero correction first; replace only for strictly lower effective macro MSE; other ties preserve original candidate order",
        "score_definition": ("Float64 sufficient-statistic score of float32(raw_beta32*returned_gain32); "
                             "returned gains also encode training inactivity. FittedReadout evaluates the "
                             "raw correction before multiplying its gain, so direct forward must verify rounding agreement."),
        "active_mask_supplied": active_variables is not None,
        "active_variables": active.tolist(),
        "inactive_variables": (~active).nonzero(as_tuple=True)[0].tolist(),
        "inactive_gain_definition": "Gain0 encodes the supplied training-support activity mask, not validation fitting; it applies even when0 is absent from the gain grid",
        "gains": gains.tolist(),
        "validation_count_eligible_count": original_report["eligible_count"],
        "eligible_count": sum(bool(row["eligible"]) for row in per_variable),
        "changed_count": int((gains != 1).sum()),
        "closed_variables": (gains == 0).nonzero(as_tuple=True)[0].tolist(),
        "closed_by_validation": ((gains == 0) & active).nonzero(as_tuple=True)[0].tolist(),
        "effective_validation": effective,
        "per_variable": per_variable,
    })
    return gains, effective, report
