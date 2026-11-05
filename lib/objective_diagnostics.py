"""Per-variable summaries of training loss coefficients."""

import math
import numbers

import torch


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return int(value)


class ObjectiveAlignmentDiagnostics:
    """O(V) detached summaries of variable loss coefficients over one epoch.

    Counts are weighted target-mask sums, not independent record counts. The
    global-macro coefficient intentionally does not average prediction samples,
    matching compute_all_losses. Batch-macro includes samples in its denominator.
    """

    _sum_fields = (
        "observed_counts", "expanded_counts", "batch_presence",
        "batch_mass", "batch_mass_square", "global_mass", "global_mass_square",
        "actual_unit_weight_sum",
    )

    def __init__(self, variables, training_counts=None, training_batches=None,
                 reduction="batch_macro"):
        self.variables = _positive_integer(variables, "variables")
        if reduction not in {"batch_macro", "global_macro"}:
            raise ValueError("reduction must be batch_macro or global_macro")
        self.reduction = reduction
        self.training_batches = (None if training_batches is None else
                                 _positive_integer(training_batches, "training_batches"))
        self.training_counts = None
        if training_counts is not None:
            counts = torch.as_tensor(training_counts)
            if counts.is_complex() or counts.ndim != 1 or counts.shape[0] != self.variables:
                raise ValueError("training_counts must be a real vector with one value per variable")
            counts = counts.detach().to(torch.float64).clone()
            if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
                raise ValueError("training_counts must be finite and nonnegative")
            self.training_counts = counts
        if reduction == "global_macro" and (self.training_counts is None or self.training_batches is None):
            raise ValueError("global_macro requires training_counts and training_batches")
        self._state = None
        self.batches = 0

    def _initialize(self, device):
        self._state = {name: torch.zeros(self.variables, dtype=torch.float64, device=device)
                       for name in self._sum_fields}
        for name in ("actual_unit_weight_min", "actual_slot_coefficient_min"):
            self._state[name] = torch.full((self.variables,), float("inf"), dtype=torch.float64, device=device)
        for name in ("actual_unit_weight_max", "actual_slot_coefficient_max"):
            self._state[name] = torch.zeros(self.variables, dtype=torch.float64, device=device)
        self._empty_batches = torch.zeros((), dtype=torch.float64, device=device)
        self._no_objective_batches = torch.zeros_like(self._empty_batches)
        if self.training_counts is not None:
            self.training_counts = self.training_counts.to(device=device)

    @torch.no_grad()
    def update(self, mask, prediction_shape=None):
        if not isinstance(mask, torch.Tensor) or mask.ndim != 3 or mask.shape[-1] != self.variables or mask.is_complex():
            raise ValueError("mask must be a real tensor with shape [B,T,V]")
        if not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()):
            raise ValueError("mask must be finite and nonnegative")
        copies = 1
        if prediction_shape is not None:
            try:
                shape = tuple(prediction_shape)
            except TypeError as error:
                raise ValueError("prediction_shape must be [B,T,V] or [S,B,T,V]") from error
            if (len(shape) not in (3, 4) or any(isinstance(x, bool) or not isinstance(x, numbers.Integral) or x < 0 for x in shape)
                    or shape[-3:] != tuple(mask.shape)):
                raise ValueError("prediction_shape must match mask [B,T,V] or [S,B,T,V]")
            if len(shape) == 4:
                copies = _positive_integer(shape[0], "prediction sample count")
        if self._state is None or self.batches == 0:
            self._initialize(mask.device)
        elif self._state["observed_counts"].device != mask.device:
            raise ValueError("mask device changed within one diagnostic epoch")
        mask = mask.detach().to(torch.float64)
        counts = mask.sum(dim=(0, 1))
        expanded = counts * copies
        if not bool(torch.isfinite(expanded).all()):
            raise ValueError("weighted target counts overflowed float64")
        present = expanded > 0
        active_variables = present.sum()
        batch_weight = torch.where(present, 1.0 / ((expanded + 1e-8) * active_variables.clamp_min(1)), 0.0)
        batch_mass = expanded * batch_weight
        global_weight = torch.zeros_like(counts)
        if self.training_counts is not None and self.training_batches is not None:
            full_present = self.training_counts > 0
            denominator = torch.where(full_present, self.training_counts, 1.0) * full_present.sum().clamp_min(1)
            global_weight = torch.where(full_present, self.training_batches / denominator, 0.0)
        global_mass = expanded * global_weight
        actual_weight = batch_weight if self.reduction == "batch_macro" else global_weight
        actual_present = present & (actual_weight > 0)
        actual_weight = torch.where(actual_present, actual_weight, 0.0)
        if mask.numel():
            slots = mask.reshape(-1, self.variables)
            smallest = torch.where(slots > 0, slots, float("inf")).amin(0)
            largest = slots.amax(0)
        else:
            smallest = torch.full_like(counts, float("inf"))
            largest = torch.zeros_like(counts)
        additions = (counts, expanded, present.to(torch.float64), batch_mass,
                     batch_mass.square(), global_mass, global_mass.square(), actual_weight)
        for name, addition in zip(self._sum_fields, additions):
            self._state[name].add_(addition)
        for name, value in (("actual_unit_weight_min", actual_weight),
                            ("actual_slot_coefficient_min", smallest * actual_weight)):
            self._state[name] = torch.minimum(self._state[name], torch.where(actual_present, value, float("inf")))
        for name, value in (("actual_unit_weight_max", actual_weight),
                            ("actual_slot_coefficient_max", largest * actual_weight)):
            self._state[name] = torch.maximum(self._state[name], torch.where(actual_present, value, 0.0))
        self._empty_batches.add_(active_variables == 0)
        self._no_objective_batches.add_(~actual_present.any())
        self.batches += 1

    @torch.no_grad()
    def summary(self):
        if self._state is None:
            device = self.training_counts.device if self.training_counts is not None else torch.device("cpu")
            self._initialize(device)
        raw = {name: value.detach().cpu().tolist() for name, value in self._state.items()}
        full = None if self.training_counts is None else self.training_counts.detach().cpu().tolist()
        full_active = None if full is None else sum(value > 0 for value in full)
        empty = int(self._empty_batches.detach().cpu())
        no_objective = int(self._no_objective_batches.detach().cpu())
        global_known = full is not None and self.training_batches is not None
        actual_key = "batch_mass" if self.reduction == "batch_macro" else "global_mass"
        mass_total = sum(raw[actual_key])
        batch_mass_total = sum(raw["batch_mass"])
        global_mass_total = sum(raw["global_mass"])
        rows = []

        def finite(value):
            return value if math.isfinite(value) else None

        def divide(numerator, denominator):
            return finite(numerator / denominator) if denominator > 0 else None

        for index in range(self.variables):
            observed = raw["observed_counts"][index]
            batch_mass = raw["batch_mass"][index]
            global_mass = raw["global_mass"][index]
            actual_mass = raw[actual_key][index]
            ideal = None if full is None or not full_active else (1.0 / full_active if full[index] > 0 else 0.0)
            batch_share = divide(batch_mass, batch_mass_total)
            relative = divide(batch_share, ideal) if batch_share is not None and ideal else None
            row = {
                "variable": index, "observed_weighted_target_count": observed,
                "prediction_expanded_weighted_count": raw["expanded_counts"][index],
                "training_weighted_target_count": None if full is None else full[index],
                "seen_to_full_count_ratio": None if full is None else divide(observed, full[index]),
                "expanded_to_full_count_ratio": None if full is None else divide(raw["expanded_counts"][index], full[index]),
                "batch_presence_count": int(raw["batch_presence"][index]),
                "batch_presence_fraction": divide(raw["batch_presence"][index], self.batches),
                "batch_macro_weight_mass": batch_mass,
                "batch_macro_mean_mass_per_seen_batch": divide(batch_mass, self.batches),
                "batch_macro_normalized_mass_share": batch_share,
                "batch_macro_batch_mass_square_sum": raw["batch_mass_square"][index],
                "batch_macro_effective_batch_support": divide(batch_mass ** 2, raw["batch_mass_square"][index]),
                "batch_macro_batch_concentration": divide(raw["batch_mass_square"][index], batch_mass ** 2),
                "global_macro_weight_mass": global_mass if global_known else None,
                "global_macro_mean_mass_per_seen_batch": divide(global_mass, self.batches) if global_known else None,
                "global_macro_normalized_mass_share": divide(global_mass, global_mass_total) if global_known else None,
                "global_macro_batch_mass_square_sum": raw["global_mass_square"][index] if global_known else None,
                "global_macro_effective_batch_support": divide(global_mass ** 2, raw["global_mass_square"][index]) if global_known else None,
                "global_macro_batch_concentration": divide(raw["global_mass_square"][index], global_mass ** 2) if global_known else None,
                "actual_objective_weight_mass": actual_mass,
                "mean_actual_objective_mass_per_seen_batch": divide(actual_mass, self.batches),
                "mean_actual_unit_mask_weight_per_seen_batch": divide(raw["actual_unit_weight_sum"][index], self.batches),
                "actual_normalized_mass_share": divide(actual_mass, mass_total),
                "ideal_uniform_full_training_share": ideal,
                "batch_macro_share_to_ideal_ratio": relative,
                "batch_macro_relative_underweighting": None if relative is None else 1.0 - relative,
            }
            for field in ("actual_unit_weight_min", "actual_unit_weight_max", "actual_slot_coefficient_min", "actual_slot_coefficient_max"):
                row[field] = finite(raw[field][index]) if actual_mass > 0 else None
            rows.append(row)
        counts_match = None if full is None else all(math.isclose(seen, total, rel_tol=1e-9, abs_tol=1e-9) for seen, total in zip(raw["observed_counts"], full))
        batches_match = None if self.training_batches is None else self.batches == self.training_batches
        known_complete = counts_match is not None and batches_match is not None
        return {
            "schema_version": 1, "reduction": self.reduction, "variables": self.variables,
            "definition": "Data-objective loss coefficients only, excluding auxiliary penalties, gradient norms, and clipping. Counts are nonnegative weighted label-mask sums, not independent records. Global macro does not average S prediction copies. Per-seen-batch means include empty batches; all-empty batch-macro objectives are undefined and excluded from mass.",
            "batch_macro_coefficient": "1 / ((S * batch_mask_sum + 1e-8) * active_batch_variables)",
            "global_macro_coefficient": "training_batches / (training_mask_sum * active_training_variables), without averaging S",
            "observed_slot_coefficient": "mask_value * per_unit_mask_coefficient; masks may be weighted",
            "effective_batch_support_definition": "sum(batch_mass)^2 / sum(batch_mass^2); concentration is its reciprocal",
            "batches_seen": self.batches, "training_batches": self.training_batches,
            "empty_target_batches": empty, "no_objective_variable_batches": no_objective,
            "active_training_variables": full_active,
            "training_counts_available": full is not None, "global_macro_comparison_available": global_known,
            "counts_match_full_pass": counts_match, "batches_match_full_pass": batches_match,
            "partial_pass": not (counts_match and batches_match) if known_complete else None,
            "coverage_definition": "Coverage compares unexpanded weighted mask counts and batch count; sample-expanded counts are reported separately.",
            "actual_total_weight_mass": finite(mass_total), "batch_macro_total_weight_mass": finite(batch_mass_total),
            "global_macro_total_weight_mass": finite(global_mass_total) if global_known else None,
            "per_variable": rows,
        }
