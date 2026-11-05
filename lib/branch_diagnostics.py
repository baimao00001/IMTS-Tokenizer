"""Single-model branch errors and actual gate values on observed forecast targets."""

import math
import torch

from lib.reliability_diagnostics import ReliabilityDiagnostics
from lib.diagnostic_metrics import _flush_counter_updates


def _packed_statistics(values):
    packed = torch.cat([value.reshape(-1) for value in values]).cpu()
    return [part.reshape(value.shape) for part, value in
            zip(packed.split([value.numel() for value in values]), values)]


def _statistics_on_cpu(values):
    return values if all(value.device.type == "cpu" for value in values) else _packed_statistics(values)


def _flush_statistics(pending):
    if not pending:
        return
    try:
        updates = _statistics_on_cpu([update for _, _, update in pending])
        for (destination, name, _), update in zip(pending, updates):
            # Preserve the legacy JSON integer type for empty selected groups.
            if not update[0].any():
                update = update.long()
            destination[name] = destination.get(name, torch.zeros_like(update)) + update
    finally:
        pending.clear()


class BranchDiagnostics:
    accepts_target_tensors = True

    def __init__(self, history_width):
        if not math.isfinite(history_width) or history_width <= 0:
            raise ValueError("history_width must be finite and positive")
        self.history_width = history_width
        self.names = ["all"] + [f"count_{count}/age_{age}" for count in ("0", "1", "2_to_4", "over_4")
                                for age in ("0_to_quarter", "quarter_to_half", "half_to_one", "over_one", "unavailable")]
        self.errors, self.gates, self.nonfinite = {}, {}, {}
        self.corrections, self.tails = {}, {}
        self.batches = 0
        self.negative_age_targets = 0
        self.reliability = ReliabilityDiagnostics(history_width)

    @torch.no_grad()
    def record(self, batch, prediction, components=None, gates=None, target_tensors=None):
        pending_counts, pending_statistics = [], []
        try:
            self._record(batch, prediction, components, gates, target_tensors,
                         pending_counts, pending_statistics)
        finally:
            try:
                _flush_counter_updates(pending_counts)
            finally:
                _flush_statistics(pending_statistics)

    def _record(self, batch, prediction, components, gates, target_tensors,
                pending_counts, pending_statistics):
        target = batch["data_to_predict"].detach()
        target_mask = batch["mask_predicted_data"]
        valid = (target_mask > 0) & torch.isfinite(target_mask) & torch.isfinite(target)
        b, t, v = target.shape
        batch_index, time_index, variables = valid.nonzero(as_tuple=True)
        target_values = target[batch_index, time_index, variables].double()
        values = batch["observed_data"].detach().reshape(b, -1, v).double()
        observed = batch["observed_mask"].reshape_as(values) > 0
        times = batch["observed_tp"].detach().reshape_as(values).double()
        observed = observed & torch.isfinite(values) & torch.isfinite(times)
        counts = observed.sum(1)
        mean = torch.where(observed, values, 0).sum(1) / counts.clamp_min(1)
        if values.shape[1]:
            last_time, index = times.masked_fill(~observed, -torch.inf).max(1)
            last = torch.where(observed, values, 0).gather(1, index.unsqueeze(1)).squeeze(1)
        else:
            last_time = mean.new_full(mean.shape, -torch.inf)
            last = mean
        query = batch["tp_to_predict"].detach().double()
        if query.ndim == 1:
            query = query.unsqueeze(0).expand(b, -1)
        target_counts = counts[batch_index, variables]
        age = (query[batch_index, time_index] - last_time[batch_index, variables]) / self.history_width
        pending_counts.append((self.__dict__, "negative_age_targets", (torch.isfinite(age) & (age < 0)).sum()))
        age_group = (age >= .25).long() + (age >= .5).long() + (age >= 1).long()
        age_group = torch.where(torch.isfinite(age) & (target_counts > 0), age_group, 4)
        count_group = (target_counts > 0).long() + (target_counts > 1).long() + (target_counts > 4).long()
        groups = 1 + count_group * 5 + age_group
        combined = groups * v + variables
        size = len(self.names) * v

        def reduce(weight, chosen):
            selected = torch.where(chosen, weight, 0.0)
            return (torch.bincount(combined, weights=selected, minlength=size) +
                    torch.bincount(variables, weights=selected, minlength=size)).reshape(len(self.names), v)

        def reduce_statistics(weights, chosen):
            return torch.stack([reduce(weight, chosen) for weight in weights])

        def observed_targets(value):
            return value.detach().expand_as(target)[batch_index, time_index, variables].double()

        parts = {"forecast": prediction, "history_mean": mean[:, None, :],
                 "last_observation": last[:, None, :]}
        parts.update(components or {})
        # Select target observations once; all branch arithmetic stays sparse.
        forecast = observed_targets(prediction)
        parts = {name: forecast if part is prediction else observed_targets(part) for name, part in parts.items()}
        ones = torch.ones_like(target_values)
        centered = torch.where(observed, values - mean[:, None, :], 0.0)
        variance = centered.square().sum(1) / counts.clamp_min(1)
        scale = (variance + 1e-4).sqrt()
        target_distance = (target_values - mean[batch_index, variables]).abs() / scale[batch_index, variables]
        tail_groups = (("history_supported", target_counts > 0),
                       ("over_2_history_std", (target_counts > 1) & (target_distance > 2)),
                       ("over_4_history_std", (target_counts > 1) & (target_distance > 4)))
        for name, part in parts.items():
            finite = torch.isfinite(part)
            chosen = finite
            if name in ("history_mean", "last_observation"):
                chosen = chosen & (target_counts > 0)
            pending_counts.append((self.nonfinite, name, (~finite).sum()))
            difference = part - target_values
            squared, absolute = difference.square(), difference.abs()
            update = reduce_statistics([ones, squared, absolute], chosen)
            pending_statistics.append((self.errors, name, update))
            # Use only historical scale to define tail groups, not a fitted target threshold.
            for tail_name, selector in tail_groups:
                selected = chosen & selector
                tail_update = reduce_statistics([ones, squared, absolute], selected)
                key = name + "/" + tail_name
                pending_statistics.append((self.tails, key, tail_update))

        reference_names = [name for name in (
            "reference", "core_forecast", "ridge_core", "training_prior", "last_observation",
            "shared_readout", "fitted_core", "fitted_shared_only", "fitted_local_only",
            "fitted_without_extra", "fitted_extra_only", "fitted_without_context", "fitted_context_only",
            "fitted_without_shrinkage") if name in parts]
        for reference_name in reference_names:
            reference = parts[reference_name]
            chosen = torch.isfinite(reference) & torch.isfinite(forecast)
            if reference_name == "last_observation":
                chosen = chosen & (target_counts > 0)
            needed = target_values - reference
            correction = forecast - reference
            product = needed * correction
            baseline_sse, energy = needed.square(), correction.square()
            improvement = 2 * product - energy
            update = reduce_statistics([ones, baseline_sse, energy, product,
                                        (improvement > 0).double(), (improvement < 0).double(),
                                        (product < 0).double()], chosen)
            pending_statistics.append((self.corrections, reference_name, update))
        for name, gate in (gates or {}).items():
            gate = observed_targets(gate)
            finite = torch.isfinite(gate)
            chosen = finite
            key = "gate/" + name
            pending_counts.append((self.nonfinite, key, (~finite).sum()))
            if name == "reference":
                chosen = chosen & (target_counts > 0)
            probability = gate.clamp(1e-12, 1 - 1e-12)
            entropy = -(probability * probability.log() + (1 - probability) * (1 - probability).log())
            update = reduce_statistics([ones, gate, ((gate < .05) | (gate > .95)).double(), entropy], chosen)
            pending_statistics.append((self.gates, name, update))
        self.batches += 1
        self.reliability.record(batch, prediction, components, target_tensors)

    def summary(self):
        def number(value):
            value = float(value)
            return value if math.isfinite(value) else None
        result = {"batches": self.batches, "history_width": self.history_width,
                  "negative_age_targets": self.negative_age_targets,
                  "definitions": {"age": "(query_time - last observed time of this variable) / history_width; unavailable for no history",
                                  "count": "usable history slots of this variable",
                                  "components": "Prediction components of the same model and shared head",
                                  "reference_gate": "only targets with own-variable history; missing-history fallback does not use this gate"},
                  "groups": {}}
        for index, group in enumerate(self.names):
            row = {"components": {}, "gates": {}, "correction_alignment": {}, "tail_errors": {}}
            for name, values in self.errors.items():
                count, sse, sae = values[:, index]
                present = count > 0
                row["components"][name] = {"count": int(count.sum()),
                    "macro_mse": number((sse[present] / count[present]).mean()) if present.any() else None,
                    "micro_mse": number(sse.sum() / count.sum()) if count.sum() else None,
                    "macro_mae": number((sae[present] / count[present]).mean()) if present.any() else None,
                    "per_variable_count": count.tolist(), "per_variable_sse": [number(x) for x in sse],
                    "per_variable_sae": [number(x) for x in sae],
                    "overflowed_statistics": int((~torch.isfinite(values[:, index])).sum())}
            for name, values in self.gates.items():
                by_variable = values[:, index]
                count, summed, saturated, entropy = by_variable.sum(-1)
                row["gates"][name] = {"count": int(count), "mean": number(summed / count) if count else None,
                                      "saturated_fraction": number(saturated / count) if count else None,
                                      "entropy": number(entropy / count) if count else None,
                                      "per_variable_count": by_variable[0].tolist(),
                                      "per_variable_mean": [number(total / n) if n else None for n, total in zip(by_variable[0], by_variable[1])],
                                      "per_variable_saturated_fraction": [number(total / n) if n else None for n, total in zip(by_variable[0], by_variable[2])]}
            for name, values in self.corrections.items():
                count, baseline, energy, cross, helped, harmed, wrong_direction = values[:, index]
                present = count > 0
                n = count.sum()
                normalized = values[:, index, present] / count[present] if present.any() else None
                mean_stats = normalized.mean(-1) if normalized is not None else None
                macro_energy = float(mean_stats[2]) if mean_stats is not None else 0.0
                alpha = float(mean_stats[3]) / macro_energy if macro_energy > 0 else None
                clipped_alpha = max(0.0, min(1.0, alpha)) if alpha is not None and math.isfinite(alpha) else None
                oracle_mse = None
                if mean_stats is not None:
                    if macro_energy == 0:
                        oracle_mse = number(mean_stats[1])
                    elif clipped_alpha is not None:
                        oracle_mse = number(mean_stats[1] - 2 * clipped_alpha * mean_stats[3] + clipped_alpha ** 2 * mean_stats[2])
                row["correction_alignment"][name] = {
                    "count": int(n), "helped_fraction": number(helped.sum() / n) if n else None,
                    "harmed_fraction": number(harmed.sum() / n) if n else None,
                    "wrong_direction_fraction": number(wrong_direction.sum() / n) if n else None,
                    "macro_reference_mse": number(mean_stats[1]) if mean_stats is not None else None,
                    "macro_correction_energy": number(mean_stats[2]) if mean_stats is not None else None,
                    "macro_residual_cross": number(mean_stats[3]) if mean_stats is not None else None,
                    "macro_mse_gain": number(2 * mean_stats[3] - mean_stats[2]) if mean_stats is not None else None,
                    "oracle_alpha_macro": number(alpha) if alpha is not None else None,
                    "oracle_shrink_alpha_macro": clipped_alpha,
                    "oracle_shrink_macro_mse": oracle_mse,
                    "per_variable_count": count.tolist(),
                    "per_variable_reference_sse": [number(x) for x in baseline],
                    "per_variable_correction_energy": [number(x) for x in energy],
                    "per_variable_required_cross": [number(x) for x in cross],
                    "per_variable_oracle_alpha": [number(c / e) if e > 0 else None for c, e in zip(cross, energy)],
                    "overflowed_statistics": int((~torch.isfinite(values[:, index])).sum()),
                }
            for name, values in self.tails.items():
                count, sse, sae = values[:, index]
                present = count > 0
                row["tail_errors"][name] = {
                    "count": int(count.sum()),
                    "macro_mse": number((sse[present] / count[present]).mean()) if present.any() else None,
                    "macro_mae": number((sae[present] / count[present]).mean()) if present.any() else None,
                    "per_variable_count": count.tolist(),
                    "per_variable_sse": [number(x) for x in sse],
                    "overflowed_statistics": int((~torch.isfinite(values[:, index])).sum()),
                }
            result["groups"][group] = row
        result["nonfinite_component_predictions"] = self.nonfinite
        result["definitions"]["correction_alignment"] = "forecast = reference + correction; oracle alpha is an observed-target diagnostic"
        result["definitions"]["correction_energy"] = "Macro averages per-variable target means: E=mean(correction^2), C=mean((target-reference)*correction), gain=2*C-E, oracle alpha=C/E. The reference key names the prediction subtracted from the full forecast; fitted_shared_only and fitted_local_only include the fitted core."
        result["definitions"]["tail_errors"] = "abs(target - history_mean) > k*sqrt(history_variance + 1e-4), count>1; thresholds are historical and identical for every branch"
        result["reliability"] = self.reliability.summary()
        return result
