"""Detached forecast diagnostics accumulated over every batch."""

import math

import torch


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _divide(numerator, denominator):
    return _number(numerator / denominator) if denominator > 0 else None


def _packed_counter_values(values):
    return torch.stack(values).cpu().tolist()


def _counter_values(values):
    if all(value.device.type == "cpu" for value in values):
        return [int(value) for value in values]
    return _packed_counter_values(values)


def _flush_counter_updates(pending):
    if not pending:
        return
    try:
        counts = _counter_values([value for _, _, value in pending])
        for (destination, name, _), count in zip(pending, counts):
            destination[name] = destination.get(name, 0) + int(count)
    finally:
        pending.clear()


class _Moments:
    """Per-variable additive statistics, kept on CPU between batches."""

    fields = (
        "count", "sse", "sae", "signed_error_sum", "prediction_sum",
        "target_sum", "prediction_square_sum", "target_square_sum",
        "prediction_target_sum", "absolute_percentage_sum",
        "percentage_count",
    )

    def __init__(self, dimensions):
        self.totals = torch.zeros(len(self.fields), dimensions, dtype=torch.float64)

    @staticmethod
    def values(prediction, target, valid):
        # Reuse moments for every subgroup of the same predictions.
        prediction = torch.where(valid, prediction, 0.0)
        target = torch.where(valid, target, 0.0)
        error = prediction - target
        percentage_valid = valid & (target != 0)
        safe_denominator = torch.where(percentage_valid, target.abs(), 1.0)
        return (
            None, error.square(), error.abs(), error,
            prediction, target, prediction.square(), target.square(),
            prediction * target,
            torch.where(percentage_valid, error.abs() / safe_denominator, 0.0),
            percentage_valid.to(torch.float64),
        )

    @staticmethod
    def reduce(values, valid):
        dimensions = valid.shape[-1]
        return torch.stack([
            valid.reshape(-1, dimensions).sum(0, dtype=torch.float64)
            if value is None else
            torch.where(valid, value, 0.0).reshape(-1, dimensions).sum(0)
            for value in values
        ])

    def summary(self):
        raw = dict(zip(self.fields, self.totals.tolist()))
        dimensions = self.totals.shape[1]
        variables = []
        for index in range(dimensions):
            values = {name: raw[name][index] for name in self.fields}
            count = values["count"]
            pred_mean = _divide(values["prediction_sum"], count)
            target_mean = _divide(values["target_sum"], count)
            pred_variance = target_variance = correlation = None
            if count > 0 and pred_mean is not None and target_mean is not None:
                pred_variance = _number(values["prediction_square_sum"] / count - pred_mean * pred_mean)
                target_variance = _number(values["target_square_sum"] / count - target_mean * target_mean)
                pred_variance = max(0.0, pred_variance) if pred_variance is not None else None
                target_variance = max(0.0, target_variance) if target_variance is not None else None
                covariance = _number(values["prediction_target_sum"] / count - pred_mean * target_mean)
                if count > 1 and pred_variance and target_variance and covariance is not None:
                    denominator = math.sqrt(pred_variance) * math.sqrt(target_variance)
                    candidate = _number(covariance / denominator)
                    correlation = max(-1.0, min(1.0, candidate)) if candidate is not None else None
            variables.append({
                "variable": index, "count": int(count),
                "mse": _divide(values["sse"], count),
                "mae": _divide(values["sae"], count),
                "bias": _divide(values["signed_error_sum"], count),
                "prediction_mean": pred_mean, "target_mean": target_mean,
                "prediction_variance": _number(pred_variance) if pred_variance is not None else None,
                "target_variance": _number(target_variance) if target_variance is not None else None,
                "correlation": correlation,
                "corrected_mape": _divide(values["absolute_percentage_sum"], values["percentage_count"]),
            })
        present = [row for row in variables if row["count"] > 0]
        total_count = sum(raw["count"])
        def macro(values):
            return _divide(sum(values), len(values)) if all(value is not None for value in values) else None
        macro_mse = macro([row["mse"] for row in present])
        macro_mae = macro([row["mae"] for row in present])
        present_percentage = [row["corrected_mape"] for index, row in enumerate(variables)
                              if raw["percentage_count"][index] > 0]
        return {
            "count": int(total_count), "present_variables": len(present),
            "macro_mse": macro_mse, "macro_mae": macro_mae,
            "macro_rmse": math.sqrt(macro_mse) if macro_mse is not None else None,
            "micro_mse": _divide(sum(raw["sse"]), total_count),
            "micro_mae": _divide(sum(raw["sae"]), total_count),
            "micro_bias": _divide(sum(raw["signed_error_sum"]), total_count),
            "corrected_macro_mape": macro(present_percentage),
            "nonfinite_sufficient_statistics": int((~torch.isfinite(self.totals)).sum().item()),
            "overflowed_variables": torch.where((~torch.isfinite(self.totals)).any(0))[0].tolist(),
            "per_variable": variables,
            "sufficient_statistics": {
                name: [_number(value) for value in vector] for name, vector in raw.items()
            },
        }


class ForecastDiagnostics:
    """Accumulate all batches of one split; construct a fresh object per split.

    Horizon bins use model timestamp units, with lead relative to the last
    observed timestamp in each sample. Default boundaries are fractions of the
    loader's time normalization scale, not fractions of the forecast window.
    """

    def __init__(self, horizon_edges=(0.0, 0.125, 0.25, 0.5, float("inf"))):
        self.horizon_edges = tuple(float(edge) for edge in horizon_edges)
        if (len(self.horizon_edges) < 2 or self.horizon_edges[0] != 0.0
                or self.horizon_edges[-1] != float("inf")
                or any(not math.isfinite(edge) for edge in self.horizon_edges[:-1])
                or any(right <= left for left, right in zip(self.horizon_edges, self.horizon_edges[1:]))):
            raise ValueError("horizon_edges must increase strictly from zero to positive infinity")
        self.reset()

    def reset(self):
        self.groups = {}
        self.batches = 0
        self.samples = 0
        self.target_slots = 0
        self.observed_targets = 0
        self.nonfinite_predictions = 0
        self.nonfinite_targets = 0
        self.invalid_target_mask = 0
        self.history_slots = 0
        self.history_observations = 0
        self.samples_without_history = 0
        self.sample_variables_without_history = 0
        self.lead_min = None
        self.lead_max = None
        self.dimensions = None

    def _update(self, name, values, valid, pending):
        if name not in self.groups:
            self.groups[name] = _Moments(valid.shape[-1])
        pending.append((name, _Moments.reduce(values, valid)))

    @torch.no_grad()
    def record(self, batch, prediction):
        pending_counts = []
        try:
            self._record(batch, prediction, pending_counts)
        finally:
            # Commit earlier counters even if a later input check rejects the batch.
            _flush_counter_updates(pending_counts)

    def _record(self, batch, prediction, pending_counts):
        prediction = prediction.detach()
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(0)
        if prediction.ndim != 4:
            raise ValueError("prediction must have shape [B,T,V] or [S,B,T,V]")
        samples, batch_size, horizon, dimensions = prediction.shape
        if self.dimensions is not None and dimensions != self.dimensions:
            raise ValueError("variable count changed within one diagnostic split")
        self.dimensions = dimensions
        target = batch["data_to_predict"].detach()
        target_mask = batch["mask_predicted_data"].detach()
        if target.shape != (batch_size, horizon, dimensions) or target_mask.shape != target.shape:
            raise ValueError("target and target mask must match prediction [B,T,V]")
        prediction = prediction.to(torch.float64)
        target = target.to(torch.float64).unsqueeze(0).expand_as(prediction)
        target_mask = target_mask.unsqueeze(0).expand_as(prediction)
        requested = target_mask > 0
        valid = requested & torch.isfinite(prediction) & torch.isfinite(target)
        pending = []
        forecast_values = _Moments.values(prediction, target, valid)
        self.batches += 1
        self.samples += batch_size
        self.target_slots += target_mask.numel()
        pending_counts.extend((
            (self.__dict__, "observed_targets", requested.sum()),
            (self.__dict__, "nonfinite_predictions", (requested & ~torch.isfinite(prediction)).sum()),
            (self.__dict__, "nonfinite_targets", (requested & ~torch.isfinite(target)).sum()),
            (self.__dict__, "invalid_target_mask", ((target_mask != 0) & (target_mask != 1)).sum()),
        ))
        self._update("forecast", forecast_values, valid, pending)

        history = batch["observed_data"].detach().reshape(batch_size, -1, dimensions).to(torch.float64)
        history_mask = batch["observed_mask"].detach().reshape_as(history) > 0
        history_time = batch["observed_tp"].detach()
        if history_time.shape == batch["observed_data"].shape:
            history_time = history_time.reshape_as(history)
        elif history_time.numel() == batch_size * history.shape[1]:
            history_time = history_time.reshape(batch_size, -1, 1).expand_as(history)
        elif history_time.ndim == 1 and history_time.numel() == history.shape[1]:
            history_time = history_time.reshape(1, -1, 1).expand_as(history)
        else:
            raise ValueError("observation timestamps do not match observation slots")
        history_time = history_time.to(torch.float64)
        history_valid = history_mask & torch.isfinite(history) & torch.isfinite(history_time)
        counts = history_valid.sum(1)
        available = counts > 0
        self.history_slots += history_mask.numel()
        pending_counts.extend((
            (self.__dict__, "history_observations", history_mask.sum()),
            (self.__dict__, "samples_without_history", (~available.any(-1)).sum()),
            (self.__dict__, "sample_variables_without_history", (~available).sum()),
        ))
        clean_history = torch.where(history_valid, history, 0.0)
        history_mean = clean_history.sum(1) / counts.clamp_min(1)
        if history.shape[1] > 0:
            # Timestamp argmax handles unsorted input and padded zero timestamps.
            valid_times = torch.where(history_valid, history_time, -float("inf"))
            last_times, last_indices = valid_times.max(1)
            history_last = clean_history.gather(1, last_indices.unsqueeze(1)).squeeze(1)
            last_sample_time = last_times.max(-1).values
        else:
            history_last = history_mean
            last_sample_time = prediction.new_full((batch_size,), -float("inf"))
        baseline_valid = valid & available.unsqueeze(0).unsqueeze(2)
        for name, baseline in (("history_mean", history_mean), ("last_observation", history_last)):
            baseline = baseline.unsqueeze(0).unsqueeze(2).expand_as(prediction)
            baseline_values = _Moments.values(baseline, target, baseline_valid)
            self._update("baseline_" + name, baseline_values, baseline_valid, pending)
        self._update("forecast_with_history", forecast_values, baseline_valid, pending)
        self._update("forecast_without_history", forecast_values, valid & ~available.unsqueeze(0).unsqueeze(2), pending)

        # Denominator excludes rows with no usable measurements in any variable.
        valid_rows = history_valid.any(-1).sum(-1, keepdim=True)
        density = counts.to(torch.float64) / valid_rows.clamp_min(1)
        for index, (lower, upper) in enumerate(((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0000001))):
            selected = (density >= lower) & (density < upper)
            self._update("history_slot_occupancy_" + str(index), forecast_values, valid & selected.unsqueeze(0).unsqueeze(2), pending)
        for name, selected in (
            ("0", counts == 0), ("1", counts == 1),
            ("2_to_4", (counts >= 2) & (counts <= 4)),
            ("5_to_16", (counts >= 5) & (counts <= 16)), ("over_16", counts > 16),
        ):
            self._update("history_count_" + name, forecast_values, valid & selected.unsqueeze(0).unsqueeze(2), pending)

        query_time = batch["tp_to_predict"].detach().to(torch.float64)
        if query_time.ndim == 1:
            query_time = query_time.unsqueeze(0).expand(batch_size, -1)
        if query_time.shape != (batch_size, horizon):
            raise ValueError("forecast timestamps must have shape [T] or [B,T]")
        lead = query_time - last_sample_time.unsqueeze(-1)
        lead_valid = torch.isfinite(lead) & available.any(-1).unsqueeze(-1)
        active_lead = lead_valid & requested.any(0).any(-1)
        if active_lead.any():
            minimum = float(lead[active_lead].min().item())
            maximum = float(lead[active_lead].max().item())
            self.lead_min = minimum if self.lead_min is None else min(self.lead_min, minimum)
            self.lead_max = maximum if self.lead_max is None else max(self.lead_max, maximum)
        for index, (lower, upper) in enumerate(zip(self.horizon_edges, self.horizon_edges[1:])):
            selected = lead_valid & (lead >= lower) & (lead < upper)
            self._update("horizon_" + str(index), forecast_values, valid & selected.unsqueeze(0).unsqueeze(-1), pending)
        self._update("negative_lead", forecast_values, valid & (lead_valid & (lead < 0)).unsqueeze(0).unsqueeze(-1), pending)
        self._update("unavailable_lead", forecast_values, valid & (~lead_valid).unsqueeze(0).unsqueeze(-1), pending)
        # Transfer all group reductions together, avoiding one GPU sync per group.
        updates = torch.stack([update for _, update in pending]).cpu()
        for (name, _), update in zip(pending, updates):
            self.groups[name].totals += update

    def summary(self):
        groups = {name: accumulator.summary() for name, accumulator in self.groups.items()}
        baseline_comparisons = {}
        forecast = groups.get("forecast_with_history", {})
        for name in ("history_mean", "last_observation"):
            baseline = groups.get("baseline_" + name, {})
            baseline_mse = baseline.get("micro_mse")
            forecast_mse = forecast.get("micro_mse")
            baseline_comparisons[name] = {
                "matched_count": baseline.get("count", 0),
                "micro_mse_skill": _number(1.0 - forecast_mse / baseline_mse)
                if baseline_mse is not None and baseline_mse > 0 and forecast_mse is not None else None,
            }
        return {
            "schema_version": 1,
            "definitions": {
                "aggregation": "All batches; per-variable sums are additive; macro averages only observed variables; micro weights observations.",
                "finite_policy": "Statistics use finite observed prediction-target pairs; nonfinite and invalid-mask counts are reported separately. Overflowed sufficient statistics and dependent metrics become null and have explicit counters.",
                "variance": "Population variance over finite observed prediction-target pairs.",
                "mape": "Absolute error / absolute nonzero target in model value units; no inverse normalization; ratio, not percent.",
                "horizon": "Query timestamp minus last usable observation timestamp of each sample, in model timestamp units; boundaries are fixed across batches.",
                "history_slot_occupancy": "Variable usable history count / input slot rows occupied by any variable. Packed patch rows align local ordinal slots, not physical timestamps, so this is packed-slot relative occupancy, not time coverage. Empty history gives zero.",
                "history_count": "Usable observation slots per sample and variable: 0, 1, 2-4, 5-16, over 16.",
                "history_slots": "Model input slots; overlapping patches count repeated observations repeatedly for occupancy, counts and mean baseline.",
                "baseline": "Mean and timestamp-selected last value per sample and variable; only variables with usable history; skill uses matched observations.",
                "target_coverage": "Observed target entries / padded prediction tensor entries, including trajectory sample repetitions.",
            },
            "coverage": {
                "batches": self.batches, "samples": self.samples,
                "target_slots": self.target_slots, "observed_targets": self.observed_targets,
                "target_observed_fraction": _divide(self.observed_targets, self.target_slots),
                "nonfinite_predictions": self.nonfinite_predictions, "nonfinite_targets": self.nonfinite_targets,
                "invalid_target_mask": self.invalid_target_mask,
                "history_slots": self.history_slots, "history_observations": self.history_observations,
                "samples_without_history": self.samples_without_history,
                "sample_variables_without_history": self.sample_variables_without_history,
            },
            "horizon_edges": [_number(edge) for edge in self.horizon_edges],
            "horizon_last_edge_is_infinity": math.isinf(self.horizon_edges[-1]),
            "lead_min": self.lead_min, "lead_max": self.lead_max,
            "history_slot_occupancy_edges": [0.0, 0.25, 0.5, 0.75, 1.0],
            "groups": groups, "baseline_comparisons": baseline_comparisons,
        }
