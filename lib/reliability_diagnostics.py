"""Bounded record-concentration and predictor-state diagnostics."""

import math

import torch


STATE_NAMES = (
    "count_reliability", "recency_reliability", "history_span",
    "last_mean_deviation", "relative_variation", "latent_strength",
)
STATE_EDGES = (
    (0.0, 0.25, 0.5, 0.75), (0.0, 0.25, 0.5, 0.75),
    (0.0, 0.25, 0.5, 1.0), (-0.75, -0.25, 0.25, 0.75),
    (0.0, 0.25, 0.5, 0.75), (0.0, 0.25, 0.5, 0.75),
)


def _numbers(tensor):
    def clean(value):
        return value if math.isfinite(value) else None
    return [clean(float(value)) for value in tensor]


def _packed_reductions(values):
    groups = {}
    for index, value in enumerate(values):
        groups.setdefault((value.device, value.dtype), []).append((index, value))
    result = [None] * len(values)
    for group in groups.values():
        packed = torch.cat([value.reshape(-1) for _, value in group]).cpu()
        pieces = packed.split([value.numel() for _, value in group])
        for (index, value), piece in zip(group, pieces):
            result[index] = piece.reshape(value.shape)
    return result


def _reductions_on_cpu(values):
    return values if all(value.device.type == "cpu" for value in values) else _packed_reductions(values)


def _defer_update(value):
    return value.device.type != "cpu"


class ReliabilityDiagnostics:
    """A record is an input window, not necessarily an independent person."""

    def __init__(self, history_width=None):
        if history_width is not None and (not math.isfinite(history_width) or history_width <= 0):
            raise ValueError("history_width must be finite and positive")
        self.history_width = history_width
        self.batches = 0
        self.input_windows = 0
        self.variables = None
        self.records = {}
        self.states = {}
        self.context_available_batches = 0
        self.context_unavailable_batches = 0
        self.query_slots = {"eligible": 0, "excluded": 0, "observed_targets_outside_scope": 0}

    def _records(self, name, values, pending=None):
        # Each column holds one nonnegative contribution per window/variable.
        values = values.double()
        sums = values.sum(0)
        squares = values.square().sum(0)
        maxima = values.amax(0) if values.shape[0] else torch.zeros_like(sums)
        positive = (values > 0).sum(0)
        updates = (sums, squares, maxima, positive)
        if not _defer_update(sums):
            self._update_records(name, updates)
        elif pending is None:
            self._flush_updates([("records", name, None, updates)])
        else:
            pending.append(("records", name, None, updates))

    def _update_records(self, name, values):
        sums, squares, maxima, positive = values
        if name not in self.records:
            self.records[name] = [torch.zeros_like(sums), torch.zeros_like(squares),
                                  torch.zeros_like(maxima), torch.zeros_like(positive)]
        row = self.records[name]
        row[0] += sums
        row[1] += squares
        row[2] = torch.maximum(row[2], maxima)
        row[3] += positive

    def _apply_update(self, kind, name, keys, values):
        if kind == "records":
            self._update_records(name, values)
        elif kind == "query":
            self.query_slots[name] += int(values[0])
        else:
            dest = self.states.setdefault(name, {})
            for key, value in zip(keys, values):
                dest[key] = dest.get(key, torch.zeros_like(value)) + value

    def _flush_updates(self, pending):
        if not pending:
            return
        try:
            values = _reductions_on_cpu([value for _, _, _, updates in pending for value in updates])
            offset = 0
            for kind, name, keys, updates in pending:
                current = values[offset:offset + len(updates)]
                offset += len(updates)
                self._apply_update(kind, name, keys, current)
        finally:
            pending.clear()

    @torch.no_grad()
    def record(self, batch, prediction, components=None, target_tensors=None):
        pending = []
        try:
            self._record(batch, prediction, components, target_tensors, pending)
        finally:
            # Commit completed reductions on failures too, in their original order.
            self._flush_updates(pending)

    def _record(self, batch, prediction, components, target_tensors, pending):
        target = batch["data_to_predict"].detach()
        mask = batch["mask_predicted_data"].detach()
        if target.ndim != 3 or mask.shape != target.shape:
            raise ValueError("reliability targets and masks must have shape [batch,time,variable]")
        b, t, v = target.shape
        if self.variables is not None and v != self.variables:
            raise ValueError("reliability variable count changed within one observer")
        self.variables = v
        valid = (mask > 0) & torch.isfinite(mask) & torch.isfinite(target)
        bi, ti, vi = valid.nonzero(as_tuple=True)
        truth = target[bi, ti, vi].double()
        forecast = prediction.detach().expand_as(target)[bi, ti, vi].double()
        finite = torch.isfinite(forecast)
        slots = bi * v + vi

        def records(weight, chosen=finite):
            weight = torch.where(chosen, weight, 0.0)
            return torch.bincount(slots, weights=weight, minlength=b * v).reshape(b, v)

        ones = torch.ones_like(truth)
        error = forecast - truth
        self._records("target_count", records(ones, torch.ones_like(finite)), pending)
        self._records("forecast_count", records(ones), pending)
        self._records("forecast_sse", records(error.square()), pending)
        self._records("forecast_sae", records(error.abs()), pending)
        parts = components or {}
        reference_name = next((key for key in (
            "fitted_core", "ridge_core", "core_forecast", "reference", "shared_readout"
        ) if key in parts), None)
        reference = None
        if reference_name is not None:
            reference = parts[reference_name].detach().expand_as(target)[bi, ti, vi].double()
            joint = finite & torch.isfinite(reference)
            needed, correction = truth - reference, forecast - reference
            energy, cross = correction.square(), needed * correction
            record_energy, record_cross = records(energy, joint), records(cross, joint)
            record_gain = 2 * record_cross - record_energy
            prefix = "against/" + reference_name + "/"
            for name, values in (
                ("count", records(ones, joint)),
                ("core_sse", records(needed.square(), joint)),
                ("correction_energy", record_energy),
                ("absolute_record_cross", record_cross.abs()),
                ("positive_record_cross", record_cross.clamp_min(0)),
                ("negative_record_cross", (-record_cross).clamp_min(0)),
                ("positive_record_gain", record_gain.clamp_min(0)),
                ("negative_record_gain", (-record_gain).clamp_min(0)),
            ):
                self._records(prefix + name, values, pending)
        payload = (target_tensors or {}).get("model/fitted_readout", {})
        context = payload.get("context")
        if context is None:
            self.context_unavailable_batches += 1
        else:
            if tuple(payload.get("context_names", ())) != STATE_NAMES:
                raise ValueError("reliability context names must match deployed state ordering")
            if context.shape != (*target.shape, len(STATE_NAMES)):
                raise ValueError("reliability context must have shape [batch,time,variable,6]")
            if self.history_width is None or "tp_to_predict" not in batch:
                raise ValueError("query-slot diagnostics require history_width and prediction times")
            raw_query = batch["tp_to_predict"].detach()
            query = raw_query.double()
            if query.ndim == 1:
                query = query.unsqueeze(0).expand(b, -1)
            if query.shape != (b, t):
                raise ValueError("prediction times must have shape [batch,time] or [time]")
            start = torch.as_tensor(batch.get("window_start", 0.0), device=query.device, dtype=query.dtype)
            if start.ndim == 0 or start.numel() == 1:
                start = start.reshape(1).expand(b)
            elif start.numel() == b:
                start = start.reshape(b)
            else:
                raise ValueError("window_start must be scalar or have one value per input window")
            boundary_time = start + self.history_width
            eps = torch.finfo(raw_query.dtype).eps if raw_query.is_floating_point() else 0.0
            tolerance = 8 * eps * boundary_time.abs().clamp_min(1.0)
            eligible = (torch.isfinite(query) & torch.isfinite(boundary_time[:, None])
                        & (query >= boundary_time[:, None] - tolerance[:, None]))
            eligible = eligible[:, :, None].expand(b, t, v)
            if _defer_update(eligible):
                pending.append(("query", "eligible", None, (eligible.sum(),)))
                pending.append(("query", "excluded", None, ((~eligible).sum(),)))
                pending.append(("query", "observed_targets_outside_scope", None, ((valid & ~eligible).sum(),)))
            else:
                self.query_slots["eligible"] += int(eligible.sum())
                self.query_slots["excluded"] += int((~eligible).sum())
                self.query_slots["observed_targets_outside_scope"] += int((valid & ~eligible).sum())
            self.context_available_batches += 1
            flat_variables = torch.arange(v, device=context.device).repeat(b * t)
            for state_index, (name, edges) in enumerate(zip(STATE_NAMES, STATE_EDGES)):
                state = context.detach()[..., state_index].double()
                all_state = state.reshape(-1)
                boundary = state.new_tensor(edges)
                all_bin = torch.bucketize(all_state.contiguous(), boundary, right=False)
                query_scope = eligible.reshape(-1)
                all_finite = torch.isfinite(all_state) & query_scope
                chosen_state = state[bi, ti, vi]
                chosen_bin = torch.bucketize(chosen_state.contiguous(), boundary, right=False)
                state_finite = torch.isfinite(chosen_state)
                keys = {}

                def bins(index, variable, weight, choose):
                    return torch.bincount(index * v + variable,
                                          weights=torch.where(choose, weight, 0.0),
                                          minlength=5 * v).reshape(5, v)

                keys["query_count"] = bins(all_bin, flat_variables, torch.ones_like(all_state), all_finite)
                keys["query_sum"] = bins(all_bin, flat_variables, all_state, all_finite)
                keys["query_square_sum"] = bins(all_bin, flat_variables, all_state.square(), all_finite)
                keys["query_nonfinite_count"] = bins(all_bin, flat_variables, torch.ones_like(all_state), ~torch.isfinite(all_state) & query_scope)
                keys["target_count"] = bins(chosen_bin, vi, ones, state_finite)
                keys["target_sum"] = bins(chosen_bin, vi, chosen_state, state_finite)
                keys["target_square_sum"] = bins(chosen_bin, vi, chosen_state.square(), state_finite)
                keys["target_nonfinite_count"] = bins(chosen_bin, vi, ones, ~state_finite)
                keys["forecast_count"] = bins(chosen_bin, vi, ones, state_finite & finite)
                keys["forecast_sse"] = bins(chosen_bin, vi, error.square(), state_finite & finite)
                if reference is not None:
                    selected = state_finite & joint
                    prefix = "against/" + reference_name + "/"
                    keys[prefix + "count"] = bins(chosen_bin, vi, ones, selected)
                    keys[prefix + "core_sse"] = bins(chosen_bin, vi, needed.square(), selected)
                    keys[prefix + "correction_energy"] = bins(chosen_bin, vi, energy, selected)
                    keys[prefix + "residual_cross"] = bins(chosen_bin, vi, cross, selected)
                if _defer_update(state):
                    pending.append(("state", name, tuple(keys), tuple(keys.values())))
                else:
                    dest = self.states.setdefault(name, {})
                    for key, value in keys.items():
                        dest[key] = dest.get(key, torch.zeros_like(value)) + value
        self.batches += 1
        self.input_windows += b

    def summary(self):
        records = {}
        for name, (summed, squared, maximum, positive) in self.records.items():
            effective = torch.where(squared > 0, summed.square() / squared, 0.0)
            share = torch.where(summed > 0, maximum / summed, 0.0)
            records[name] = {
                "sum": _numbers(summed), "sum_square": _numbers(squared),
                "max_record_contribution": _numbers(maximum),
                "positive_records": positive.tolist(),
                "contribution_effective_records": _numbers(effective),
                "maximum_record_fraction": _numbers(share),
                "overflowed_statistics": int((~torch.isfinite(torch.stack((summed, squared, maximum)))).sum()),
            }
        states = {}
        for name, edges in zip(STATE_NAMES, STATE_EDGES):
            if name not in self.states:
                states[name] = {"available": False, "edges": list(edges)}
                continue
            states[name] = {"available": True, "edges": list(edges),
                            "statistics": {k: [_numbers(row) for row in values]
                                           for k, values in self.states[name].items()}}
        return {
            "batches": self.batches, "input_windows": self.input_windows,
            "variables": self.variables, "context_available_batches": self.context_available_batches,
            "context_unavailable_batches": self.context_unavailable_batches,
            "query_slots": dict(self.query_slots),
            "definitions": {
                "record": "One input window/sample, aggregated over forecast time within a variable; not a unique patient or independent sample.",
                "concentration": "Per variable, effective=(sum record contribution)^2/sum(record contribution^2); maximum fraction=max/sum. Descriptive concentration, not inferential effective sample size.",
                "signed_moments": "Cross and gain are summed within a record first, then decomposed into positive and negative magnitudes. Positive-minus-negative recovers the signed total.",
                "query_scope": "Finite query>=window_start+history_width, with8 source-dtype eps relative tolerance, broadcast over variables; padded earlier queries excluded. Target scope independently uses finite observed targets; outside-scope count is reported.",
                "states": "Predictor context payload; unavailable when no payload is supplied.",
                "bins": "Five bins (-inf,e0],(e0,e1],(e1,e2],(e2,e3],(e3,inf); each statistic is [bin,variable]. Nonfinite state counts are separate and excluded from moments.",
                "reference": "First available complete fitted_core/ridge_core/core_forecast, then reference/shared_readout; explicit against/<name> keys identify the subtraction.",
            },
            "records": records, "states": states,
        }
