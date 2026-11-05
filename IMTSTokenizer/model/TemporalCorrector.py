"""A zero-start historical trend/prior correction inside one forecasting model."""

import math

import torch
from torch import nn

from .IMTS_Tokenizer import _diagnostic


class TemporalCorrector(nn.Module):
    uses_window_start = True

    def __init__(self, core, args):
        super().__init__()
        self.core, self.args = core, args
        self.correction_only = bool(getattr(args, "correction_only", False))
        if self.correction_only:
            self.core.requires_grad_(False)
            self.core.eval()
        self.history_width = args.history / args.time_max
        self.seasonal_period = float(getattr(args, "seasonal_period", 0.0))
        self.seasonal_bandwidth = float(getattr(args, "seasonal_bandwidth", 1.0))
        if (not math.isfinite(self.seasonal_period) or not 0 <= self.seasonal_period <= args.history
                or not math.isfinite(self.seasonal_bandwidth) or self.seasonal_bandwidth <= 0):
            raise ValueError("Seasonal period must be in [0,history] and bandwidth positive")
        prior = torch.as_tensor(args.value_prior, dtype=torch.float32)
        if prior.shape != (args.num_var,) or not torch.isfinite(prior).all():
            raise ValueError("Temporal correction requires a finite train-only variable prior")
        self.register_buffer("training_prior", prior.clone())
        selected = getattr(args, "correction_variables", None)
        if selected is not None:
            if (not selected or len(set(selected)) != len(selected)
                    or any(type(index) is not int or not 0 <= index < args.num_var for index in selected)):
                raise ValueError("correction_variables must be unique valid variable indices")
            active = torch.zeros(args.num_var, dtype=torch.float32)
            active[selected] = 1.0
            self.register_buffer("active_variables", active)
        else:
            self.active_variables = None
        with torch.random.fork_rng(devices=[]):
            self.variable_embedding = nn.Parameter(torch.empty(args.num_var, 8))
            nn.init.normal_(self.variable_embedding, std=.02)
            self.decoder = nn.Sequential(nn.Linear(20 if self.seasonal_period else 18, 32), nn.SiLU(),
                                         nn.Linear(32, 4 if self.seasonal_period else 3))
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        if self.correction_only:
            self.core.eval()
        return self

    def _seasonal_readout(self, query, values, times, observed, start):
        center = query[:, :, None, None] - self.seasonal_period / self.args.time_max
        width = self.seasonal_bandwidth / self.args.time_max
        difference = times[:, None, :, :] - center
        slack = 8 * torch.finfo(times.dtype).eps * (1 + times[:, None, :, :].abs() + center.abs())
        allowed = observed[:, None, :, :] & (difference.abs() <= 2 * width + slack)
        allowed = allowed & (times[:, None, :, :] <= start[:, None, None, None] + self.history_width + slack)
        allowed = allowed & (times[:, None, :, :] <= query[:, :, None, None] + slack)
        present = allowed.any(dim=-2)
        logits = -.5 * (difference / width).square()
        logits = logits.masked_fill(~allowed, -torch.inf)
        logits = torch.where(present.unsqueeze(-2), logits, torch.zeros_like(logits))
        weights = logits.softmax(dim=-2) * allowed
        reference = (weights * values[:, None, :, :]).sum(dim=-2)
        return reference, present, weights, allowed

    def forecasting(self, time_predict, x, truth_time_steps, mask, window_start=None):
        self._diagnostic_predictions, self._diagnostic_gates = {}, {}
        self._diagnostic_target_tensors = {}
        self.core.diagnostics = getattr(self, "diagnostics", None)
        capture = getattr(self, "collect_diagnostic_predictions", False)
        self.core.collect_diagnostic_predictions = capture
        if getattr(self.core, "uses_window_start", False):
            core_prediction = self.core.forecasting(time_predict, x, truth_time_steps, mask, window_start)
        else:
            core_prediction = self.core.forecasting(time_predict, x, truth_time_steps, mask)
        core_target_tensors = dict(getattr(self.core, "_diagnostic_target_tensors", {}))
        self.core._diagnostic_target_tensors = {}
        batch, _, _, variables = x.shape
        observed = (mask > 0).reshape(batch, -1, variables)
        values = torch.where(mask > 0, x, 0.0).reshape(batch, -1, variables)
        times = truth_time_steps.reshape_as(values)
        start = x.new_zeros(batch) if window_start is None else torch.as_tensor(window_start, device=x.device, dtype=x.dtype).reshape(-1)
        if start.numel() == 1:
            start = start.expand(batch)
        count = observed.sum(1)
        supported = count > 0
        mean = values.sum(1) / count.clamp_min(1)
        normalized_time = torch.where(observed, (times - start[:, None, None]) / self.history_width, 0.0)
        mean_time = normalized_time.sum(1) / count.clamp_min(1)
        centered_value = torch.where(observed, values - mean[:, None, :], 0.0)
        centered_time = torch.where(observed, normalized_time - mean_time[:, None, :], 0.0)
        variance = centered_value.square().sum(1) / count.clamp_min(1)
        time_variance = centered_time.square().sum(1) / count.clamp_min(1)
        covariance = (centered_time * centered_value).sum(1) / count.clamp_min(1)
        slope = covariance / (time_variance + .01)
        scale = (variance + 1e-4).sqrt()
        if values.shape[1]:
            last_time, index = times.masked_fill(~observed, -torch.inf).max(1)
            first_time = times.masked_fill(~observed, torch.inf).amin(1)
            last = values.gather(1, index[:, None, :]).squeeze(1)
            span = torch.where(supported, (last_time - first_time) / self.history_width, 0.0)
            last_time = torch.where(supported, last_time, start[:, None])
        else:
            last, span = mean, mean
            last_time = start[:, None].expand_as(mean)
        age = ((time_predict[:, :, None] - last_time[:, None, :]) / self.history_width).clamp_min(0)
        age_fraction = age / (1 + age)
        prior = self.training_prior[None, None, :].expand_as(core_prediction)
        last = torch.where(supported, last, self.training_prior)
        mean = torch.where(supported, mean, self.training_prior)
        stable_slope = 2 * scale * (slope / (2 * scale)).tanh()
        trend = stable_slope[:, None, :] * age_fraction
        trend = torch.where(((count > 1) & (time_variance > 1e-8))[:, None, :], trend, 0.0)
        shape = core_prediction.shape
        expand = lambda value: value[:, None, :].expand(shape)
        features = torch.stack((
            expand(supported.to(x.dtype)), expand(count.to(x.dtype).log1p() / math.log(33)),
            expand(span), age_fraction, expand(scale.log1p()),
            expand(((last - mean) / scale).tanh()), expand((slope / scale).tanh()),
            ((core_prediction - last[:, None, :]) / scale[:, None, :]).tanh(),
            ((core_prediction - prior) / scale[:, None, :]).tanh(),
            expand(((mean - self.training_prior) / scale).tanh()),
        ), dim=-1)
        seasonal = None
        if self.seasonal_period:
            seasonal, seasonal_available, seasonal_weights, seasonal_allowed = self._seasonal_readout(
                time_predict, values, times, observed, start)
            seasonal = torch.where(seasonal_available, seasonal, core_prediction)
            seasonal_offset = torch.where(seasonal_available, seasonal - core_prediction, 0.0)
            features = torch.cat((features, (seasonal_offset / scale[:, None, :]).tanh().unsqueeze(-1),
                                  seasonal_available.to(x.dtype).unsqueeze(-1)), dim=-1)
        identity = self.variable_embedding[None, None].expand(batch, shape[1], -1, -1)
        hidden = self.decoder[1](self.decoder[0](torch.cat((features, identity), dim=-1)))
        logits = self.decoder[2](hidden)
        coefficients = .5 * logits.tanh()
        last_offset = torch.where(supported[:, None, :], last[:, None, :] - core_prediction, 0.0)
        offsets = torch.stack((last_offset, trend, prior - core_prediction), dim=-1)
        if seasonal is not None:
            offsets = torch.cat((offsets, seasonal_offset.unsqueeze(-1)), dim=-1)
        contributions = coefficients * offsets
        if self.active_variables is not None:
            contributions = contributions * self.active_variables[None, None, :, None]
        correction = contributions.sum(-1)
        prediction = core_prediction + correction
        contribution_names = ["last", "trend", "prior"] + (["seasonal"] if seasonal is not None else [])
        recorder = _diagnostic(self)
        if recorder is not None or capture:
            self._diagnostic_target_tensors = core_target_tensors
            self._diagnostic_predictions = dict(getattr(self.core, "_diagnostic_predictions", {}))
            self._diagnostic_predictions.update({
                "core_forecast": core_prediction.detach(), "training_prior": prior.detach(),
                "temporal_trend": (last[:, None, :] + trend).detach(),
            })
            for index, name in enumerate(contribution_names):
                self._diagnostic_predictions["correction_without_" + name] = (
                    prediction.detach() - contributions[..., index].detach())
            self._diagnostic_gates = dict(getattr(self.core, "_diagnostic_gates", {}))
            if self.active_variables is not None:
                self._diagnostic_gates["correction_active"] = self.active_variables[None, None, :].expand_as(prediction).detach()
            if seasonal is not None:
                self._diagnostic_predictions["seasonal_reference"] = seasonal.detach()
                self._diagnostic_gates["seasonal_available"] = seasonal_available.to(x.dtype).detach()
        self.core._diagnostic_predictions, self.core._diagnostic_gates = {}, {}
        if recorder is not None:
            self._diagnostic_target_tensors["model/correction"] = {
                "coefficients": coefficients, "contributions": contributions,
                "hidden": hidden, "branch_names": contribution_names,
                "saturation_threshold": .45}
            recorder.scalar("model/correction/definition", "signed coefficients in [-0.5,0.5], not probability gates; one shared core encoder")
            if self.active_variables is not None:
                recorder.tensor("model/correction/active_variables", self.active_variables)
            for name, value in (("features", features), ("hidden", hidden), ("logits", logits), ("coefficients", coefficients),
                                ("offsets", offsets), ("contributions", contributions),
                                ("correction", correction), ("prediction", prediction)):
                recorder.tensor("model/correction/" + name, value, grad=True)
            for index, name in enumerate(contribution_names):
                # Gradients attach to the complete contributions tensor used by sum.
                recorder.tensor("model/correction/" + name + "_contribution", contributions[..., index])
            recorder.features("model/correction/hidden_features", hidden)
            if seasonal is not None:
                recorder.tensor("model/correction/seasonal_reference", seasonal, grad=True)
                recorder.tensor("model/correction/seasonal_available", seasonal_available)
                recorder.tensor("model/correction/seasonal_weights", seasonal_weights, grad=True)
                recorder.distribution("model/correction/seasonal_weight_distribution", seasonal_weights,
                                      dim=-2, mask=seasonal_allowed)
            with torch.no_grad():
                absolute = contributions.detach().abs().sum(-1)
                total = absolute.sum()
                recorder.scalar("model/correction/cancellation_fraction",
                                1 - correction.detach().abs().sum() / total if total > 0 else 0.0)
                recorder.scalar("model/correction/zero_contribution_energy", total == 0)
        return prediction
