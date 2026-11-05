"""Train-fitted linear residual readouts around a single fixed forecasting model."""

import math

import torch
from torch import nn

from .IMTS_Tokenizer import _diagnostic


class RidgeCorrector(nn.Module):
    uses_window_start = True
    HISTORY_NAMES = ("bias", "last", "mean", "prior", "trend",
                     "age_last", "age_mean", "age_prior")

    def __init__(self, core, args):
        super().__init__()
        self.core, self.args = core, args
        self.ridge_mode = getattr(args, "ridge_mode", "history")
        if self.ridge_mode not in {"history", "readout", "hybrid"}:
            raise ValueError("ridge_mode must be history, readout or hybrid")
        self.ridge_rank = getattr(args, "ridge_rank", 4)
        if type(self.ridge_rank) is not int or self.ridge_rank < 1:
            raise ValueError("ridge_rank must be a positive integer")
        if type(args.num_var) is not int or args.num_var < 1:
            raise ValueError("num_var must be a positive integer")
        if (not math.isfinite(float(args.history)) or float(args.history) <= 0
                or not math.isfinite(float(args.time_max)) or float(args.time_max) <= 0):
            raise ValueError("history and time_max must be finite and positive")
        self.history_width = float(args.history) / float(args.time_max)
        if not math.isfinite(self.history_width) or self.history_width <= 0:
            raise ValueError("Normalized history width must be finite and positive")
        prior = torch.as_tensor(args.value_prior, dtype=torch.float32)
        if prior.shape != (args.num_var,) or not torch.isfinite(prior).all():
            raise ValueError("Ridge correction requires one finite train-only prior per variable")
        self.register_buffer("training_prior", prior.detach().clone())
        self.core.requires_grad_(False)
        self.core.eval()

        if self.ridge_mode == "history":
            projection = torch.empty(0, 0)
            self.basis_names = self.HISTORY_NAMES
        else:
            leaf = self._readout_leaf()
            width = leaf.head.in_features
            if self.ridge_rank > width:
                raise ValueError("ridge_rank cannot exceed the core readout feature dimension")
            generator = torch.Generator(device="cpu").manual_seed(730127)
            sample = torch.randn(width, self.ridge_rank, generator=generator, dtype=torch.float64)
            projection, upper = torch.linalg.qr(sample, mode="reduced")
            # Fix QR's column-sign ambiguity without touching global RNG state.
            signs = torch.where(upper.diagonal() < 0, -1.0, 1.0)
            projection = (projection * signs).float()
            leading = self.HISTORY_NAMES if self.ridge_mode == "hybrid" else ("bias",)
            self.basis_names = leading + tuple("latent_" + str(i) for i in range(self.ridge_rank))
        self.register_buffer("projection", projection)
        self.coefficients = nn.Parameter(torch.zeros(args.num_var, len(self.basis_names)))
        self._diagnostic_predictions, self._diagnostic_gates = {}, {}
        self._core_predictions, self._core_gates = {}, {}
        self._diagnostic_target_tensors, self._core_target_tensors = {}, {}

    def _readout_leaf(self):
        node, seen, leaf = self.core, set(), None
        while isinstance(node, nn.Module):
            if id(node) in seen:
                raise ValueError("Cyclic core wrappers cannot expose readout features")
            seen.add(id(node))
            if isinstance(getattr(node, "head", None), nn.Linear):
                leaf = node
            node = getattr(node, "core", None)
        if leaf is None:
            raise ValueError("Readout correction requires a core with a linear head")
        return leaf

    def train(self, mode=True):
        super().train(mode)
        self.core.eval()
        return self

    def _history_basis(self, prediction, query, values, times, mask, start):
        batch, _, _, variables = values.shape
        observed = (mask > 0).reshape(batch, -1, variables)
        values = torch.where(mask > 0, values, 0.0).reshape(batch, -1, variables)
        times = times.reshape_as(values)
        count = observed.sum(dim=1)
        supported = count > 0
        mean = values.sum(dim=1) / count.clamp_min(1)
        local_time = torch.where(observed, (times - start[:, None, None]) / self.history_width, 0.0)
        mean_time = local_time.sum(dim=1) / count.clamp_min(1)
        centered_values = torch.where(observed, values - mean[:, None, :], 0.0)
        centered_times = torch.where(observed, local_time - mean_time[:, None, :], 0.0)
        variance = centered_values.square().sum(dim=1) / count.clamp_min(1)
        time_variance = centered_times.square().sum(dim=1) / count.clamp_min(1)
        covariance = (centered_times * centered_values).sum(dim=1) / count.clamp_min(1)
        slope = covariance / (time_variance + .01)
        scale = (variance + 1e-4).sqrt()
        if values.shape[1]:
            last_time, index = times.masked_fill(~observed, -torch.inf).max(dim=1)
            last = values.gather(1, index[:, None, :]).squeeze(1)
            last_time = torch.where(supported, last_time, start[:, None])
        else:
            last = mean
            last_time = start[:, None].expand_as(mean)
        age = ((query[:, :, None] - last_time[:, None, :]) / self.history_width).clamp_min(0)
        age_fraction = age / (1 + age)
        trend = (2 * scale * (slope / (2 * scale)).tanh())[:, None, :] * age_fraction
        trend = torch.where(((count > 1) & (time_variance > 1e-8))[:, None, :], trend, 0.0)
        last_offset = torch.where(supported[:, None, :], last[:, None, :] - prediction, 0.0)
        mean_offset = torch.where(supported[:, None, :], mean[:, None, :] - prediction, 0.0)
        prior_offset = self.training_prior[None, None, :] - prediction
        return torch.stack((torch.ones_like(prediction), last_offset, mean_offset,
                            prior_offset, trend, age_fraction * last_offset,
                            age_fraction * mean_offset, age_fraction * prior_offset), dim=-1)

    def design(self, time_predict, x, times, mask, window_start=None, *, return_readout=False,
               return_components=False):
        """Return the fixed forecast and fixed raw basis, with exactly one backbone call."""
        if return_components and not return_readout:
            raise ValueError("return_components requires return_readout=True")
        if x.ndim != 4 or times.shape != x.shape or mask.shape != x.shape:
            raise ValueError("values, times and mask must match [batch, patch, time, variable]")
        if x.shape[-1] != self.args.num_var:
            raise ValueError("Input variable count differs from the ridge model")
        if time_predict.ndim != 2 or time_predict.shape[0] != x.shape[0]:
            raise ValueError("prediction times must match [batch, horizon]")
        start = (x.new_zeros(x.shape[0]) if window_start is None else
                 torch.as_tensor(window_start, device=x.device, dtype=x.dtype).reshape(-1))
        if start.numel() == 1:
            start = start.expand(x.shape[0])
        if start.numel() != x.shape[0]:
            raise ValueError("window_start must be scalar or contain one value per sample")
        recorder = _diagnostic(self)
        capture = recorder is not None or getattr(self, "collect_diagnostic_predictions", False)
        self._diagnostic_predictions, self._diagnostic_gates = {}, {}
        self._core_predictions, self._core_gates = {}, {}
        self._diagnostic_target_tensors, self._core_target_tensors = {}, {}
        self.core.diagnostics = getattr(self, "diagnostics", None)
        self.core.collect_diagnostic_predictions = capture
        self.core.eval()
        leaf = self._readout_leaf() if self.ridge_mode != "history" else None
        if return_readout and leaf is None:
            raise ValueError("Raw readout capture requires readout or hybrid mode")
        sentinel = object()
        previous_flag = getattr(leaf, "capture_readout_features", sentinel) if leaf is not None else sentinel
        previous_components = (getattr(leaf, "capture_readout_components", sentinel)
                               if leaf is not None and return_components else sentinel)
        if leaf is not None:
            leaf.capture_readout_features = True
            leaf._readout_features = None
            if return_components:
                leaf.capture_readout_components = True
                leaf._readout_query = leaf._readout_context = None
        try:
            with torch.no_grad():
                if getattr(self.core, "uses_window_start", False):
                    prediction = self.core.forecasting(time_predict, x, times, mask, window_start)
                else:
                    prediction = self.core.forecasting(time_predict, x, times, mask)
                expected_shape = (x.shape[0], time_predict.shape[1], x.shape[-1])
                if prediction.shape != expected_shape:
                    raise ValueError("Core prediction must match [batch, horizon, variable]")
                if capture:
                    self._core_predictions = dict(getattr(self.core, "_diagnostic_predictions", {}))
                    self._core_gates = dict(getattr(self.core, "_diagnostic_gates", {}))
                if recorder is not None:
                    self._core_target_tensors = dict(getattr(self.core, "_diagnostic_target_tensors", {}))
                if self.ridge_mode in {"history", "hybrid"}:
                    basis = self._history_basis(prediction, time_predict, x, times, mask, start)
                else:
                    basis = torch.ones_like(prediction).unsqueeze(-1)
                if leaf is not None:
                    features = getattr(leaf, "_readout_features", None)
                    if (not torch.is_tensor(features) or
                            features.shape != prediction.shape + (self.projection.shape[0],)):
                        raise ValueError("Core did not expose matching query*key readout features")
                    latent = features @ self.projection
                    basis = torch.cat((basis, latent), dim=-1)
                    if return_components:
                        query = getattr(leaf, "_readout_query", None)
                        context = getattr(leaf, "_readout_context", None)
                        if (not torch.is_tensor(query) or query.shape != features.shape
                                or not torch.is_tensor(context)
                                or context.shape != (x.shape[0], x.shape[-1], features.shape[-1])
                                or query.dtype != features.dtype or context.dtype != features.dtype
                                or query.device != features.device or context.device != features.device):
                            raise ValueError("Core did not expose matching final query and unexpanded context")
                    if recorder is not None:
                        recorder.features("model/ridge/readout_features", features)
                        recorder.features("model/ridge/projected_features", latent)
        finally:
            self.core._diagnostic_predictions, self.core._diagnostic_gates = {}, {}
            self.core._diagnostic_target_tensors = {}
            if leaf is not None:
                leaf._readout_features = None
                if return_components:
                    leaf._readout_query = leaf._readout_context = None
                    if previous_components is sentinel:
                        del leaf.capture_readout_components
                    else:
                        leaf.capture_readout_components = previous_components
                if previous_flag is sentinel:
                    del leaf.capture_readout_features
                else:
                    leaf.capture_readout_features = previous_flag
        if recorder is not None:
            recorder.scalar("model/ridge/mode", self.ridge_mode)
            recorder.scalar("model/ridge/basis_names", list(self.basis_names))
            recorder.tensor("model/ridge/basis", basis)
            recorder.features("model/ridge/basis_geometry", basis)
            recorder.tensor("model/ridge/projection", self.projection)
        if return_components:
            return prediction.detach(), basis.detach(), features.detach(), query.detach(), context.detach()
        if return_readout:
            return prediction.detach(), basis.detach(), features.detach()
        return prediction.detach(), basis.detach()

    def forecasting(self, time_predict, x, truth_time_steps, mask, window_start=None):
        core_prediction, basis = self.design(time_predict, x, truth_time_steps, mask, window_start)
        coefficient_view = self.coefficients[None, None, :, :].expand_as(basis)
        contributions = basis * coefficient_view
        correction = contributions.sum(dim=-1)
        prediction = core_prediction + correction
        recorder = _diagnostic(self)
        if recorder is not None or getattr(self, "collect_diagnostic_predictions", False):
            self._diagnostic_predictions = self._core_predictions
            self._diagnostic_gates = self._core_gates
            self._diagnostic_predictions["ridge_core"] = core_prediction
            for index, name in enumerate(self.basis_names):
                contribution = contributions[..., index].detach()
                self._diagnostic_predictions["ridge_only_" + name] = core_prediction + contribution
                self._diagnostic_predictions["ridge_without_" + name] = prediction.detach() - contribution
        self._core_predictions, self._core_gates = {}, {}
        if recorder is not None:
            self._diagnostic_target_tensors = self._core_target_tensors
            self._diagnostic_target_tensors["model/ridge"] = {
                "coefficients": coefficient_view, "contributions": contributions,
                "hidden": basis, "branch_names": list(self.basis_names),
            }
            recorder.scalar("model/ridge/coefficient_definition", "signed raw-basis coefficients, not probability gates")
            recorder.tensor("model/ridge/coefficients", self.coefficients, grad=True)
            recorder.tensor("model/ridge/contributions", contributions, grad=True)
            recorder.tensor("model/ridge/correction", correction, grad=True)
            recorder.tensor("model/ridge/prediction", prediction, grad=True)
            with torch.no_grad():
                total = contributions.detach().abs().sum()
                recorder.scalar("model/ridge/cancellation_fraction",
                                1 - correction.detach().abs().sum() / total if total > 0 else 0.0)
                recorder.scalar("model/ridge/zero_contribution_energy", total == 0)
        self._core_target_tensors = {}
        return prediction
