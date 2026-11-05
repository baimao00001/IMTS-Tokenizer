"""Historical reliability states and compact conditional readout features."""

import math

import torch
from torch import nn

from .InteractionBasis import _orthogonal_columns


CONTEXT_NAMES = ("count_reliability", "recency_reliability", "history_span",
                 "last_mean_deviation", "relative_variation", "latent_strength")


def context_state_features(raw_features, query_times, x, times, mask, history_width, window_start=None):
    """Return B,T,V,6 states using only supplied history and current latent values."""
    if isinstance(history_width, bool) or not math.isfinite(float(history_width)) or history_width <= 0:
        raise ValueError("history_width must be finite and positive")
    if x.ndim != 4 or x.shape != times.shape or x.shape != mask.shape:
        raise ValueError("values, times and mask must match [batch, patch, time, variable]")
    batch, _, _, variables = x.shape
    if query_times.ndim != 2 or query_times.shape[0] != batch:
        raise ValueError("query_times must match [batch, horizon]")
    if (raw_features.ndim != 4 or raw_features.shape[:3] != (batch, query_times.shape[1], variables)
            or raw_features.shape[-1] <= 0):
        raise ValueError("raw_features must match [batch, horizon, variable, feature]")
    if any(not tensor.is_floating_point() for tensor in (raw_features, query_times, x, times)):
        raise ValueError("features, query times, values and history times must be real floating-point tensors")
    if any(tensor.device != x.device for tensor in (raw_features, query_times, times, mask)):
        raise ValueError("context inputs must share a device")
    if any(tensor.dtype != x.dtype for tensor in (raw_features, query_times, times)):
        raise ValueError("features, query times, values and history times must share a dtype")
    if mask.is_complex() or not torch.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("history mask must be real, finite and nonnegative")
    observed = (mask > 0).reshape(batch, -1, variables)
    values = torch.where(mask > 0, x, 0.).reshape(batch, -1, variables)
    historical_times = torch.where(mask > 0, times, 0.).reshape_as(values)
    if any(not torch.isfinite(tensor).all() for tensor in (raw_features, query_times, values, historical_times)):
        raise ValueError("observed history, query times and latent features must be finite")
    start = x.new_zeros(batch) if window_start is None else torch.as_tensor(window_start, dtype=x.dtype, device=x.device)
    if start.numel() == 1:
        start = start.reshape(1).expand(batch)
    elif start.numel() == batch:
        start = start.reshape(batch)
    else:
        raise ValueError("window_start must be scalar or have one value per sample")
    if not torch.isfinite(start).all():
        raise ValueError("window_start must be finite")
    count = observed.sum(1)
    supported = count > 0
    denominator = count.clamp_min(1)
    mean = values.sum(1) / denominator
    centered = torch.where(observed, values - mean[:, None, :], 0.)
    variance = centered.square().sum(1) / denominator
    # Evaluate sqrt away from zero on the inactive branch, preserving the exact
    # population standard deviation and finite gradients for constant histories.
    positive_variance = variance > 0
    standard_deviation = torch.where(positive_variance, torch.sqrt(torch.where(positive_variance, variance, 1.)), 0.)
    if values.shape[1]:
        last_time, last_index = historical_times.masked_fill(~observed, -torch.inf).max(1)
        first_time = historical_times.masked_fill(~observed, torch.inf).amin(1)
        last = values.gather(1, last_index[:, None, :]).squeeze(1)
        span = torch.where(supported, (last_time - first_time) / history_width, 0.)
        last_time = torch.where(supported, last_time, start[:, None])
    else:
        last, span = mean, torch.zeros_like(mean)
        last_time = start[:, None].expand_as(mean)
    age = ((query_times[:, :, None] - last_time[:, None, :]) / history_width).clamp_min(0.)
    shape = raw_features.shape[:3]
    expand = lambda tensor: tensor[:, None, :].expand(shape)
    count_reliability = count.to(x.dtype) / (count + 4)
    deviation = torch.tanh((last - mean) / (standard_deviation + 1e-4))
    variation = standard_deviation / (standard_deviation + mean.abs() + 1e-3)
    history = torch.stack((expand(count_reliability), 1 / (1 + age), expand(span),
                           expand(deviation), expand(variation)), dim=-1)
    history = torch.where(supported[:, None, :, None], history, 0.)
    latent_energy = raw_features.square().mean(-1)
    positive_energy = latent_energy > 0
    latent_rms = torch.where(positive_energy, torch.sqrt(torch.where(positive_energy, latent_energy, 1.)), 0.)
    states = torch.cat((history, (latent_rms / (1 + latent_rms)).unsqueeze(-1)), dim=-1)
    if not torch.isfinite(states).all():
        raise ValueError("context state computation produced nonfinite values")
    return states


class ContextReadoutBasis(nn.Module):
    projection_seed = 730127
    context_names = CONTEXT_NAMES

    def __init__(self, feature_dim, history_width, rank=4):
        super().__init__()
        if type(feature_dim) is not int or feature_dim < 1 or type(rank) is not int or not 1 <= rank <= feature_dim:
            raise ValueError("feature_dim and rank must be positive integers with rank <= feature_dim")
        if isinstance(history_width, bool) or not math.isfinite(float(history_width)) or history_width <= 0:
            raise ValueError("history_width must be finite and positive")
        self.raw_feature_dim, self.history_width, self.rank = feature_dim, float(history_width), rank
        self.context_dim = len(CONTEXT_NAMES)
        self.feature_dim = feature_dim + self.context_dim * (rank + 1)
        self.output_dim, self.core_feature_dim = self.feature_dim, self.raw_feature_dim
        self.feature_slices = {"raw": (0, feature_dim), "state": (feature_dim, feature_dim + self.context_dim),
                               "interaction": (feature_dim + self.context_dim, self.feature_dim)}
        self.basis_names = (tuple("feature_" + str(i) for i in range(feature_dim)) + CONTEXT_NAMES
                            + tuple("latent_" + str(i) + "_x_" + name for i in range(rank) for name in CONTEXT_NAMES))
        self.register_buffer("projection", _orthogonal_columns(feature_dim, rank, self.projection_seed).float())

    def compute(self, raw_features, query_times, x, times, mask, window_start=None):
        if raw_features.ndim != 4 or raw_features.shape[-1] != self.raw_feature_dim:
            raise ValueError("raw_features width differs from the context basis")
        if raw_features.device != self.projection.device or raw_features.dtype != self.projection.dtype:
            raise ValueError("features and projection buffer must share a device and dtype")
        states = context_state_features(raw_features, query_times, x, times, mask, self.history_width, window_start)
        projected = raw_features @ self.projection
        interactions = (projected.unsqueeze(-1) * states.unsqueeze(-2)).flatten(-2)
        expanded = torch.cat((raw_features, states, interactions), dim=-1)
        if not torch.isfinite(expanded).all():
            raise ValueError("context interactions produced nonfinite values")
        return expanded, states

    def forward(self, raw_features, query_times, x, times, mask, window_start=None):
        return self.compute(raw_features, query_times, x, times, mask, window_start)[0]
