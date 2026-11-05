"""History-conditioned source bias for the first attention layer."""

import math

import torch


def source_bias_features(times, mask, history_width, window_start=None):
    """Compute support, observation count, time span and recency features."""
    batch, variables = times.shape[0], times.shape[-1]
    if window_start is None:
        start = times.new_zeros(batch)
    else:
        start = torch.as_tensor(window_start, dtype=times.dtype, device=times.device)
        if start.numel() == 1:
            start = start.reshape(1).expand(batch)
        elif start.numel() == batch:
            start = start.reshape(batch)
        else:
            raise ValueError("window_start must be scalar or contain one value per sample")
    observed = (mask > 0).reshape(batch, -1, variables)
    flat_time = times.reshape(batch, -1, variables)
    count = observed.sum(dim=1)
    supported = count > 0
    if flat_time.shape[1]:
        last_time = flat_time.masked_fill(~observed, -torch.inf).max(dim=1).values
        last_time = torch.where(supported, last_time, start.unsqueeze(1))
        first_time = flat_time.masked_fill(~observed, torch.inf).amin(dim=1)
        span = torch.where(supported, last_time - first_time, 0.0) / history_width
    else:
        last_time = start.unsqueeze(1).expand(batch, variables)
        span = torch.zeros_like(last_time)
    age = ((start[:, None] + history_width - last_time) / history_width).clamp_min(0)
    features = torch.stack((supported.to(times.dtype), count.to(times.dtype).log1p() / math.log(33),
                            span, age / (1 + age)), dim=-1)
    return features, supported, count, age


def _parameter_profile(recorder, prefix, parameter, axis_labels=None):
    recorder.scalar(prefix + "/gradient_definition",
                    "Current backward parameter gradient before clipping, summed over all target paths; not own-target attribution or accumulated .grad")
    recorder.scalar(prefix + "/parameter_shape", list(parameter.shape))
    recorder.scalar(prefix + "/gradient_hook_registered", bool(parameter.requires_grad))
    recorder.scalar(prefix + "/values", parameter.detach().cpu().tolist())
    if axis_labels is not None:
        recorder.scalar(prefix + "/axis_labels", list(axis_labels))
    if parameter.requires_grad:
        def capture(gradient):
            if recorder.active:
                recorder.scalar(prefix + "/gradients", gradient.detach().cpu().tolist())
                recorder.scalar(prefix + "/gradient_zero", (gradient.detach() == 0).cpu().tolist())
            return None

        recorder._hooks.append(parameter.register_hook(capture))


def first_source_bias(module, times, mask, window_start, recorder):
    features, supported, count, age = source_bias_features(
        times, mask, module.args.history / module.args.time_max, window_start)
    bias = module.source_reliability_logits + (features * module.source_reliability_weight).sum(dim=-1)
    if recorder is not None:
        prefix = "model/source_reliability"
        recorder.scalar(prefix + "/mode", "first_bias")
        recorder.scalar(prefix + "/definition", "Unbounded additive first-layer source attention logits, not probability gates")
        recorder.scalar(prefix + "/applied_layers", [0])
        recorder.tensor(prefix + "/features", features)
        recorder.tensor(prefix + "/weight", module.source_reliability_weight, grad=True)
        recorder.tensor(prefix + "/variable_logits", module.source_reliability_logits, grad=True)
        _parameter_profile(recorder, prefix + "/variable_parameter", module.source_reliability_logits)
        _parameter_profile(recorder, prefix + "/feature_parameter", module.source_reliability_weight,
                           axis_labels=("support", "log_count", "span", "bounded_age"))
        recorder.tensor(prefix + "/logit_bias", bias, grad=True)
        recorder.tensor(prefix + "/bias_observed", bias, mask=supported)
        recorder.tensor(prefix + "/bias_no_history", bias, mask=~supported)
        recorder.tensor(prefix + "/count", count)
        recorder.tensor(prefix + "/age_at_history_end", age, mask=supported)
    return bias, supported
