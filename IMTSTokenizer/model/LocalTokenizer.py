"""Local temporal kernels and an optional history-centered residual decoder.

The legacy model remains in IMTS_Tokenizer.py. The local variant changes only
the token construction; local_residual also represents observation timing and
coverage, and forecasts a correction to a learned mean/last-value reference.
"""

import math

import torch
from .variable_readout_support import initialize_variable_readout, apply_variable_readout
from torch import nn
from torch.nn import functional as F

from .IMTS_Tokenizer import (
    SelfAttentionLayer, _bind_diagnostics, _diagnostic, _norm_ratio, _record_weight_geometry,
    learn_angle,
)


def _record_parameter_profile(recorder, prefix, parameter, state_labels=None, axis_labels=None):
    """Summarize parameter rows and capture gradients on the full parameter."""
    if recorder is None or not recorder.active:
        return
    recorder.scalar(prefix + "/gradient_definition",
                    "Current backward parameter gradient before clipping, summed over all target paths; not own-target attribution or accumulated .grad")
    recorder.scalar(prefix + "/parameter_shape", list(parameter.shape))
    recorder.scalar(prefix + "/gradient_hook_registered", bool(parameter.requires_grad))
    if state_labels is not None:
        recorder.scalar(prefix + "/state_labels", list(state_labels))
        rows = parameter.detach().double()
        if rows.ndim == 2:
            rows = rows.unsqueeze(0)
        recorder.scalar(prefix + "/value_row_l2", rows.norm(dim=-1).cpu().tolist())
    else:
        recorder.scalar(prefix + "/values", parameter.detach().cpu().tolist())
        if axis_labels is not None:
            recorder.scalar(prefix + "/axis_labels", list(axis_labels))
    if not parameter.requires_grad:
        return

    def capture(gradient):
        if recorder.active:
            with torch.no_grad():
                if state_labels is not None:
                    rows = gradient.detach().double()
                    if rows.ndim == 2:
                        rows = rows.unsqueeze(0)
                    recorder.scalar(prefix + "/gradient_row_l2", rows.norm(dim=-1).cpu().tolist())
                    recorder.scalar(prefix + "/gradient_row_zero", (rows == 0).all(dim=-1).cpu().tolist())
                else:
                    recorder.scalar(prefix + "/gradients", gradient.detach().cpu().tolist())
                    recorder.scalar(prefix + "/gradient_zero", (gradient.detach() == 0).cpu().tolist())
        return None

    recorder._hooks.append(parameter.register_hook(capture))


class LocalKernel(nn.Module):
    def __init__(self, anchors, variables, enriched=False):
        super().__init__()
        grid = (torch.arange(anchors, dtype=torch.float32) + 0.5) / anchors
        grid = grid.view(1, 1, anchors, 1, 1).expand(1, 1, anchors, 1, variables)
        self.anchor_logits = nn.Parameter(torch.logit(grid).clone())
        self.minimum_bandwidth = 0.02
        initial_bandwidth = max(1.5 / anchors, 0.08)
        raw = math.log(math.expm1(initial_bandwidth - self.minimum_bandwidth))
        self.bandwidth_raw = nn.Parameter(torch.full_like(grid, raw))
        self.enriched = enriched
        if enriched:
            self.time_amplitude = nn.Parameter(torch.full_like(grid, 0.05))
            self.support_projection = nn.Linear(4, anchors, bias=False)
            nn.init.normal_(self.support_projection.weight, std=0.02)

    def forward(self, values, relative_time, mask):
        observed = mask > 0
        clean_values = torch.where(observed, values, 0.0)
        relative_time = torch.where(observed, relative_time, 0.0)
        anchors = self.anchor_logits.sigmoid()
        bandwidth = self.minimum_bandwidth + F.softplus(self.bandwidth_raw)
        standardized_offset = (relative_time.unsqueeze(2) - anchors) / bandwidth
        logits = -0.5 * standardized_offset.square()
        present = observed.any(dim=-2, keepdim=True).unsqueeze(2)
        masked_logits = logits.masked_fill(~observed.unsqueeze(2), -torch.inf)
        # All-empty rows use finite logits before softmax, then exactly zero mass.
        masked_logits = torch.where(present, masked_logits, torch.zeros_like(masked_logits))
        weights = masked_logits.softmax(dim=-2) * observed.unsqueeze(2)
        value_tokens = (weights * clean_values.unsqueeze(2)).sum(dim=-2)
        tokens = value_tokens
        info = {"anchors": anchors, "bandwidth": bandwidth,
                "relative_time": relative_time, "offset": standardized_offset,
                "logits": logits, "weights": weights, "value_tokens": value_tokens}
        if self.enriched:
            time_tokens = (weights * standardized_offset.tanh()).sum(dim=-2)
            time_tokens = time_tokens * self.time_amplitude.squeeze(-2)
            count = observed.sum(dim=-2)
            supported = count > 0
            if relative_time.shape[-2]:
                lower = relative_time.masked_fill(~observed, torch.inf).amin(dim=-2)
                upper = relative_time.masked_fill(~observed, -torch.inf).amax(dim=-2)
                lower = torch.where(supported, lower, 0.0)
                upper = torch.where(supported, upper, 0.0)
            else:
                lower = upper = torch.zeros_like(count, dtype=relative_time.dtype)
            support = torch.stack((supported.to(values.dtype),
                                   count.to(values.dtype).log1p() / math.log(33),
                                   torch.where(supported, 1 - upper, 0.0),
                                   upper - lower), dim=-1)
            support_tokens = self.support_projection(support).permute(0, 1, 3, 2)
            tokens = value_tokens + time_tokens + support_tokens
            info.update(time_tokens=time_tokens, support_features=support,
                        support_tokens=support_tokens)
        return tokens.permute(0, 3, 1, 2), info


class LocalTokenizer(nn.Module):
    uses_window_start = True

    def __init__(self, args):
        super().__init__()
        if args.variant not in {"local", "local_residual"}:
            raise ValueError("LocalTokenizer variant must be local or local_residual")
        if args.multi < 1 or args.hid_dim < 1 or args.num_var < 1:
            raise ValueError("multi, hid_dim, and num_var must be positive")
        if args.npatch != 2 ** (args.multi - 1):
            raise ValueError("npatch must equal 2 ** (multi - 1)")
        for name in ("time_max", "patch_size", "history"):
            if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0:
                raise ValueError(name + " must be finite and positive")
        if not math.isclose(args.patch_size * args.npatch, args.history, rel_tol=1e-6):
            raise ValueError("patch_size * npatch must equal history")
        dim = args.hid_dim * args.npatch * args.multi
        if dim % 2 or (args.num_layers > 0 and dim % 8):
            raise ValueError("feature dimension must be even and divisible by attention heads")
        self.args = args
        self.residual = args.variant == "local_residual"
        self.scales = nn.ModuleList([
            LocalKernel(args.hid_dim * 2 ** i, args.num_var, self.residual)
            for i in range(args.multi)
        ])
        self.query = learn_angle(num_var=args.num_var, dim=dim)
        self.layer = nn.ModuleList([
            SelfAttentionLayer(dim, dim, num_heads=8, dropout=args.dropout)
            for _ in range(args.num_layers)
        ])
        normalization = getattr(args, "qk_normalization", "none")
        if normalization not in {"none", "layer", "adaptive"}:
            raise ValueError("qk_normalization must be none, layer or adaptive")
        for layer in self.layer:
            layer.attn.qk_normalization = normalization
            if normalization == "adaptive":
                layer.attn.qk_mix = nn.Parameter(torch.zeros(()))
        self.head = nn.Linear(dim, 1)
        if self.residual:
            self.reference_mix_logits = nn.Parameter(torch.zeros(args.num_var))
            supplied_prior = getattr(args, "value_prior", None)
            if supplied_prior is None:
                fallback = 0.0 if getattr(args, "dataset", None) == "ushcn" else 0.5
                prior = torch.full((args.num_var,), fallback)
            else:
                prior = torch.as_tensor(supplied_prior, dtype=torch.float32)
                if prior.shape != (args.num_var,) or not torch.isfinite(prior).all():
                    raise ValueError("value_prior must contain one finite value per variable")
            self.missing_reference = nn.Parameter(prior.detach().clone())
            self.level_projection = nn.Linear(1, dim, bias=False)
            self.age_projection = nn.Linear(1, dim, bias=False)
            nn.init.normal_(self.level_projection.weight, std=0.02)
            nn.init.normal_(self.age_projection.weight, std=0.02)
            # A small nonzero head starts near the reference and keeps gradients
            # flowing into kernels, attention, and temporal features immediately.
            nn.init.normal_(self.head.weight, std=0.01 / math.sqrt(dim))
            nn.init.zeros_(self.head.bias)
        self.source_reliability_enabled = bool(getattr(args, "source_reliability", False))
        self.source_reliability_mode = getattr(args, "source_reliability_mode", "gated")
        if self.source_reliability_mode not in {"gated", "first_bias"}:
            raise ValueError("source_reliability_mode must be gated or first_bias")
        if self.source_reliability_mode == "first_bias" and not self.source_reliability_enabled:
            raise ValueError("first_bias requires source_reliability")
        if self.source_reliability_enabled:
            if not self.residual or not self.layer:
                raise ValueError("source_reliability requires local_residual with attention layers")
            if self.source_reliability_mode == "first_bias":
                self.source_reliability_weight = nn.Parameter(torch.zeros(4))
                self.source_reliability_logits = nn.Parameter(torch.zeros(args.num_var))
            else:
                self.source_reliability_weight = nn.Parameter(torch.tensor([3.0, 0.3, 0.0, -0.5]))
                self.source_reliability_logits = nn.Parameter(torch.full((args.num_var,), -1.0))
        self.target_query_mode = getattr(args, "target_query_mode", "none")
        if self.target_query_mode not in {"none", "missing", "all", "conditional"}:
            raise ValueError("target_query_mode must be none, missing, all or conditional")
        if self.target_query_mode != "none":
            if not self.residual or not self.layer:
                raise ValueError("target query identity requires local_residual with attention layers")
            shape = (2, args.num_var, dim) if self.target_query_mode == "conditional" else (args.num_var, dim)
            self.target_query_embedding = nn.Parameter(torch.zeros(shape))
        initialize_variable_readout(self, args, dim)
        self._readout_features = self._readout_query = self._readout_context = None

    @staticmethod
    def _start(window_start, values):
        if window_start is None:
            return values.new_zeros(values.shape[0])
        start = torch.as_tensor(window_start, dtype=values.dtype, device=values.device)
        if start.numel() == 1:
            return start.reshape(1).expand(values.shape[0])
        if start.numel() != values.shape[0]:
            raise ValueError("window_start must be scalar or contain one value per sample")
        return start.reshape(values.shape[0])

    def _reference(self, values, times, mask, start):
        batch, _, _, variables = values.shape
        observed = (mask > 0).reshape(batch, -1, variables)
        flat_values = torch.where(mask > 0, values, 0.0).reshape(batch, -1, variables)
        flat_times = times.reshape(batch, -1, variables)
        count = observed.sum(dim=1)
        supported = count > 0
        history_mean = flat_values.sum(dim=1) / count.clamp_min(1)
        if flat_values.shape[1]:
            last_time, index = flat_times.masked_fill(~observed, -torch.inf).max(dim=1)
            last_value = flat_values.gather(1, index.unsqueeze(1)).squeeze(1)
            last_time = torch.where(supported, last_time, start.unsqueeze(1))
        else:
            last_time = start.unsqueeze(1).expand_as(history_mean)
            last_value = history_mean
        mix = self.reference_mix_logits.sigmoid()
        reference = mix * last_value + (1 - mix) * history_mean
        reference = torch.where(supported, reference, self.missing_reference)
        return reference, last_time, supported, history_mean, last_value

    def _source_reliability(self, times, mask, start, last_time, recorder=None):
        observed = (mask > 0).reshape(times.shape[0], -1, times.shape[-1])
        flat_time = times.reshape(times.shape[0], -1, times.shape[-1])
        count = observed.sum(dim=1)
        supported = count > 0
        width = self.args.history / self.args.time_max
        if flat_time.shape[1]:
            first_time = flat_time.masked_fill(~observed, torch.inf).amin(dim=1)
            span = torch.where(supported, last_time - first_time, 0.0) / width
        else:
            span = torch.zeros_like(last_time)
        age = ((start[:, None] + width - last_time) / width).clamp_min(0)
        features = torch.stack((supported.to(times.dtype), count.to(times.dtype).log1p() / math.log(33),
                                span, age / (1 + age)), dim=-1)
        logits = self.source_reliability_logits + (features * self.source_reliability_weight).sum(dim=-1)
        if self.source_reliability_mode == "first_bias":
            if recorder is not None:
                prefix = "model/source_reliability"
                recorder.scalar(prefix + "/mode", "first_bias")
                recorder.scalar(prefix + "/definition", "Unbounded additive first-layer source attention logits, not probability gates")
                recorder.scalar(prefix + "/applied_layers", [0])
                recorder.tensor(prefix + "/features", features)
                recorder.tensor(prefix + "/weight", self.source_reliability_weight, grad=True)
                recorder.tensor(prefix + "/variable_logits", self.source_reliability_logits, grad=True)
                _record_parameter_profile(recorder, prefix + "/variable_parameter",
                                          self.source_reliability_logits)
                _record_parameter_profile(recorder, prefix + "/feature_parameter",
                                          self.source_reliability_weight,
                                          axis_labels=("support", "log_count", "span", "bounded_age"))
                recorder.tensor(prefix + "/logit_bias", logits, grad=True)
                recorder.tensor(prefix + "/bias_observed", logits, mask=supported)
                recorder.tensor(prefix + "/bias_no_history", logits, mask=~supported)
                recorder.tensor(prefix + "/count", count)
                recorder.tensor(prefix + "/age_at_history_end", age, mask=supported)
            return logits, None, supported
        # Log-sigmoid stays finite for very negative logits, without hard masking.
        bias = F.logsigmoid(logits)
        gate = logits.sigmoid()
        if recorder is not None:
            prefix = "model/source_reliability"
            recorder.tensor(prefix + "/features", features)
            recorder.tensor(prefix + "/logits", logits, grad=True)
            recorder.tensor(prefix + "/logit_bias", bias, grad=True)
            recorder.tensor(prefix + "/gate", gate, grad=True)
            recorder.tensor(prefix + "/count", count)
            recorder.tensor(prefix + "/age_at_history_end", age, mask=supported)
            recorder.tensor(prefix + "/gate_observed", gate, mask=supported)
            recorder.tensor(prefix + "/gate_no_history", gate, mask=~supported)
            recorder.scalar(prefix + "/saturated_fraction", ((gate < 0.05) | (gate > 0.95)).float().mean())
            with torch.no_grad():
                p = gate.detach().float().clamp(1e-7, 1-1e-7)
                entropy = -(p*p.log() + (1-p)*(1-p).log()) / math.log(2)
                recorder.tensor(prefix + "/normalized_entropy", entropy)
                recorder.scalar(prefix + "/gate_mean_by_variable", gate.detach().mean(dim=0).tolist())
        return bias, gate, supported

    def encode_history(self, values, times, mask, window_start=None, recorder=None):
        """Return the pre-attention tokens; timestamps and start use model units."""
        start = self._start(window_start, values)
        embeddings = []
        for i, scale in enumerate(self.scales):
            width = self.args.patch_size * 2 ** i / self.args.time_max
            patch_index = torch.arange(values.shape[1], dtype=times.dtype, device=times.device)
            origin = start[:, None, None, None] + patch_index[None, :, None, None] * width
            relative = (times - origin) / width
            tokens, info = scale(values, relative, mask)
            embeddings.append(tokens.reshape(tokens.shape[0], tokens.shape[1], -1))
            if recorder is not None:
                self._record_scale(recorder, i, values, mask, tokens, info)
            if i + 1 < len(self.scales):
                values = torch.cat((values[:, ::2], values[:, 1::2]), dim=2)
                times = torch.cat((times[:, ::2], times[:, 1::2]), dim=2)
                mask = torch.cat((mask[:, ::2], mask[:, 1::2]), dim=2)
        key = torch.cat(embeddings, dim=-1)
        if recorder is not None:
            recorder.tensor("model/multiscale/concatenated", key, grad=True)
            recorder.features("model/multiscale/features", key)
            with torch.no_grad():
                energy = torch.stack([item.detach().square().sum() for item in embeddings])
                recorder.tensor("model/multiscale/energy_fraction", energy / energy.sum().clamp_min(1e-12))
        return key

    def _record_scale(self, recorder, i, values, mask, tokens, info):
        prefix = "model/scale_{}".format(i)
        observed = mask > 0
        for name in ("anchors", "bandwidth", "logits", "weights", "value_tokens"):
            recorder.tensor(prefix + "/" + name, info[name], grad=True)
        recorder.tensor(prefix + "/relative_times", info["relative_time"], mask=observed)
        recorder.tensor(prefix + "/standardized_offset", info["offset"], mask=observed.unsqueeze(2))
        recorder.distribution(prefix + "/weight_distribution", info["weights"], dim=-2,
                              mask=observed.unsqueeze(2))
        recorder.tensor(prefix + "/weighted_tokens", tokens, grad=True)
        recorder.features(prefix + "/token_features", tokens)
        if self.residual:
            recorder.tensor(prefix + "/time_amplitude", self.scales[i].time_amplitude, grad=True)
            for name in ("time_tokens", "support_features", "support_tokens"):
                recorder.tensor(prefix + "/" + name, info[name], grad=True)
            recorder.scalar(prefix + "/time_to_value_norm", _norm_ratio(info["time_tokens"], info["value_tokens"]))
            recorder.scalar(prefix + "/support_to_value_norm", _norm_ratio(info["support_tokens"], info["value_tokens"]))
        with torch.no_grad():
            count = observed.sum(dim=-2)
            recorder.tensor(prefix + "/observed_count", count)
            recorder.scalar(prefix + "/empty_patch_variable_fraction", (count == 0).float().mean())
            recorder.scalar(prefix + "/single_observation_fraction", (count == 1).float().mean())
            recorder.tensor(prefix + "/weight_mass", info["weights"].sum(dim=-2))
            recorder.scalar(prefix + "/mask_leakage_mass", (info["weights"] * (~observed).unsqueeze(2)).sum())
        _record_weight_geometry(recorder, prefix, info["weights"], mask)

    def forecasting(self, time_predict, x, truth_time_steps, mask, window_start=None):
        self._readout_features = self._readout_query = self._readout_context = None
        self._diagnostic_predictions = {}
        self._diagnostic_gates = {}
        self._diagnostic_target_tensors = {}
        self._variable_readout_pending = None
        if x.ndim != 4 or truth_time_steps.shape != x.shape or mask.shape != x.shape:
            raise ValueError("values, times, and mask must match [batch, patch, time, variable]")
        if x.shape[1] != self.args.npatch or x.shape[-1] != self.args.num_var:
            raise ValueError("input patch or variable count differs from the model")
        if time_predict.ndim != 2 or time_predict.shape[0] != x.shape[0]:
            raise ValueError("prediction times must match [batch, horizon]")
        start = self._start(window_start, x)
        recorder = _diagnostic(self)
        _bind_diagnostics(self.query, recorder, "model/query")
        for i, layer in enumerate(self.layer):
            _bind_diagnostics(layer, recorder, "model/layer_{}".format(i))
        if recorder is not None:
            recorder.tensor("model/input/values", x, mask=mask > 0)
            recorder.tensor("model/input/times", truth_time_steps, mask=mask > 0)
            recorder.tensor("model/input/window_start", start)
        if self.residual:
            reference, last_time, supported, mean, last = self._reference(x, truth_time_steps, mask, start)
            values = torch.where(mask > 0, x - reference[:, None, None, :], 0.0)
        else:
            values = x
        key = self.encode_history(values, truth_time_steps, mask, start, recorder)
        if self.residual:
            # Cross-variable attention needs observed absolute levels as well as
            # centered dynamics to predict variables with no own history.
            level_feature = self.level_projection(reference.unsqueeze(-1))
            if recorder is not None:
                recorder.scalar("model/residual/level_to_dynamic_norm", _norm_ratio(level_feature, key))
            key = key + level_feature
        query = self.query(time_predict)
        if self.residual:
            age = (time_predict[:, :, None] - last_time[:, None, :]) / (self.args.history / self.args.time_max)
            age = age.clamp_min(0)
            age_feature = self.age_projection(age.unsqueeze(-1))
            query = query + age_feature
        source_options = {}
        if recorder is not None and self.residual:
            source_options["source_supported"] = supported
        if self.source_reliability_enabled:
            bias, gate, supported = self._source_reliability(truth_time_steps, mask, start, last_time, recorder)
            source_options = dict(source_logit_bias=bias, source_value_gate=gate, source_supported=supported)
        first_bias_effect = target_query_effect = source_value_effect = None
        for index, layer in enumerate(self.layer):
            options = source_options
            if self.source_reliability_enabled and self.source_reliability_mode == "first_bias":
                if index:
                    options = {"source_supported": supported} if recorder is not None else {}
                if recorder is not None:
                    recorder.scalar("model/layer_{}/attention/source_bias_applied".format(index), index == 0)
            if index == 0 and self.target_query_mode != "none":
                applied = ~supported if self.target_query_mode == "missing" else torch.ones_like(supported)
                if self.target_query_mode == "conditional":
                    # Table 0 handles absent own history; table 1 handles present history.
                    offset = torch.where(supported.unsqueeze(-1), self.target_query_embedding[1],
                                         self.target_query_embedding[0])
                else:
                    offset = self.target_query_embedding.unsqueeze(0).expand_as(key) * applied.unsqueeze(-1)
                options = dict(options, target_query_offset=offset)
                if recorder is not None:
                    recorder.scalar("model/target_query_identity/mode", self.target_query_mode)
                    recorder.scalar("model/target_query_identity/definition",
                                    "First-layer Q-only offset after input normalization; counterfactual attention fixes K/V and source bias")
                    recorder.tensor("model/target_query_identity/embedding", self.target_query_embedding, grad=True)
                    recorder.tensor("model/target_query_identity/applied", applied)
                    labels = (("no_own_history", "own_history_present")
                              if self.target_query_mode == "conditional" else
                              ("no_own_history_only",) if self.target_query_mode == "missing" else ("all_targets",))
                    _record_parameter_profile(recorder, "model/target_query_identity/parameter_rows",
                                              self.target_query_embedding, state_labels=labels)
                    if self.target_query_mode == "conditional":
                        recorder.scalar("model/target_query_identity/table_definition",
                                        ["no_own_history", "own_history_present"])
                        recorder.tensor("model/target_query_identity/history_state", supported)
            key, _ = layer(key, key, key, **options)
            if index == 0 and recorder is not None:
                first_bias_effect = getattr(layer.attn, "_diagnostic_source_bias_effect", None)
                target_query_effect = getattr(layer.attn, "_diagnostic_target_query_effect", None)
                source_value_effect = getattr(layer.attn, "_diagnostic_source_value_effect", None)
        readout_context = key
        key = key.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
        product = query * key
        correction = self.head(product).squeeze(-1)
        output = correction + reference[:, None, :] if self.residual else correction
        if self.residual and (recorder is not None or getattr(self, "collect_diagnostic_predictions", False)):
            self._diagnostic_predictions = {"reference": reference.detach()[:, None, :].expand_as(output)}
        output = apply_variable_readout(self, product, output, recorder)
        if first_bias_effect is not None:
            self._diagnostic_target_tensors["model/first_source_bias"] = {
                "observed_measures": {name: value[:, None, :].expand_as(output)
                                      for name, value in first_bias_effect.items()}}
        if target_query_effect is not None:
            self._diagnostic_target_tensors["model/target_query_identity"] = {
                "observed_measures": {name: value[:, None, :].expand_as(output)
                                      for name, value in target_query_effect.items()}}
        if source_value_effect is not None:
            self._diagnostic_target_tensors["model/first_source_values"] = {
                "observed_measures": {name: value[:, None, :].expand_as(output)
                                      for name, value in source_value_effect.items()}}
        if recorder is not None:
            for name, tensor in (("key", key), ("query", query), ("product", product)):
                recorder.tensor("model/modulation/" + name, tensor, grad=True)
            recorder.scalar("model/modulation/product_to_key_norm", _norm_ratio(product, key))
            recorder.tensor("model/output/prediction", output, grad=True)
            if self.residual:
                for name, tensor in (("reference", reference), ("history_mean", mean),
                                     ("last_value", last), ("supported", supported),
                                     ("reference_mix_logits", self.reference_mix_logits),
                                     ("missing_reference", self.missing_reference),
                                     ("centered_values", values), ("query_age", age),
                                     ("level_feature", level_feature),
                                     ("age_feature", age_feature), ("correction", correction)):
                    recorder.tensor("model/residual/" + name, tensor, grad=True)
                recorder.tensor("model/residual/reference_mix", self.reference_mix_logits.detach().sigmoid())
                recorder.scalar("model/residual/correction_to_reference_norm",
                                _norm_ratio(correction, reference[:, None, :].expand_as(correction)))
        # Store readout features after the forecast completes.
        if getattr(self, "capture_readout_features", False):
            self._readout_features = product
        if getattr(self, "capture_readout_components", False):
            self._readout_query, self._readout_context = query, readout_context
        return output
