"""A local encoder with reference/context gates and optional correction shrinkage."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .IMTS_Tokenizer import _bind_diagnostics, _diagnostic, _norm_ratio
from .LocalTokenizer import LocalTokenizer


class GatedTokenizer(LocalTokenizer):
    uses_window_start = True
    supports_readout_components = False

    def __init__(self, args):
        if args.variant not in {"local_gated", "local_adaptive"}:
            raise ValueError("GatedTokenizer variant must be local_gated or local_adaptive")
        core_args = copy.copy(args)
        core_args.variant = "local_residual"
        super().__init__(core_args)
        self.args = args
        self.history_width = args.history / args.time_max
        if not math.isfinite(self.history_width) or self.history_width <= 0:
            raise ValueError("normalized history width must be finite and positive")
        if self.history_width / args.npatch <= 0:
            raise ValueError("normalized patch width must be positive")
        dim = self.head.in_features
        self.variable_embedding = nn.Parameter(torch.empty(args.num_var, dim))
        nn.init.normal_(self.variable_embedding, std=0.02)
        self.local_norm = nn.LayerNorm(dim)
        self.reference_gate = nn.Linear(4, 1, bias=False)
        self.context_gate = nn.Linear(4, 1, bias=False)
        self.local_mix_logits = nn.Parameter(torch.full((args.num_var,), -math.log(3)))
        with torch.no_grad():
            self.reference_gate.weight.copy_(torch.tensor([[0.0, 0.05, 0.05, -0.1]]))
            self.context_gate.weight.copy_(torch.tensor([[0.2, 0.1, 0.1, -0.2]]))
        self.adaptive = args.variant == "local_adaptive"
        if self.adaptive:
            # The overwritten Linear initializer must not shift data/dropout RNG.
            with torch.random.fork_rng(devices=[]):
                self.correction_gate = nn.Linear(4, 1, bias=False)
            self.correction_mix_logits = nn.Parameter(torch.full((args.num_var,), -math.log(3)))
            with torch.no_grad():
                self.correction_gate.weight.copy_(torch.tensor([[3.0, 1.0, 0.0, -0.5]]))

    def _history_support(self, times, mask):
        batch, _, _, variables = times.shape
        observed = (mask > 0).reshape(batch, -1, variables)
        flat_time = times.reshape(batch, -1, variables)
        count = observed.sum(dim=1)
        if flat_time.shape[1]:
            first = flat_time.masked_fill(~observed, torch.inf).amin(dim=1)
            last = flat_time.masked_fill(~observed, -torch.inf).amax(dim=1)
            span = torch.where(count > 0, last - first, 0.0) / self.history_width
        else:
            span = times.new_zeros(batch, variables)
        return count, span

    def _gate_features(self, count, span, age):
        shape = age.shape
        return torch.stack((
            (count > 0).to(age.dtype)[:, None, :].expand(shape),
            (count.to(age.dtype).log1p() / math.log(33))[:, None, :].expand(shape),
            span[:, None, :].expand(shape),
            age / (1 + age),
        ), dim=-1)

    @staticmethod
    def _record_gate(recorder, name, logits, gate, count, age):
        prefix = "model/gates/" + name
        # Hooks attach to the exact gate tensors used by the forecast graph.
        recorder.tensor(prefix + "/logits", logits, grad=True)
        recorder.tensor(prefix + "/value", gate, grad=True)
        with torch.no_grad():
            probability = gate.detach().float().clamp(1e-7, 1 - 1e-7)
            entropy = -(probability * probability.log() +
                        (1 - probability) * (1 - probability).log()) / math.log(2)
            saturation = (gate.detach() < 0.05) | (gate.detach() > 0.95)
            recorder.tensor(prefix + "/normalized_entropy", entropy)
            recorder.scalar(prefix + "/saturation_fraction", saturation.float().mean())
            expanded_count = count[:, None, :].expand_as(gate)

            def summary(selected):
                size = int(selected.sum().item())
                return {
                    "query_variable_count": size,
                    "gate_mean": gate.detach()[selected].mean().item() if size else None,
                    "normalized_entropy_mean": entropy[selected].mean().item() if size else None,
                    "saturation_fraction": saturation[selected].float().mean().item() if size else None,
                }

            recorder.scalar(prefix + "/by_count", {
                "empty": summary(expanded_count == 0),
                "one": summary(expanded_count == 1),
                "two_to_three": summary((expanded_count >= 2) & (expanded_count <= 3)),
                "four_plus": summary(expanded_count >= 4),
            })
            supported = expanded_count > 0
            recorder.scalar(prefix + "/by_age", {
                "up_to_quarter_history": summary(supported & (age <= 0.25)),
                "quarter_to_one_history": summary(supported & (age > 0.25) & (age <= 1)),
                "over_one_history": summary(supported & (age > 1)),
            })

    def forecasting(self, time_predict, x, truth_time_steps, mask, window_start=None):
        self._readout_features = self._readout_query = self._readout_context = None
        self._diagnostic_predictions = {}
        self._diagnostic_gates = {}
        self._diagnostic_target_tensors = {}
        if getattr(self, "capture_readout_components", False):
            raise ValueError("GatedTokenizer has a time-dependent context; unexpanded readout component capture is unsupported")
        if x.ndim != 4 or truth_time_steps.shape != x.shape or mask.shape != x.shape:
            raise ValueError("values, times, and mask must match [batch, patch, time, variable]")
        if x.shape[1] != self.args.npatch or x.shape[-1] != self.args.num_var:
            raise ValueError("input patch or variable count differs from the model")
        if time_predict.ndim != 2 or time_predict.shape[0] != x.shape[0]:
            raise ValueError("prediction times must match [batch, horizon]")
        if not torch.isfinite(time_predict).all():
            raise ValueError("prediction times must be finite")
        start = self._start(window_start, x)
        if not torch.isfinite(start).all():
            raise ValueError("window_start must be finite")
        observed = mask > 0
        if not torch.isfinite(truth_time_steps[observed]).all():
            raise ValueError("observed history times must be finite")
        recorder = _diagnostic(self)
        _bind_diagnostics(self.query, recorder, "model/query")
        for i, layer in enumerate(self.layer):
            _bind_diagnostics(layer, recorder, "model/layer_{}".format(i))

        center, last_time, supported, mean, last = self._reference(x, truth_time_steps, mask, start)
        values = torch.where(observed, x - center[:, None, None, :], 0.0)
        dynamic_key = self.encode_history(values, truth_time_steps, mask, start, recorder)
        level_feature = self.level_projection(center.unsqueeze(-1))
        history_key = dynamic_key + level_feature + self.variable_embedding.unsqueeze(0)
        local = self.local_norm(history_key)
        context = history_key
        for layer in self.layer:
            context, _ = layer(context, context, context)
        if not self.layer:
            context = local

        count, span = self._history_support(truth_time_steps, mask)
        age = ((time_predict[:, :, None] - last_time[:, None, :]) / self.history_width).clamp_min(0)
        features = self._gate_features(count, span, age)
        reference_logits = self.reference_mix_logits + self.reference_gate(features).squeeze(-1)
        reference_gate = reference_logits.sigmoid()
        observed_reference = reference_gate * last[:, None, :] + (1 - reference_gate) * mean[:, None, :]
        reference = torch.where(supported[:, None, :], observed_reference, self.missing_reference)
        context_logits = self.local_mix_logits + self.context_gate(features).squeeze(-1)
        local_gate = context_logits.sigmoid()
        local_branch = local_gate.unsqueeze(-1) * local.unsqueeze(1)
        context_branch = (1 - local_gate).unsqueeze(-1) * context.unsqueeze(1)
        key = local_branch + context_branch
        age_feature = self.age_projection(age.unsqueeze(-1))
        query = self.query(time_predict) + age_feature
        product = query * key
        raw_correction = self.head(product).squeeze(-1)
        correction = raw_correction
        if self.adaptive:
            correction_logits = self.correction_mix_logits + self.correction_gate(features).squeeze(-1)
            correction_gate = correction_logits.sigmoid()
            correction = correction_gate * raw_correction
        output = reference + correction

        if recorder is not None or getattr(self, "collect_diagnostic_predictions", False):
            with torch.no_grad():
                context_correction = F.linear(query * context[:, None, :, :], self.head.weight, self.head.bias).squeeze(-1)
                local_correction = F.linear(query * local[:, None, :, :], self.head.weight, self.head.bias).squeeze(-1)
                if self.adaptive:
                    context_correction = correction_gate * context_correction
                    local_correction = correction_gate * local_correction
                self._diagnostic_predictions = {
                    "reference": reference.detach(),
                    "history_mean": torch.where(supported, mean, self.missing_reference)[:, None, :].expand_as(output).detach(),
                    "last_observation": torch.where(supported, last, self.missing_reference)[:, None, :].expand_as(output).detach(),
                    "context_path": reference + context_correction,
                    "local_path": reference + local_correction,
                }
                self._diagnostic_gates = {"reference": reference_gate.detach(), "local": local_gate.detach()}
                if self.adaptive:
                    self._diagnostic_predictions["raw_forecast"] = reference + raw_correction
                    self._diagnostic_gates["correction"] = correction_gate.detach()

        if recorder is not None:
            recorder.scalar("model/gates/query_scope", "all query slots; branch_diagnostics restricts gate summaries to observed targets")
            recorder.tensor("model/input/values", x, mask=observed)
            recorder.tensor("model/input/times", truth_time_steps, mask=observed)
            recorder.tensor("model/input/window_start", start)
            recorder.tensor("model/gates/features", features)
            recorder.tensor("model/gates/observed_count", count)
            recorder.tensor("model/gates/observed_span", span)
            self._record_gate(recorder, "reference", reference_logits, reference_gate, count, age)
            self._record_gate(recorder, "local_context", context_logits, local_gate, count, age)
            if self.adaptive:
                self._record_gate(recorder, "correction", correction_logits, correction_gate, count, age)
                recorder.tensor("model/residual/raw_correction", raw_correction, grad=True)
                recorder.tensor("model/residual/gated_correction", correction, grad=True)
                recorder.scalar("model/residual/raw_correction_norm", raw_correction.detach().float().norm())
                recorder.scalar("model/residual/gated_correction_norm", correction.detach().float().norm())
            for name, tensor in (("key", key), ("query", query), ("product", product)):
                recorder.tensor("model/modulation/" + name, tensor, grad=True)
            recorder.scalar("model/modulation/product_to_key_norm", _norm_ratio(product, key))
            for name, tensor in (("center", center), ("reference", reference), ("history_mean", mean),
                                 ("last_value", last), ("supported", supported),
                                 ("reference_mix_logits", self.reference_mix_logits),
                                 ("missing_reference", self.missing_reference),
                                 ("centered_values", values), ("query_age", age),
                                 ("level_feature", level_feature), ("age_feature", age_feature),
                                 ("correction", correction)):
                recorder.tensor("model/residual/" + name, tensor, grad=True)
            for name, tensor in (("dynamic", dynamic_key), ("level", level_feature),
                                 ("variable_embedding", self.variable_embedding),
                                 ("local", local), ("context", context),
                                 ("weighted_local", local_branch), ("weighted_context", context_branch),
                                 ("reference", reference), ("correction", correction)):
                recorder.tensor("model/branches/" + name, tensor, grad=True)
                recorder.scalar("model/branches/" + name + "_norm", tensor.detach().float().norm())
            recorder.scalar("model/residual/level_to_dynamic_norm", _norm_ratio(level_feature, dynamic_key))
            recorder.scalar("model/residual/correction_to_reference_norm", _norm_ratio(correction, reference))
            recorder.scalar("model/branches/local_to_context_norm", _norm_ratio(local, context))
            recorder.tensor("model/output/prediction", output, grad=True)
        if getattr(self, "capture_readout_features", False):
            self._readout_features = product
        return output
