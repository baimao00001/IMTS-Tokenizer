"""Detached output-gradient exposure and scalar clipping proxies per epoch."""

import math

import torch

from .target_diagnostic_groups import SUPPORT_NAMES, history_support_masks


class OptimizationDiagnostics:
    """Observe one prediction/backward/update sequence at a time.

    Call complete_step after the optimizer update. The first backward through
    each prediction supplies output-gradient statistics; clipping is summarized
    by a scalar proxy. The pending hook retains detached eligibility/history
    masks, and epoch accumulators use O(history_groups * variables) storage.
    """

    fields = (
        "requested_target_count", "observed_target_count", "excluded_target_count",
        "invalid_mask_count", "nonfinite_target_count", "nonfinite_prediction_count",
        "finite_gradient_count", "nonfinite_gradient_count", "zero_gradient_count",
        "preclip_l1", "preclip_squared_l2", "postclip_proxy_count",
        "postclip_proxy_l1", "postclip_proxy_squared_l2",
        "clip_known_preclip_l1", "clip_known_preclip_squared_l2",
    )

    def __init__(self, variables):
        if type(variables) is not int or variables <= 0:
            raise ValueError("variables must be a positive integer")
        self.variables = variables
        self.observed_forwards = 0
        self.completed_steps = 0
        self.discarded_steps = 0
        self.fixed_prediction_steps = 0
        self.hook_not_called_steps = 0
        self._state = None
        self._pending = None
        self._handle = None

    def _initialize(self, device):
        shape = (len(SUPPORT_NAMES), self.variables)
        self._state = {name: torch.zeros(shape, device=device, dtype=torch.float64) for name in self.fields}
        self._scalars = {name: torch.zeros((), device=device, dtype=torch.float64) for name in (
            "valid_scale_steps", "invalid_scale_steps", "clipped_steps", "scale_sum",
            "valid_norm_steps", "invalid_norm_steps", "missing_norm_steps", "norm_sum", "norm_max")}
        self._scalars["scale_min"] = torch.full((), float("inf"), device=device, dtype=torch.float64)

    @staticmethod
    def _group(values_by_sample, groups):
        return (values_by_sample.unsqueeze(0) * groups).sum(1)

    def _remove_hook(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.no_grad()
    def observe_prediction(self, prediction, batch):
        if self._pending is not None:
            raise RuntimeError("Complete or discard the previous observed step before observing another")
        if (not torch.is_tensor(prediction) or prediction.ndim not in (3, 4)
                or prediction.shape[-1] != self.variables or not prediction.is_floating_point()):
            raise ValueError("prediction must be a real floating [B,T,V] or [S,B,T,V] tensor")
        if prediction.ndim == 4 and prediction.shape[0] <= 0:
            raise ValueError("prediction sample count must be positive")
        try:
            target, mask, history = (batch[name] for name in ("data_to_predict", "mask_predicted_data", "observed_mask"))
        except (KeyError, TypeError) as error:
            raise ValueError("batch must supply targets, target mask and observed_mask") from error
        shape = tuple(prediction.shape[-3:])
        if any(not torch.is_tensor(value) or tuple(value.shape) != shape or value.is_complex() for value in (target, mask)):
            raise ValueError("target and target mask must match prediction [B,T,V]")
        if (not torch.is_tensor(history) or history.ndim not in (3, 4) or history.is_complex()
                or history.shape[0] != shape[0] or history.shape[-1] != self.variables):
            raise ValueError("observed_mask must match [B,L,V] or [B,P,L,V]")
        if any(value.device != prediction.device for value in (target, mask, history)):
            raise ValueError("prediction, targets and masks must share a device")
        if self._state is None or self.observed_forwards == 0:
            self._initialize(prediction.device)
        elif self._state["preclip_l1"].device != prediction.device:
            raise ValueError("prediction device changed within one diagnostic epoch")
        groups = history_support_masks(history.detach(), shape[0], self.variables)
        pred = prediction.detach().unsqueeze(0) if prediction.ndim == 3 else prediction.detach()
        requested = (mask.detach() > 0).unsqueeze(0).expand_as(pred)
        mask_finite = torch.isfinite(mask.detach()).unsqueeze(0).expand_as(pred)
        target_finite = torch.isfinite(target.detach()).unsqueeze(0).expand_as(pred)
        prediction_finite = torch.isfinite(pred)
        observed = requested & mask_finite & target_finite & prediction_finite

        def grouped_count(eligible):
            return self._group(eligible.sum((0, 2), dtype=torch.float64), groups)

        pending = {name: torch.zeros_like(self._state[name]) for name in self.fields}
        pending.update(
            requested_target_count=grouped_count(requested),
            observed_target_count=grouped_count(observed),
            excluded_target_count=grouped_count(requested & ~observed),
            invalid_mask_count=grouped_count((~mask_finite) | (mask.detach() < 0).unsqueeze(0)),
            nonfinite_target_count=grouped_count(requested & ~target_finite),
            nonfinite_prediction_count=grouped_count(requested & ~prediction_finite),
        )
        pending["requires_grad"] = prediction.requires_grad
        pending["hook_called"] = False
        self._pending = pending
        self.observed_forwards += 1
        if not prediction.requires_grad:
            return
        three_dimensional = prediction.ndim == 3

        def capture(gradient):
            if self._pending is not pending:
                return None
            with torch.no_grad():
                values = gradient.detach().unsqueeze(0) if three_dimensional else gradient.detach()
                finite = observed & torch.isfinite(values)
                clean = torch.where(finite, values.double(), 0.0)
                pending["finite_gradient_count"] = grouped_count(finite)
                pending["nonfinite_gradient_count"] = grouped_count(observed & ~torch.isfinite(values))
                pending["zero_gradient_count"] = grouped_count(finite & (values == 0))
                pending["preclip_l1"] = self._group(clean.abs().sum((0, 2)), groups)
                pending["preclip_squared_l2"] = self._group(clean.square().sum((0, 2)), groups)
                pending["hook_called"] = True
            self._remove_hook()
            return None

        self._handle = prediction.register_hook(capture)

    def _scalar(self, value, name):
        if torch.is_tensor(value) and value.is_complex():
            raise ValueError(name + " must be a real scalar")
        try:
            value = torch.as_tensor(value, device=self._state["preclip_l1"].device, dtype=torch.float64).detach()
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(name + " must be a real scalar") from error
        if value.numel() != 1:
            raise ValueError(name + " must be a scalar")
        return value.reshape(())

    @torch.no_grad()
    def complete_step(self, preclip_norm, clip_scale):
        if self._pending is None:
            raise RuntimeError("No observed step is pending")
        scale = self._scalar(clip_scale, "clip_scale")
        norm = None if preclip_norm is None else self._scalar(preclip_norm, "preclip_norm")
        self._remove_hook()
        pending = self._pending
        scale_valid = torch.isfinite(scale) & (scale >= 0) & (scale <= 1)
        clean_scale = torch.where(scale_valid, scale, 0.)
        # Keep a matched preclip denominator when clipping metadata is missing.
        for suffix in ("l1", "squared_l2"):
            paired_pre = torch.where(scale_valid, pending["preclip_" + suffix], 0.)
            pending["clip_known_preclip_" + suffix] = paired_pre
            power = clean_scale if suffix == "l1" else clean_scale.square()
            pending["postclip_proxy_" + suffix] = torch.where(clean_scale == 0, torch.zeros_like(paired_pre), paired_pre * power)
        pending["postclip_proxy_count"] = torch.where(scale_valid, pending["finite_gradient_count"], 0.)
        for name in self.fields:
            self._state[name].add_(pending[name])
        scalars = self._scalars
        scalars["valid_scale_steps"].add_(scale_valid)
        scalars["invalid_scale_steps"].add_(~scale_valid)
        scalars["clipped_steps"].add_(scale_valid & (scale < 1))
        scalars["scale_sum"].add_(clean_scale)
        scalars["scale_min"] = torch.minimum(scalars["scale_min"], torch.where(scale_valid, scale, float("inf")))
        if norm is None:
            scalars["missing_norm_steps"].add_(1)
        else:
            norm_valid = torch.isfinite(norm) & (norm >= 0)
            clean_norm = torch.where(norm_valid, norm, 0.)
            scalars["valid_norm_steps"].add_(norm_valid)
            scalars["invalid_norm_steps"].add_(~norm_valid)
            scalars["norm_sum"].add_(clean_norm)
            scalars["norm_max"] = torch.maximum(scalars["norm_max"], clean_norm)
        self.fixed_prediction_steps += not pending["requires_grad"]
        self.hook_not_called_steps += pending["requires_grad"] and not pending["hook_called"]
        self.completed_steps += 1
        self._pending = None

    def close(self):
        """Remove an unconsumed hook and discard an incomplete caller step."""
        self._remove_hook()
        if self._pending is not None:
            self.discarded_steps += 1
            self._pending = None

    @torch.no_grad()
    def summary(self):
        if self._state is None:
            self._initialize(torch.device("cpu"))
        raw = {name: value.detach().cpu().tolist() for name, value in self._state.items()}
        scalar = {name: float(value.detach().cpu()) for name, value in self._scalars.items()}

        def number(value):
            return value if math.isfinite(value) else None

        def ratio(a, b):
            return number(a / b) if b > 0 else None

        groups = {}
        for index, name in enumerate(SUPPORT_NAMES):
            pre_total = sum(raw["preclip_l1"][index])
            post_total = sum(raw["postclip_proxy_l1"][index])
            rows = []
            for variable in range(self.variables):
                row = {"variable": variable, **{field: number(raw[field][index][variable]) for field in self.fields}}
                for prefix, count_name in (("preclip", "finite_gradient_count"), ("postclip_proxy", "postclip_proxy_count")):
                    square = raw[prefix + "_squared_l2"][index][variable]
                    count = raw[count_name][index][variable]
                    row[prefix + "_l2"] = number(math.sqrt(square)) if square >= 0 else None
                    row[prefix + "_mean_abs"] = ratio(raw[prefix + "_l1"][index][variable], count)
                    mean_square = ratio(square, count)
                    row[prefix + "_rms"] = math.sqrt(mean_square) if mean_square is not None else None
                row["preclip_l1_share"] = ratio(raw["preclip_l1"][index][variable], pre_total)
                row["postclip_proxy_l1_share"] = ratio(raw["postclip_proxy_l1"][index][variable], post_total)
                row["clip_l1_retained_ratio"] = ratio(raw["postclip_proxy_l1"][index][variable], raw["clip_known_preclip_l1"][index][variable])
                row["clip_squared_l2_retained_ratio"] = ratio(raw["postclip_proxy_squared_l2"][index][variable], raw["clip_known_preclip_squared_l2"][index][variable])
                rows.append(row)
            groups[name] = {"per_variable": rows, "preclip_l1_total": number(pre_total),
                            "postclip_proxy_l1_total": number(post_total)}
        valid_norms, valid_scales = scalar["valid_norm_steps"], scalar["valid_scale_steps"]
        return {
            "schema_version": 1, "variables": self.variables,
            "definition": "Observed finite-target prediction gradients from one backward per forward. Output-gradient clipping proxies use the caller's actual global parameter clip scalar; they are NOT per-variable parameter-gradient attribution or optimizer updates.",
            "count_definition": "Unweighted positive finite-mask prediction elements, including S copies; gradient values already contain loss weights and are not multiplied by the target mask again. History groups count positive packed observation-mask slots.",
            "moment_definition": "L1 and squared L2 are additive across observed completed steps; L2 is the square root of concatenated squared norms, not a norm of summed gradients. Postclip moments include only valid clip-scale steps; retention ratios use matched preclip moments.",
            "completed_steps_definition": "Caller must invoke complete_step after its successful optimizer update; the observer does not execute or verify that update.",
            "observed_forwards": self.observed_forwards, "completed_steps": self.completed_steps,
            "discarded_steps": self.discarded_steps, "pending_step": self._pending is not None,
            "fixed_prediction_steps": self.fixed_prediction_steps,
            "hook_not_called_steps": self.hook_not_called_steps,
            "parameter_clipping": {**{key: number(value) for key, value in scalar.items()},
                "norm_mean": ratio(scalar["norm_sum"], valid_norms),
                "norm_max": number(scalar["norm_max"]) if valid_norms else None,
                "scale_min": number(scalar["scale_min"]) if valid_scales else None,
                "scale_mean": ratio(scalar["scale_sum"], valid_scales),
                "clipped_fraction": ratio(scalar["clipped_steps"], valid_scales)},
            "nonfinite_accumulated_moments": sum(not math.isfinite(value) for values in raw.values() for group in values for value in group),
            "groups": groups,
        }
