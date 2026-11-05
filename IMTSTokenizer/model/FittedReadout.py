"""A fixed full-feature affine residual decoder around one forecasting model."""

import torch
from torch import nn

from .IMTS_Tokenizer import _diagnostic
from .InteractionBasis import ReadoutInteractionBasis
from .ContextReadoutBasis import CONTEXT_NAMES, ContextReadoutBasis, context_state_features
from .SpectralContextBasis import SpectralContextBasis


class FittedReadout(nn.Module):
    uses_window_start = True

    def __init__(self, core, args):
        super().__init__()
        if getattr(args, "fitted_readout", None) not in {"full", "hierarchical"}:
            raise ValueError("fitted_readout must be full or hierarchical")
        if type(args.num_var) is not int or args.num_var < 1:
            raise ValueError("num_var must be a positive integer")
        self.core, self.args = core, args
        self.raw_feature_dim = self._readout_leaf().head.in_features
        self.readout_basis = getattr(args, "readout_basis", "full")
        if self.readout_basis not in {"full", "diagonal", "interaction", "context", "spectral", "spectral_nested"}:
            raise ValueError("readout_basis must be full, diagonal, interaction, context, spectral or spectral_nested")
        if self.readout_basis == "full":
            self.feature_dim = self.raw_feature_dim
            self.basis_names = ("bias",) + tuple("feature_" + str(index) for index in range(self.feature_dim))
        elif self.readout_basis == "context":
            self.context_basis = ContextReadoutBasis(
                self.raw_feature_dim, args.history / args.time_max, getattr(args, "ridge_rank", 4))
            self.feature_dim = self.context_basis.feature_dim
            self.basis_names = ("bias",) + self.context_basis.basis_names
        elif self.readout_basis in {"spectral", "spectral_nested"}:
            self.spectral_basis = SpectralContextBasis(self.raw_feature_dim, getattr(args, "ridge_rank", 4),
                                                       nested=self.readout_basis == "spectral_nested")
            self.feature_dim = self.spectral_basis.feature_dim
            self.basis_names = ("bias",) + self.spectral_basis.basis_names
        else:
            self.interaction_basis = ReadoutInteractionBasis(
                self.raw_feature_dim, getattr(args, "ridge_rank", 4), getattr(args, "readout_extra_rank", 4))
            self.feature_dim = self.interaction_basis.core_rank + self.interaction_basis.extra_rank
            self.basis_names = (("bias",) + tuple("latent_" + str(index) for index in range(self.interaction_basis.core_rank))
                                + tuple(self.readout_basis + "_" + str(index) for index in range(self.interaction_basis.extra_rank)))
        self.register_buffer("coefficients", torch.zeros(args.num_var, self.feature_dim + 1))
        self.register_buffer("shared_coefficients", torch.zeros(self.feature_dim + 1))
        self.register_buffer("active_variables", torch.ones(args.num_var, dtype=torch.bool))
        self.readout_shrinkage = getattr(args, "readout_shrinkage", "none")
        if self.readout_shrinkage not in {"none", "variable", "block"}:
            raise ValueError("readout_shrinkage must be none, variable or block")
        if self.readout_shrinkage == "variable":
            self.register_buffer("readout_gain", torch.ones(args.num_var))
        elif self.readout_shrinkage == "block":
            if self.readout_basis not in {"spectral", "spectral_nested"}:
                raise ValueError("Block readout shrinkage requires a spectral basis")
            self.register_buffer("readout_block_gain", torch.ones(args.num_var, 2))
        self.core.requires_grad_(False).eval()
        self._diagnostic_predictions, self._diagnostic_gates, self._diagnostic_target_tensors = {}, {}, {}
        self._core_predictions, self._core_gates, self._core_target_tensors = {}, {}, {}
        self._diagnostic_context = None

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
            raise ValueError("Full readout requires a core with a linear head")
        return leaf

    def train(self, mode=True):
        super().train(mode)
        self.core.eval()
        return self

    @torch.no_grad()
    def design(self, time_predict, x, times, mask, window_start=None):
        """Return the complete outer prediction and selected readout features, once."""
        self._diagnostic_predictions, self._diagnostic_gates, self._diagnostic_target_tensors = {}, {}, {}
        self._core_predictions, self._core_gates, self._core_target_tensors = {}, {}, {}
        self._diagnostic_context = None
        if x.ndim != 4 or times.shape != x.shape or mask.shape != x.shape:
            raise ValueError("values, times and mask must match [batch, patch, time, variable]")
        if x.shape[-1] != self.args.num_var:
            raise ValueError("Input variable count differs from the full readout model")
        if time_predict.ndim != 2 or time_predict.shape[0] != x.shape[0]:
            raise ValueError("prediction times must match [batch, horizon]")
        recorder = _diagnostic(self)
        capture = recorder is not None or getattr(self, "collect_diagnostic_predictions", False)
        self.core.diagnostics = getattr(self, "diagnostics", None)
        self.core.collect_diagnostic_predictions = capture
        self.core.eval()
        leaf = self._readout_leaf()
        sentinel = object()
        previous_flag = getattr(leaf, "capture_readout_features", sentinel)
        need_components = self.readout_basis in {"diagonal", "interaction", "spectral", "spectral_nested"}
        previous_components = getattr(leaf, "capture_readout_components", sentinel) if need_components else sentinel
        leaf.capture_readout_features, leaf._readout_features = True, None
        if need_components:
            leaf.capture_readout_components = True
            leaf._readout_query = leaf._readout_context = None
        completed = False
        try:
            if getattr(self.core, "uses_window_start", False):
                prediction = self.core.forecasting(time_predict, x, times, mask, window_start)
            else:
                prediction = self.core.forecasting(time_predict, x, times, mask)
            expected = (x.shape[0], time_predict.shape[1], x.shape[-1])
            if prediction.shape != expected:
                raise ValueError("Core prediction must match [batch, horizon, variable]")
            features = getattr(leaf, "_readout_features", None)
            if not torch.is_tensor(features) or features.shape != expected + (self.raw_feature_dim,):
                raise ValueError("Core did not expose matching raw query*key readout features")
            if (capture and self.readout_basis != "context"
                    and hasattr(self.args, "history") and hasattr(self.args, "time_max")):
                self._diagnostic_context = context_state_features(
                    features, time_predict, x, times, mask, self.args.history / self.args.time_max, window_start)
            if need_components:
                query = getattr(leaf, "_readout_query", None)
                context = getattr(leaf, "_readout_context", None)
                if not torch.is_tensor(query) or not torch.is_tensor(context):
                    raise ValueError("Core did not expose final query and unexpanded context")
                if self.readout_basis in {"spectral", "spectral_nested"}:
                    features = self.spectral_basis(features, query, context, time_predict)
                else:
                    features = self.interaction_basis(features, query, context, self.readout_basis)
            elif self.readout_basis == "context":
                features, self._diagnostic_context = self.context_basis.compute(
                    features, time_predict, x, times, mask, window_start)
            if capture:
                self._core_predictions = dict(getattr(self.core, "_diagnostic_predictions", {}))
                self._core_gates = dict(getattr(self.core, "_diagnostic_gates", {}))
            if capture:
                self._core_target_tensors = dict(getattr(self.core, "_diagnostic_target_tensors", {}))
            completed = True
            return prediction.detach(), features.detach()
        finally:
            if not completed:
                self._diagnostic_context = None
                self._core_predictions, self._core_gates, self._core_target_tensors = {}, {}, {}
            node, seen = self.core, set()
            while isinstance(node, nn.Module) and id(node) not in seen:
                seen.add(id(node))
                node._diagnostic_predictions, node._diagnostic_gates = {}, {}
                node._diagnostic_target_tensors = {}
                node = getattr(node, "core", None)
            leaf._readout_features = None
            if need_components:
                leaf._readout_query = leaf._readout_context = None
                if previous_components is sentinel:
                    del leaf.capture_readout_components
                else:
                    leaf.capture_readout_components = previous_components
            if previous_flag is sentinel:
                del leaf.capture_readout_features
            else:
                leaf.capture_readout_features = previous_flag

    def _projected_contributions(self, features, coefficients):
        rank = self.interaction_basis.core_rank
        original = torch.cat((torch.ones_like(features[..., :1]), features[..., :rank]), dim=-1)
        original = (original * coefficients[None, None, :, :rank + 1]).sum(-1)
        extra = (features[..., rank:] * coefficients[None, None, :, rank + 1:]).sum(-1)
        return original, extra

    @torch.no_grad()
    def forecasting(self, time_predict, x, truth_time_steps, mask, window_start=None):
        completed = False
        try:
            prediction = self._forecasting(time_predict, x, truth_time_steps, mask, window_start)
            completed = True
            return prediction
        finally:
            self._core_predictions, self._core_gates, self._core_target_tensors = {}, {}, {}
            self._diagnostic_context = None
            if not completed:
                self._diagnostic_predictions, self._diagnostic_gates, self._diagnostic_target_tensors = {}, {}, {}

    def _forecasting(self, time_predict, x, truth_time_steps, mask, window_start=None):
        core, features = self.design(time_predict, x, truth_time_steps, mask, window_start)
        beta = torch.where(self.active_variables[:, None], self.coefficients, 0.0)
        block_gain = None
        if self.readout_shrinkage == "block":
            block_gain = self.readout_block_gain
            if not torch.isfinite(block_gain).all() or ((block_gain < 0) | (block_gain > 1)).any():
                raise ValueError("readout_block_gain must contain finite values in [0,1]")
            unshrunk_beta = beta
            boundary = self.raw_feature_dim + 1
            beta = torch.cat((beta[:, :boundary] * block_gain[:, :1],
                              beta[:, boundary:] * block_gain[:, 1:]), dim=-1)
        if self.readout_basis in {"full", "context", "spectral", "spectral_nested"}:
            correction = torch.einsum("btvd,vd->btv", features, beta[:, 1:]) + beta[None, None, :, 0]
        else:
            original, extra = self._projected_contributions(features, beta)
            correction = original + extra
        correction = torch.where(self.active_variables[None, None, :], correction, 0.0)
        unshrunk_correction = correction
        gain = None
        if self.readout_shrinkage == "variable":
            if not torch.isfinite(self.readout_gain).all() or ((self.readout_gain < 0) | (self.readout_gain > 1)).any():
                raise ValueError("readout_gain must contain finite values in [0,1]")
            gain = self.readout_gain[None, None, :]
            correction = gain * correction
        prediction = core + correction
        recorder = _diagnostic(self)
        capture = recorder is not None or getattr(self, "collect_diagnostic_predictions", False)
        if capture:
            if block_gain is not None:
                shared_beta = self.shared_coefficients[None, :].expand(self.args.num_var, -1)
                shared_beta = torch.cat((shared_beta[:, :boundary] * block_gain[:, :1],
                                         shared_beta[:, boundary:] * block_gain[:, 1:]), dim=-1)
                shared = torch.einsum("btvd,vd->btv", features, shared_beta[:, 1:]) + shared_beta[None, None, :, 0]
                unshrunk_correction = (torch.einsum("btvd,vd->btv", features, unshrunk_beta[:, 1:])
                                      + unshrunk_beta[None, None, :, 0])
            elif self.readout_basis in {"full", "context", "spectral", "spectral_nested"}:
                shared = torch.einsum("btvd,d->btv", features, self.shared_coefficients[1:]) + self.shared_coefficients[0]
            else:
                shared_original, shared_extra = self._projected_contributions(
                    features, self.shared_coefficients[None, :].expand(self.args.num_var, -1))
                shared = shared_original + shared_extra
            shared = torch.where(self.active_variables[None, None, :], shared, 0.0)
            if gain is not None:
                shared = gain * shared
            local = correction - shared
            self._diagnostic_predictions = self._core_predictions
            self._diagnostic_gates = self._core_gates
            self._diagnostic_target_tensors = self._core_target_tensors
            if self._diagnostic_context is not None:
                self._diagnostic_target_tensors["model/fitted_readout"] = {
                    "context": self._diagnostic_context, "context_names": list(CONTEXT_NAMES)}
            self._diagnostic_predictions.update({"fitted_core": core, "fitted_shared_only": core + shared,
                                                "fitted_local_only": core + local})
            if gain is not None:
                self._diagnostic_predictions["fitted_without_shrinkage"] = core + unshrunk_correction
                self._diagnostic_gates["readout_shrinkage"] = gain.expand_as(prediction)
            if block_gain is not None:
                self._diagnostic_predictions["fitted_without_shrinkage"] = core + unshrunk_correction
                for index, name in enumerate(("raw", "spectral")):
                    self._diagnostic_gates["readout_" + name + "_shrinkage"] = block_gain[None, None, :, index].expand_as(prediction)
            if self.readout_basis in {"diagonal", "interaction"}:
                original = torch.where(self.active_variables[None, None, :], original, 0.0)
                extra = torch.where(self.active_variables[None, None, :], extra, 0.0)
                if gain is not None:
                    original, extra = gain * original, gain * extra
                self._diagnostic_predictions.update({"fitted_without_extra": core + original,
                                                    "fitted_extra_only": core + extra})
            if self.readout_basis == "context":
                raw = torch.einsum("btvd,vd->btv", features[..., :self.raw_feature_dim], beta[:, 1:self.raw_feature_dim + 1])
                raw = raw + beta[None, None, :, 0]
                if gain is not None:
                    raw = gain * raw
                extra = correction - raw
                self._diagnostic_predictions.update({"fitted_without_context": core + raw,
                                                    "fitted_context_only": core + extra})
            if self.readout_basis in {"spectral", "spectral_nested"}:
                raw = torch.einsum("btvd,vd->btv", features[..., :self.raw_feature_dim], beta[:, 1:self.raw_feature_dim + 1])
                raw = raw + beta[None, None, :, 0]
                if gain is not None:
                    raw = gain * raw
                extra = correction - raw
                spectral_terms = (features[..., self.raw_feature_dim:]
                                  * beta[None, None, :, self.raw_feature_dim + 1:])
                spectral_terms = spectral_terms.reshape(*prediction.shape, self.spectral_basis.rank, 8)
                bands = (spectral_terms[..., :4] + spectral_terms[..., 4:]).sum(dim=-2)
                if gain is not None:
                    bands = gain.unsqueeze(-1) * bands
                self._diagnostic_predictions.update({"fitted_raw_only": core + raw,
                                                    "fitted_without_extra": core + raw,
                                                    "fitted_spectral_only": core + extra})
                for index, cycle in enumerate((1, 2, 4, 8)):
                    self._diagnostic_predictions["fitted_without_spectral_band_" + str(cycle)] = prediction - bands[..., index]
            if recorder is not None:
                contributions = torch.stack((shared, local), dim=-1)
                self._diagnostic_target_tensors.setdefault("model/fitted_readout", {}).update(
                    contributions=contributions, hidden=features, branch_names=["shared", "local"])
                recorder.scalar("model/fitted_readout/definition", "Fixed readout residual; shared/local signed contributions, not gates; one core call")
                recorder.scalar("model/fitted_readout/readout_basis", self.readout_basis)
                recorder.tensor("model/fitted_readout/coefficients", self.coefficients)
                recorder.tensor("model/fitted_readout/shared_coefficients", self.shared_coefficients)
                recorder.tensor("model/fitted_readout/active_variables", self.active_variables)
                recorder.tensor("model/fitted_readout/contributions", contributions)
                recorder.tensor("model/fitted_readout/correction", correction)
                recorder.tensor("model/fitted_readout/prediction", prediction)
                if gain is not None:
                    recorder.tensor("model/fitted_readout/readout_gain", self.readout_gain)
                    recorder.tensor("model/fitted_readout/unshrunk_correction", unshrunk_correction)
                    recorder.scalar("model/fitted_readout/shrinkage_definition",
                                    "Validation-selected gain scales train-fitted coefficients; normal-gradient diagnostics use the original coefficients.")
                if block_gain is not None:
                    recorder.tensor("model/fitted_readout/readout_block_gain", block_gain)
                    recorder.tensor("model/fitted_readout/effective_coefficients", beta)
                    recorder.tensor("model/fitted_readout/unshrunk_correction", unshrunk_correction)
                    recorder.scalar("model/fitted_readout/shrinkage_definition",
                                    "Two validation-selected gains per variable scale raw+bias and spectral coefficient blocks before prediction. Stored coefficients stay train-fitted; shared coefficients receive the same per-variable block gains before shared/local diagnostics.")
                geometry_name = "raw_feature_geometry" if self.readout_basis == "full" else "basis_feature_geometry"
                recorder.features("model/fitted_readout/" + geometry_name, features)
                if self.readout_basis in {"diagonal", "interaction"}:
                    self._diagnostic_target_tensors["model/fitted_readout/extra_basis"] = {
                        "contributions": torch.stack((original, extra), dim=-1), "hidden": features,
                        "branch_names": ["original", self.readout_basis]}
                    recorder.scalar("model/fitted_readout/without_extra_definition",
                                    "Internal contribution ablation with jointly fitted original coefficients; not the independently fitted old control")
                    recorder.scalar("model/fitted_readout/original_extra_cross_energy", (2 * original * extra).mean())
                if self.readout_basis == "context":
                    self._diagnostic_target_tensors["model/fitted_readout"].update(
                        context=self._diagnostic_context, context_names=list(self.context_basis.context_names))
                    self._diagnostic_target_tensors["model/fitted_readout/context_basis"] = {
                        "contributions": torch.stack((raw, extra), dim=-1), "hidden": self._diagnostic_context,
                        "branch_names": ["raw", "context"]}
                    recorder.tensor("model/fitted_readout/context", self._diagnostic_context)
                    recorder.scalar("model/fitted_readout/context_names", list(self.context_basis.context_names))
                    recorder.scalar("model/fitted_readout/context_definition",
                                    "Fixed history and latent states, plus state times fixed latent projection; coefficients are signed corrections, not probability gates")
                    recorder.scalar("model/fitted_readout/raw_context_cross_energy", (2 * raw * extra).mean())
                if self.readout_basis in {"spectral", "spectral_nested"}:
                    self._diagnostic_target_tensors["model/fitted_readout/spectral_basis"] = {
                        "contributions": torch.cat((raw.unsqueeze(-1), bands), dim=-1), "hidden": features,
                        "branch_names": ["raw", "spectral_band_1", "spectral_band_2", "spectral_band_4", "spectral_band_8"]}
                    recorder.scalar("model/fitted_readout/spectral_definition",
                                    "Fixed [sin,cos](2*pi*[1,2,4,8]*query_time) times fixed projected context; original normalized time units, not variable-specific calendar periods")
                    recorder.scalar("model/fitted_readout/spectral_basis_names", list(self.spectral_basis.basis_names))
                    recorder.scalar("model/fitted_readout/without_extra_definition",
                                    "Internal contribution ablation with jointly fitted raw coefficients; not the independently fitted raw control")
                    recorder.scalar("model/fitted_readout/spectral_band_ablation_definition",
                                    "Subtract one jointly fitted frequency-band contribution after active masking and gain; all other coefficients stay fixed, with no independent retraining")
                    recorder.tensor("model/fitted_readout/spectral_projection", self.spectral_basis.projection)
                    recorder.tensor("model/fitted_readout/spectral_cycles", self.spectral_basis.cycles)
                    if self.readout_basis == "spectral_nested":
                        recorder.scalar("model/fitted_readout/spectral_projection_definition",
                                        "Original rank4 projection and matmul preserved; private-seed orthogonal completion; larger ranks extend the same feature prefix")
                        recorder.scalar("model/fitted_readout/spectral_context_rank", self.spectral_basis.rank)
                    recorder.tensor("model/fitted_readout/spectral_band_contributions", bands)
                    recorder.scalar("model/fitted_readout/raw_spectral_cross_energy", (2 * raw * extra).mean())
                total = contributions.abs().sum()
                recorder.scalar("model/fitted_readout/cancellation_fraction", 1 - correction.abs().sum() / total if total > 0 else 0.0)
                recorder.scalar("model/fitted_readout/shared_local_cross_energy", (2 * shared * local).mean())
                if self._diagnostic_context is not None and self.readout_basis != "context":
                    recorder.tensor("model/fitted_readout/context", self._diagnostic_context)
                    recorder.scalar("model/fitted_readout/context_names", list(CONTEXT_NAMES))
        return prediction
