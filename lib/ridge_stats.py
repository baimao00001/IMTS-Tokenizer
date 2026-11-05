"""Weighted, per-variable sufficient statistics for a fixed-model residual fit."""
import math
import operator

import torch


class RidgeStatistics:
    def __init__(self, variables, features):
        if isinstance(variables, bool) or isinstance(features, bool):
            raise ValueError("variables and features must be positive integers")
        self.variables, self.features = operator.index(variables), operator.index(features)
        if self.variables <= 0 or self.features <= 0:
            raise ValueError("variables and features must be positive integers")
        shape = (self.variables, self.features)
        self.counts = torch.zeros(self.variables, dtype=torch.float64)
        self.gram = torch.zeros(*shape, self.features, dtype=torch.float64)
        self.cross = torch.zeros(shape, dtype=torch.float64)
        self.reference_sse = torch.zeros_like(self.counts)
        self.residual_sum = torch.zeros_like(self.counts)
        self.batches = 0

    @torch.no_grad()
    def update(self, basis, core, target, mask):
        expected = tuple(target.shape)
        if target.ndim != 3 or target.shape[-1] != self.variables:
            raise ValueError("target must have shape [batch, time, variables]")
        if tuple(core.shape) != expected or tuple(mask.shape) != expected:
            raise ValueError("core, target and mask shapes must match exactly")
        if tuple(basis.shape) != expected + (self.features,):
            raise ValueError("basis must have shape [batch, time, variables, features]")
        if any(value.device != target.device for value in (basis, core, mask)):
            raise ValueError("basis, core, target and mask must share a device")
        if any(value.is_complex() for value in (basis, core, target, mask)):
            raise ValueError("ridge inputs must be real-valued")
        weights = mask.detach().double()
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("mask weights must be finite and nonnegative")
        observed = weights > 0
        valid = torch.isfinite(core) & torch.isfinite(target) & torch.isfinite(basis).all(-1)
        if (observed & ~valid).any():
            raise ValueError("observed targets, core predictions and basis must be finite")
        x = torch.where(observed[..., None], basis.detach().double(), 0.0)
        residual = torch.where(observed, target.detach().double() - core.detach().double(), 0.0)
        gram = torch.einsum("btvf,btvg,btv->vfg", x, x, weights)
        cross = torch.einsum("btvf,btv,btv->vf", x, residual, weights)
        # One device-to-host transfer per batch for all additive statistics.
        packed = torch.cat((weights.sum((0, 1))[:, None], gram.flatten(1), cross,
                            (weights * residual.square()).sum((0, 1))[:, None],
                            (weights * residual).sum((0, 1))[:, None]), dim=1).cpu()
        if not torch.isfinite(packed).all():
            raise ValueError("nonfinite weighted sufficient statistics")
        end = 1 + self.features * self.features
        updates = (packed[:, 0], packed[:, 1:end].reshape_as(self.gram),
                   packed[:, end:end + self.features], packed[:, -2], packed[:, -1])
        names = ("counts", "gram", "cross", "reference_sse", "residual_sum")
        accumulated = [getattr(self, name) + value for name, value in zip(names, updates)]
        if not all(torch.isfinite(value).all() for value in accumulated):
            raise ValueError("nonfinite accumulated sufficient statistics")
        for name, value in zip(names, accumulated):
            setattr(self, name, value)
        self.batches += 1
        return self

    def select(self, indices):
        indices = list(indices)
        if not indices or any(isinstance(index, bool) for index in indices):
            raise ValueError("indices must be a nonempty sequence of unique feature indices")
        indices = [operator.index(index) for index in indices]
        if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= self.features:
            raise ValueError("feature indices must be unique and in range")
        result = RidgeStatistics(self.variables, len(indices))
        result.counts = self.counts.clone()
        result.gram = self.gram[:, indices][:, :, indices].clone()
        result.cross = self.cross[:, indices].clone()
        result.reference_sse = self.reference_sse.clone()
        result.residual_sum = self.residual_sum.clone()
        result.batches = self.batches
        return result

    @torch.no_grad()
    def score(self, coefficients):
        beta = torch.as_tensor(coefficients, device="cpu", dtype=torch.float64).detach()
        if tuple(beta.shape) != (self.variables, self.features) or not torch.isfinite(beta).all():
            raise ValueError("coefficients must be finite with shape [variables, features]")
        linear = (beta * self.cross).sum(-1)
        quadratic = torch.einsum("vf,vfg,vg->v", beta, self.gram, beta)
        sse = self.reference_sse - 2 * linear + quadratic
        tolerance = 64 * torch.finfo(torch.float64).eps * (
            self.reference_sse.abs() + 2 * linear.abs() + quadratic.abs()).clamp_min(1.0)
        if not torch.isfinite(sse).all() or (sse < -tolerance).any():
            raise ValueError("invalid residual SSE from sufficient statistics")
        sse = sse.clamp_min(0)
        active = self.counts > 0
        mse = sse / torch.where(active, self.counts, 1.0)
        return {"macro_mse": mse[active].mean().item() if active.any() else None,
                "micro_mse": (sse.sum() / self.counts.sum()).item() if active.any() else None,
                "per_variable_mse": [value.item() if used else None for value, used in zip(mse, active)],
                "per_variable_sse": sse.tolist(), "counts": self.counts.tolist(),
                "count": self.counts.sum().item(), "observed_variables": int(active.sum())}

    @torch.no_grad()
    def fit(self, ridge, support_shrink=True):
        ridge = float(ridge)
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and strictly positive")
        if not all(torch.isfinite(value).all() for value in
                   (self.counts, self.gram, self.cross, self.reference_sse, self.residual_sum)):
            raise ValueError("sufficient statistics must be finite")
        if (self.counts < 0).any() or (self.reference_sse < 0).any():
            raise ValueError("counts and reference SSE must be nonnegative")
        active = self.counts > 0
        count = torch.where(active, self.counts, 1.0)
        scales = (self.gram.diagonal(dim1=-2, dim2=-1).clamp_min(0) / count[:, None]).sqrt().clamp_min(1e-8)
        gram = self.gram / (count[:, None, None] * scales[:, :, None] * scales[:, None, :])
        gram = (gram + gram.transpose(-1, -2)) / 2
        cross = self.cross / (count[:, None] * scales)
        median = torch.quantile(self.counts[active], .5).item() if active.any() else 0.0
        strength = torch.full_like(count, ridge)
        if support_shrink:
            strength = strength * (median / count).clamp_min(1.0)
        matrix = gram + strength[:, None, None] * torch.eye(self.features, dtype=torch.float64)
        normalized = torch.linalg.solve(matrix, cross.unsqueeze(-1)).squeeze(-1)
        normalized = torch.where(active[:, None], normalized, 0.0)
        coefficients = normalized / scales
        if not torch.isfinite(coefficients).all():
            raise ValueError("ridge solve produced nonfinite coefficients")
        eig = torch.linalg.eigvalsh(gram).clamp_min(0)
        eigen_sum = eig.sum(-1)
        probabilities = eig / eigen_sum.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
        effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(-1))
        effective_rank = torch.where(eigen_sum > 0, effective_rank, 0.0)
        threshold = self.features * torch.finfo(torch.float64).eps * eig[:, -1]
        full_rank = eig[:, 0] > threshold
        normal_residual = torch.einsum("vfg,vg->vf", matrix, normalized) - cross
        unpenalized = 2 * (torch.einsum("vfg,vg->vf", self.gram, coefficients) - self.cross) / count[:, None]
        regularized = unpenalized + 2 * strength[:, None] * scales.square() * coefficients
        diagnostics = {
            "definition": "Per-variable mean squared residual plus lambda*sum((feature_RMS*raw_beta)^2); all coefficients, including any intercept, are regularized.",
            "ridge": ridge, "support_shrink": bool(support_shrink), "counts": self.counts.tolist(),
            "median_positive_count": median, "lambda_per_variable": strength.tolist(),
            "feature_rms": scales.tolist(), "gram_eigenvalues": eig.tolist(),
            "gram_condition": [(eig[i, -1] / eig[i, 0]).item() if full_rank[i] else None
                               for i in range(self.variables)],
            "gram_rank": (eig > threshold[:, None]).sum(-1).tolist(),
            "gram_effective_rank": effective_rank.tolist(),
            "regularized_condition": ((eig[:, -1] + strength) / (eig[:, 0] + strength)).tolist(),
            "normal_equation_residual": normal_residual.tolist(),
            "normal_equation_residual_norm": normal_residual.norm(dim=-1).tolist(),
            "raw_unpenalized_gradient": unpenalized.tolist(),
            "raw_regularized_gradient": regularized.tolist(),
            "raw_unpenalized_gradient_norm": unpenalized.norm(dim=-1).tolist(),
            "raw_regularized_gradient_norm": regularized.norm(dim=-1).tolist(),
            "coefficient_norm": coefficients.norm(dim=-1).tolist(),
            "normalized_coefficient_norm": normalized.norm(dim=-1).tolist(),
            "train_score": self.score(coefficients), "reference_score": self.score(torch.zeros_like(coefficients)),
            "batches": self.batches,
        }
        return coefficients, diagnostics
