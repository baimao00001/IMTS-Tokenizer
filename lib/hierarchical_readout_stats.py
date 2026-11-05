"""Observed-row statistics and convex full-feature single-model readout fitting."""

import math
import operator
from contextlib import contextmanager

import torch

from lib.ridge_stats import RidgeStatistics


class FullReadoutStatistics(RidgeStatistics):
    def __init__(self, variables, feature_dim):
        if isinstance(feature_dim, bool):
            raise ValueError("feature_dim must be a positive integer")
        self.feature_dim = operator.index(feature_dim)
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        super().__init__(variables, self.feature_dim + 1)
        self._fitting_cache = None

    @contextmanager
    def fitting_cache(self):
        """Reuse read-only fitting work inside one bounded candidate sweep."""
        previous = self._fitting_cache
        current = {}
        self._fitting_cache = current
        try:
            yield self
        finally:
            current.clear()
            self._fitting_cache = previous

    def _current_fitting_cache(self):
        cache = self._fitting_cache
        if cache is None:
            return None
        fields = tuple(getattr(self, name) for name in
                       ("counts", "gram", "cross", "reference_sse", "residual_sum"))
        # Tensor versions cover ordinary in-place changes and views; replacing
        # statistics or completing another update also invalidates the scope.
        try:
            signature = (self.variables, self.features, self.batches,
                         tuple((id(value), value._version, value.shape, value.dtype, value.device)
                               for value in fields))
        except RuntimeError:
            # Inference tensors have no version counter, so do not cache them.
            cache.clear()
            return None
        if cache.get("signature") != signature:
            cache.clear()
            cache["signature"] = signature
        return cache

    @torch.no_grad()
    def update_features(self, features, core, target, mask, chunk_rows=4096):
        """Gather observed rows before float64 conversion; commit a batch atomically."""
        if isinstance(chunk_rows, bool):
            raise ValueError("chunk_rows must be a positive integer")
        chunk_rows = operator.index(chunk_rows)
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be a positive integer")
        if target.ndim != 3 or target.shape[-1] != self.variables:
            raise ValueError("target must have shape [batch, time, variables]")
        if core.shape != target.shape or mask.shape != target.shape:
            raise ValueError("core, target and mask shapes must match exactly")
        if features.shape != target.shape + (self.feature_dim,):
            raise ValueError("features must match [batch, time, variables, feature_dim]")
        if any(value.device != target.device for value in (features, core, mask)):
            raise ValueError("features, core, target and mask must share a device")
        if any(value.is_complex() for value in (features, core, target, mask)):
            raise ValueError("readout inputs must be real-valued")
        if not features.is_floating_point() or not core.is_floating_point() or not target.is_floating_point():
            raise ValueError("features, core and target must be floating-point tensors")
        if not torch.isfinite(mask).all() or (mask < 0).any():
            raise ValueError("mask weights must be finite and nonnegative")
        fields = ("counts", "gram", "cross", "reference_sse", "residual_sum")
        updates = {name: torch.zeros_like(getattr(self, name)) for name in fields}
        for variable in range(self.variables):
            coordinates = (mask[..., variable] > 0).nonzero(as_tuple=False)
            for start in range(0, len(coordinates), chunk_rows):
                rows = coordinates[start:start + chunk_rows]
                batch, time = rows.unbind(dim=-1)
                values = features[batch, time, variable].detach().double()
                original = core[batch, time, variable].detach().double()
                truth = target[batch, time, variable].detach().double()
                weights = mask[batch, time, variable].detach().double()
                if not all(torch.isfinite(value).all() for value in (values, original, truth)):
                    raise ValueError("observed targets, predictions and features must be finite")
                basis = torch.cat((torch.ones_like(values[:, :1]), values), dim=-1)
                residual = truth - original
                weighted = basis * weights[:, None]
                gram = basis.T @ weighted
                cross = basis.T @ (weights * residual)
                packed = torch.cat((weights.sum().reshape(1), gram.reshape(-1), cross,
                                    (weights * residual.square()).sum().reshape(1),
                                    (weights * residual).sum().reshape(1))).cpu()
                if not torch.isfinite(packed).all():
                    raise ValueError("nonfinite weighted readout statistics")
                end = 1 + self.features * self.features
                updates["counts"][variable] += packed[0]
                updates["gram"][variable] += packed[1:end].reshape(self.features, self.features)
                updates["cross"][variable] += packed[end:end + self.features]
                updates["reference_sse"][variable] += packed[-2]
                updates["residual_sum"][variable] += packed[-1]
        accumulated = {name: getattr(self, name) + updates[name] for name in fields}
        if not all(torch.isfinite(value).all() for value in accumulated.values()):
            raise ValueError("nonfinite accumulated readout statistics")
        for name, value in accumulated.items():
            setattr(self, name, value)
        self.batches += 1
        return self

    @staticmethod
    def _positive(value, name):
        if isinstance(value, bool):
            raise ValueError(name + " must be finite and strictly positive")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and strictly positive")
        return value

    def _normalized(self, support_shrink, common_rms):
        if type(support_shrink) is not bool or type(common_rms) is not bool:
            raise ValueError("support_shrink and common_rms must be boolean")
        cache = self._current_fitting_cache()
        key = ("normalized", support_shrink, common_rms)
        if cache is not None and key in cache:
            return cache[key]
        if not all(torch.isfinite(getattr(self, name)).all() for name in
                   ("counts", "gram", "cross", "reference_sse", "residual_sum")):
            raise ValueError("sufficient statistics must be finite")
        if (self.counts < 0).any() or (self.reference_sse < 0).any():
            raise ValueError("counts and reference SSE must be nonnegative")
        active = self.counts > 0
        counts = torch.where(active, self.counts, 1.0)
        raw_gram = self.gram / counts[:, None, None]
        diagonal = raw_gram.diagonal(dim1=-2, dim2=-1).clamp_min(0)
        if common_rms:
            square = diagonal[active].mean(0) if active.any() else torch.zeros(self.features, dtype=torch.float64)
            scales = square.sqrt().clamp_min(1e-8).expand(self.variables, -1)
        else:
            scales = diagonal.sqrt().clamp_min(1e-8)
        gram = raw_gram / (scales[:, :, None] * scales[:, None, :])
        gram = (gram + gram.transpose(-1, -2)) / 2
        cross = self.cross / (counts[:, None] * scales)
        median = torch.quantile(self.counts[active], .5).item() if active.any() else 0.0
        support = (median / counts).clamp_min(1.0) if support_shrink else torch.ones_like(counts)
        support = torch.where(active, support, 0.0)
        result = active, counts, scales, gram, cross, support, median
        if cache is not None:
            cache[key] = result
        return result

    @staticmethod
    def _geometry(gram, active, strength, *, eigenvalues=None):
        eig = (torch.linalg.eigvalsh(gram[active]).clamp_min(0)
               if eigenvalues is None else eigenvalues)
        totals = eig.sum(-1)
        probability = eig / totals.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
        effective = (-(probability * probability.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(-1)).exp()
        effective = torch.where(totals > 0, effective, 0.0)
        threshold = gram.shape[-1] * torch.finfo(torch.float64).eps * eig[:, -1]
        condition = (eig[:, -1] + strength[active]) / (eig[:, 0] + strength[active])
        return {"active_gram_eigenvalues": eig.tolist(), "active_gram_effective_rank": effective.tolist(),
                "active_gram_rank": (eig > threshold[:, None]).sum(-1).tolist(),
                "active_regularized_condition": condition.tolist()}

    def _fit_geometry(self, gram, active, strength):
        cache = self._current_fitting_cache()
        eigenvalues = None
        if cache is not None:
            key = ("geometry", id(gram), gram._version, id(active), active._version)
            if key not in cache:
                cache[key] = torch.linalg.eigvalsh(gram[active]).clamp_min(0)
            eigenvalues = cache[key]
        return self._geometry(gram, active, strength, eigenvalues=eigenvalues)

    def _report(self, mode, raw, normalized, shared, local, active, scales, gram,
                cross, strength, median, support_shrink, common_rms, global_ridge=0.0,
                reduced=None, effective_df=None):
        raw_shared = shared / scales[0]
        prediction = normalized
        unpenalized = (gram @ prediction.unsqueeze(-1)).squeeze(-1) - cross
        local_normal = 2 * (unpenalized + strength[:, None] * local)
        local_normal = torch.where(active[:, None], local_normal, 0.0)
        shared_normal = (2 * (unpenalized[active].mean(0) + global_ridge * shared)
                         if mode == "hierarchical" and active.any() else torch.zeros_like(shared))
        active_mean = lambda value: value[active].mean().item() if active.any() else 0.0
        local_energy = strength * local.square().sum(-1)
        active_count = int(active.sum())
        if not torch.isfinite(raw.float()).all() or not torch.isfinite(raw_shared.float()).all():
            raise ValueError("readout coefficients cannot be represented as finite float32")
        deployed = raw.float().double() * scales
        deployed_shared = raw_shared.float().double() * scales[0]
        deployed_local = torch.where(active[:, None], deployed - deployed_shared, 0.0)
        deployed_unpenalized = (gram @ deployed.unsqueeze(-1)).squeeze(-1) - cross
        deployed_local_gradient = torch.where(active[:, None],
                                             2 * (deployed_unpenalized + strength[:, None] * deployed_local), 0.0)
        deployed_shared_gradient = (2 * (deployed_unpenalized[active].mean(0) + global_ridge * deployed_shared)
                                    if mode == "hierarchical" and active.any() else torch.zeros_like(shared))
        result = {
            "kind": "full_feature_" + mode,
            "definition": ("Mean active-variable residual MSE plus mean(lambda_local*support*||local||^2) "
                           "+ lambda_global*||shared||^2 in "
                           + ("common" if common_rms else "per-variable")
                           + " train RMS coordinates; intercept regularized."),
            "counts": self.counts.tolist(), "active_variables": active.tolist(),
            "active_variable_indices": active.nonzero(as_tuple=True)[0].tolist(),
            "zero_support_variables": (~active).nonzero(as_tuple=True)[0].tolist(),
            "median_positive_count": median, "support_shrink": support_shrink,
            "common_rms": common_rms, "feature_rms": scales.tolist(),
            "common_feature_rms": scales[0].tolist() if common_rms else None,
            "lambda_per_variable": strength.tolist(), "global_ridge": global_ridge,
            "shared_coefficients": raw_shared.tolist(),
            "local_coefficients": (raw - torch.where(active[:, None], raw_shared, 0.0)).tolist(),
            "normalized_shared_coefficients": shared.tolist(),
            "normalized_local_coefficients": local.tolist(),
            "local_normal_gradient": local_normal.tolist(),
            "local_normal_gradient_norm": local_normal.norm(dim=-1).tolist(),
            "local_normal_gradient_definition": "Normalized-coordinate per-variable gradient; divide by active variable count for the full objective gradient.",
            "local_objective_gradient": (local_normal / max(1, active_count)).tolist(),
            "shared_normal_gradient": shared_normal.tolist(),
            "shared_normal_gradient_norm": shared_normal.norm().item(),
            "per_variable_shared_gradient_contribution": torch.where(active[:, None], 2 * unpenalized / max(1, active_count), 0.0).tolist(),
            "float32_deployed_local_normal_gradient": deployed_local_gradient.tolist(),
            "float32_deployed_local_normal_gradient_norm": deployed_local_gradient.norm(dim=-1).tolist(),
            "float32_deployed_shared_normal_gradient": deployed_shared_gradient.tolist(),
            "float32_deployed_shared_normal_gradient_norm": deployed_shared_gradient.norm().item(),
            "local_regularizer_per_variable": local_energy.tolist(),
            "local_regularizer": active_mean(local_energy),
            "shared_regularizer": global_ridge * shared.square().sum().item(),
            "coefficient_norm": raw.norm(dim=-1).tolist(),
            "effective_degrees_of_freedom": effective_df,
            "effective_degrees_of_freedom_definition": "Per-variable diagonal trace of the training-target linear smoother, including shared-parameter influence.",
            "train_score": self.score(raw), "reference_score": self.score(torch.zeros_like(raw)),
            "shared_only_train_score": self.score(torch.where(active[:, None], raw_shared, 0.0)),
            "local_only_train_score": self.score(raw - torch.where(active[:, None], raw_shared, 0.0)),
            "batches": self.batches,
        }
        result.update(self._fit_geometry(gram, active, strength))
        if reduced is not None:
            eigenvalues = torch.linalg.eigvalsh(reduced)
            result["shared_system_eigenvalues"] = eigenvalues.tolist()
            result["shared_system_condition"] = (eigenvalues[-1] / eigenvalues[0]).item()
        return result

    @torch.no_grad()
    def fit_independent(self, ridge, support_shrink=True, common_rms=True, *, solver="lu"):
        ridge = self._positive(ridge, "ridge")
        if not isinstance(solver, str) or solver not in {"lu", "cholesky"}:
            raise ValueError("solver must be lu or cholesky")
        active, _, scales, gram, cross, support, median = self._normalized(support_shrink, common_rms)
        strength = ridge * support
        normalized = torch.zeros_like(self.cross)
        degrees = torch.zeros_like(self.counts)
        if active.any():
            matrix = gram[active] + strength[active, None, None] * torch.eye(self.features, dtype=torch.float64)
            if solver == "cholesky":
                factor = torch.linalg.cholesky(matrix)
                normalized[active] = torch.cholesky_solve(cross[active, :, None], factor).squeeze(-1)
                degrees[active] = torch.cholesky_solve(gram[active], factor).diagonal(dim1=-2, dim2=-1).sum(-1)
            else:
                normalized[active] = torch.linalg.solve(matrix, cross[active, :, None]).squeeze(-1)
                degrees[active] = torch.linalg.solve(matrix, gram[active]).diagonal(dim1=-2, dim2=-1).sum(-1)
        raw = normalized / scales
        if not torch.isfinite(raw).all():
            raise ValueError("full independent solve produced nonfinite coefficients")
        zero = torch.zeros(self.features, dtype=torch.float64)
        report = self._report("independent", raw, normalized, zero, normalized, active, scales, gram,
                              cross, strength, median, support_shrink, common_rms,
                              effective_df=degrees.tolist())
        report["ridge"] = ridge
        if solver != "lu":
            report["solver"] = solver
        return raw, report

    @torch.no_grad()
    def fit_hierarchical(self, local_ridge, global_ridge, support_shrink=True, *, solver="lu"):
        local_ridge = self._positive(local_ridge, "local_ridge")
        global_ridge = self._positive(global_ridge, "global_ridge")
        if not isinstance(solver, str) or solver not in {"lu", "cholesky"}:
            raise ValueError("solver must be lu or cholesky")
        active, _, scales, gram, cross, support, median = self._normalized(support_shrink, True)
        strength = local_ridge * support
        eye = torch.eye(self.features, dtype=torch.float64)
        shared = torch.zeros(self.features, dtype=torch.float64)
        local = torch.zeros_like(self.cross)
        degrees = torch.zeros_like(self.counts)
        reduced = global_ridge * eye
        if active.any():
            cache = self._current_fitting_cache()
            key = (id(gram), id(cross), id(active), local_ridge, solver)
            local_system = cache.get("local_system") if cache is not None else None
            if local_system is not None and local_system["key"] == key:
                used_gram, used_cross, alpha, matrix, factor, solved_gram, solved_cross = local_system["values"]
            else:
                if cache is not None:
                    cache.pop("local_system", None)
                local_system = None
                used_gram, used_cross, alpha = gram[active], cross[active], strength[active]
                matrix = used_gram + alpha[:, None, None] * eye
                factor = None
                if solver == "cholesky":
                    factor = torch.linalg.cholesky(matrix)
                    solved_gram = torch.cholesky_solve(used_gram, factor)
                    solved_cross = torch.cholesky_solve(used_cross[:, :, None], factor).squeeze(-1)
                else:
                    solved_gram = torch.linalg.solve(matrix, used_gram)
                    solved_cross = torch.linalg.solve(matrix, used_cross[:, :, None]).squeeze(-1)
                local_system = {"key": key, "values": (used_gram, used_cross, alpha, matrix, factor,
                                                       solved_gram, solved_cross)}
                if cache is not None:
                    # Only the current local penalty is retained. The global
                    # grid reuses it without keeping five large inverses alive.
                    cache["local_system"] = local_system
            # Equivalent to G-G*A^-1*G, without cancellation for a weak penalty.
            reduced = (alpha[:, None, None] * solved_gram).mean(0)
            reduced = (reduced + reduced.T) / 2 + global_ridge * eye
            rhs = (alpha[:, None] * solved_cross).mean(0)
            if solver == "cholesky":
                shared_factor = torch.linalg.cholesky(reduced)
                shared = torch.cholesky_solve(rhs[:, None], shared_factor).squeeze(-1)
                local[active] = torch.cholesky_solve((used_cross - used_gram @ shared)[:, :, None], factor).squeeze(-1)
                inverse = local_system.get("inverse")
                if inverse is None:
                    inverse = torch.cholesky_solve(eye.expand_as(matrix), factor)
                response = inverse + alpha[:, None, None].square() / int(active.sum()) * (
                    inverse @ torch.cholesky_solve(inverse, shared_factor.expand_as(inverse)))
            else:
                shared = torch.linalg.solve(reduced, rhs)
                local[active] = torch.linalg.solve(matrix, (used_cross - used_gram @ shared)[:, :, None]).squeeze(-1)
                inverse = local_system.get("inverse")
                if inverse is None:
                    inverse = torch.linalg.solve(matrix, eye.expand_as(matrix))
                response = inverse + alpha[:, None, None].square() / int(active.sum()) * (
                    inverse @ torch.linalg.solve(reduced, inverse))
            degrees[active] = (used_gram @ response).diagonal(dim1=-2, dim2=-1).sum(-1)
            if cache is not None:
                local_system["inverse"] = inverse
        normalized = torch.where(active[:, None], shared + local, 0.0)
        raw = normalized / scales
        if not torch.isfinite(raw).all() or not torch.isfinite(shared).all():
            raise ValueError("hierarchical solve produced nonfinite coefficients")
        report = self._report("hierarchical", raw, normalized, shared, local, active, scales, gram,
                              cross, strength, median, support_shrink, True, global_ridge,
                              reduced, degrees.tolist())
        report["local_ridge"] = local_ridge
        if solver != "lu":
            report["solver"] = solver
        return raw, report
