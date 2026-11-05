"""Train-only support and feature-block penalties for a fixed affine readout."""

import math

import torch

from lib.hierarchical_readout_stats import FullReadoutStatistics


class AdaptiveReadoutStatistics(FullReadoutStatistics):
    @torch.no_grad()
    def fit_adaptive(self, ridge, support_power=1.5, feature_penalty=None):
        ridge = self._positive(ridge, "ridge")
        if isinstance(support_power, bool):
            raise ValueError("support_power must be finite and nonnegative")
        support_power = float(support_power)
        if not math.isfinite(support_power) or support_power < 0:
            raise ValueError("support_power must be finite and nonnegative")
        if feature_penalty is None:
            penalty = torch.ones(self.features, dtype=torch.float64)
        else:
            penalty = torch.as_tensor(feature_penalty, device="cpu")
            if penalty.is_complex() or penalty.dtype == torch.bool:
                raise ValueError("feature_penalty must be real-valued and non-boolean")
            penalty = torch.as_tensor(feature_penalty, device="cpu", dtype=torch.float64).detach()
            if penalty.shape != (self.features,) or not torch.isfinite(penalty).all() or (penalty <= 0).any():
                raise ValueError("feature_penalty must have one finite positive entry per feature, including bias")
        active, _, scales, gram, cross, support, median = self._normalized(True, False)
        support = torch.where(active, support.pow(support_power), 0.)
        strength = ridge * support
        effective_penalty = strength[:, None] * penalty
        if not torch.isfinite(effective_penalty).all():
            raise ValueError("support and feature penalties overflowed")
        normalized = torch.zeros_like(cross)
        degrees = torch.zeros_like(self.counts)
        normal = torch.zeros_like(cross)
        condition = torch.zeros_like(self.counts)
        minimum = torch.zeros_like(self.counts)
        maximum = torch.zeros_like(self.counts)
        if active.any():
            matrix = gram[active] + torch.diag_embed(effective_penalty[active])
            normalized[active] = torch.linalg.solve(matrix, cross[active, :, None]).squeeze(-1)
            degrees[active] = torch.linalg.solve(matrix, gram[active]).diagonal(dim1=-2, dim2=-1).sum(-1)
            eigenvalues = torch.linalg.eigvalsh(matrix)
            minimum[active], maximum[active] = eigenvalues[:, 0], eigenvalues[:, -1]
            if (minimum[active] <= 0).any():
                raise ValueError("regularized readout system is not positive definite at solve precision")
            condition[active] = maximum[active] / minimum[active]
            normal[active] = 2 * ((matrix @ normalized[active, :, None]).squeeze(-1) - cross[active])
        raw = normalized / scales
        if not torch.isfinite(raw).all() or not torch.isfinite(raw.float()).all():
            raise ValueError("adaptive readout coefficients must be finite in float64 and float32")
        deployed = raw.float().double() * scales
        residual_gradient = 2 * ((gram @ deployed.unsqueeze(-1)).squeeze(-1) - cross
                                 + effective_penalty * deployed)
        residual_gradient = torch.where(active[:, None], residual_gradient, 0.)
        active_count = int(active.sum())
        geometry = self._fit_geometry(gram, active, strength)
        # A nonuniform feature penalty has a different condition number from
        # G + lambda*I; report the eigenvalues of the actual fitted system.
        geometry["active_regularized_condition"] = condition[active].tolist()
        report = {
            "kind": "adaptive_full_feature_independent",
            "definition": ("Mean active-variable residual MSE plus mean(lambda*max(1,median_positive_count/Nv)"
                           "**support_power*sum(feature_penalty*(train_RMS*raw_beta)^2)); bias regularized."),
            "ridge": ridge, "support_power": support_power,
            "support_shrink": True, "common_rms": False,
            "counts": self.counts.tolist(), "active_variables": active.tolist(),
            "active_variable_indices": active.nonzero(as_tuple=True)[0].tolist(),
            "zero_support_variables": (~active).nonzero(as_tuple=True)[0].tolist(),
            "median_positive_count": median, "support_multiplier": support.tolist(),
            "feature_rms": scales.tolist(), "feature_penalty": penalty.tolist(),
            "lambda_per_variable": strength.tolist(),
            "effective_feature_penalty": effective_penalty.tolist(),
            "normalized_coefficients": normalized.tolist(), "coefficient_norm": raw.norm(dim=-1).tolist(),
            "shared_coefficients": torch.zeros(self.features, dtype=torch.float64).tolist(),
            "local_normal_gradient": normal.tolist(),
            "local_normal_gradient_norm": normal.norm(dim=-1).tolist(),
            "local_normal_gradient_definition": "Per-variable normalized-coordinate gradient; divide by active variable count for the complete objective.",
            "local_objective_gradient": (normal / max(1, active_count)).tolist(),
            "float32_deployed_local_normal_gradient": residual_gradient.tolist(),
            "float32_deployed_local_normal_gradient_norm": residual_gradient.norm(dim=-1).tolist(),
            "local_regularizer_per_variable": (effective_penalty * normalized.square()).sum(-1).tolist(),
            "regularized_system_min_eigenvalue": minimum.tolist(),
            "regularized_system_max_eigenvalue": maximum.tolist(),
            "effective_degrees_of_freedom": degrees.tolist(),
            "effective_degrees_of_freedom_definition": "Trace of (G+diag(effective_penalty))^-1 G for each variable; zero without training support.",
            "train_score": self.score(raw), "reference_score": self.score(torch.zeros_like(raw)),
            "batches": self.batches,
        }
        report.update(geometry)
        return raw, report
