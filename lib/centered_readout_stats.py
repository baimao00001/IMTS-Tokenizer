"""Train-only affine fitting with separate mean-level and slope penalties."""

import math

import torch

from lib.hierarchical_readout_stats import FullReadoutStatistics


class CenteredReadoutStatistics(FullReadoutStatistics):
    @staticmethod
    def _nonnegative(value, name):
        if isinstance(value, bool):
            raise ValueError(name + " must be finite and nonnegative")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(name + " must be finite and nonnegative")
        return value

    @torch.no_grad()
    def fit_centered(self, ridge, bias_penalty=.1, support_power=1., center_features=True):
        ridge = self._positive(ridge, "ridge")
        bias_penalty = self._nonnegative(bias_penalty, "bias_penalty")
        support_power = self._nonnegative(support_power, "support_power")
        if type(center_features) is not bool:
            raise ValueError("center_features must be boolean")
        active, counts, scales, gram, cross, support, median = self._normalized(True, False)
        observed_mean = self.gram[:, 0, 1:] / counts[:, None]
        means = torch.zeros_like(observed_mean)
        constant = torch.zeros_like(observed_mean, dtype=torch.bool)
        if center_features:
            means = observed_mean
            second = self.gram[:, 1:, 1:] / counts[:, None, None]
            covariance = second - means[:, :, None] * means[:, None, :]
            covariance = (covariance + covariance.transpose(-2, -1)) / 2
            variance = covariance.diagonal(dim1=-2, dim2=-1)
            magnitude = torch.maximum(second.diagonal(dim1=-2, dim2=-1).abs(), means.square())
            tolerance = 64 * torch.finfo(torch.float64).eps * magnitude.clamp_min(torch.finfo(torch.float64).tiny)
            if ((variance < -tolerance) & active[:, None]).any():
                raise ValueError("centered statistics contain a negative variance beyond cancellation tolerance")
            constant = variance.abs() <= tolerance
            covariance = torch.where(constant[:, :, None] | constant[:, None, :], 0., covariance)
            scale = variance.clamp_min(0).sqrt().clamp_min(1e-8)
            scales = torch.cat((torch.ones_like(counts[:, None]), scale), dim=-1)
            gram = torch.zeros_like(self.gram)
            gram[:, 0, 0] = active.double()
            gram[:, 1:, 1:] = covariance / scale[:, :, None] / scale[:, None, :]
            centered_cross = self.cross[:, 1:] / counts[:, None] - means * (
                self.cross[:, 0] / counts)[:, None]
            centered_cross = torch.where(constant, 0., centered_cross)
            cross = torch.cat(((self.cross[:, 0] / counts)[:, None], centered_cross / scale), dim=-1)
        support = torch.where(active, support.pow(support_power), 0.)
        strength = ridge * support
        penalty = strength[:, None].expand_as(cross).clone()
        penalty[:, 0] *= bias_penalty
        if not torch.isfinite(penalty).all():
            raise ValueError("centered support penalty overflowed")
        normalized = torch.zeros_like(cross)
        normal = torch.zeros_like(cross)
        degrees_by_feature = torch.zeros_like(cross)
        minimum = torch.zeros_like(counts)
        maximum = torch.zeros_like(counts)
        if active.any():
            matrix = gram[active] + torch.diag_embed(penalty[active])
            normalized[active] = torch.linalg.solve(matrix, cross[active, :, None]).squeeze(-1)
            degrees_by_feature[active] = torch.linalg.solve(matrix, gram[active]).diagonal(dim1=-2, dim2=-1)
            eigenvalues = torch.linalg.eigvalsh(matrix)
            minimum[active], maximum[active] = eigenvalues[:, 0], eigenvalues[:, -1]
            if (minimum[active] <= 0).any():
                raise ValueError("centered regularized system is not positive definite at solve precision")
            normal[active] = 2 * ((matrix @ normalized[active, :, None]).squeeze(-1) - cross[active])
        raw = normalized / scales
        raw[:, 0] -= (means * raw[:, 1:]).sum(-1)
        if not torch.isfinite(raw).all() or not torch.isfinite(raw.float()).all():
            raise ValueError("centered readout coefficients must be finite in float64 and float32")
        deployed = raw.float().double() * scales
        deployed[:, 0] = raw.float().double()[:, 0] + (means * raw.float().double()[:, 1:]).sum(-1)
        deployed_normal = 2 * ((gram @ deployed[:, :, None]).squeeze(-1) - cross + penalty * deployed)
        deployed_normal = torch.where(active[:, None], deployed_normal, 0.)
        active_count = max(1, int(active.sum()))
        report = {
            "kind": "centered_independent_affine",
            "definition": ("Mean active-variable train residual MSE plus mean(lambda*support**power*"
                           "(bias_penalty*mean_level**2 + sum((train_scale*slope)**2))); "
                           "train means and scales are folded into raw affine coefficients."),
            "ridge": ridge, "support_power": support_power, "bias_penalty": bias_penalty,
            "center_features": center_features, "support_shrink": True, "common_rms": False,
            "normalization": "train_standard_deviation" if center_features else "train_RMS",
            "counts": self.counts.tolist(), "active_variables": active.tolist(),
            "active_variable_indices": active.nonzero(as_tuple=True)[0].tolist(),
            "zero_support_variables": (~active).nonzero(as_tuple=True)[0].tolist(),
            "median_positive_count": median, "support_multiplier": support.tolist(),
            "observed_feature_mean": observed_mean.tolist(), "centering_mean": means.tolist(),
            "feature_scale": scales.tolist(), "constant_feature_mask": constant.tolist(),
            "constant_feature_definition": "Centered second moment is within 64 float64 eps of its subtraction scale; zeroed to avoid cancellation-only slope information.",
            "lambda_per_variable": strength.tolist(), "effective_feature_penalty": penalty.tolist(),
            "normalized_coefficients": normalized.tolist(),
            "centered_mean_level": normalized[:, 0].tolist(),
            "raw_intercept": raw[:, 0].tolist(), "raw_slopes": raw[:, 1:].tolist(),
            "coefficient_norm": raw.norm(dim=-1).tolist(),
            "shared_coefficients": torch.zeros(self.features, dtype=torch.float64).tolist(),
            "local_normal_gradient": normal.tolist(), "local_normal_gradient_norm": normal.norm(dim=-1).tolist(),
            "local_normal_gradient_definition": "Per-variable gradient in train-normalized centered coordinates; divide by active variable count for complete objective.",
            "local_objective_gradient": (normal / active_count).tolist(),
            "float32_deployed_centered_coefficients": deployed.tolist(),
            "float32_deployed_local_normal_gradient": deployed_normal.tolist(),
            "float32_deployed_local_normal_gradient_norm": deployed_normal.norm(dim=-1).tolist(),
            "bias_regularizer_per_variable": (penalty[:, 0] * normalized[:, 0].square()).tolist(),
            "slope_regularizer_per_variable": (penalty[:, 1:] * normalized[:, 1:].square()).sum(-1).tolist(),
            "effective_degrees_of_freedom": degrees_by_feature.sum(-1).tolist(),
            "bias_effective_degrees_of_freedom": degrees_by_feature[:, 0].tolist(),
            "slope_effective_degrees_of_freedom": degrees_by_feature[:, 1:].sum(-1).tolist(),
            "effective_degrees_of_freedom_definition": "Trace of (G+diag(penalty))^-1 G in train-centered coordinates, partitioned into bias and slope diagonal traces.",
            "regularized_system_min_eigenvalue": minimum.tolist(),
            "regularized_system_max_eigenvalue": maximum.tolist(),
            "active_regularized_condition": (maximum[active] / minimum[active]).tolist(),
            "train_score": self.score(raw), "reference_score": self.score(torch.zeros_like(raw)),
            "batches": self.batches,
        }
        return raw, report
