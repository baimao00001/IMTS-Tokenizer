"""Zero-start, support-regularized variable-specific residual prediction head."""

import torch
from torch import nn
from torch.nn import functional as F


class VariableReadout(nn.Module):
    def __init__(self, feature_dim, variables, rank=4, learned_projection=True,
                 training_counts=None):
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("variables", variables), ("rank", rank)):
            if type(value) is not int or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if rank > feature_dim:
            raise ValueError("rank cannot exceed feature_dim")
        if type(learned_projection) is not bool:
            raise ValueError("learned_projection must be boolean")
        if training_counts is None:
            raise ValueError("VariableReadout requires complete train-only target counts")
        raw_counts = torch.as_tensor(training_counts)
        if raw_counts.is_complex() or raw_counts.dtype == torch.bool:
            raise ValueError("training_counts must be real nonnegative counts")
        # Convert the original input, not a default-float32 temporary, to retain
        # exact training counts when callers supply a Python float sequence.
        counts = torch.as_tensor(training_counts, device="cpu", dtype=torch.float64).detach().clone()
        if counts.shape != (variables,) or not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("training_counts must contain one finite nonnegative count per variable")
        self.feature_dim, self.variables, self.rank = feature_dim, variables, rank
        self.learned_projection = learned_projection
        self.basis_names = ("bias",) + tuple("latent_" + str(index) for index in range(rank))
        positive = counts > 0
        support = torch.zeros_like(counts)
        if positive.any():
            median = torch.quantile(counts[positive], .5)
            support[positive] = (median / counts[positive]).clamp_min(1.0)
        if not torch.isfinite(support).all() or (support > torch.finfo(torch.float32).max).any():
            raise ValueError("training support ratios are outside the finite float32 range")
        self.register_buffer("training_counts", counts)
        self.register_buffer("support_weights", support.float())
        generator = torch.Generator(device="cpu").manual_seed(730227)
        sample = torch.randn(feature_dim, rank, generator=generator, dtype=torch.float64)
        projection, upper = torch.linalg.qr(sample, mode="reduced")
        projection = (projection * torch.where(upper.diagonal() < 0, -1.0, 1.0)).float()
        if learned_projection:
            self.projection = nn.Parameter(projection)
        else:
            self.register_buffer("projection", projection)
        self.coefficients = nn.Parameter(torch.zeros(variables, rank + 1))

    def normalized_projection(self):
        """Expose the exact projection used by forward for active diagnostics."""
        epsilon = max(1e-12, torch.finfo(self.projection.dtype).tiny)
        return F.normalize(self.projection, p=2, dim=0, eps=epsilon)

    def forward(self, features, return_details=False):
        if features.ndim != 4 or features.shape[-2:] != (self.variables, self.feature_dim):
            raise ValueError("features must have shape [batch, time, variables, feature_dim]")
        if not features.is_floating_point():
            raise ValueError("features must be real floating-point tensors")
        latent = F.layer_norm(features, (self.feature_dim,)) @ self.normalized_projection()
        basis = torch.cat((torch.ones_like(latent[..., :1]), latent), dim=-1)
        expanded_coefficients = self.coefficients[None, None, :, :].expand_as(basis)
        contributions = basis * expanded_coefficients
        delta = contributions.sum(dim=-1)
        if return_details:
            # Only the caller retains these references; inactive diagnostics keep no graph.
            return delta, {"coefficients": expanded_coefficients,
                           "contributions": contributions, "hidden": latent,
                           "branch_names": list(self.basis_names)}
        return delta

    def regularization(self, mask, delta, training_batches=None):
        """Support-weighted mean correction energy in the original prediction units."""
        if delta.ndim != 3 or delta.shape[-1] != self.variables or mask.shape != delta.shape:
            raise ValueError("mask and delta must match [batch, time, variables]")
        if mask.device != delta.device or delta.device != self.support_weights.device:
            raise ValueError("mask, delta and support_weights must share a device")
        if not delta.is_floating_point() or mask.is_complex():
            raise ValueError("delta must be floating-point and mask must be real")
        if not torch.isfinite(mask).all() or (mask < 0).any():
            raise ValueError("mask weights must be finite and nonnegative")
        # Accumulate low-precision energy in float32; masked NaNs must not leak in.
        dtype = torch.float64 if delta.dtype == torch.float64 else torch.float32
        weights = mask.to(dtype=dtype)
        clean_delta = torch.where(mask > 0, delta, 0.0).to(dtype=dtype)
        counts = weights.sum(dim=(0, 1))
        present = counts > 0
        energy = (clean_delta.square() * weights).sum(dim=(0, 1))
        if training_batches is not None:
            if type(training_batches) is not int or training_batches < 1:
                raise ValueError("training_batches must be a positive integer")
            train_counts = self.training_counts.to(dtype=dtype)
            supported = train_counts > 0
            per_variable = energy / torch.where(supported, train_counts, 1.0)
            weighted = per_variable * self.support_weights.to(dtype=dtype)
            return weighted.sum() * training_batches / supported.sum().clamp_min(1)
        per_variable = energy / (counts + 1e-8)
        weighted = per_variable * self.support_weights.to(dtype=dtype)
        return weighted.sum() / present.sum().clamp_min(1)
