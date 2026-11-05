"""Fixed diagonal and off-diagonal bilinear feature banks for one readout."""

import torch
from torch import nn


def _orthogonal_columns(width, rank, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn(width, rank, generator=generator, device="cpu", dtype=torch.float64)
    columns, upper = torch.linalg.qr(sample, mode="reduced")
    return columns * torch.where(upper.diagonal() < 0, -1.0, 1.0)


class ReadoutInteractionBasis(nn.Module):
    projection_seed = 730127
    extra_seed = 730327
    query_seed = 730527
    context_seed = 730627

    def __init__(self, feature_dim, core_rank=4, extra_rank=4):
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("core_rank", core_rank), ("extra_rank", extra_rank)):
            if type(value) is not int or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if core_rank + extra_rank > feature_dim:
            raise ValueError("core_rank + extra_rank cannot exceed feature_dim")
        self.feature_dim, self.core_rank, self.extra_rank = feature_dim, core_rank, extra_rank
        original = _orthogonal_columns(feature_dim, core_rank, self.projection_seed)
        extra = _orthogonal_columns(feature_dim, extra_rank, self.extra_seed)
        # Orthogonalize twice to control cancellation near a full-dimensional bank.
        for _ in range(2):
            extra = extra - original @ (original.T @ extra)
            extra, upper = torch.linalg.qr(extra, mode="reduced")
            extra = extra * torch.where(upper.diagonal() < 0, -1.0, 1.0)
        self.register_buffer("projection", original.float())
        self.register_buffer("extra_projection", extra.float())
        self.register_buffer("query_projection", _orthogonal_columns(feature_dim, extra_rank, self.query_seed).float())
        self.register_buffer("context_projection", _orthogonal_columns(feature_dim, extra_rank, self.context_seed).float())

    def forward(self, product, query, context, kind="interaction"):
        if kind not in {"diagonal", "interaction"}:
            raise ValueError("kind must be diagonal or interaction")
        if product.ndim != 4 or product.shape[-1] != self.feature_dim or query.shape != product.shape:
            raise ValueError("product and query must match [batch, time, variables, feature_dim]")
        if context.shape != (product.shape[0], product.shape[2], self.feature_dim):
            raise ValueError("context must match [batch, variables, feature_dim] before time expansion")
        if any(not value.is_floating_point() for value in (product, query, context)):
            raise ValueError("product, query and context must be real floating-point tensors")
        if any(value.device != self.projection.device for value in (product, query, context)):
            raise ValueError("inputs and projection buffers must share a device")
        if any(value.dtype != self.projection.dtype for value in (product, query, context)):
            raise ValueError("inputs and projection buffers must share a dtype")
        if any(not torch.isfinite(value).all() for value in (product, query, context)):
            raise ValueError("interaction inputs must be finite")
        # Separate matmuls retain the projection's floating-point reduction order.
        original = product @ self.projection
        if kind == "diagonal":
            additional = product @ self.extra_projection
        else:
            projected_query = query @ self.query_projection
            projected_context = context @ self.context_projection
            diagonal = product @ (self.query_projection * self.context_projection)
            additional = projected_query * projected_context[:, None, :, :] - diagonal
        result = torch.cat((original, additional), dim=-1)
        if not torch.isfinite(result).all():
            raise ValueError("interaction feature calculation produced nonfinite values")
        return result
