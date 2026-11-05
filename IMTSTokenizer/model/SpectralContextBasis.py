"""Fixed multiscale query-time interactions with fixed history context."""

import math

import torch
from torch import nn

from .InteractionBasis import _orthogonal_columns


class SpectralContextBasis(nn.Module):
    projection_seed = 730127

    def __init__(self, raw_feature_dim, rank=4, *, nested=False):
        super().__init__()
        for name, value in (("raw_feature_dim", raw_feature_dim), ("rank", rank)):
            if type(value) is not int or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if rank > raw_feature_dim:
            raise ValueError("rank cannot exceed raw_feature_dim")
        if type(nested) is not bool:
            raise ValueError("nested must be a boolean")
        if nested and rank < 4:
            raise ValueError("nested spectral rank must be at least four")
        self.raw_feature_dim = raw_feature_dim
        self.rank = rank
        self.nested = nested
        self.feature_dim = raw_feature_dim + 8 * rank
        if nested:
            projection = self._nested_projection(raw_feature_dim, rank)
        else:
            projection = _orthogonal_columns(raw_feature_dim, rank, self.projection_seed).float()
        self.register_buffer("projection", projection)
        self.register_buffer("cycles", torch.tensor([1., 2., 4., 8.], dtype=torch.float32, device="cpu"))
        phase_names = tuple(kind + "_" + str(cycle) for kind in ("sin", "cos") for cycle in (1, 2, 4, 8))
        self.basis_names = (tuple("feature_" + str(index) for index in range(raw_feature_dim))
                            + tuple("context_" + str(index) + "_" + phase for index in range(rank) for phase in phase_names))

    @classmethod
    def _nested_projection(cls, dimension, rank):
        first = _orthogonal_columns(dimension, 4, cls.projection_seed).float()
        columns = [first[:, index].clone() for index in range(4)]
        generator = torch.Generator(device="cpu").manual_seed(730128)
        while len(columns) < rank:
            vector = torch.randn(dimension, dtype=torch.double, generator=generator, device="cpu")
            basis = torch.stack(columns, dim=-1).double()
            # Account for the original float32 QR rounding without changing its
            # four columns. Sequential completion makes every rank a prefix.
            for _ in range(2):
                vector = vector - basis @ torch.linalg.solve(basis.T @ basis, basis.T @ vector)
            columns.append((vector / torch.linalg.vector_norm(vector)).float())
        return torch.stack(columns, dim=-1)

    def forward(self, product, query, context, time_predict):
        if product.ndim != 4 or product.shape[-1] != self.raw_feature_dim or query.shape != product.shape:
            raise ValueError("product and query must match [batch, time, variables, raw_feature_dim]")
        if context.shape != (product.shape[0], product.shape[2], self.raw_feature_dim):
            raise ValueError("context must match [batch, variables, raw_feature_dim] before time expansion")
        if time_predict.shape != product.shape[:2]:
            raise ValueError("prediction times must match [batch, time]")
        inputs = (product, query, context, time_predict)
        if any(not value.is_floating_point() for value in inputs):
            raise ValueError("spectral inputs must be real floating-point tensors")
        if any(value.device != self.projection.device for value in inputs):
            raise ValueError("spectral inputs and projection buffers must share a device")
        if any(value.dtype != self.projection.dtype for value in inputs):
            raise ValueError("spectral inputs and projection buffers must share a dtype")
        if any(not torch.isfinite(value).all() for value in inputs):
            raise ValueError("spectral inputs must be finite")
        # Phases use absolute normalized query times, not window-relative times.
        phase = time_predict[:, :, None] * (2 * math.pi * self.cycles)
        phi = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if self.nested:
            projected = [context @ self.projection[:, :4].contiguous()]
            for index in range(4, self.rank):
                projected.append((context * self.projection[:, index]).sum(dim=-1, keepdim=True))
            projected_context = torch.cat(projected, dim=-1)
        else:
            projected_context = context @ self.projection
        extra = (projected_context[:, None, :, :, None] * phi[:, :, None, None, :]).flatten(-2)
        result = torch.cat((product, extra), dim=-1)
        if not torch.isfinite(result).all():
            raise ValueError("spectral feature calculation produced nonfinite values")
        return result
