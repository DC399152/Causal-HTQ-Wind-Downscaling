"""Continuous physical-height embeddings for wind profile tokens."""

from __future__ import annotations

import math

import torch
from torch import nn


class PhysicalHeightEncoder(nn.Module):
    """Encode physical heights in metres into ``d_model`` features."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        height_center_m: float = 300.0,
        height_scale_m: float = 100.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        if not math.isfinite(height_center_m):
            raise ValueError("height_center_m must be finite")
        if not math.isfinite(height_scale_m) or height_scale_m <= 0:
            raise ValueError("height_scale_m must be finite and positive")

        self.height_center_m = float(height_center_m)
        self.height_scale_m = float(height_scale_m)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, height_values: torch.Tensor) -> torch.Tensor:
        """Return embeddings with shape ``[B, H, D]`` from metre values ``[B, H]``."""

        if height_values.ndim != 2:
            raise ValueError("height_values must have shape [B, H]")
        if not torch.isfinite(height_values).all():
            raise ValueError("height_values contains NaN or Inf")
        normalized = (
            height_values.to(dtype=torch.float32) - self.height_center_m
        ) / self.height_scale_m
        return self.mlp(normalized.unsqueeze(-1))
