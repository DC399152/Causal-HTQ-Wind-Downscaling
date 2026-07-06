"""Target 10min-height query construction."""

from __future__ import annotations

import torch
from torch import nn


class TargetQueryBuilder(nn.Module):
    """Build learned target queries for 10min bins and height levels."""

    def __init__(self, d_model: int = 64, target_steps: int = 6, height_levels: int = 6) -> None:
        super().__init__()
        self.d_model = d_model
        self.target_steps = target_steps
        self.height_levels = height_levels
        self.target_bin_embedding = nn.Embedding(target_steps, d_model)
        self.height_embedding = nn.Embedding(height_levels, d_model)

    def forward(
        self,
        batch_size: int,
        height_levels: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return target queries with shape [B, T_out * H, d_model]."""

        height_levels = self.height_levels if height_levels is None else height_levels
        if height_levels > self.height_levels:
            raise ValueError("Requested more height levels than configured")

        target_ids = torch.arange(self.target_steps, device=device)
        height_ids = torch.arange(height_levels, device=device)
        target_emb = self.target_bin_embedding(target_ids).view(self.target_steps, 1, self.d_model)
        height_emb = self.height_embedding(height_ids).view(1, height_levels, self.d_model)
        queries = target_emb + height_emb
        queries = queries.reshape(1, self.target_steps * height_levels, self.d_model)
        return queries.expand(batch_size, -1, -1)

    def build(self, batch_size: int, height_levels: int):
        """Compatibility alias for older scaffold code."""

        return self.forward(batch_size=batch_size, height_levels=height_levels)
