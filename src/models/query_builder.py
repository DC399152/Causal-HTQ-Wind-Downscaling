"""Target 10min-height query construction."""

from __future__ import annotations

import torch
from torch import nn


class FixedTargetQueryBuilder(nn.Module):
    """Build learned target queries shared across all samples.

    query[j, h] = target_bin_embedding[j] + height_embedding[h]
    """

    def __init__(self, d_model: int = 64, target_steps: int = 6, height_levels: int = 6) -> None:
        super().__init__()
        self.d_model = d_model
        self.target_steps = target_steps
        self.height_levels = height_levels
        self.target_bin_embedding = nn.Embedding(target_steps, d_model)
        self.height_embedding = nn.Embedding(height_levels, d_model)

    def forward(
        self,
        encoder_memory: torch.Tensor | None = None,
        *,
        batch_size: int | None = None,
        height_levels: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return target queries with shape [B, T_out * H, d_model]."""

        if encoder_memory is not None:
            batch_size = int(encoder_memory.shape[0])
            device = encoder_memory.device
        if batch_size is None:
            raise ValueError("batch_size is required when encoder_memory is not provided")
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


class ContextConditionedQueryBuilder(nn.Module):
    """Build target queries conditioned on encoder memory.

    query[j, h] =
        target_bin_embedding[j]
      + height_embedding[h]
      + context_embedding[h]

    ``context_embedding`` is taken from the current-hour height tokens in
    ``encoder_memory`` after the Transformer encoder.
    """

    def __init__(
        self,
        d_model: int = 64,
        target_steps: int = 6,
        context_hours: int = 6,
        height_levels: int = 6,
        *,
        use_context_projection: bool = True,
        use_context_layernorm: bool = True,
        use_trend_context: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.target_steps = target_steps
        self.context_hours = context_hours
        self.height_levels = height_levels
        self.use_context_projection = use_context_projection
        self.use_context_layernorm = use_context_layernorm
        self.use_trend_context = use_trend_context
        self.target_bin_embedding = nn.Embedding(target_steps, d_model)
        self.height_embedding = nn.Embedding(height_levels, d_model)
        self.context_proj = nn.Linear(d_model, d_model) if use_context_projection else nn.Identity()
        self.context_norm = nn.LayerNorm(d_model) if use_context_layernorm else nn.Identity()
        self.trend_proj = nn.Linear(d_model, d_model) if use_trend_context else None

    def forward(
        self,
        encoder_memory: torch.Tensor,
        *,
        height_levels: int | None = None,
    ) -> torch.Tensor:
        """Return context-conditioned queries [B, T_out * H, d_model]."""

        if encoder_memory.ndim != 3:
            raise ValueError("encoder_memory must have shape [B, L*H, D]")
        batch_size, num_tokens, d_model = encoder_memory.shape
        if d_model != self.d_model:
            raise ValueError(f"encoder_memory hidden dimension must be {self.d_model}")
        height_levels = self.height_levels if height_levels is None else height_levels
        if height_levels > self.height_levels:
            raise ValueError("Requested more height levels than configured")
        expected_tokens = self.context_hours * self.height_levels
        if num_tokens != expected_tokens:
            raise ValueError(f"encoder_memory token dimension must be {expected_tokens}")

        memory_4d = encoder_memory.reshape(batch_size, self.context_hours, self.height_levels, self.d_model)
        current_height_context = memory_4d[:, -1, :height_levels, :]
        context_embedding = self.context_norm(self.context_proj(current_height_context))

        if self.use_trend_context:
            if self.context_hours < 2:
                raise ValueError("Trend context requires at least two context hours")
            if self.trend_proj is None:
                raise RuntimeError("trend_proj is not initialized")
            trend_context = memory_4d[:, -1, :height_levels, :] - memory_4d[:, -2, :height_levels, :]
            context_embedding = context_embedding + self.trend_proj(trend_context)

        target_ids = torch.arange(self.target_steps, device=encoder_memory.device)
        height_ids = torch.arange(height_levels, device=encoder_memory.device)
        target_emb = self.target_bin_embedding(target_ids)
        height_emb = self.height_embedding(height_ids)

        queries = (
            target_emb[None, :, None, :]
            + height_emb[None, None, :, :]
            + context_embedding[:, None, :, :]
        )
        return queries.reshape(batch_size, self.target_steps * height_levels, self.d_model)


# Backward-compatible name for old imports. The default remains fixed queries
# unless HTQConfig selects ContextConditionedQueryBuilder explicitly.
TargetQueryBuilder = FixedTargetQueryBuilder
