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


class TemporalContextPooling(nn.Module):
    """Attention-pool encoder memory along the hourly context axis.

    Input memory uses semantic shape [B, L, H, D]. The output keeps one
    context vector per height: [B, H, D].
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, memory_4d: torch.Tensor) -> torch.Tensor:
        if memory_4d.ndim != 4:
            raise ValueError("memory_4d must have shape [B, L, H, D]")

        scores = self.score(memory_4d)  # [B, L, H, 1].
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights * memory_4d, dim=1)  # [B, H, D].


class MultiScaleTrendEmbedding(nn.Module):
    """Encode current-minus-past trends at configurable hourly scales."""

    def __init__(self, d_model: int, trend_scales: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        if not trend_scales:
            raise ValueError("trend_scales must contain at least one scale")
        self.d_model = d_model
        self.trend_scales = tuple(int(scale) for scale in trend_scales)
        if any(scale <= 0 for scale in self.trend_scales):
            raise ValueError("trend_scales must be positive integers")

        self.scale_projections = nn.ModuleList(
            nn.Linear(d_model, d_model) for _ in self.trend_scales
        )
        self.output_projection = nn.Linear(d_model * len(self.trend_scales), d_model)

    def forward(self, memory_4d: torch.Tensor) -> torch.Tensor:
        if memory_4d.ndim != 4:
            raise ValueError("memory_4d must have shape [B, L, H, D]")
        if memory_4d.shape[-1] != self.d_model:
            raise ValueError(f"memory_4d hidden dimension must be {self.d_model}")

        context_hours = memory_4d.shape[1]
        current = memory_4d[:, -1]
        projected_trends = []
        for scale, projection in zip(self.trend_scales, self.scale_projections):
            if context_hours > scale:
                raw_trend = current - memory_4d[:, -(scale + 1)]
            else:
                raw_trend = torch.zeros_like(current)
            projected_trends.append(projection(raw_trend))

        return self.output_projection(torch.cat(projected_trends, dim=-1))


class ContextConditionedQueryBuilder(nn.Module):
    """Build target queries with optional context-conditioned components.

    Four ablation modes are supported via config flags:
    - Time + Height only.
    - Time + Height + multi-scale trend.
    - Time + Height + temporal context pooling.
    - Time + Height + trend + temporal context pooling.
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
        use_temporal_context: bool = False,
        use_multiscale_trend: bool = False,
        trend_scales: tuple[int, ...] = (1, 3, 5),
        use_trend_context: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.target_steps = target_steps
        self.context_hours = context_hours
        self.height_levels = height_levels
        self.use_context_projection = use_context_projection
        self.use_context_layernorm = use_context_layernorm
        self.use_temporal_context = use_temporal_context
        # Backward compatibility: old configs used use_trend_context for a
        # single-step trend. Treat it as enabling the new multi-scale trend.
        self.use_multiscale_trend = use_multiscale_trend or use_trend_context
        self.trend_scales = tuple(int(scale) for scale in trend_scales)
        self.target_bin_embedding = nn.Embedding(target_steps, d_model)
        self.height_embedding = nn.Embedding(height_levels, d_model)

        self.temporal_pooling = (
            TemporalContextPooling(d_model) if self.use_temporal_context else None
        )
        self.trend_embedding = (
            MultiScaleTrendEmbedding(d_model, self.trend_scales)
            if self.use_multiscale_trend
            else None
        )
        self.context_fusion = (
            nn.Linear(2 * d_model, d_model)
            if self.use_temporal_context and self.use_multiscale_trend
            else None
        )
        self.context_proj = (
            nn.Linear(d_model, d_model)
            if (use_context_projection and self.uses_encoder_context)
            else nn.Identity()
        )
        self.context_norm = (
            nn.LayerNorm(d_model)
            if (use_context_layernorm and self.uses_encoder_context)
            else nn.Identity()
        )
        self.fused_context_norm = (
            nn.LayerNorm(d_model)
            if self.use_temporal_context and self.use_multiscale_trend
            else nn.Identity()
        )

    @property
    def uses_encoder_context(self) -> bool:
        return self.use_temporal_context or self.use_multiscale_trend

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

        context_embedding = None
        if self.uses_encoder_context:
            memory_4d = encoder_memory.reshape(
                batch_size,
                self.context_hours,
                self.height_levels,
                self.d_model,
            )
            memory_4d = memory_4d[:, :, :height_levels, :]

            temporal_context = None
            if self.temporal_pooling is not None:
                temporal_context = self.temporal_pooling(memory_4d)

            trend_context = None
            if self.trend_embedding is not None:
                trend_context = self.trend_embedding(memory_4d)

            if temporal_context is not None and trend_context is not None:
                if self.context_fusion is None:
                    raise RuntimeError("context_fusion is not initialized")
                context_embedding = self.context_fusion(
                    torch.cat([temporal_context, trend_context], dim=-1)
                )
                context_embedding = self.fused_context_norm(context_embedding)
            elif temporal_context is not None:
                context_embedding = temporal_context
            elif trend_context is not None:
                context_embedding = trend_context

            if context_embedding is None:
                raise RuntimeError("Context components are enabled but no context was built")
            context_embedding = self.context_norm(self.context_proj(context_embedding))

        target_ids = torch.arange(self.target_steps, device=encoder_memory.device)
        height_ids = torch.arange(height_levels, device=encoder_memory.device)
        target_emb = self.target_bin_embedding(target_ids)
        height_emb = self.height_embedding(height_ids)

        queries = target_emb[None, :, None, :] + height_emb[None, None, :, :]
        if context_embedding is not None:
            queries = queries + context_embedding[:, None, :, :]
        else:
            queries = queries.expand(batch_size, -1, -1, -1)
        return queries.reshape(batch_size, self.target_steps * height_levels, self.d_model)


# Backward-compatible name for old imports. The default remains fixed queries
# unless HTQConfig selects ContextConditionedQueryBuilder explicitly.
TargetQueryBuilder = FixedTargetQueryBuilder
