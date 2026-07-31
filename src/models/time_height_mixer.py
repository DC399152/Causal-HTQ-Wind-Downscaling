"""Axis-specific MLP mixers for structured height-time features."""

from __future__ import annotations

import torch
from torch import nn


def _validate_4d(
    x: torch.Tensor,
    *,
    context_hours: int,
    height_levels: int,
    d_model: int,
) -> None:
    if x.ndim != 4:
        raise ValueError("x must have shape [B, L, H, D]")
    expected = (context_hours, height_levels, d_model)
    if tuple(x.shape[1:]) != expected:
        raise ValueError(
            f"x must have trailing shape {expected}, got {tuple(x.shape[1:])}"
        )


class TemporalMixingMLP(nn.Module):
    """Mix only the historical-time axis of ``[B, L, H, D]`` features."""

    def __init__(
        self,
        context_hours: int,
        height_levels: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_hours = context_hours
        self.height_levels = height_levels
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(context_hours, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, context_hours),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_4d(
            x,
            context_hours=self.context_hours,
            height_levels=self.height_levels,
            d_model=self.d_model,
        )
        mixed = self.norm(x).permute(0, 2, 3, 1).contiguous()
        mixed = self.mlp(mixed).permute(0, 3, 1, 2).contiguous()
        return x + self.residual_dropout(mixed)


class HeightMixingMLP(nn.Module):
    """Mix only the physical-height axis of ``[B, L, H, D]`` features."""

    def __init__(
        self,
        context_hours: int,
        height_levels: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_hours = context_hours
        self.height_levels = height_levels
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(height_levels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, height_levels),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_4d(
            x,
            context_hours=self.context_hours,
            height_levels=self.height_levels,
            d_model=self.d_model,
        )
        mixed = self.norm(x).permute(0, 1, 3, 2).contiguous()
        mixed = self.mlp(mixed).permute(0, 1, 3, 2).contiguous()
        return x + self.residual_dropout(mixed)


class ChannelMixingMLP(nn.Module):
    """Mix only hidden channels at each independent time-height position."""

    def __init__(
        self,
        context_hours: int,
        height_levels: int,
        d_model: int,
        expansion_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_hours = context_hours
        self.height_levels = height_levels
        self.d_model = d_model
        hidden_dim = expansion_ratio * d_model
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_4d(
            x,
            context_hours=self.context_hours,
            height_levels=self.height_levels,
            d_model=self.d_model,
        )
        return x + self.residual_dropout(self.mlp(self.norm(x)))


class TimeHeightMixerBlock(nn.Module):
    """Apply temporal, height, then hidden-channel mixing."""

    def __init__(
        self,
        context_hours: int,
        height_levels: int,
        d_model: int,
        temporal_hidden_dim: int,
        height_hidden_dim: int,
        channel_expansion_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal_mixing = TemporalMixingMLP(
            context_hours,
            height_levels,
            d_model,
            temporal_hidden_dim,
            dropout,
        )
        self.height_mixing = HeightMixingMLP(
            context_hours,
            height_levels,
            d_model,
            height_hidden_dim,
            dropout,
        )
        self.channel_mixing = ChannelMixingMLP(
            context_hours,
            height_levels,
            d_model,
            channel_expansion_ratio,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_mixing(x)
        x = self.height_mixing(x)
        return self.channel_mixing(x)


class TemporalTargetProjection(nn.Module):
    """Project L historical positions to T target horizons for each H-D pair."""

    def __init__(
        self,
        context_hours: int,
        target_steps: int,
        height_levels: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_hours = context_hours
        self.target_steps = target_steps
        self.height_levels = height_levels
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(context_hours, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_steps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_4d(
            x,
            context_hours=self.context_hours,
            height_levels=self.height_levels,
            d_model=self.d_model,
        )
        projected = self.norm(x).permute(0, 2, 3, 1).contiguous()
        projected = self.projection(projected)
        return projected.permute(0, 3, 1, 2).contiguous()
