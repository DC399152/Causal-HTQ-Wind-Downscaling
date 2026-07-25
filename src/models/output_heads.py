"""Configurable residual output heads shared by both HTQ architectures."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


OUTPUT_HEAD_TYPES = {
    "shared_linear",
    "shared_mlp",
    "multi_horizon_shared_trunk",
    "multi_horizon_independent_mlp",
}


@dataclass(frozen=True)
class ResidualHeadConfig:
    type: str
    hidden_dim: int = 64
    dropout: float = 0.05
    final_weight_std: float = 0.001
    identical_horizon_init: bool = True
    share_across_heights: bool = True

    def validate(self, *, target_steps: int, output_channels: int) -> None:
        if self.type not in OUTPUT_HEAD_TYPES:
            raise ValueError(
                f"Unknown output_head.type {self.type!r}; expected one of "
                f"{sorted(OUTPUT_HEAD_TYPES)}"
            )
        if self.hidden_dim <= 0:
            raise ValueError("output_head.hidden_dim must be positive")
        if target_steps <= 0:
            raise ValueError("target_steps must be positive")
        if output_channels <= 0:
            raise ValueError("output_channels must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("output_head.dropout must be in [0, 1)")
        if self.final_weight_std < 0.0:
            raise ValueError("output_head.final_weight_std must be non-negative")
        if not self.share_across_heights:
            raise ValueError(
                "output_head.share_across_heights=false is not supported; "
                "multi-horizon heads are shared across heights"
            )


def residual_head_config_from_mapping(
    config: Mapping[str, Any] | None,
    *,
    default_type: str,
    default_hidden_dim: int = 64,
    default_dropout: float = 0.05,
    default_final_weight_std: float = 0.001,
) -> ResidualHeadConfig:
    """Parse a YAML-style output-head mapping with architecture defaults."""

    values = dict(config or {})
    return ResidualHeadConfig(
        type=str(values.get("type", default_type)),
        hidden_dim=int(values.get("hidden_dim", default_hidden_dim)),
        dropout=float(values.get("dropout", default_dropout)),
        final_weight_std=float(
            values.get("final_weight_std", default_final_weight_std)
        ),
        identical_horizon_init=bool(values.get("identical_horizon_init", True)),
        share_across_heights=bool(values.get("share_across_heights", True)),
    )


def output_head_config_fields(config: ResidualHeadConfig) -> dict[str, Any]:
    """Return flat model-config fields suitable for checkpoint metadata."""

    return {
        "output_head_type": config.type,
        "output_head_hidden_dim": config.hidden_dim,
        "output_head_dropout": config.dropout,
        "output_head_final_weight_std": config.final_weight_std,
        "output_head_identical_horizon_init": config.identical_horizon_init,
        "output_head_share_across_heights": config.share_across_heights,
    }


def residual_head_config_from_model_fields(
    *,
    output_head_type: str | None,
    output_head_hidden_dim: int | None,
    output_head_dropout: float | None,
    output_head_final_weight_std: float | None,
    output_head_identical_horizon_init: bool,
    output_head_share_across_heights: bool,
    default_type: str,
    default_hidden_dim: int,
    default_dropout: float,
    default_final_weight_std: float,
) -> ResidualHeadConfig:
    """Resolve flat dataclass fields, including legacy architecture defaults."""

    return ResidualHeadConfig(
        type=output_head_type or default_type,
        hidden_dim=(
            default_hidden_dim
            if output_head_hidden_dim is None
            else int(output_head_hidden_dim)
        ),
        dropout=(
            default_dropout
            if output_head_dropout is None
            else float(output_head_dropout)
        ),
        final_weight_std=(
            default_final_weight_std
            if output_head_final_weight_std is None
            else float(output_head_final_weight_std)
        ),
        identical_horizon_init=bool(output_head_identical_horizon_init),
        share_across_heights=bool(output_head_share_across_heights),
    )


def _shared_mlp(
    d_model: int,
    output_channels: int,
    config: ResidualHeadConfig,
) -> nn.Sequential:
    head = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, config.hidden_dim),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, output_channels),
    )
    _initialize_final_layer(head[-1], config.final_weight_std)
    return head


def _initialize_final_layer(layer: nn.Linear, weight_std: float) -> None:
    nn.init.normal_(layer.weight, mean=0.0, std=weight_std)
    nn.init.zeros_(layer.bias)


class MultiHorizonSharedTrunkResidualHead(nn.Module):
    """Shared nonlinear trunk followed by one final linear map per horizon."""

    def __init__(
        self,
        d_model: int,
        target_steps: int,
        output_channels: int,
        config: ResidualHeadConfig,
    ) -> None:
        super().__init__()
        self.target_steps = target_steps
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        base_head = nn.Linear(config.hidden_dim, output_channels)
        _initialize_final_layer(base_head, config.final_weight_std)
        if config.identical_horizon_init:
            self.horizon_heads = nn.ModuleList(
                copy.deepcopy(base_head) for _ in range(target_steps)
            )
        else:
            self.horizon_heads = nn.ModuleList(
                nn.Linear(config.hidden_dim, output_channels)
                for _ in range(target_steps)
            )
            for head in self.horizon_heads:
                _initialize_final_layer(head, config.final_weight_std)

    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        _validate_target_features(target_features, self.target_steps)
        shared_features = self.trunk(target_features)
        return torch.stack(
            [
                head(shared_features[:, horizon])
                for horizon, head in enumerate(self.horizon_heads)
            ],
            dim=1,
        )


class MultiHorizonIndependentMLPResidualHead(nn.Module):
    """One complete MLP per horizon, shared only across heights."""

    def __init__(
        self,
        d_model: int,
        target_steps: int,
        output_channels: int,
        config: ResidualHeadConfig,
    ) -> None:
        super().__init__()
        self.target_steps = target_steps
        base_head = _shared_mlp(d_model, output_channels, config)
        if config.identical_horizon_init:
            self.horizon_mlps = nn.ModuleList(
                copy.deepcopy(base_head) for _ in range(target_steps)
            )
        else:
            self.horizon_mlps = nn.ModuleList(
                _shared_mlp(d_model, output_channels, config)
                for _ in range(target_steps)
            )

    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        _validate_target_features(target_features, self.target_steps)
        return torch.stack(
            [
                head(target_features[:, horizon])
                for horizon, head in enumerate(self.horizon_mlps)
            ],
            dim=1,
        )


def _validate_target_features(
    target_features: torch.Tensor,
    target_steps: int,
) -> None:
    if target_features.ndim != 4:
        raise ValueError("target_features must have shape [B, T, H, D]")
    if target_features.shape[1] != target_steps:
        raise ValueError(
            f"target_features time dimension must be {target_steps}, "
            f"got {target_features.shape[1]}"
        )


def build_residual_head(
    *,
    d_model: int,
    target_steps: int,
    output_channels: int,
    config: ResidualHeadConfig,
) -> nn.Module:
    """Build one residual head while preserving legacy module key layouts."""

    config.validate(target_steps=target_steps, output_channels=output_channels)
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    if config.type == "shared_linear":
        # Direct Linear preserves legacy residual_head.weight/bias keys.
        return nn.Linear(d_model, output_channels)
    if config.type == "shared_mlp":
        # Direct Sequential preserves legacy encoder-only residual_head.N keys.
        return _shared_mlp(d_model, output_channels, config)
    if config.type == "multi_horizon_shared_trunk":
        return MultiHorizonSharedTrunkResidualHead(
            d_model,
            target_steps,
            output_channels,
            config,
        )
    return MultiHorizonIndependentMLPResidualHead(
        d_model,
        target_steps,
        output_channels,
        config,
    )


def number_of_horizon_heads(
    config: ResidualHeadConfig,
    target_steps: int,
) -> int:
    return (
        1
        if config.type in {"shared_linear", "shared_mlp"}
        else target_steps
    )


def residual_head_parameter_counts(
    *,
    d_model: int,
    target_steps: int,
    output_channels: int,
    hidden_dim: int,
) -> dict[str, int]:
    """Return parameter counts for all head variants without creating modules."""

    if min(d_model, target_steps, output_channels, hidden_dim) <= 0:
        raise ValueError("Residual head dimensions must be positive")
    final_linear = hidden_dim * output_channels + output_channels
    shared_trunk = 2 * d_model + d_model * hidden_dim + hidden_dim
    shared_mlp = shared_trunk + final_linear
    return {
        "shared_linear": d_model * output_channels + output_channels,
        "shared_mlp": shared_mlp,
        "multi_horizon_shared_trunk": (
            shared_trunk + target_steps * final_linear
        ),
        "multi_horizon_independent_mlp": target_steps * shared_mlp,
    }
