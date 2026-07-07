"""Pressure-level ERA5 token encoder for multimodal HTQ input."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MeteoEncoderConfig:
    """Configuration for pressure-level meteorological tokens."""

    d_model: int = 64
    context_hours: int = 6
    num_pressure_levels: int = 5
    num_meteo_channels: int = 2
    pressure_levels_hpa: tuple[int, ...] = (1000, 975, 950, 925, 900)
    use_delta: bool = True
    use_mask_channels: bool = False


class MeteoPressureLevelEncoder(nn.Module):
    """Encode ERA5 station-interpolated pressure-level inputs.

    Inputs use semantic shape [B, L, P, C_m], where L is hourly context,
    P is pressure level, and C_m is meteorological channel count.
    Pressure levels remain pressure-level tokens; no conversion to lidar height
    is performed.
    """

    def __init__(self, config: MeteoEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or MeteoEncoderConfig()

        feature_dim = self.config.num_meteo_channels
        if self.config.use_delta:
            feature_dim += self.config.num_meteo_channels
        if self.config.use_mask_channels:
            feature_dim += self.config.num_meteo_channels
        self.feature_dim = feature_dim

        self.projection = nn.Linear(feature_dim, self.config.d_model)
        self.hour_embedding = nn.Embedding(self.config.context_hours, self.config.d_model)
        self.pressure_level_embedding = nn.Embedding(
            self.config.num_pressure_levels,
            self.config.d_model,
        )

    def forward(
        self,
        x_meteo: torch.Tensor,
        meteo_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return meteo tokens with shape [B, L*P, d_model]."""

        self._validate_inputs(x_meteo, meteo_mask)
        batch_size, context_hours, num_pressure_levels, _ = x_meteo.shape

        values = torch.nan_to_num(x_meteo, nan=0.0, posinf=0.0, neginf=0.0)
        if meteo_mask is not None:
            values = torch.where(meteo_mask, values, torch.zeros_like(values))
        feature_parts = [values]
        if self.config.use_delta:
            # delta_meteo: [B, L, P, C_m], with delta[:, 0] fixed to zero.
            delta = torch.zeros_like(values)
            delta[:, 1:] = values[:, 1:] - values[:, :-1]
            if meteo_mask is not None:
                delta_valid = meteo_mask[:, 1:] & meteo_mask[:, :-1]
                delta[:, 1:] = torch.where(delta_valid, delta[:, 1:], torch.zeros_like(delta[:, 1:]))
            feature_parts.append(delta)

        if self.config.use_mask_channels:
            if meteo_mask is None:
                mask_features = torch.ones_like(feature_parts[0])
            else:
                mask_features = meteo_mask.to(dtype=x_meteo.dtype)
            feature_parts.append(mask_features)

        # token_features: [B, L, P, F], default F=4 for [temp, humidity, dtemp, dhumidity].
        token_features = torch.cat(feature_parts, dim=-1)
        projected = self.projection(token_features)

        hour_ids = torch.arange(context_hours, device=x_meteo.device)
        pressure_ids = torch.arange(num_pressure_levels, device=x_meteo.device)
        hour_emb = self.hour_embedding(hour_ids).view(1, context_hours, 1, self.config.d_model)
        pressure_emb = self.pressure_level_embedding(pressure_ids).view(
            1,
            1,
            num_pressure_levels,
            self.config.d_model,
        )
        embedded = projected + hour_emb + pressure_emb
        meteo_tokens = embedded.reshape(
            batch_size,
            context_hours * num_pressure_levels,
            self.config.d_model,
        )
        return torch.nan_to_num(meteo_tokens, nan=0.0, posinf=0.0, neginf=0.0)

    def _validate_inputs(
        self,
        x_meteo: torch.Tensor,
        meteo_mask: torch.Tensor | None,
    ) -> None:
        if x_meteo.ndim != 4:
            raise ValueError("x_meteo must have shape [B, L, P, C_m]")
        _, context_hours, num_pressure_levels, channels = x_meteo.shape
        if context_hours > self.config.context_hours:
            raise ValueError("x_meteo has more context hours than configured")
        if num_pressure_levels > self.config.num_pressure_levels:
            raise ValueError("x_meteo has more pressure levels than configured")
        if channels != self.config.num_meteo_channels:
            raise ValueError(
                f"x_meteo channel dimension must be {self.config.num_meteo_channels}"
            )
        if meteo_mask is not None and meteo_mask.shape != x_meteo.shape:
            raise ValueError("meteo_mask must have the same shape as x_meteo")
