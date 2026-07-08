"""Mask-aware height-time tokenization for hourly wind profiles."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TokenizedHeightTime:
    """Tokenized hourly context.

    Attributes
    ----------
    token_features:
        [B, L, H, C_value + C_mask] when ``include_mask_features=True``.
    token_valid:
        [B, L, H], True when at least one channel is valid.
    """

    token_features: object
    token_valid: object
    token_embeddings: object | None = None


class HeightTimeTokenizer(nn.Module):
    """Prepare [B, L, H, C] hourly profiles for HTQ tokenization.

    Invalid value positions should already be filled with 0.0 by the Dataset.
    This tokenizer keeps the corresponding mask explicit so the model does not
    confuse a missing placeholder with a physical zero wind component.
    """

    def __init__(
        self,
        include_mask_features: bool = True,
        include_delta_features: bool = False,
        d_model: int | None = None,
        context_hours: int = 6,
        height_levels: int = 6,
        input_channels: int = 2,
    ) -> None:
        super().__init__()
        self.include_mask_features = include_mask_features
        self.include_delta_features = include_delta_features
        self.d_model = d_model
        self.context_hours = context_hours
        self.height_levels = height_levels
        self.input_channels = input_channels

        feature_dim = input_channels
        if include_delta_features:
            feature_dim += input_channels
        if include_mask_features:
            feature_dim += input_channels
        self.feature_dim = feature_dim

        if d_model is None:
            self.projection = None
            self.hour_embedding = None
            self.height_embedding = None
        else:
            self.projection = nn.Linear(feature_dim, d_model)
            self.hour_embedding = nn.Embedding(context_hours, d_model)
            self.height_embedding = nn.Embedding(height_levels, d_model)

    def forward(self, x_hourly, x_mask=None) -> TokenizedHeightTime:
        """Return token features and token-level validity mask.

        Parameters
        ----------
        x_hourly:
            [B, L, H, C] normalized hourly input.
        x_mask:
            [B, L, H, C] boolean validity mask. Required when
            ``include_mask_features=True``.
        """

        if x_hourly.ndim != 4:
            raise ValueError("x_hourly must have shape [B, L, H, C]")
        batch_size, context_hours, height_levels, channels = x_hourly.shape
        if channels != self.input_channels:
            raise ValueError(f"x_hourly last dimension must be {self.input_channels}")
        if context_hours > self.context_hours:
            raise ValueError("x_hourly has more context hours than configured")
        if height_levels > self.height_levels:
            raise ValueError("x_hourly has more height levels than configured")

        if x_mask is None:
            token_valid = _ones_like_token_grid(x_hourly)
            feature_parts = [x_hourly]
        else:
            if x_mask.shape != x_hourly.shape:
                raise ValueError("x_mask must have the same shape as x_hourly")
            token_valid = x_mask.any(dim=-1)
            feature_parts = [x_hourly]

        if self.include_delta_features:
            # delta: [B, L, H, C], with delta[:, 0] fixed to zero.
            delta = torch.zeros_like(x_hourly)
            raw_delta = x_hourly[:, 1:] - x_hourly[:, :-1]
            if x_mask is None:
                delta[:, 1:] = raw_delta
            else:
                valid_delta = x_mask[:, 1:] & x_mask[:, :-1]
                delta[:, 1:] = torch.where(valid_delta, raw_delta, torch.zeros_like(raw_delta))
            feature_parts.append(delta)

        if self.include_mask_features:
            if x_mask is None:
                mask_features = torch.ones_like(x_hourly)
            else:
                mask_features = x_mask.to(dtype=x_hourly.dtype)
            feature_parts.append(mask_features)

        # token_features: [B, L, H, F], where F is 6 for [u, v, du, dv, mask_u, mask_v].
        token_features = torch.cat(feature_parts, dim=-1)

        token_embeddings = None
        if self.projection is not None:
            projected = self.projection(token_features)
            hour_ids = torch.arange(context_hours, device=x_hourly.device)
            height_ids = torch.arange(height_levels, device=x_hourly.device)
            hour_emb = self.hour_embedding(hour_ids).view(1, context_hours, 1, self.d_model)
            height_emb = self.height_embedding(height_ids).view(1, 1, height_levels, self.d_model)
            embedded = projected + hour_emb + height_emb
            token_embeddings = embedded.reshape(batch_size, context_hours * height_levels, self.d_model)

        return TokenizedHeightTime(
            token_features=token_features,
            token_valid=token_valid,
            token_embeddings=token_embeddings,
        )


def _ones_like_token_grid(x_hourly):
    return torch.ones(
        x_hourly.shape[:-1],
        dtype=torch.bool,
        device=x_hourly.device,
    )
