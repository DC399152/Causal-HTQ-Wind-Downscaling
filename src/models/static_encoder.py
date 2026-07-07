"""Static feature encoder for station-level multimodal inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StaticEncoderConfig:
    """Configuration for station-level static feature tokenization."""

    input_dim: int = 17
    d_model: int = 64
    hidden_dim: int = 128
    dropout: float = 0.1
    n_static_tokens: int = 1


class StaticFeatureEncoder(nn.Module):
    """Encode station-level static features into one or more static tokens.

    Input:
    - x_static: [B, C_static]

    Output:
    - static_tokens: [B, N_static, D]
    """

    def __init__(self, config: StaticEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or StaticEncoderConfig()
        self.mlp = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.d_model * self.config.n_static_tokens),
        )

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        if x_static.ndim != 2:
            raise ValueError("x_static must have shape [B, C_static]")
        if x_static.shape[-1] != self.config.input_dim:
            raise ValueError(f"x_static feature dimension must be {self.config.input_dim}")

        batch_size = x_static.shape[0]
        tokens = self.mlp(x_static.to(dtype=torch.float32))
        tokens = tokens.reshape(batch_size, self.config.n_static_tokens, self.config.d_model)
        if not torch.isfinite(tokens).all():
            raise ValueError("StaticFeatureEncoder produced NaN or Inf tokens")
        return tokens
