"""Minimal Causal Height-Time Query Transformer forward pass."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.query_builder import TargetQueryBuilder
from src.models.tokenizer import HeightTimeTokenizer


@dataclass(frozen=True)
class HTQConfig:
    """Shape and module defaults for CausalHTQTransformer."""

    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    context_hours: int = 6
    target_steps: int = 6
    height_levels: int = 6
    input_channels: int = 2
    output_channels: int = 2


class CausalHTQTransformer(nn.Module):
    """Causal HTQ-Transformer with semantic [B, L, H, C] tensors.

    Forward inputs
    --------------
    x_hourly:
        [B, L=6, H=6, 2], normalized hourly context.
    x_mask:
        [B, L=6, H=6, 2], True for valid values and False for invalid values.
    """

    def __init__(self, config: HTQConfig | None = None) -> None:
        super().__init__()
        self.config = config or HTQConfig()

        self.tokenizer = HeightTimeTokenizer(
            include_mask_features=True,
            include_delta_features=True,
            d_model=self.config.d_model,
            context_hours=self.config.context_hours,
            height_levels=self.config.height_levels,
            input_channels=self.config.input_channels,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_encoder_layers,
        )

        self.query_builder = TargetQueryBuilder(
            d_model=self.config.d_model,
            target_steps=self.config.target_steps,
            height_levels=self.config.height_levels,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.config.num_decoder_layers,
        )
        self.residual_head = nn.Linear(self.config.d_model, self.config.output_channels)

    def forward(self, x_hourly: torch.Tensor, x_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run minimal HTQ forward.

        Returns
        -------
        dict
            pred: [B, T_out=6, H=6, 2]
            residual: [B, T_out=6, H=6, 2].
            encoder_memory: [B, L*H=36, d_model=64]
        """

        self._validate_inputs(x_hourly, x_mask)
        batch_size, context_hours, height_levels, _ = x_hourly.shape

        tokenized = self.tokenizer(x_hourly, x_mask)
        if tokenized.token_embeddings is None:
            raise RuntimeError("HTQ tokenizer must be configured with d_model")

        # encoder_input: [B, L*H, d_model].
        encoder_input = tokenized.token_embeddings
        token_valid = tokenized.token_valid.reshape(batch_size, context_hours * height_levels)
        src_key_padding_mask = ~token_valid
        encoder_memory = self.encoder(
            encoder_input,
            src_key_padding_mask=src_key_padding_mask,
        )

        # target_queries: [B, T_out*H, d_model].
        target_queries = self.query_builder(
            batch_size=batch_size,
            height_levels=height_levels,
            device=x_hourly.device,
        )
        decoded = self.decoder(
            tgt=target_queries,
            memory=encoder_memory,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # residual: [B, T_out, H, 2].
        residual = self.residual_head(decoded)
        residual = residual.reshape(
            batch_size,
            self.config.target_steps,
            height_levels,
            self.config.output_channels,
        )

        # pred: current hourly profile plus learned intra-hour residual.
        current_hourly = x_hourly[:, -1]
        pred = current_hourly.unsqueeze(1) + residual
        return {
            "pred": pred,
            "residual": residual,
            "encoder_memory": encoder_memory,
        }

    def _validate_inputs(self, x_hourly: torch.Tensor, x_mask: torch.Tensor) -> None:
        if x_hourly.ndim != 4:
            raise ValueError("x_hourly must have shape [B, L, H, C]")
        if x_mask.ndim != 4:
            raise ValueError("x_mask must have shape [B, L, H, C]")
        if x_mask.shape != x_hourly.shape:
            raise ValueError("x_mask must have the same shape as x_hourly")
        _, context_hours, height_levels, channels = x_hourly.shape
        if context_hours != self.config.context_hours:
            raise ValueError(f"x_hourly context dimension must be {self.config.context_hours}")
        if height_levels != self.config.height_levels:
            raise ValueError(f"x_hourly height dimension must be {self.config.height_levels}")
        if channels != self.config.input_channels:
            raise ValueError(f"x_hourly channel dimension must be {self.config.input_channels}")
