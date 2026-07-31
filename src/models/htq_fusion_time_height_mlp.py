"""Fusion-fronted time-height MLP for deterministic wind downscaling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.meteo_encoder import MeteoEncoderConfig, MeteoPressureLevelEncoder
from src.models.multimodal_fusion import GatedCrossAttentionFusion
from src.models.output_heads import (
    build_residual_head,
    residual_head_config_from_model_fields,
)
from src.models.static_encoder import StaticEncoderConfig, StaticFeatureEncoder
from src.models.time_height_mixer import (
    TemporalTargetProjection,
    TimeHeightMixerBlock,
)
from src.models.tokenizer import HeightTimeTokenizer


@dataclass(frozen=True)
class FusionTimeHeightMLPConfig:
    """Configuration for :class:`HTQFusionTimeHeightMLP`."""

    architecture: str = "htq_fusion_time_height_mlp"
    name: str = "htq_fusion_time_height_mlp_v1"
    context_hours: int = 12
    target_steps: int = 6
    height_levels: int = 6
    input_channels: int = 2
    output_channels: int = 2
    d_model: int = 128
    mlp_d_model: int = 96
    num_mixer_blocks: int = 3
    temporal_mixing_hidden_dim: int = 32
    height_mixing_hidden_dim: int = 16
    channel_expansion_ratio: int = 4
    mixer_dropout: float = 0.1
    target_projection_hidden_dim: int = 24
    target_projection_dropout: float = 0.05
    use_target_time_embedding: bool = True
    include_mask_features: bool = True
    include_delta_features: bool = True
    use_meteo: bool = False
    use_static: bool = False
    meteo_context_hours: int = 12
    meteo_pressure_levels_hpa: tuple[int, ...] = (1000, 975, 950, 925, 900)
    num_meteo_channels: int = 2
    meteo_use_delta: bool = True
    meteo_use_mask_channels: bool = False
    fusion_nhead: int = 4
    fusion_dropout: float = 0.1
    fusion_gate_init_bias: float = -2.0
    static_input_dim: int = 17
    static_n_tokens: int = 1
    static_hidden_dim: int = 128
    static_dropout: float = 0.1
    output_head_type: str | None = None
    output_head_hidden_dim: int | None = None
    output_head_dropout: float | None = None
    output_head_final_weight_std: float | None = None
    output_head_identical_horizon_init: bool = True
    output_head_share_across_heights: bool = True


class HTQFusionTimeHeightMLP(nn.Module):
    """Reuse the existing multimodal front-end and replace HTQ with MLP mixers."""

    def __init__(self, config: FusionTimeHeightMLPConfig | None = None) -> None:
        super().__init__()
        self.config = config or FusionTimeHeightMLPConfig()
        self._validate_config()

        self.tokenizer = HeightTimeTokenizer(
            include_mask_features=self.config.include_mask_features,
            include_delta_features=self.config.include_delta_features,
            d_model=self.config.d_model,
            context_hours=self.config.context_hours,
            height_levels=self.config.height_levels,
            input_channels=self.config.input_channels,
        )

        self.meteo_encoder = None
        if self.config.use_meteo:
            self.meteo_encoder = MeteoPressureLevelEncoder(
                MeteoEncoderConfig(
                    d_model=self.config.d_model,
                    context_hours=self.config.meteo_context_hours,
                    num_pressure_levels=len(
                        self.config.meteo_pressure_levels_hpa
                    ),
                    num_meteo_channels=self.config.num_meteo_channels,
                    pressure_levels_hpa=self.config.meteo_pressure_levels_hpa,
                    use_delta=self.config.meteo_use_delta,
                    use_mask_channels=self.config.meteo_use_mask_channels,
                )
            )

        self.static_encoder = None
        if self.config.use_static:
            self.static_encoder = StaticFeatureEncoder(
                StaticEncoderConfig(
                    input_dim=self.config.static_input_dim,
                    d_model=self.config.d_model,
                    hidden_dim=self.config.static_hidden_dim,
                    dropout=self.config.static_dropout,
                    n_static_tokens=self.config.static_n_tokens,
                )
            )

        self.fusion = None
        if self.config.use_meteo or self.config.use_static:
            self.fusion = GatedCrossAttentionFusion(
                d_model=self.config.d_model,
                nhead=self.config.fusion_nhead,
                dropout=self.config.fusion_dropout,
                gate_init_bias=self.config.fusion_gate_init_bias,
            )

        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self.config.mlp_d_model),
        )
        self.mixer_blocks = nn.ModuleList(
            [
                TimeHeightMixerBlock(
                    context_hours=self.config.context_hours,
                    height_levels=self.config.height_levels,
                    d_model=self.config.mlp_d_model,
                    temporal_hidden_dim=(
                        self.config.temporal_mixing_hidden_dim
                    ),
                    height_hidden_dim=self.config.height_mixing_hidden_dim,
                    channel_expansion_ratio=(
                        self.config.channel_expansion_ratio
                    ),
                    dropout=self.config.mixer_dropout,
                )
                for _ in range(self.config.num_mixer_blocks)
            ]
        )
        self.backbone_output_norm = nn.LayerNorm(self.config.mlp_d_model)
        self.target_projection = TemporalTargetProjection(
            context_hours=self.config.context_hours,
            target_steps=self.config.target_steps,
            height_levels=self.config.height_levels,
            d_model=self.config.mlp_d_model,
            hidden_dim=self.config.target_projection_hidden_dim,
            dropout=self.config.target_projection_dropout,
        )
        self.target_time_embedding = (
            nn.Embedding(self.config.target_steps, self.config.mlp_d_model)
            if self.config.use_target_time_embedding
            else None
        )
        if self.target_time_embedding is not None:
            nn.init.normal_(
                self.target_time_embedding.weight,
                mean=0.0,
                std=0.02,
            )

        self.output_head_config = residual_head_config_from_model_fields(
            output_head_type=self.config.output_head_type,
            output_head_hidden_dim=self.config.output_head_hidden_dim,
            output_head_dropout=self.config.output_head_dropout,
            output_head_final_weight_std=(
                self.config.output_head_final_weight_std
            ),
            output_head_identical_horizon_init=(
                self.config.output_head_identical_horizon_init
            ),
            output_head_share_across_heights=(
                self.config.output_head_share_across_heights
            ),
            default_type="multi_horizon_shared_trunk",
            default_hidden_dim=64,
            default_dropout=0.05,
            default_final_weight_std=0.001,
        )
        self.residual_head = build_residual_head(
            d_model=self.config.mlp_d_model,
            target_steps=self.config.target_steps,
            output_channels=self.config.output_channels,
            config=self.output_head_config,
        )

    def forward(
        self,
        x_hourly: torch.Tensor,
        x_mask: torch.Tensor,
        x_meteo: torch.Tensor | None = None,
        meteo_mask: torch.Tensor | None = None,
        x_static: torch.Tensor | None = None,
        current_hourly_reference: torch.Tensor | None = None,
        height_values: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict[str, object]:
        """Return residual and prediction tensors with shape ``[B, T, H, C]``."""

        del height_values  # Kept for the shared train/evaluate call signature.
        self._validate_inputs(x_hourly, x_mask)
        batch_size, context_hours, height_levels, _ = x_hourly.shape
        tokenized = self.tokenizer(x_hourly, x_mask)
        if tokenized.token_embeddings is None:
            raise RuntimeError("Tokenizer must produce projected wind tokens")

        wind_tokens = tokenized.token_embeddings
        aux_tokens: list[torch.Tensor] = []
        aux_padding_masks: list[torch.Tensor] = []
        if self.config.use_meteo:
            if x_meteo is None or meteo_mask is None:
                raise ValueError(
                    "use_meteo=True requires x_meteo and meteo_mask"
                )
            if self.meteo_encoder is None:
                raise RuntimeError("Meteo encoder is not initialized")
            meteo_tokens = self.meteo_encoder(x_meteo, meteo_mask)
            aux_tokens.append(meteo_tokens)
            meteo_valid = meteo_mask.any(dim=-1).reshape(batch_size, -1)
            aux_padding_masks.append(~meteo_valid)

        if self.config.use_static:
            if x_static is None:
                raise ValueError("use_static=True requires x_static")
            if self.static_encoder is None:
                raise RuntimeError("Static encoder is not initialized")
            static_tokens = self.static_encoder(x_static)
            aux_tokens.append(static_tokens)
            aux_padding_masks.append(
                torch.zeros(
                    static_tokens.shape[:2],
                    dtype=torch.bool,
                    device=static_tokens.device,
                )
            )

        fusion_info = None
        fused_tokens = wind_tokens
        if aux_tokens:
            if self.fusion is None:
                raise RuntimeError("Multimodal fusion module is not initialized")
            fused_tokens, fusion_info = self.fusion(
                wind_tokens,
                torch.cat(aux_tokens, dim=1),
                torch.cat(aux_padding_masks, dim=1),
            )

        token_valid = tokenized.token_valid.unsqueeze(-1)
        fused_features = fused_tokens.reshape(
            batch_size,
            context_hours,
            height_levels,
            self.config.d_model,
        )
        fused_features = torch.where(
            token_valid,
            fused_features,
            torch.zeros_like(fused_features),
        )
        context_features = self.input_projection(fused_features)
        context_features = torch.where(
            token_valid,
            context_features,
            torch.zeros_like(context_features),
        )

        block_outputs: list[torch.Tensor] = []
        for block in self.mixer_blocks:
            context_features = block(context_features)
            context_features = torch.where(
                token_valid,
                context_features,
                torch.zeros_like(context_features),
            )
            if return_features:
                block_outputs.append(context_features)
        context_features = self.backbone_output_norm(context_features)
        context_features = torch.where(
            token_valid,
            context_features,
            torch.zeros_like(context_features),
        )

        target_projection_output = self.target_projection(context_features)
        target_features = target_projection_output
        if self.target_time_embedding is not None:
            time_ids = torch.arange(
                self.config.target_steps,
                device=target_features.device,
            )
            target_features = target_features + self.target_time_embedding(
                time_ids
            ).view(1, self.config.target_steps, 1, self.config.mlp_d_model)

        residual = self.residual_head(target_features)
        current_hourly = (
            current_hourly_reference
            if current_hourly_reference is not None
            else x_hourly[:, -1]
        )
        if current_hourly.shape != x_hourly[:, -1].shape:
            raise ValueError(
                "current_hourly_reference must have shape [B, H, C], "
                f"got {tuple(current_hourly.shape)}"
            )
        prediction = current_hourly.unsqueeze(1) + residual
        if not torch.isfinite(prediction).all():
            raise ValueError(
                "HTQFusionTimeHeightMLP produced NaN or Inf predictions"
            )

        output: dict[str, object] = {
            "pred": prediction,
            "residual": residual,
            "fusion_info": fusion_info,
        }
        if return_features:
            output.update(
                {
                    "fused_features": fused_features,
                    "context_features": context_features,
                    "target_projection_output": target_projection_output,
                    "target_features": target_features,
                    "mixer_block_outputs": block_outputs,
                }
            )
        return output

    def _validate_config(self) -> None:
        if self.config.architecture != "htq_fusion_time_height_mlp":
            raise ValueError(
                "architecture must be 'htq_fusion_time_height_mlp'"
            )
        positive = {
            "context_hours": self.config.context_hours,
            "target_steps": self.config.target_steps,
            "height_levels": self.config.height_levels,
            "input_channels": self.config.input_channels,
            "output_channels": self.config.output_channels,
            "d_model": self.config.d_model,
            "mlp_d_model": self.config.mlp_d_model,
            "num_mixer_blocks": self.config.num_mixer_blocks,
            "temporal_mixing_hidden_dim": (
                self.config.temporal_mixing_hidden_dim
            ),
            "height_mixing_hidden_dim": self.config.height_mixing_hidden_dim,
            "channel_expansion_ratio": self.config.channel_expansion_ratio,
            "target_projection_hidden_dim": (
                self.config.target_projection_hidden_dim
            ),
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "mixer_dropout": self.config.mixer_dropout,
            "target_projection_dropout": (
                self.config.target_projection_dropout
            ),
            "fusion_dropout": self.config.fusion_dropout,
            "static_dropout": self.config.static_dropout,
        }.items():
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if (
            self.config.use_meteo or self.config.use_static
        ) and self.config.d_model % self.config.fusion_nhead != 0:
            raise ValueError("d_model must be divisible by fusion_nhead")
        if (
            self.config.use_meteo
            and self.config.meteo_context_hours != self.config.context_hours
        ):
            raise ValueError(
                "meteo_context_hours must equal context_hours"
            )

    def _validate_inputs(
        self,
        x_hourly: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> None:
        if x_hourly.ndim != 4:
            raise ValueError("x_hourly must have shape [B, L, H, C]")
        if x_mask.shape != x_hourly.shape:
            raise ValueError("x_mask must have the same shape as x_hourly")
        expected = (
            self.config.context_hours,
            self.config.height_levels,
            self.config.input_channels,
        )
        if tuple(x_hourly.shape[1:]) != expected:
            raise ValueError(
                f"x_hourly must have trailing shape {expected}, "
                f"got {tuple(x_hourly.shape[1:])}"
            )
