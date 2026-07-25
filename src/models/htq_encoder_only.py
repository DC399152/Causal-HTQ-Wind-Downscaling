"""Target-token Transformer Encoder-Only model for wind downscaling."""

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
from src.models.physical_height_encoder import PhysicalHeightEncoder
from src.models.static_encoder import StaticEncoderConfig, StaticFeatureEncoder
from src.models.target_token_builder import TargetTokenBuilder
from src.models.tokenizer import HeightTimeTokenizer


@dataclass(frozen=True)
class EncoderOnlyConfig:
    """Configuration for :class:`HTQTargetTokenEncoderOnly`."""

    architecture: str = "htq_target_token_encoder_only"
    name: str = "htq_encoder_only_v1"
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True
    context_hours: int = 12
    target_steps: int = 6
    height_levels: int = 6
    input_channels: int = 2
    output_channels: int = 2
    include_mask_features: bool = True
    include_delta_features: bool = True
    use_physical_height_embedding: bool = True
    physical_height_hidden_dim: int = 64
    height_center_m: float = 300.0
    height_scale_m: float = 100.0
    condition_on_current_height: bool = True
    context_gate_init_bias: float = -1.0
    use_block_attention_mask: bool = True
    allow_target_to_target_attention: bool = True
    residual_head_hidden_dim: int = 64
    residual_head_dropout: float = 0.05
    residual_head_final_weight_std: float = 0.001
    output_head_type: str | None = None
    output_head_hidden_dim: int | None = None
    output_head_dropout: float | None = None
    output_head_final_weight_std: float | None = None
    output_head_identical_horizon_init: bool = True
    output_head_share_across_heights: bool = True
    use_meteo: bool = False
    use_static: bool = False
    meteo_context_hours: int = 12
    meteo_pressure_levels_hpa: tuple[int, ...] = (1000, 975, 950, 925, 900)
    num_meteo_channels: int = 2
    meteo_use_delta: bool = True
    meteo_use_mask_channels: bool = False
    fusion_nhead: int = 8
    fusion_dropout: float = 0.1
    fusion_gate_init_bias: float = -2.0
    static_input_dim: int = 17
    static_n_tokens: int = 1
    static_hidden_dim: int = 128
    static_dropout: float = 0.1


class HTQTargetTokenEncoderOnly(nn.Module):
    """Encode observed and target tokens in one block-masked Transformer."""

    def __init__(self, config: EncoderOnlyConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderOnlyConfig()
        self._validate_config()

        self.tokenizer = HeightTimeTokenizer(
            include_mask_features=self.config.include_mask_features,
            include_delta_features=self.config.include_delta_features,
            d_model=None,
            context_hours=self.config.context_hours,
            height_levels=self.config.height_levels,
            input_channels=self.config.input_channels,
        )
        self.wind_projection = nn.Linear(self.tokenizer.feature_dim, self.config.d_model)
        self.context_hour_embedding = nn.Embedding(self.config.context_hours, self.config.d_model)
        self.input_type_embedding = nn.Embedding(1, self.config.d_model)
        self.physical_height_encoder = PhysicalHeightEncoder(
            d_model=self.config.d_model,
            hidden_dim=self.config.physical_height_hidden_dim,
            height_center_m=self.config.height_center_m,
            height_scale_m=self.config.height_scale_m,
        )
        self.target_token_builder = TargetTokenBuilder(
            d_model=self.config.d_model,
            target_steps=self.config.target_steps,
            height_levels=self.config.height_levels,
            condition_on_current_height=self.config.condition_on_current_height,
            context_gate_init_bias=self.config.context_gate_init_bias,
        )

        self.meteo_encoder = None
        if self.config.use_meteo:
            self.meteo_encoder = MeteoPressureLevelEncoder(
                MeteoEncoderConfig(
                    d_model=self.config.d_model,
                    context_hours=self.config.meteo_context_hours,
                    num_pressure_levels=len(self.config.meteo_pressure_levels_hpa),
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation=self.config.activation,
            batch_first=True,
            norm_first=self.config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_encoder_layers,
            norm=nn.LayerNorm(self.config.d_model),
            enable_nested_tensor=False,
        )
        self.output_head_config = residual_head_config_from_model_fields(
            output_head_type=self.config.output_head_type,
            output_head_hidden_dim=self.config.output_head_hidden_dim,
            output_head_dropout=self.config.output_head_dropout,
            output_head_final_weight_std=self.config.output_head_final_weight_std,
            output_head_identical_horizon_init=(
                self.config.output_head_identical_horizon_init
            ),
            output_head_share_across_heights=(
                self.config.output_head_share_across_heights
            ),
            default_type="shared_mlp",
            default_hidden_dim=self.config.residual_head_hidden_dim,
            default_dropout=self.config.residual_head_dropout,
            default_final_weight_std=self.config.residual_head_final_weight_std,
        )
        self.residual_head = build_residual_head(
            d_model=self.config.d_model,
            target_steps=self.config.target_steps,
            output_channels=self.config.output_channels,
            config=self.output_head_config,
        )

    @property
    def input_token_count(self) -> int:
        return self.config.context_hours * self.config.height_levels

    @property
    def target_token_count(self) -> int:
        return self.config.target_steps * self.config.height_levels

    @staticmethod
    def build_attention_mask(
        input_token_count: int,
        target_token_count: int,
        *,
        use_block_attention_mask: bool,
        allow_target_to_target_attention: bool,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build a bool mask where ``True`` means attention is forbidden."""

        if not use_block_attention_mask:
            return None
        total = input_token_count + target_token_count
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:input_token_count, input_token_count:] = True
        if not allow_target_to_target_attention:
            target_slice = slice(input_token_count, total)
            mask[target_slice, target_slice] = True
            target_ids = torch.arange(input_token_count, total, device=device)
            mask[target_ids, target_ids] = False
        return mask

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
    ) -> dict[str, torch.Tensor | dict[str, object] | None]:
        """Return residual and prediction tensors with shape ``[B, T, H, C]``."""

        self._validate_inputs(x_hourly, x_mask, current_hourly_reference, height_values)
        batch_size, context_hours, height_levels, _ = x_hourly.shape
        if height_values is None or current_hourly_reference is None:
            raise RuntimeError("Validated required inputs are unexpectedly missing")

        tokenized = self.tokenizer(x_hourly, x_mask)
        token_features = tokenized.token_features
        wind_4d = self.wind_projection(token_features)
        hour_ids = torch.arange(context_hours, device=x_hourly.device)
        hour_embedding = self.context_hour_embedding(hour_ids).view(
            1, context_hours, 1, self.config.d_model
        )
        physical_height = self.physical_height_encoder(height_values)
        input_type = self.input_type_embedding.weight.view(1, 1, 1, self.config.d_model)
        wind_4d = wind_4d + hour_embedding + physical_height[:, None, :, :] + input_type
        wind_tokens = wind_4d.reshape(batch_size, self.input_token_count, self.config.d_model)

        aux_tokens: list[torch.Tensor] = []
        aux_padding_masks: list[torch.Tensor] = []
        if self.config.use_meteo:
            if x_meteo is None or meteo_mask is None:
                raise ValueError("use_meteo=True requires x_meteo and meteo_mask")
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
                torch.zeros(static_tokens.shape[:2], dtype=torch.bool, device=x_hourly.device)
            )

        fusion_info = None
        if aux_tokens:
            if self.fusion is None:
                raise RuntimeError("Multimodal fusion module is not initialized")
            wind_tokens, fusion_info = self.fusion(
                wind_tokens,
                torch.cat(aux_tokens, dim=1),
                torch.cat(aux_padding_masks, dim=1),
            )

        fused_4d = wind_tokens.reshape(
            batch_size, context_hours, height_levels, self.config.d_model
        )
        target_tokens = self.target_token_builder(
            physical_height,
            current_height_context=fused_4d[:, -1]
            if self.config.condition_on_current_height
            else None,
        )
        all_tokens = torch.cat([wind_tokens, target_tokens], dim=1)

        input_valid = tokenized.token_valid.reshape(batch_size, self.input_token_count)
        target_valid = torch.ones(
            (batch_size, self.target_token_count), dtype=torch.bool, device=x_hourly.device
        )
        key_padding_mask = ~torch.cat([input_valid, target_valid], dim=1)
        attention_mask = self.build_attention_mask(
            self.input_token_count,
            self.target_token_count,
            use_block_attention_mask=self.config.use_block_attention_mask,
            allow_target_to_target_attention=self.config.allow_target_to_target_attention,
            device=x_hourly.device,
        )
        encoded = self.encoder(
            all_tokens,
            mask=attention_mask,
            src_key_padding_mask=key_padding_mask,
        )
        target_features_flat = encoded[:, self.input_token_count :, :]
        target_features = target_features_flat.reshape(
            batch_size,
            self.config.target_steps,
            height_levels,
            self.config.d_model,
        )
        residual = self.residual_head(target_features)
        pred = current_hourly_reference.unsqueeze(1) + residual
        if not torch.isfinite(pred).all():
            raise ValueError("HTQTargetTokenEncoderOnly produced NaN or Inf predictions")
        return {
            "pred": pred,
            "residual": residual,
            "target_features": target_features,
            "encoder_memory": encoded[:, : self.input_token_count, :],
            "fusion_info": fusion_info,
        }

    def _validate_config(self) -> None:
        if self.config.architecture != "htq_target_token_encoder_only":
            raise ValueError(f"Unsupported encoder-only architecture {self.config.architecture!r}")
        for name in (
            "d_model", "nhead", "num_encoder_layers", "dim_feedforward",
            "context_hours", "target_steps", "height_levels", "input_channels",
            "output_channels", "residual_head_hidden_dim",
        ):
            if int(getattr(self.config, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.config.d_model % self.config.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.config.d_model % self.config.fusion_nhead != 0:
            raise ValueError("d_model must be divisible by fusion_nhead")
        if self.config.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")
        if not self.config.use_physical_height_embedding:
            raise ValueError("Encoder-only model requires use_physical_height_embedding=True")
        if self.config.residual_head_final_weight_std < 0:
            raise ValueError("residual_head_final_weight_std must be non-negative")

    def _validate_inputs(
        self,
        x_hourly: torch.Tensor,
        x_mask: torch.Tensor,
        current_hourly_reference: torch.Tensor | None,
        height_values: torch.Tensor | None,
    ) -> None:
        expected = (
            self.config.context_hours,
            self.config.height_levels,
            self.config.input_channels,
        )
        if x_hourly.ndim != 4 or tuple(x_hourly.shape[1:]) != expected:
            raise ValueError(f"x_hourly must have shape [B, {expected[0]}, {expected[1]}, {expected[2]}]")
        if x_mask.shape != x_hourly.shape:
            raise ValueError("x_mask must have the same shape as x_hourly")
        expected_reference = (x_hourly.shape[0], self.config.height_levels, self.config.output_channels)
        if current_hourly_reference is None or tuple(current_hourly_reference.shape) != expected_reference:
            raise ValueError(f"current_hourly_reference must have shape {expected_reference}")
        expected_heights = (x_hourly.shape[0], self.config.height_levels)
        if height_values is None or tuple(height_values.shape) != expected_heights:
            raise ValueError(f"height_values must have shape {expected_heights}")
