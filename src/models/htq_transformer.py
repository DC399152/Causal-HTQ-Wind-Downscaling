"""Minimal Causal Height-Time Query Transformer forward pass."""

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
from src.models.query_builder import ContextConditionedQueryBuilder, FixedTargetQueryBuilder
from src.models.static_encoder import StaticEncoderConfig, StaticFeatureEncoder
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
    enforce_zero_mean_residual: bool = False
    use_meteo: bool = False
    use_static: bool = False
    meteo_context_hours: int = 6
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
    query_builder_type: str = "context_conditioned"
    query_use_context_projection: bool = True
    query_use_context_layernorm: bool = True
    query_use_temporal_context: bool = False
    query_use_multiscale_trend: bool = False
    query_trend_scales: tuple[int, ...] = (1, 3, 5)
    query_use_trend_context: bool = False
    output_head_type: str | None = None
    output_head_hidden_dim: int | None = None
    output_head_dropout: float | None = None
    output_head_final_weight_std: float | None = None
    output_head_identical_horizon_init: bool = True
    output_head_share_across_heights: bool = True


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
        if self.config.enforce_zero_mean_residual:
            raise ValueError(
                "enforce_zero_mean_residual=True is disabled for this task. "
                "Residuals must be allowed to have non-zero target-time mean."
            )

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

        if self.config.query_builder_type == "fixed":
            self.query_builder = FixedTargetQueryBuilder(
                d_model=self.config.d_model,
                target_steps=self.config.target_steps,
                height_levels=self.config.height_levels,
            )
        elif self.config.query_builder_type == "context_conditioned":
            self.query_builder = ContextConditionedQueryBuilder(
                d_model=self.config.d_model,
                target_steps=self.config.target_steps,
                context_hours=self.config.context_hours,
                height_levels=self.config.height_levels,
                use_context_projection=self.config.query_use_context_projection,
                use_context_layernorm=self.config.query_use_context_layernorm,
                use_temporal_context=self.config.query_use_temporal_context,
                use_multiscale_trend=self.config.query_use_multiscale_trend,
                trend_scales=self.config.query_trend_scales,
                use_trend_context=self.config.query_use_trend_context,
            )
        else:
            raise ValueError(
                "query_builder_type must be 'fixed' or 'context_conditioned', "
                f"got {self.config.query_builder_type!r}"
            )

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
        else:
            self.meteo_encoder = None

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
        else:
            self.static_encoder = None

        if self.config.use_meteo or self.config.use_static:
            self.fusion = GatedCrossAttentionFusion(
                d_model=self.config.d_model,
                nhead=self.config.fusion_nhead,
                dropout=self.config.fusion_dropout,
                gate_init_bias=self.config.fusion_gate_init_bias,
            )
        else:
            self.fusion = None

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
            default_type="shared_linear",
            default_hidden_dim=64,
            default_dropout=0.05,
            default_final_weight_std=0.001,
        )
        self.residual_head = build_residual_head(
            d_model=self.config.d_model,
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
    ) -> dict[str, torch.Tensor | dict[str, object] | None]:
        """Run minimal HTQ forward.

        Returns
        -------
        dict
            pred: [B, T_out=6, H=6, 2]
            residual: [B, T_out=6, H=6, 2].
            encoder_memory: [B, L*H=36, d_model=64]
            fusion_info: gate diagnostics when multimodal fusion is used, else None.
        """

        self._validate_inputs(x_hourly, x_mask)
        batch_size, context_hours, height_levels, _ = x_hourly.shape

        tokenized = self.tokenizer(x_hourly, x_mask)
        if tokenized.token_embeddings is None:
            raise RuntimeError("HTQ tokenizer must be configured with d_model")

        # encoder_input: [B, L*H, d_model].
        wind_tokens = tokenized.token_embeddings
        fusion_info = None
        aux_tokens_list: list[torch.Tensor] = []
        aux_padding_masks: list[torch.Tensor] = []
        if self.config.use_meteo:
            if x_meteo is None:
                raise ValueError("HTQConfig.use_meteo=True but x_meteo is missing from the batch")
            if meteo_mask is None:
                raise ValueError("HTQConfig.use_meteo=True but meteo_mask is missing from the batch")
            if self.meteo_encoder is None:
                raise RuntimeError("Meteo encoder is not initialized")
            meteo_tokens = self.meteo_encoder(x_meteo, meteo_mask)
            aux_tokens_list.append(meteo_tokens)
            # meteo_token_valid: [B, L, P], True if any meteo channel is valid.
            meteo_token_valid = meteo_mask.any(dim=-1)
            aux_padding_masks.append(~meteo_token_valid.reshape(
                meteo_token_valid.shape[0],
                meteo_token_valid.shape[1] * meteo_token_valid.shape[2],
            ))

        if self.config.use_static:
            if x_static is None:
                raise ValueError("HTQConfig.use_static=True but x_static is missing from the batch")
            if self.static_encoder is None:
                raise RuntimeError("Static encoder is not initialized")
            static_tokens = self.static_encoder(x_static)
            aux_tokens_list.append(static_tokens)
            aux_padding_masks.append(
                torch.zeros(
                    static_tokens.shape[:2],
                    dtype=torch.bool,
                    device=static_tokens.device,
                )
            )

        if aux_tokens_list:
            if self.fusion is None:
                raise RuntimeError("Multimodal fusion module is not initialized")
            aux_tokens = torch.cat(aux_tokens_list, dim=1)
            aux_key_padding_mask = None
            if aux_padding_masks and len(aux_padding_masks) == len(aux_tokens_list):
                aux_key_padding_mask = torch.cat(aux_padding_masks, dim=1)
            encoder_input, fusion_info = self.fusion(wind_tokens, aux_tokens, aux_key_padding_mask)
        else:
            encoder_input = wind_tokens

        token_valid = tokenized.token_valid.reshape(batch_size, context_hours * height_levels)
        src_key_padding_mask = ~token_valid
        encoder_memory = self.encoder(
            encoder_input,
            src_key_padding_mask=src_key_padding_mask,
        )

        # target_queries: [B, T_out*H, d_model].
        target_queries = self.query_builder(
            encoder_memory,
            token_valid=token_valid.reshape(batch_size, context_hours, height_levels),
            height_levels=height_levels,
        )
        decoded = self.decoder(
            tgt=target_queries,
            memory=encoder_memory,
            memory_key_padding_mask=src_key_padding_mask,
        )

        target_features = decoded.reshape(
            batch_size,
            self.config.target_steps,
            height_levels,
            self.config.d_model,
        )
        # residual: [B, T_out, H, 2].
        residual = self.residual_head(target_features)
        # pred: reference hourly profile plus learned intra-hour residual.
        # When training in normalized space, callers can pass a y-normalized
        # current_hourly_reference so pred and y_10min use the same reference
        # normalization. Falling back to x_hourly[:, -1] preserves old callers.
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
        pred = current_hourly.unsqueeze(1) + residual
        output = {
            "pred": pred,
            "residual": residual,
            "encoder_memory": encoder_memory,
            "fusion_info": fusion_info,
        }
        if return_features:
            output["target_features"] = target_features
        return output

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
