"""Central model construction for training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

from torch import nn

from src.models.htq_cross_attention_reader import (
    CrossAttentionReaderConfig,
    HTQCrossAttentionReader,
)
from src.models.htq_encoder_only import EncoderOnlyConfig, HTQTargetTokenEncoderOnly
from src.models.htq_fusion_time_height_mlp import (
    FusionTimeHeightMLPConfig,
    HTQFusionTimeHeightMLP,
)
from src.models.htq_transformer import CausalHTQTransformer, HTQConfig
from src.models.output_heads import (
    output_head_config_fields,
    residual_head_config_from_mapping,
)


ModelConfig = (
    HTQConfig
    | EncoderOnlyConfig
    | CrossAttentionReaderConfig
    | FusionTimeHeightMLPConfig
)


def architecture_from_config(config: ModelConfig | Mapping[str, Any]) -> str:
    """Return architecture, defaulting historical configs to encoder-decoder."""

    if isinstance(config, Mapping):
        return str(config.get("architecture", "htq_encoder_decoder"))
    return str(getattr(config, "architecture", "htq_encoder_decoder"))


def model_config_from_dict(config: Mapping[str, Any]) -> ModelConfig:
    """Create the architecture-specific dataclass while ignoring unrelated keys."""

    architecture = architecture_from_config(config)
    config = dict(config)
    nested_output_head = config.get("output_head")
    if isinstance(nested_output_head, Mapping):
        default_type = {
            "htq_target_token_encoder_only": "shared_mlp",
            "htq_cross_attention_reader": "multi_horizon_shared_trunk",
            "htq_fusion_time_height_mlp": "multi_horizon_shared_trunk",
        }.get(architecture, "shared_linear")
        config.update(
            output_head_config_fields(
                residual_head_config_from_mapping(
                    nested_output_head,
                    default_type=default_type,
                    default_hidden_dim=int(
                        config.get("residual_head_hidden_dim", 64)
                    ),
                    default_dropout=float(
                        config.get("residual_head_dropout", 0.05)
                    ),
                    default_final_weight_std=float(
                        config.get("residual_head_final_weight_std", 0.001)
                    ),
                )
            )
        )
    if architecture == "htq_encoder_decoder":
        allowed = {field.name for field in fields(HTQConfig)}
        return HTQConfig(**{key: value for key, value in config.items() if key in allowed})
    if architecture == "htq_target_token_encoder_only":
        allowed = {field.name for field in fields(EncoderOnlyConfig)}
        payload = {key: value for key, value in config.items() if key in allowed}
        payload["architecture"] = architecture
        if "meteo_pressure_levels_hpa" in payload:
            payload["meteo_pressure_levels_hpa"] = tuple(payload["meteo_pressure_levels_hpa"])
        return EncoderOnlyConfig(**payload)
    if architecture == "htq_cross_attention_reader":
        allowed = {field.name for field in fields(CrossAttentionReaderConfig)}
        payload = {key: value for key, value in config.items() if key in allowed}
        payload["architecture"] = architecture
        for key in ("meteo_pressure_levels_hpa", "query_trend_scales"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return CrossAttentionReaderConfig(**payload)
    if architecture == "htq_fusion_time_height_mlp":
        allowed = {field.name for field in fields(FusionTimeHeightMLPConfig)}
        payload = {key: value for key, value in config.items() if key in allowed}
        payload["architecture"] = architecture
        if "meteo_pressure_levels_hpa" in payload:
            payload["meteo_pressure_levels_hpa"] = tuple(
                payload["meteo_pressure_levels_hpa"]
            )
        return FusionTimeHeightMLPConfig(**payload)
    raise ValueError(
        f"Unknown model architecture {architecture!r}; expected "
        "'htq_encoder_decoder', 'htq_target_token_encoder_only', or "
        "'htq_cross_attention_reader', or 'htq_fusion_time_height_mlp'"
    )


def build_model(config: ModelConfig | Mapping[str, Any]) -> nn.Module:
    """Build one supported model architecture."""

    resolved = config if is_dataclass(config) else model_config_from_dict(config)
    architecture = architecture_from_config(resolved)
    if architecture == "htq_encoder_decoder":
        if not isinstance(resolved, HTQConfig):
            raise TypeError("Encoder-decoder architecture requires HTQConfig")
        return CausalHTQTransformer(resolved)
    if architecture == "htq_target_token_encoder_only":
        if not isinstance(resolved, EncoderOnlyConfig):
            raise TypeError("Encoder-only architecture requires EncoderOnlyConfig")
        return HTQTargetTokenEncoderOnly(resolved)
    if architecture == "htq_cross_attention_reader":
        if not isinstance(resolved, CrossAttentionReaderConfig):
            raise TypeError(
                "Cross-attention Reader architecture requires "
                "CrossAttentionReaderConfig"
            )
        return HTQCrossAttentionReader(resolved)
    if architecture == "htq_fusion_time_height_mlp":
        if not isinstance(resolved, FusionTimeHeightMLPConfig):
            raise TypeError(
                "Fusion time-height MLP architecture requires "
                "FusionTimeHeightMLPConfig"
            )
        return HTQFusionTimeHeightMLP(resolved)
    raise ValueError(f"Unknown model architecture {architecture!r}")
