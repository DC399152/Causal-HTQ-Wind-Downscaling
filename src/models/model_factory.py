"""Central model construction for training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

from torch import nn

from src.models.htq_encoder_only import EncoderOnlyConfig, HTQTargetTokenEncoderOnly
from src.models.htq_transformer import CausalHTQTransformer, HTQConfig


ModelConfig = HTQConfig | EncoderOnlyConfig


def architecture_from_config(config: ModelConfig | Mapping[str, Any]) -> str:
    """Return architecture, defaulting historical configs to encoder-decoder."""

    if isinstance(config, Mapping):
        return str(config.get("architecture", "htq_encoder_decoder"))
    return str(getattr(config, "architecture", "htq_encoder_decoder"))


def model_config_from_dict(config: Mapping[str, Any]) -> ModelConfig:
    """Create the architecture-specific dataclass while ignoring unrelated keys."""

    architecture = architecture_from_config(config)
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
    raise ValueError(
        f"Unknown model architecture {architecture!r}; expected "
        "'htq_encoder_decoder' or 'htq_target_token_encoder_only'"
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
    raise ValueError(f"Unknown model architecture {architecture!r}")
