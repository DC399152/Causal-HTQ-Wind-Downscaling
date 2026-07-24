"""Minimal Causal HTQ-Transformer training entry point.

Training loss is computed in normalized space. Validation and test metrics are
computed after denormalizing predictions and targets to physical m/s units.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import WindDownscalingDataset, load_norm_stats, require_torch
from src.models.htq_encoder_only import EncoderOnlyConfig
from src.models.htq_transformer import HTQConfig
from src.models.model_factory import ModelConfig, architecture_from_config, build_model
from src.training.losses import residual_physics_loss
from src.training.metrics import (
    add_metric_sums,
    empty_physical_metric_sums,
    finalize_physical_metrics,
    physical_metric_sums,
)
from src.training.utils import get_device, set_seed, y_denormalize


DEFAULT_CONFIG = "configs/model/htq_meteo.yaml"
DEFAULT_DATASET_DIR = "data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1"
DEFAULT_RUN_DIR = "runs/htq_minimal"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with a clear dependency error."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read training config files.") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model_config(config: dict[str, Any], args: argparse.Namespace) -> ModelConfig:
    """Map YAML keys to the architecture-specific model config."""

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    multimodal_cfg = config.get("multimodal", {})
    meteo_cfg = config.get("meteo", {})
    static_cfg = config.get("static", {})
    fusion_cfg = config.get("fusion", {})
    query_cfg = config.get("query_builder", {})
    pressure_levels = tuple(int(v) for v in meteo_cfg.get("pressure_levels_hpa", [1000, 975, 950, 925, 900]))
    trend_scales = tuple(int(v) for v in query_cfg.get("trend_scales", [1, 3, 5]))
    architecture = str(model_cfg.get("architecture", "htq_encoder_decoder"))
    if architecture == "htq_target_token_encoder_only":
        target_cfg = model_cfg.get("target_tokens", {})
        residual_head_cfg = model_cfg.get("residual_head", {})
        return EncoderOnlyConfig(
            architecture=architecture,
            name=str(model_cfg.get("name", "htq_encoder_only_v1")),
            d_model=args.d_model or int(model_cfg.get("d_model", 128)),
            nhead=args.nhead or int(model_cfg.get("n_heads", model_cfg.get("nhead", 8))),
            num_encoder_layers=args.encoder_layers
            or int(model_cfg.get("encoder_layers", model_cfg.get("num_encoder_layers", 4))),
            dim_feedforward=args.dim_feedforward or int(model_cfg.get("dim_feedforward", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            activation=str(model_cfg.get("activation", "gelu")),
            norm_first=bool(model_cfg.get("norm_first", True)),
            context_hours=int(model_cfg.get("context_hours", data_cfg.get("context_hours", 12))),
            target_steps=int(model_cfg.get("target_steps", data_cfg.get("target_steps", 6))),
            height_levels=int(model_cfg.get("height_levels", 6)),
            input_channels=int(model_cfg.get("input_channels", 2)),
            output_channels=int(model_cfg.get("output_channels", 2)),
            include_mask_features=bool(model_cfg.get("include_mask_features", True)),
            include_delta_features=bool(model_cfg.get("include_delta_features", True)),
            use_physical_height_embedding=bool(model_cfg.get("use_physical_height_embedding", True)),
            physical_height_hidden_dim=int(model_cfg.get("physical_height_hidden_dim", 64)),
            height_center_m=float(model_cfg.get("height_center_m", 300.0)),
            height_scale_m=float(model_cfg.get("height_scale_m", 100.0)),
            condition_on_current_height=bool(target_cfg.get("condition_on_current_height", True)),
            context_gate_init_bias=float(target_cfg.get("context_gate_init_bias", -1.0)),
            use_block_attention_mask=bool(target_cfg.get("use_block_attention_mask", True)),
            allow_target_to_target_attention=bool(
                target_cfg.get("allow_target_to_target_attention", True)
            ),
            residual_head_hidden_dim=int(residual_head_cfg.get("hidden_dim", 64)),
            residual_head_dropout=float(residual_head_cfg.get("dropout", 0.05)),
            residual_head_final_weight_std=float(
                residual_head_cfg.get("final_weight_std", 0.001)
            ),
            use_meteo=bool(multimodal_cfg.get("use_meteo", model_cfg.get("use_meteo", False))),
            use_static=bool(multimodal_cfg.get("use_static", model_cfg.get("use_static", False))),
            meteo_context_hours=int(meteo_cfg.get("context_hours", data_cfg.get("context_hours", 12))),
            meteo_pressure_levels_hpa=pressure_levels,
            num_meteo_channels=int(meteo_cfg.get("num_meteo_channels", 2)),
            meteo_use_delta=bool(meteo_cfg.get("use_delta", True)),
            meteo_use_mask_channels=bool(meteo_cfg.get("use_mask_channels", False)),
            fusion_nhead=int(fusion_cfg.get("nhead", model_cfg.get("nhead", 8))),
            fusion_dropout=float(fusion_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
            fusion_gate_init_bias=float(fusion_cfg.get("gate_init_bias", -2.0)),
            static_input_dim=int(static_cfg.get("input_dim", 17)),
            static_n_tokens=int(static_cfg.get("n_static_tokens", 1)),
            static_hidden_dim=int(static_cfg.get("hidden_dim", 128)),
            static_dropout=float(static_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        )
    if architecture != "htq_encoder_decoder":
        raise ValueError(f"Unknown model architecture {architecture!r}")
    return HTQConfig(
        d_model=args.d_model or int(model_cfg.get("d_model", 64)),
        nhead=args.nhead or int(model_cfg.get("n_heads", model_cfg.get("nhead", 4))),
        num_encoder_layers=args.encoder_layers
        or int(model_cfg.get("encoder_layers", model_cfg.get("num_encoder_layers", 2))),
        num_decoder_layers=args.decoder_layers
        or int(model_cfg.get("decoder_layers", model_cfg.get("num_decoder_layers", 2))),
        dim_feedforward=args.dim_feedforward or int(model_cfg.get("dim_feedforward", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        context_hours=int(data_cfg.get("context_hours", 6)),
        target_steps=int(data_cfg.get("target_steps", 6)),
        height_levels=int(model_cfg.get("height_levels", 6)),
        input_channels=int(model_cfg.get("input_channels", 2)),
        output_channels=int(model_cfg.get("output_channels", 2)),
        enforce_zero_mean_residual=bool(model_cfg.get("enforce_zero_mean_residual", False)),
        use_meteo=bool(multimodal_cfg.get("use_meteo", model_cfg.get("use_meteo", False))),
        use_static=bool(multimodal_cfg.get("use_static", model_cfg.get("use_static", False))),
        meteo_context_hours=int(meteo_cfg.get("context_hours", data_cfg.get("context_hours", 6))),
        meteo_pressure_levels_hpa=pressure_levels,
        num_meteo_channels=int(meteo_cfg.get("num_meteo_channels", 2)),
        meteo_use_delta=bool(meteo_cfg.get("use_delta", True)),
        meteo_use_mask_channels=bool(meteo_cfg.get("use_mask_channels", False)),
        fusion_nhead=int(fusion_cfg.get("nhead", model_cfg.get("n_heads", model_cfg.get("nhead", 4)))),
        fusion_dropout=float(fusion_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        fusion_gate_init_bias=float(fusion_cfg.get("gate_init_bias", -2.0)),
        static_input_dim=int(static_cfg.get("input_dim", 17)),
        static_n_tokens=int(static_cfg.get("n_static_tokens", 1)),
        static_hidden_dim=int(static_cfg.get("hidden_dim", 128)),
        static_dropout=float(static_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        query_builder_type=str(query_cfg.get("type", model_cfg.get("query_builder_type", "context_conditioned"))),
        query_use_context_projection=bool(query_cfg.get("use_context_projection", True)),
        query_use_context_layernorm=bool(query_cfg.get("use_context_layernorm", True)),
        query_use_temporal_context=bool(query_cfg.get("use_temporal_context", False)),
        query_use_multiscale_trend=bool(
            query_cfg.get("use_multiscale_trend", query_cfg.get("use_trend_context", False))
        ),
        query_trend_scales=trend_scales,
        query_use_trend_context=bool(query_cfg.get("use_trend_context", False)),
    )


def make_loader(
    dataset_dir: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    """Create a normalized DataLoader for one split."""

    torch = require_torch()
    dataset = WindDownscalingDataset(
        dataset_dir,
        split=split,
        normalize=True,
        return_metadata=False,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch: dict[str, Any], device):
    """Move tensor batch values to the selected device."""

    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def model_forward(model, batch: dict[str, Any]):
    """Run HTQ with optional multimodal inputs when present in the batch."""

    return model(
        x_hourly=batch["x_hourly"],
        x_mask=batch["x_mask"],
        x_meteo=batch.get("x_meteo"),
        meteo_mask=batch.get("meteo_mask"),
        x_static=batch.get("x_static"),
        current_hourly_reference=batch.get("current_hourly_y_norm"),
        height_values=batch.get("height_values", batch.get("height")),
    )


def compute_loss_parts(outputs: dict[str, Any], batch: dict[str, Any], loss_config: dict[str, float | str]):
    """Compute the configured normalized-space training loss."""

    loss_type = str(loss_config.get("type", "residual_physics"))
    if loss_type != "residual_physics":
        raise ValueError(
            f"Unknown loss.type {loss_type!r}; only 'residual_physics' is supported. "
            "Use lambda_xxx=0 inside residual_physics for ablations."
        )
    pred = outputs["pred"]
    return residual_physics_loss(
        pred,
        outputs.get("residual"),
        batch["y_10min"],
        batch["y_mask"],
        batch.get("current_hourly_y_norm"),
        batch.get("height"),
        lambda_wind=float(loss_config["lambda_wind"]),
        lambda_extreme=float(loss_config["lambda_extreme"]),
        lambda_residual_weighted=float(loss_config["lambda_residual_weighted"]),
        lambda_temporal=float(loss_config["lambda_temporal"]),
        lambda_temporal_weighted=float(loss_config["lambda_temporal_weighted"]),
        lambda_roughness=float(loss_config["lambda_roughness"]),
        lambda_amplitude=float(loss_config["lambda_amplitude"]),
        lambda_gradient_amplitude=float(loss_config["lambda_gradient_amplitude"]),
        lambda_residual_corr=float(loss_config["lambda_residual_corr"]),
        lambda_temporal_gradient_corr=float(loss_config["lambda_temporal_gradient_corr"]),
        lambda_vertical=float(loss_config["lambda_vertical"]),
        lambda_consistency=float(loss_config["lambda_consistency"]),
        extreme_beta=float(loss_config["extreme_beta"]),
        extreme_threshold=float(loss_config["extreme_threshold"]),
        extreme_scale=float(loss_config["extreme_scale"]),
        extreme_max_weight=float(loss_config["extreme_max_weight"]),
        residual_weight_alpha=float(loss_config["residual_weight_alpha"]),
        residual_weight_gamma=float(loss_config["residual_weight_gamma"]),
        residual_weight_q_ref=float(loss_config["residual_weight_q_ref"]),
        residual_weight_max=float(loss_config["residual_weight_max"]),
        temporal_weight_alpha=float(loss_config["temporal_weight_alpha"]),
        temporal_weight_gamma=float(loss_config["temporal_weight_gamma"]),
        temporal_weight_q_ref=float(loss_config["temporal_weight_q_ref"]),
        temporal_weight_max=float(loss_config["temporal_weight_max"]),
        amplitude_eps=float(loss_config["amplitude_eps"]),
        y_mean=loss_config.get("y_mean"),
        y_std=loss_config.get("y_std"),
    )


def _add_loss_totals(total: dict[str, float], loss_parts: dict[str, Any]) -> None:
    for key, value in loss_parts.items():
        if key not in total:
            total[key] = 0.0
        if hasattr(value, "detach"):
            total[key] += float(value.detach().item())
        else:
            total[key] += float(value)


def _average_loss_totals(prefix: str, total: dict[str, float], total_batches: int) -> dict[str, float]:
    return {f"{prefix}_loss_norm_{key}": value / total_batches for key, value in total.items()}


def build_scheduler(optimizer, train_cfg: dict[str, Any], steps_per_epoch: int, max_epochs: int, base_lr: float):
    """Build optional step-level warmup + cosine scheduler."""

    scheduler_cfg = train_cfg.get("scheduler", {}) or {}
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type in {"none", "off", ""}:
        return None, {"type": "none"}
    if scheduler_type != "warmup_cosine":
        raise ValueError(f"Unsupported scheduler.type: {scheduler_type!r}")

    warmup_epochs = float(scheduler_cfg.get("warmup_epochs", 0))
    min_lr = float(scheduler_cfg.get("min_learning_rate", 0.0))
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive to build scheduler")
    total_steps = max(1, int(max_epochs * steps_per_epoch))
    warmup_steps = min(total_steps, max(0, int(warmup_epochs * steps_per_epoch)))
    min_factor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(min_factor, float(step + 1) / float(warmup_steps))
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    torch = require_torch()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    info = {
        "type": "warmup_cosine",
        "warmup_epochs": warmup_epochs,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "min_learning_rate": min_lr,
    }
    return scheduler, info


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_config: dict[str, float | str],
    gradient_clip_norm: float | None = None,
    scheduler=None,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Train for one epoch using normalized-space weighted HTQ losses."""

    model.train()
    total: dict[str, float] = {}
    grad_norm_total = 0.0
    grad_norm_count = 0
    lr_total = 0.0
    total_batches = 0
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model_forward(model, batch)
        loss_parts = compute_loss_parts(out, batch, loss_config)
        loss = loss_parts["loss"]
        torch = require_torch()
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at batch {batch_idx}: {float(loss.detach().cpu())}")
        loss.backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip_norm,
                error_if_nonfinite=True,
            )
            grad_norm_total += float(grad_norm)
            grad_norm_count += 1
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        lr_total += float(optimizer.param_groups[0]["lr"])

        _add_loss_totals(total, loss_parts)
        total_batches += 1

    if total_batches == 0:
        raise ValueError("No training batches were processed")
    averaged = _average_loss_totals("train", total, total_batches)
    averaged["train_loss_norm_total"] = averaged.pop("train_loss_norm_loss")
    averaged["train_lr"] = lr_total / total_batches
    if grad_norm_count:
        averaged["train_grad_norm"] = grad_norm_total / grad_norm_count
    return averaged


def evaluate(
    model,
    loader,
    norm_stats: dict[str, Any],
    device,
    loss_config: dict[str, float | str] | None = None,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate normalized weighted losses and physical-unit m/s metrics."""

    torch = require_torch()
    model.eval()
    loss_config = loss_config or default_loss_config()
    total: dict[str, float] = {}
    total_batches = 0
    metric_sums = empty_physical_metric_sums()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            batch = move_batch(batch, device)
            out = model_forward(model, batch)
            loss_parts = compute_loss_parts(out, batch, loss_config)

            pred_ms = y_denormalize(out["pred"], norm_stats)
            target_ms = y_denormalize(batch["y_10min"], norm_stats)

            _add_loss_totals(total, loss_parts)
            total_batches += 1
            add_metric_sums(metric_sums, physical_metric_sums(pred_ms, target_ms, batch["y_mask"]))

    if total_batches == 0:
        raise ValueError("No evaluation batches were processed")

    metrics = _average_loss_totals("", total, total_batches)
    metrics = {key.removeprefix("_"): value for key, value in metrics.items()}
    metrics["loss_norm_total"] = metrics.pop("loss_norm_loss")
    metrics.update(finalize_physical_metrics(metric_sums))
    return metrics


def default_loss_config() -> dict[str, float | str]:
    """Default normalized-space HTQ training loss configuration."""

    return {
        "type": "residual_physics",
        "lambda_temporal": 0.2,
        "lambda_vertical": 0.05,
        "lambda_wind": 1.0,
        "lambda_extreme": 0.3,
        "lambda_residual_weighted": 0.0,
        "lambda_temporal_weighted": 0.0,
        "lambda_roughness": 0.1,
        "lambda_amplitude": 0.0,
        "lambda_gradient_amplitude": 0.0,
        "lambda_residual_corr": 0.0,
        "lambda_temporal_gradient_corr": 0.0,
        "amplitude_eps": 1e-4,
        "lambda_consistency": 0.1,
        "extreme_beta": 1.0,
        "extreme_threshold": 10.0,
        "extreme_scale": 2.0,
        "extreme_max_weight": 5.0,
        "residual_weight_alpha": 1.0,
        "residual_weight_gamma": 1.0,
        "residual_weight_q_ref": 1.0,
        "residual_weight_max": 5.0,
        "temporal_weight_alpha": 1.0,
        "temporal_weight_gamma": 1.0,
        "temporal_weight_q_ref": 1.0,
        "temporal_weight_max": 5.0,
    }


def describe_loss_type(loss_config: dict[str, float | str]) -> str:
    """Human-readable loss formula for run metadata."""

    loss_type = str(loss_config.get("type", "residual_physics"))
    if loss_type == "residual_physics":
        return (
            "lambda_wind*wind_l1 + lambda_extreme*extreme_weighted_l1 + "
            "lambda_residual_weighted*residual_weighted_l1 + "
            "lambda_temporal*residual_temporal_l1 + "
            "lambda_temporal_weighted*residual_temporal_weighted_l1 + "
            "lambda_roughness*residual_second_order_l1 + lambda_vertical*vertical_shear_l1 + "
            "lambda_amplitude*residual_amplitude_l1 + "
            "lambda_gradient_amplitude*residual_gradient_amplitude_smooth_l1 + "
            "lambda_residual_corr*residual_correlation_loss + "
            "lambda_temporal_gradient_corr*temporal_gradient_correlation_loss + "
            "lambda_consistency*hourly_consistency_l1"
        )
    return f"unsupported loss type: {loss_type}"


def attach_norm_stats_to_loss_config(
    loss_config: dict[str, Any],
    norm_stats: dict[str, Any],
) -> dict[str, Any]:
    """Attach y-normalization stats for losses that need physical thresholds."""

    updated = dict(loss_config)
    updated.setdefault("y_mean", norm_stats.get("y_mean"))
    updated.setdefault("y_std", norm_stats.get("y_std"))
    return updated


def loss_config_from_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, float | str]:
    """Read loss config and allow CLI overrides for standard weights."""

    loss_cfg = config.get("loss", {})
    extreme_cfg = loss_cfg.get("extreme", {})
    residual_weight_cfg = loss_cfg.get("residual_weight", {})
    temporal_weight_cfg = loss_cfg.get("temporal_weight", {})
    defaults = default_loss_config()
    return {
        "type": str(loss_cfg.get("type", defaults["type"])),
        "lambda_wind": float(loss_cfg.get("lambda_wind", defaults["lambda_wind"])),
        "lambda_extreme": float(loss_cfg.get("lambda_extreme", defaults["lambda_extreme"])),
        "lambda_residual_weighted": float(
            loss_cfg.get("lambda_residual_weighted", defaults["lambda_residual_weighted"])
        ),
        "lambda_temporal": args.lambda_temporal
        if args.lambda_temporal is not None
        else float(loss_cfg.get("lambda_temporal", defaults["lambda_temporal"])),
        "lambda_temporal_weighted": float(
            loss_cfg.get("lambda_temporal_weighted", defaults["lambda_temporal_weighted"])
        ),
        "lambda_roughness": float(loss_cfg.get("lambda_roughness", defaults["lambda_roughness"])),
        "lambda_amplitude": float(loss_cfg.get("lambda_amplitude", defaults["lambda_amplitude"])),
        "lambda_gradient_amplitude": float(
            loss_cfg.get("lambda_gradient_amplitude", defaults["lambda_gradient_amplitude"])
        ),
        "lambda_residual_corr": float(
            loss_cfg.get("lambda_residual_corr", defaults["lambda_residual_corr"])
        ),
        "lambda_temporal_gradient_corr": float(
            loss_cfg.get("lambda_temporal_gradient_corr", defaults["lambda_temporal_gradient_corr"])
        ),
        "amplitude_eps": float(loss_cfg.get("amplitude_eps", defaults["amplitude_eps"])),
        "lambda_vertical": args.lambda_vertical
        if args.lambda_vertical is not None
        else float(loss_cfg.get("lambda_vertical", defaults["lambda_vertical"])),
        "lambda_consistency": float(loss_cfg.get("lambda_consistency", defaults["lambda_consistency"])),
        "extreme_beta": float(extreme_cfg.get("beta", loss_cfg.get("extreme_beta", defaults["extreme_beta"]))),
        "extreme_threshold": float(
            extreme_cfg.get("threshold", loss_cfg.get("extreme_threshold", defaults["extreme_threshold"]))
        ),
        "extreme_scale": float(extreme_cfg.get("scale", loss_cfg.get("extreme_scale", defaults["extreme_scale"]))),
        "extreme_max_weight": float(
            extreme_cfg.get("max_weight", loss_cfg.get("extreme_max_weight", defaults["extreme_max_weight"]))
        ),
        "residual_weight_alpha": float(
            residual_weight_cfg.get("alpha", loss_cfg.get("residual_weight_alpha", defaults["residual_weight_alpha"]))
        ),
        "residual_weight_gamma": float(
            residual_weight_cfg.get("gamma", loss_cfg.get("residual_weight_gamma", defaults["residual_weight_gamma"]))
        ),
        "residual_weight_q_ref": float(
            residual_weight_cfg.get("q_ref", loss_cfg.get("residual_weight_q_ref", defaults["residual_weight_q_ref"]))
        ),
        "residual_weight_max": float(
            residual_weight_cfg.get("max_weight", loss_cfg.get("residual_weight_max", defaults["residual_weight_max"]))
        ),
        "temporal_weight_alpha": float(
            temporal_weight_cfg.get("alpha", loss_cfg.get("temporal_weight_alpha", defaults["temporal_weight_alpha"]))
        ),
        "temporal_weight_gamma": float(
            temporal_weight_cfg.get("gamma", loss_cfg.get("temporal_weight_gamma", defaults["temporal_weight_gamma"]))
        ),
        "temporal_weight_q_ref": float(
            temporal_weight_cfg.get("q_ref", loss_cfg.get("temporal_weight_q_ref", defaults["temporal_weight_q_ref"]))
        ),
        "temporal_weight_max": float(
            temporal_weight_cfg.get("max_weight", loss_cfg.get("temporal_weight_max", defaults["temporal_weight_max"]))
        ),
    }


def early_stopping_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Read early-stopping settings.

    Early stopping monitors validation MAE in physical m/s by default. Lower is
    better, matching the checkpoint selection criterion used by this script.
    """

    training_cfg = config.get("training", {})
    early_cfg = training_cfg.get("early_stopping", {})
    patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else early_cfg.get("patience")
    )
    return {
        "enabled": patience is not None and int(patience) > 0,
        "patience": int(patience) if patience is not None else None,
        "min_delta": args.early_stopping_min_delta
        if args.early_stopping_min_delta is not None
        else float(early_cfg.get("min_delta", 0.0)),
        "monitor": args.early_stopping_monitor or early_cfg.get("monitor", "MAE_ms"),
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict[str, float],
    model_config: ModelConfig,
    loss_config: dict[str, float | str],
    scheduler=None,
    norm_stats: dict[str, Any] | None = None,
) -> None:
    """Save a compact PyTorch checkpoint."""

    torch = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture_from_config(model_config),
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "model_config": model_config.__dict__,
            "loss_weights": {
                "lambda_wind": float(loss_config["lambda_wind"]),
                "lambda_extreme": float(loss_config["lambda_extreme"]),
                "lambda_residual_weighted": float(loss_config["lambda_residual_weighted"]),
                "lambda_temporal": float(loss_config["lambda_temporal"]),
                "lambda_temporal_weighted": float(loss_config["lambda_temporal_weighted"]),
                "lambda_roughness": float(loss_config["lambda_roughness"]),
                "lambda_amplitude": float(loss_config["lambda_amplitude"]),
                "lambda_gradient_amplitude": float(loss_config["lambda_gradient_amplitude"]),
                "lambda_residual_corr": float(loss_config["lambda_residual_corr"]),
                "lambda_temporal_gradient_corr": float(loss_config["lambda_temporal_gradient_corr"]),
                "lambda_vertical": float(loss_config["lambda_vertical"]),
                "lambda_consistency": float(loss_config["lambda_consistency"]),
            },
            "loss_config": loss_config,
            "norm_stats": norm_stats,
        },
        path,
    )


def write_metrics_json(path: str | Path, summary: dict[str, Any]) -> None:
    """Atomically persist training history so interruptions do not lose completed epochs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    temporary_path.replace(path)


def validate_monitor_value(monitor_name: str, monitor_value: float) -> None:
    """Fail fast when the validation monitor cannot define a best checkpoint."""

    if not math.isfinite(monitor_value):
        raise ValueError(f"Validation monitor {monitor_name} is not finite: {monitor_value}")


def maybe_save_named_checkpoints(
    run_dir: Path,
    model,
    optimizer,
    epoch: int,
    row: dict[str, Any],
    model_config: ModelConfig,
    loss_config: dict[str, float | str],
    best_values: dict[str, float],
    best_epochs: dict[str, int],
    scheduler=None,
    norm_stats: dict[str, Any] | None = None,
) -> None:
    """Save additional validation-best checkpoints without changing early stopping."""

    specs = {
        "best_mae.pt": ("val_MAE_ms", "min"),
        "best_loss.pt": ("val_loss_norm_total", "min"),
        "best_residual_acc.pt": ("val_residual_ACC", "max"),
        "best_temporal_gradient_acc.pt": ("val_temporal_gradient_ACC", "max"),
        "best_composite.pt": ("val_residual_temporal_composite", "max"),
    }
    composite = float(row["val_residual_ACC"]) + float(row["val_temporal_gradient_ACC"])
    row["val_residual_temporal_composite"] = composite

    for filename, (metric_name, mode) in specs.items():
        value = float(row[metric_name])
        validate_monitor_value(metric_name, value)
        previous = best_values.get(filename)
        improved = previous is None or (value < previous if mode == "min" else value > previous)
        if improved:
            best_values[filename] = value
            best_epochs[filename] = epoch
            save_checkpoint(
                run_dir / filename,
                model,
                optimizer,
                epoch,
                row,
                model_config,
                loss_config,
                scheduler=scheduler,
                norm_stats=norm_stats,
            )


def load_best_checkpoint_for_test(best_checkpoint_path: str | Path, model, device) -> dict[str, Any]:
    """Load best.pt into ``model`` before final test evaluation."""

    torch = require_torch()
    best_checkpoint_path = Path(best_checkpoint_path)
    if not best_checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint was not created: {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--lambda-temporal", type=float, default=None)
    parser.add_argument("--lambda-vertical", type=float, default=None)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop if validation monitor does not improve for this many epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=None,
        help="Minimum validation improvement required to reset patience.",
    )
    parser.add_argument(
        "--early-stopping-monitor",
        default=None,
        help="Validation metric to monitor, e.g. MAE_ms or RMSE_ms. Lower is better.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 32))
    max_epochs = args.max_epochs or int(train_cfg.get("max_epochs") or 1)
    learning_rate = args.learning_rate or float(train_cfg.get("learning_rate") or 1e-4)
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(train_cfg.get("weight_decay", 0.01))
    )
    num_workers = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 0))
    gradient_clip_norm = (
        args.gradient_clip_norm
        if args.gradient_clip_norm is not None
        else train_cfg.get("gradient_clip_norm")
    )
    gradient_clip_norm = None if gradient_clip_norm is None else float(gradient_clip_norm)
    dataset_dir = args.dataset_dir or data_cfg.get("dataset_dir", DEFAULT_DATASET_DIR)
    run_dir = Path(args.run_dir)
    loss_config = loss_config_from_config(config, args)
    early_stopping = early_stopping_config(config, args)

    set_seed(seed)
    torch = require_torch()
    device = get_device(args.device)
    norm_stats = load_norm_stats(Path(dataset_dir) / "norm_stats.json")
    loss_config = attach_norm_stats_to_loss_config(loss_config, norm_stats)

    model_config = build_model_config(config, args)
    model = build_model(model_config).to(device)
    architecture = architecture_from_config(model_config)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    input_token_count = model_config.context_hours * model_config.height_levels
    target_token_count = model_config.target_steps * model_config.height_levels
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = make_loader(dataset_dir, "train", batch_size, True, num_workers)
    val_loader = make_loader(dataset_dir, "val", batch_size, False, num_workers)
    test_loader = make_loader(dataset_dir, "test", batch_size, False, num_workers)
    scheduler, scheduler_info = build_scheduler(
        optimizer,
        train_cfg,
        len(train_loader),
        max_epochs,
        learning_rate,
    )

    print(f"dataset_dir: {dataset_dir}")
    print(f"run_dir: {run_dir}")
    print(f"device: {device}")
    print(f"architecture: {architecture}")
    print(f"total_params: {total_params}")
    print(f"trainable_params: {trainable_params}")
    print(f"d_model: {model_config.d_model}")
    print(f"nhead: {model_config.nhead}")
    print(f"num_encoder_layers: {model_config.num_encoder_layers}")
    print(f"input_token_count: {input_token_count}")
    print(f"target_token_count: {target_token_count}")
    print(f"total_token_count: {input_token_count + target_token_count}")
    print(f"use_meteo: {model_config.use_meteo}")
    print(f"use_static: {model_config.use_static}")
    print(
        "use_physical_height_embedding: "
        f"{getattr(model_config, 'use_physical_height_embedding', False)}"
    )
    print(
        "use_block_attention_mask: "
        f"{getattr(model_config, 'use_block_attention_mask', False)}"
    )
    print(
        "condition_on_current_height: "
        f"{getattr(model_config, 'condition_on_current_height', False)}"
    )
    print(f"epochs: {max_epochs}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"weight_decay: {weight_decay}")
    print(f"gradient_clip_norm: {gradient_clip_norm}")
    print(f"scheduler: {scheduler_info}")
    print(f"num_workers: {num_workers}")
    print(
        f"loss: {loss_config['type']} normalized loss "
        f"(lambda_wind={loss_config['lambda_wind']}, "
        f"lambda_extreme={loss_config['lambda_extreme']}, "
        f"lambda_residual_weighted={loss_config['lambda_residual_weighted']}, "
        f"lambda_temporal={loss_config['lambda_temporal']}, "
        f"lambda_temporal_weighted={loss_config['lambda_temporal_weighted']}, "
        f"lambda_roughness={loss_config['lambda_roughness']}, "
        f"lambda_amplitude={loss_config['lambda_amplitude']}, "
        f"lambda_gradient_amplitude={loss_config['lambda_gradient_amplitude']}, "
        f"lambda_residual_corr={loss_config['lambda_residual_corr']}, "
        f"lambda_temporal_gradient_corr={loss_config['lambda_temporal_gradient_corr']}, "
        f"lambda_vertical={loss_config['lambda_vertical']})"
    )
    print("val/test metrics: denormalized physical m/s")
    if early_stopping["enabled"]:
        print(
            "early stopping: "
            f"monitor=val_{early_stopping['monitor']} "
            f"patience={early_stopping['patience']} "
            f"min_delta={early_stopping['min_delta']}"
        )

    best_monitor = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = None
    named_checkpoint_best_values: dict[str, float] = {}
    named_checkpoint_best_epochs: dict[str, int] = {}
    history: list[dict[str, Any]] = []
    summary = {
        "status": "running",
        "config": str(args.config),
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "seed": seed,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "scheduler": scheduler_info,
        "num_workers": num_workers,
        "architecture": architecture,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "loss_config": loss_config,
        "loss_weights": {
            "lambda_wind": float(loss_config["lambda_wind"]),
            "lambda_extreme": float(loss_config["lambda_extreme"]),
            "lambda_residual_weighted": float(loss_config["lambda_residual_weighted"]),
            "lambda_temporal": float(loss_config["lambda_temporal"]),
            "lambda_temporal_weighted": float(loss_config["lambda_temporal_weighted"]),
            "lambda_roughness": float(loss_config["lambda_roughness"]),
            "lambda_amplitude": float(loss_config["lambda_amplitude"]),
            "lambda_gradient_amplitude": float(loss_config["lambda_gradient_amplitude"]),
            "lambda_residual_corr": float(loss_config["lambda_residual_corr"]),
            "lambda_temporal_gradient_corr": float(loss_config["lambda_temporal_gradient_corr"]),
            "lambda_vertical": float(loss_config["lambda_vertical"]),
            "lambda_consistency": float(loss_config["lambda_consistency"]),
        },
        "model_config": model_config.__dict__,
        "early_stopping": {},
        "history": history,
        "checkpoint_selection": {},
        "test": None,
        "test_checkpoint": None,
        "test_checkpoint_epoch": None,
        "loss_space": "normalized",
        "loss_type": describe_loss_type(loss_config),
        "metric_space": "physical_m_per_s",
    }

    def persist_summary(status: str) -> None:
        summary["status"] = status
        summary["completed_epochs"] = len(history)
        summary["early_stopping"] = {
            **early_stopping,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "best_epoch": best_epoch,
            "best_monitor_value": best_monitor if math.isfinite(best_monitor) else None,
        }
        summary["checkpoint_selection"] = {
            name: {
                "path": str(run_dir / name),
                "epoch": int(named_checkpoint_best_epochs[name]),
                "metric_value": float(named_checkpoint_best_values[name]),
            }
            for name in sorted(named_checkpoint_best_values)
        }
        write_metrics_json(run_dir / "metrics.json", summary)

    try:
        for epoch in range(1, max_epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                loss_config,
                gradient_clip_norm=gradient_clip_norm,
                scheduler=scheduler,
                limit_batches=args.limit_train_batches,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                norm_stats,
                device,
                loss_config=loss_config,
                limit_batches=args.limit_eval_batches,
            )
            row = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            print(
                f"epoch {epoch:03d} "
                f"train_loss_norm_total={row['train_loss_norm_total']:.6g} "
                f"val_loss_norm_total={row['val_loss_norm_total']:.6g} "
                f"val_MAE_ms={row['val_MAE_ms']:.6g} "
                f"val_RMSE_ms={row['val_RMSE_ms']:.6g} "
                f"val_residual_ACC={row['val_residual_ACC']:.6g}"
            )

            save_checkpoint(
                run_dir / "last.pt",
                model,
                optimizer,
                epoch,
                row,
                model_config,
                loss_config,
                scheduler=scheduler,
                norm_stats=norm_stats,
            )
            maybe_save_named_checkpoints(
                run_dir,
                model,
                optimizer,
                epoch,
                row,
                model_config,
                loss_config,
                named_checkpoint_best_values,
                named_checkpoint_best_epochs,
                scheduler=scheduler,
                norm_stats=norm_stats,
            )
            monitor_name = str(early_stopping["monitor"])
            if monitor_name not in val_metrics:
                raise KeyError(f"Validation metric {monitor_name!r} is not available")
            monitor_value = float(val_metrics[monitor_name])
            validate_monitor_value(monitor_name, monitor_value)
            improved = monitor_value < best_monitor - float(early_stopping["min_delta"])
            if improved:
                best_monitor = monitor_value
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    row,
                    model_config,
                    loss_config,
                    scheduler=scheduler,
                    norm_stats=norm_stats,
                )
            else:
                epochs_without_improvement += 1

            row["early_stopping_monitor"] = monitor_name
            row["early_stopping_monitor_value"] = monitor_value
            row["best_epoch"] = best_epoch
            row["epochs_without_improvement"] = epochs_without_improvement

            should_stop = (
                early_stopping["enabled"]
                and epochs_without_improvement >= int(early_stopping["patience"])
            )
            if should_stop:
                stopped_early = True
                stop_reason = (
                    f"val_{monitor_name} did not improve by "
                    f"{early_stopping['min_delta']} for {early_stopping['patience']} epochs"
                )
            persist_summary("stopped_early" if should_stop else "running")
            if should_stop:
                print(f"early stopping at epoch {epoch:03d}: {stop_reason}")
                break
    except KeyboardInterrupt:
        stop_reason = "KeyboardInterrupt"
        persist_summary("interrupted")
        print(f"training interrupted; saved {len(history)} completed epochs to {run_dir / 'metrics.json'}")
        return

    best_checkpoint_path = run_dir / "best.pt"
    best_checkpoint = load_best_checkpoint_for_test(best_checkpoint_path, model, device)
    print(
        f"loaded best checkpoint for test: {best_checkpoint_path} "
        f"(epoch={best_checkpoint['epoch']})"
    )
    test_metrics = evaluate(
        model,
        test_loader,
        norm_stats,
        device,
        loss_config=loss_config,
        limit_batches=args.limit_eval_batches,
    )
    print(
        "test "
        f"loss_norm_total={test_metrics['loss_norm_total']:.6g} "
        f"MAE_ms={test_metrics['MAE_ms']:.6g} "
        f"RMSE_ms={test_metrics['RMSE_ms']:.6g} "
        f"residual_ACC={test_metrics['residual_ACC']:.6g}"
    )

    summary["test"] = test_metrics
    summary["test_checkpoint"] = str(best_checkpoint_path)
    summary["test_checkpoint_epoch"] = int(best_checkpoint["epoch"])
    persist_summary("completed")


if __name__ == "__main__":
    main()
