"""Mask-aware metrics for wind profile reconstruction."""

from __future__ import annotations

import math

from src.training.losses import masked_mae, masked_mse


def mean_absolute_error(pred, target):
    """Return unmasked MAE for tensors with matching semantic shapes."""

    return (pred - target).abs().mean()


def masked_mean_absolute_error(pred, target, mask):
    """Return MAE over valid target positions only."""

    return masked_mae(pred, target, mask)


def masked_root_mean_squared_error(pred, target, mask):
    """Return RMSE over valid target positions only."""

    return masked_mse(pred, target, mask).sqrt()


def physical_metric_sums(pred, target, mask, eps: float = 1e-6) -> dict[str, float]:
    """Return additive metric sums for physical-space [B, T, H, C] tensors.

    Metrics:
    - MAE/RMSE use all valid channel values.
    - u/v MAE use channel-specific valid masks.
    - speed MAE uses positions where both u and v are valid.
    - global flattened ACC is one Pearson correlation over every valid scalar
      value in the evaluated split.
    - residual ACC is temporal correlation over the 6 target steps for each
      [sample, height, channel] series, then summed over valid series.
    - temporal gradient metrics use y[:, 1:] - y[:, :-1] over 5 changes.
    """

    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target, and mask must have the same shape [B, T, H, C]")
    if pred.ndim != 4:
        raise ValueError("pred, target, and mask must have shape [B, T, H, C]")
    if pred.shape[-1] != 2:
        raise ValueError("Expected last dimension [u, v] with C=2")

    valid = mask.to(dtype=pred.dtype)
    error = pred - target
    abs_error = error.abs()
    sq_error = error.pow(2)

    u_mask = valid[..., 0]
    v_mask = valid[..., 1]
    both_mask_bool = mask[..., 0] & mask[..., 1]
    both_mask = both_mask_bool.to(dtype=pred.dtype)

    pred_speed = _speed(pred)
    target_speed = _speed(target)

    dy_pred = pred[:, 1:] - pred[:, :-1]
    dy_target = target[:, 1:] - target[:, :-1]
    dy_mask_bool = mask[:, 1:] & mask[:, :-1]
    dy_mask = dy_mask_bool.to(dtype=pred.dtype)
    dy_error = dy_pred - dy_target

    residual_acc_sum, residual_acc_count = temporal_acc_sum(pred, target, mask, eps=eps)
    gradient_acc_sum, gradient_acc_count = temporal_acc_sum(dy_pred, dy_target, dy_mask_bool, eps=eps)
    flattened = _global_flattened_acc_sums(pred, target, mask)

    return {
        "abs_sum": float((abs_error * valid).sum().item()),
        "sq_sum": float((sq_error * valid).sum().item()),
        "count": float(valid.sum().item()),
        "u_abs_sum": float((abs_error[..., 0] * u_mask).sum().item()),
        "u_count": float(u_mask.sum().item()),
        "v_abs_sum": float((abs_error[..., 1] * v_mask).sum().item()),
        "v_count": float(v_mask.sum().item()),
        "speed_abs_sum": float(((pred_speed - target_speed).abs() * both_mask).sum().item()),
        "speed_count": float(both_mask.sum().item()),
        "gradient_abs_sum": float((dy_error.abs() * dy_mask).sum().item()),
        "gradient_count": float(dy_mask.sum().item()),
        "residual_acc_sum": residual_acc_sum,
        "residual_acc_count": residual_acc_count,
        "temporal_gradient_acc_sum": gradient_acc_sum,
        "temporal_gradient_acc_count": gradient_acc_count,
        **flattened,
    }


def finalize_physical_metrics(sums: dict[str, float], eps: float = 1e-8) -> dict[str, float]:
    """Finalize additive physical metric sums into reportable metrics."""

    count = _positive_count(sums["count"], "valid target values")
    u_count = _positive_count(sums["u_count"], "valid u target values")
    v_count = _positive_count(sums["v_count"], "valid v target values")
    speed_count = _positive_count(sums["speed_count"], "valid speed target values")
    gradient_count = _positive_count(sums["gradient_count"], "valid temporal gradients")

    return {
        "MAE_ms": sums["abs_sum"] / count,
        "RMSE_ms": math.sqrt(sums["sq_sum"] / count),
        "u_MAE_ms": sums["u_abs_sum"] / u_count,
        "v_MAE_ms": sums["v_abs_sum"] / v_count,
        "speed_MAE_ms": sums["speed_abs_sum"] / speed_count,
        "global_flattened_ACC": _finalize_global_flattened_acc(sums, eps),
        "residual_ACC": _safe_ratio(sums["residual_acc_sum"], sums["residual_acc_count"], eps),
        "temporal_gradient_MAE_ms": sums["gradient_abs_sum"] / gradient_count,
        "temporal_gradient_ACC": _safe_ratio(
            sums["temporal_gradient_acc_sum"],
            sums["temporal_gradient_acc_count"],
            eps,
        ),
        "valid_target_values": count,
        "valid_speed_values": speed_count,
        "valid_global_flattened_values": sums["global_flattened_count"],
        "valid_temporal_gradients": gradient_count,
        "valid_residual_acc_series": sums["residual_acc_count"],
        "valid_temporal_gradient_acc_series": sums["temporal_gradient_acc_count"],
    }


def empty_physical_metric_sums() -> dict[str, float]:
    """Return zero-initialized additive metric sums."""

    return {
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "count": 0.0,
        "u_abs_sum": 0.0,
        "u_count": 0.0,
        "v_abs_sum": 0.0,
        "v_count": 0.0,
        "speed_abs_sum": 0.0,
        "speed_count": 0.0,
        "gradient_abs_sum": 0.0,
        "gradient_count": 0.0,
        "residual_acc_sum": 0.0,
        "residual_acc_count": 0.0,
        "temporal_gradient_acc_sum": 0.0,
        "temporal_gradient_acc_count": 0.0,
        "global_flattened_count": 0.0,
        "global_flattened_pred_sum": 0.0,
        "global_flattened_target_sum": 0.0,
        "global_flattened_pred_sq_sum": 0.0,
        "global_flattened_target_sq_sum": 0.0,
        "global_flattened_product_sum": 0.0,
    }


def add_metric_sums(total: dict[str, float], update: dict[str, float]) -> None:
    """Add one batch of metric sums into an accumulator in place."""

    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)


def temporal_acc_sum(pred, target, mask, eps: float = 1e-6) -> tuple[float, float]:
    """Sum temporal correlations over [sample, height, channel] series.

    Correlation is computed along dimension 1. Invalid time positions are
    ignored. Series with fewer than two valid points or near-zero variance in
    either prediction or target are skipped.
    """

    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target, and mask must have the same shape")
    if pred.ndim != 4:
        raise ValueError("Expected shape [B, T, H, C]")

    valid = mask.to(dtype=pred.dtype)
    valid_count = valid.sum(dim=1)
    safe_count = valid_count.clamp_min(1.0)

    pred_mean = (pred * valid).sum(dim=1, keepdim=True) / safe_count.unsqueeze(1)
    target_mean = (target * valid).sum(dim=1, keepdim=True) / safe_count.unsqueeze(1)
    pred_centered = (pred - pred_mean) * valid
    target_centered = (target - target_mean) * valid

    numerator = (pred_centered * target_centered).sum(dim=1)
    pred_energy = pred_centered.pow(2).sum(dim=1)
    target_energy = target_centered.pow(2).sum(dim=1)
    denominator = (pred_energy * target_energy).sqrt()

    valid_series = (valid_count >= 2) & (pred_energy > eps) & (target_energy > eps) & (denominator > eps)
    if not valid_series.any():
        return 0.0, 0.0

    corr = numerator[valid_series] / denominator[valid_series]
    return float(corr.sum().item()), float(valid_series.sum().item())


def _global_flattened_acc_sums(pred, target, mask) -> dict[str, float]:
    """Return additive sufficient statistics for masked global Pearson ACC."""

    valid_pred = pred[mask].double()
    valid_target = target[mask].double()
    return {
        "global_flattened_count": float(valid_pred.numel()),
        "global_flattened_pred_sum": float(valid_pred.sum().item()),
        "global_flattened_target_sum": float(valid_target.sum().item()),
        "global_flattened_pred_sq_sum": float(valid_pred.square().sum().item()),
        "global_flattened_target_sq_sum": float(valid_target.square().sum().item()),
        "global_flattened_product_sum": float((valid_pred * valid_target).sum().item()),
    }


def _finalize_global_flattened_acc(sums: dict[str, float], eps: float) -> float:
    count = sums["global_flattened_count"]
    if count < 2:
        return float("nan")

    covariance = (
        sums["global_flattened_product_sum"]
        - sums["global_flattened_pred_sum"]
        * sums["global_flattened_target_sum"]
        / count
    )
    pred_energy = max(
        sums["global_flattened_pred_sq_sum"]
        - sums["global_flattened_pred_sum"] ** 2 / count,
        0.0,
    )
    target_energy = max(
        sums["global_flattened_target_sq_sum"]
        - sums["global_flattened_target_sum"] ** 2 / count,
        0.0,
    )
    denominator = math.sqrt(pred_energy * target_energy)
    if denominator <= eps:
        return float("nan")
    return covariance / denominator


def _speed(values):
    return values.pow(2).sum(dim=-1).sqrt()


def _positive_count(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"No {name} available for metric computation")
    return value


def _safe_ratio(numerator: float, denominator: float, eps: float) -> float:
    if denominator <= eps:
        return float("nan")
    return numerator / denominator
