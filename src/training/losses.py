"""Mask-aware losses for HTQ training."""

from __future__ import annotations

import torch
import warnings


_WARNED_EXTREME_NORMALIZED_SPACE = False
_WARNED_RESIDUAL_WEIGHT_NORMALIZED_SPACE = False


def _mask_denominator(mask, eps: float = 1e-8):
    return mask.to(dtype=float).sum().clamp_min(eps)


def masked_mse(pred, target, mask, eps: float = 1e-8):
    """Mean squared error over valid target positions only.

    Shapes:
    - pred: [B, T_out, H, C]
    - target: [B, T_out, H, C]
    - mask: [B, T_out, H, C], True=valid
    """

    valid = mask.to(dtype=pred.dtype)
    sq_error = (pred - target).pow(2) * valid
    return sq_error.sum() / valid.sum().clamp_min(eps)


def masked_mse_loss(pred, target, mask, eps: float = 1e-8):
    """Compatibility wrapper for masked MSE."""

    return masked_mse(pred, target, mask, eps=eps)


def masked_mae(pred, target, mask, eps: float = 1e-8):
    """Mean absolute error over valid target positions only."""

    return masked_l1_loss(pred, target, mask, eps=eps)


def masked_l1_loss(pred, target, mask, eps: float = 1e-8):
    """Mean absolute error over valid target positions only.

    Shapes:
    - pred: [B, T, H, C]
    - target: [B, T, H, C]
    - mask: [B, T, H, C], True=valid
    """

    valid = mask.to(dtype=pred.dtype)
    abs_error = (pred - target).abs() * valid
    return abs_error.sum() / valid.sum().clamp_min(eps)


def masked_weighted_l1_loss(pred, target, mask, weight, eps: float = 1e-8):
    """Weighted masked L1 over valid target positions.

    Shapes:
    - pred/target/mask: [B, T, H, C]
    - weight: [B, T, H, 1] or [B, T, H, C]
    """

    valid = mask.to(dtype=pred.dtype)
    weight = weight.to(dtype=pred.dtype)
    abs_error = (pred - target).abs() * valid * weight
    return abs_error.sum() / valid.sum().clamp_min(eps)


def _zero_loss_like(reference):
    return reference.sum() * 0.0


def temporal_gradient_loss(pred, target, mask, eps: float = 1e-8):
    """Masked L1 loss on adjacent target-time gradients.

    dy_pred = pred[:, 1:] - pred[:, :-1]
    dy_true = target[:, 1:] - target[:, :-1]
    dy_mask = mask[:, 1:] & mask[:, :-1]
    """

    dy_pred = pred[:, 1:] - pred[:, :-1]
    dy_true = target[:, 1:] - target[:, :-1]
    dy_mask = mask[:, 1:] & mask[:, :-1]
    return masked_l1_loss(dy_pred, dy_true, dy_mask, eps=eps)


def vertical_gradient_loss(pred, target, mask, eps: float = 1e-8):
    """Masked L1 loss on adjacent height gradients.

    dh_pred = pred[:, :, 1:] - pred[:, :, :-1]
    dh_true = target[:, :, 1:] - target[:, :, :-1]
    dh_mask = mask[:, :, 1:] & mask[:, :, :-1]
    """

    dh_pred = pred[:, :, 1:] - pred[:, :, :-1]
    dh_true = target[:, :, 1:] - target[:, :, :-1]
    dh_mask = mask[:, :, 1:] & mask[:, :, :-1]
    return masked_l1_loss(dh_pred, dh_true, dh_mask, eps=eps)


def vertical_shear_loss(pred, target, mask, height, eps: float = 1e-8):
    """Masked L1 loss on height-normalized vertical shear.

    Shapes:
    - pred/target/mask: [B, T, H, C]
    - height: [H] or [B, H], same height units for all samples/channels

    The shear is (value[h+1] - value[h]) / (z[h+1] - z[h]).
    """

    if height is None:
        raise ValueError("height is required for vertical_shear_loss")
    if height.ndim == 1:
        dz = height[1:] - height[:-1]
        dz = dz.reshape(1, 1, -1, 1)
    elif height.ndim == 2:
        dz = height[:, 1:] - height[:, :-1]
        dz = dz.reshape(height.shape[0], 1, -1, 1)
    else:
        raise ValueError("height must have shape [H] or [B, H]")

    dz = dz.to(device=pred.device, dtype=pred.dtype).abs().clamp_min(eps)
    shear_pred = (pred[:, :, 1:] - pred[:, :, :-1]) / dz
    shear_true = (target[:, :, 1:] - target[:, :, :-1]) / dz
    shear_mask = mask[:, :, 1:] & mask[:, :, :-1]
    return masked_l1_loss(shear_pred, shear_true, shear_mask, eps=eps)


def second_order_temporal_roughness_loss(pred, target, mask, eps: float = 1e-8):
    """Masked L1 loss on second-order target-time differences."""

    if pred.shape[1] < 3:
        return _zero_loss_like(pred)
    pred_second = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    target_second = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    second_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    return masked_l1_loss(pred_second, target_second, second_mask, eps=eps)


def _magnitude_weight(vector, alpha: float, gamma: float, q_ref: float, max_weight: float, eps: float):
    mag = (vector[..., :2].pow(2).sum(dim=-1, keepdim=True) + eps).sqrt()
    q_ref_tensor = vector.new_tensor(float(q_ref)).abs().clamp_min(eps)
    weight = 1.0 + float(alpha) * (mag / q_ref_tensor).pow(float(gamma))
    return weight.clamp(max=float(max_weight)).detach()


def temporal_weighted_loss(residual_pred, true_residual, mask, weight, eps: float = 1e-8):
    dy_pred = residual_pred[:, 1:] - residual_pred[:, :-1]
    dy_true = true_residual[:, 1:] - true_residual[:, :-1]
    dy_mask = mask[:, 1:] & mask[:, :-1]
    return masked_weighted_l1_loss(dy_pred, dy_true, dy_mask, weight, eps=eps)


def residual_amplitude_loss(residual_pred, true_residual, mask, eps: float = 1e-8, amplitude_eps: float = 1e-4):
    valid = mask.to(dtype=residual_pred.dtype)
    count = valid.sum(dim=1)
    pred_mean = (residual_pred * valid).sum(dim=1) / count.clamp_min(eps)
    true_mean = (true_residual * valid).sum(dim=1) / count.clamp_min(eps)
    pred_var = ((residual_pred - pred_mean.unsqueeze(1)).pow(2) * valid).sum(dim=1) / count.clamp_min(eps)
    true_var = ((true_residual - true_mean.unsqueeze(1)).pow(2) * valid).sum(dim=1) / count.clamp_min(eps)
    amp_mask = count >= 2
    amp_eps = residual_pred.new_tensor(float(amplitude_eps)).abs().clamp_min(eps)
    pred_std = (pred_var.clamp_min(0.0) + amp_eps).sqrt()
    true_std = (true_var.clamp_min(0.0) + amp_eps).sqrt()
    return masked_l1_loss(pred_std, true_std, amp_mask, eps=eps)


def residual_physics_loss(
    pred_wind,
    residual_pred,
    target,
    mask,
    current_hourly_reference,
    height=None,
    *,
    lambda_wind: float = 1.0,
    lambda_extreme: float = 0.3,
    lambda_residual_weighted: float = 0.0,
    lambda_temporal: float = 0.2,
    lambda_temporal_weighted: float = 0.0,
    lambda_roughness: float = 0.1,
    lambda_amplitude: float = 0.0,
    lambda_vertical: float = 0.05,
    lambda_consistency: float = 0.1,
    extreme_beta: float = 1.0,
    extreme_threshold: float = 10.0,
    extreme_scale: float = 2.0,
    extreme_max_weight: float = 5.0,
    residual_weight_alpha: float = 1.0,
    residual_weight_gamma: float = 1.0,
    residual_weight_q_ref: float = 1.0,
    residual_weight_max: float = 5.0,
    temporal_weight_alpha: float = 1.0,
    temporal_weight_gamma: float = 1.0,
    temporal_weight_q_ref: float = 1.0,
    temporal_weight_max: float = 5.0,
    amplitude_eps: float = 1e-4,
    y_mean=None,
    y_std=None,
    eps: float = 1e-8,
):
    """Residual-physics loss for normalized-space HTQ training.

    The model predicts residuals internally, then forms
    ``pred_wind = current_hourly_reference + residual_pred``. Since
    ``MAE(pred_wind, target)`` is mathematically identical to
    ``MAE(residual_pred, true_residual)`` under this architecture, the direct
    residual L1 term is intentionally omitted. Residuals are still used for
    temporal and roughness structure losses.
    """

    if current_hourly_reference is None:
        raise ValueError("current_hourly_reference is required for residual_physics_loss")
    if residual_pred is None:
        raise ValueError("residual_pred is required for residual_physics_loss")
    if current_hourly_reference.ndim != 3:
        raise ValueError("current_hourly_reference must have shape [B, H, C]")
    if residual_pred.shape != pred_wind.shape:
        raise ValueError("residual_pred must have the same shape as pred_wind")

    true_residual = target - current_hourly_reference.unsqueeze(1)

    wind = masked_l1_loss(pred_wind, target, mask, eps=eps)

    y_std_tensor = None
    if y_std is not None:
        y_std_tensor = pred_wind.new_tensor(y_std).reshape(1, 1, 1, -1)

    if y_mean is not None and y_std is not None:
        y_mean_tensor = pred_wind.new_tensor(y_mean).reshape(1, 1, 1, -1)
        target_for_speed = target * y_std_tensor + y_mean_tensor
    else:
        global _WARNED_EXTREME_NORMALIZED_SPACE
        if not _WARNED_EXTREME_NORMALIZED_SPACE:
            warnings.warn(
                "residual_physics_loss received no y_mean/y_std; extreme wind "
                "weights will be computed in normalized space, so threshold is "
                "not in m/s.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_EXTREME_NORMALIZED_SPACE = True
        target_for_speed = target

    if target_for_speed.shape[-1] >= 2:
        wind_speed = (target_for_speed[..., :2].pow(2).sum(dim=-1, keepdim=True) + eps).sqrt()
    else:
        wind_speed = target_for_speed.abs().mean(dim=-1, keepdim=True)
    scale = pred_wind.new_tensor(float(extreme_scale)).abs().clamp_min(eps)
    weight = 1.0 + float(extreme_beta) * torch.sigmoid(
        (wind_speed - float(extreme_threshold)) / scale
    )
    weight = weight.clamp(max=float(extreme_max_weight)).detach()
    extreme = masked_weighted_l1_loss(pred_wind, target, mask, weight, eps=eps)

    if y_std_tensor is not None:
        true_residual_for_weight = true_residual * y_std_tensor
    else:
        global _WARNED_RESIDUAL_WEIGHT_NORMALIZED_SPACE
        if not _WARNED_RESIDUAL_WEIGHT_NORMALIZED_SPACE:
            warnings.warn(
                "residual_physics_loss received no y_std; residual and temporal "
                "weights will be computed in normalized space, so q_ref is not in m/s.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_RESIDUAL_WEIGHT_NORMALIZED_SPACE = True
        true_residual_for_weight = true_residual

    residual_weight = _magnitude_weight(
        true_residual_for_weight,
        residual_weight_alpha,
        residual_weight_gamma,
        residual_weight_q_ref,
        residual_weight_max,
        eps,
    )
    residual_weighted = masked_weighted_l1_loss(
        residual_pred,
        true_residual,
        mask,
        residual_weight,
        eps=eps,
    )
    temporal = temporal_gradient_loss(residual_pred, true_residual, mask, eps=eps)
    dy_true = true_residual[:, 1:] - true_residual[:, :-1]
    dy_true_for_weight = true_residual_for_weight[:, 1:] - true_residual_for_weight[:, :-1]
    temporal_weight = _magnitude_weight(
        dy_true_for_weight,
        temporal_weight_alpha,
        temporal_weight_gamma,
        temporal_weight_q_ref,
        temporal_weight_max,
        eps,
    )
    temporal_weighted = temporal_weighted_loss(
        residual_pred,
        true_residual,
        mask,
        temporal_weight,
        eps=eps,
    )
    roughness = second_order_temporal_roughness_loss(residual_pred, true_residual, mask, eps=eps)
    amplitude = residual_amplitude_loss(residual_pred, true_residual, mask, eps=eps, amplitude_eps=amplitude_eps)

    if float(lambda_vertical) == 0.0:
        vertical = _zero_loss_like(pred_wind)
    else:
        vertical = vertical_shear_loss(pred_wind, target, mask, height, eps=eps)

    if float(lambda_consistency) == 0.0:
        consistency = _zero_loss_like(pred_wind)
    else:
        valid = mask.to(dtype=pred_wind.dtype)
        hourly_pred = (pred_wind * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(eps)
        consistency_mask = mask.any(dim=1)
        consistency = masked_l1_loss(
            hourly_pred,
            current_hourly_reference,
            consistency_mask,
            eps=eps,
        )

    total = (
        lambda_wind * wind
        + lambda_extreme * extreme
        + lambda_residual_weighted * residual_weighted
        + lambda_temporal * temporal
        + lambda_temporal_weighted * temporal_weighted
        + lambda_roughness * roughness
        + lambda_amplitude * amplitude
        + lambda_vertical * vertical
        + lambda_consistency * consistency
    )
    valid_weight = weight.expand_as(mask).masked_select(mask)
    if valid_weight.numel() == 0:
        mean_extreme_weight = _zero_loss_like(pred_wind)
        max_extreme_weight = _zero_loss_like(pred_wind)
    else:
        mean_extreme_weight = valid_weight.mean()
        max_extreme_weight = valid_weight.max()
    return {
        "loss": total,
        "wind": wind,
        "extreme": extreme,
        "residual_weighted": residual_weighted,
        "temporal": temporal,
        "temporal_weighted": temporal_weighted,
        "roughness": roughness,
        "amplitude": amplitude,
        "vertical": vertical,
        "consistency": consistency,
        "mean_extreme_weight": mean_extreme_weight,
        "max_extreme_weight": max_extreme_weight,
        "mean_residual_weight": residual_weight.expand_as(mask).masked_select(mask).mean()
        if mask.any()
        else _zero_loss_like(pred_wind),
        "max_residual_weight": residual_weight.expand_as(mask).masked_select(mask).max()
        if mask.any()
        else _zero_loss_like(pred_wind),
        "mean_temporal_weight": temporal_weight.expand_as(mask[:, 1:]).masked_select(mask[:, 1:] & mask[:, :-1]).mean()
        if (mask[:, 1:] & mask[:, :-1]).any()
        else _zero_loss_like(pred_wind),
        "max_temporal_weight": temporal_weight.expand_as(mask[:, 1:]).masked_select(mask[:, 1:] & mask[:, :-1]).max()
        if (mask[:, 1:] & mask[:, :-1]).any()
        else _zero_loss_like(pred_wind),
    }
