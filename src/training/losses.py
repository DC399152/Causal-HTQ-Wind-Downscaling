"""Mask-aware losses for HTQ training."""

from __future__ import annotations


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


def htq_reconstruction_loss(
    pred,
    target,
    mask,
    *,
    lambda_l1: float = 1.0,
    lambda_temporal: float = 0.2,
    lambda_vertical: float = 0.05,
    eps: float = 1e-8,
):
    """Weighted normalized-space HTQ reconstruction loss.

    total_loss = lambda_l1 * masked_l1_loss
               + lambda_temporal * temporal_gradient_loss
               + lambda_vertical * vertical_gradient_loss
    """

    l1 = masked_l1_loss(pred, target, mask, eps=eps)
    temporal = temporal_gradient_loss(pred, target, mask, eps=eps)
    vertical = vertical_gradient_loss(pred, target, mask, eps=eps)
    total = lambda_l1 * l1 + lambda_temporal * temporal + lambda_vertical * vertical
    return {
        "loss": total,
        "l1": l1,
        "temporal": temporal,
        "vertical": vertical,
    }


def htq_fluctuation_aware_loss(
    pred,
    target,
    mask,
    current_hourly_reference,
    *,
    lambda_l1: float = 1.0,
    lambda_weighted: float = 0.5,
    lambda_temporal: float = 0.2,
    lambda_vertical: float = 0.05,
    alpha: float = 1.0,
    gamma: float = 1.0,
    q_ref: float = 1.0,
    max_weight: float = 5.0,
    eps: float = 1e-8,
):
    """Fluctuation-aware normalized-space HTQ reconstruction loss.

    ``current_hourly_reference`` should be [B, H, C] in the same y-normalized
    space as ``target``. Weights are derived from true residual magnitude and
    detached so gradients flow only through prediction errors.
    """

    if current_hourly_reference is None:
        raise ValueError("current_hourly_reference is required for fluctuation-aware loss")
    if current_hourly_reference.ndim != 3:
        raise ValueError("current_hourly_reference must have shape [B, H, C]")

    true_residual = target - current_hourly_reference.unsqueeze(1)
    if true_residual.shape[-1] < 2:
        residual_mag = true_residual.abs().mean(dim=-1, keepdim=True)
    else:
        residual_mag = (true_residual[..., :2].pow(2).sum(dim=-1, keepdim=True) + eps).sqrt()

    q_ref_tensor = pred.new_tensor(float(q_ref)).abs().clamp_min(eps)
    weight = 1.0 + float(alpha) * (residual_mag / q_ref_tensor).clamp_min(0.0).pow(float(gamma))
    weight = weight.clamp(max=float(max_weight)).detach()

    l1 = masked_l1_loss(pred, target, mask, eps=eps)
    weighted_l1 = masked_weighted_l1_loss(pred, target, mask, weight, eps=eps)
    temporal = temporal_gradient_loss(pred, target, mask, eps=eps)
    vertical = vertical_gradient_loss(pred, target, mask, eps=eps)
    total = (
        lambda_l1 * l1
        + lambda_weighted * weighted_l1
        + lambda_temporal * temporal
        + lambda_vertical * vertical
    )
    valid_weight = weight.expand_as(mask).masked_select(mask)
    if valid_weight.numel() == 0:
        mean_weight = weight.sum() * 0.0
        max_weight_value = weight.sum() * 0.0
    else:
        mean_weight = valid_weight.mean()
        max_weight_value = valid_weight.max()
    return {
        "loss": total,
        "l1": l1,
        "weighted_l1": weighted_l1,
        "temporal": temporal,
        "vertical": vertical,
        "mean_weight": mean_weight,
        "max_weight": max_weight_value,
    }

