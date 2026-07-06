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


def zero_mean_residual_penalty(residual, mask=None, eps: float = 1e-8):
    """Penalize non-zero target-time mean residuals.

    If ``mask`` is provided, the target-time mean is computed over valid target
    positions only. The residual shape is [B, T_out, H, C].
    """

    if mask is None:
        mean_residual = residual.mean(dim=1)
        return (mean_residual * mean_residual).mean()

    valid = mask.to(dtype=residual.dtype)
    numerator = (residual * valid).sum(dim=1)
    denominator = valid.sum(dim=1).clamp_min(eps)
    mean_residual = numerator / denominator
    valid_any = mask.any(dim=1)
    if not valid_any.any():
        return residual.sum() * 0.0
    return (mean_residual.pow(2) * valid_any.to(dtype=residual.dtype)).sum() / valid_any.to(dtype=residual.dtype).sum().clamp_min(eps)
