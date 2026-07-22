import torch

from scripts.train import default_loss_config, compute_loss_parts
from src.training.losses import (
    masked_mse_loss,
    masked_temporal_correlation_loss,
    residual_amplitude_loss,
    residual_physics_loss,
    temporal_gradient_amplitude_loss,
    temporal_gradient_correlation_loss,
    vertical_shear_loss,
)


def test_residual_physics_loss_forward_and_keys():
    batch_size, target_steps, height_levels, channels = 2, 6, 6, 2
    current = torch.randn(batch_size, height_levels, channels)
    residual_pred = torch.randn(batch_size, target_steps, height_levels, channels) * 0.1
    target_residual = torch.randn(batch_size, target_steps, height_levels, channels) * 0.1
    pred = current.unsqueeze(1) + residual_pred
    target = current.unsqueeze(1) + target_residual
    mask = torch.ones(batch_size, target_steps, height_levels, channels, dtype=torch.bool)
    mask[:, 0, 0, :] = False
    height = torch.tensor([250.0, 275.0, 300.0, 325.0, 350.0, 375.0])

    parts = residual_physics_loss(
        pred,
        residual_pred,
        target,
        mask,
        current,
        height,
        y_mean=[0.0, 0.0],
        y_std=[1.0, 1.0],
    )

    expected = {
        "loss",
        "wind",
        "extreme",
        "residual_weighted",
        "temporal",
        "temporal_weighted",
        "roughness",
        "amplitude",
        "gradient_amplitude",
        "residual_corr",
        "temporal_gradient_corr",
        "vertical",
        "consistency",
        "mean_extreme_weight",
        "max_extreme_weight",
        "mean_residual_weight",
        "max_residual_weight",
        "mean_temporal_weight",
        "max_temporal_weight",
    }
    assert expected <= set(parts)
    for key in expected:
        assert torch.isfinite(parts[key])
    assert parts["loss"].ndim == 0


def test_vertical_shear_loss_accepts_batched_height():
    pred = torch.randn(2, 6, 6, 2)
    target = torch.randn(2, 6, 6, 2)
    mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)
    height = torch.tensor(
        [
            [250.0, 275.0, 300.0, 325.0, 350.0, 375.0],
            [250.0, 260.0, 270.0, 280.0, 290.0, 300.0],
        ]
    )

    loss = vertical_shear_loss(pred, target, mask, height)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_compute_loss_parts_supports_residual_physics():
    pred = torch.randn(2, 6, 6, 2)
    residual = torch.randn(2, 6, 6, 2) * 0.1
    batch = {
        "y_10min": torch.randn(2, 6, 6, 2),
        "y_mask": torch.ones(2, 6, 6, 2, dtype=torch.bool),
        "current_hourly_y_norm": torch.randn(2, 6, 2),
        "height": torch.tensor([250.0, 275.0, 300.0, 325.0, 350.0, 375.0]).repeat(2, 1),
    }
    config = default_loss_config()
    config.update({"type": "residual_physics", "y_mean": [0.0, 0.0], "y_std": [1.0, 1.0]})

    parts = compute_loss_parts({"pred": pred, "residual": residual}, batch, config)

    assert torch.isfinite(parts["loss"])
    assert "roughness" in parts


def test_residual_physics_consistency_handles_invalid_mask():
    pred = torch.randn(2, 6, 6, 2)
    residual = torch.randn(2, 6, 6, 2)
    target = torch.randn(2, 6, 6, 2)
    current = torch.randn(2, 6, 2)
    mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)
    mask[:, :, 0, :] = False
    height = torch.tensor([250.0, 275.0, 300.0, 325.0, 350.0, 375.0]).repeat(2, 1)

    parts = residual_physics_loss(
        pred,
        residual,
        target,
        mask,
        current,
        height,
        y_mean=[0.0, 0.0],
        y_std=[1.0, 1.0],
    )

    assert torch.isfinite(parts["consistency"])
    assert torch.isfinite(parts["loss"])


def test_residual_physics_extreme_weight_range_uses_physical_speed():
    pred = torch.zeros(1, 6, 6, 2)
    residual = torch.zeros(1, 6, 6, 2)
    target = torch.ones(1, 6, 6, 2)
    current = torch.zeros(1, 6, 2)
    mask = torch.ones(1, 6, 6, 2, dtype=torch.bool)
    height = torch.tensor([[250.0, 275.0, 300.0, 325.0, 350.0, 375.0]])
    max_weight = 2.0

    parts = residual_physics_loss(
        pred,
        residual,
        target,
        mask,
        current,
        height,
        y_mean=[0.0, 0.0],
        y_std=[10.0, 10.0],
        extreme_beta=10.0,
        extreme_threshold=5.0,
        extreme_scale=1.0,
        extreme_max_weight=max_weight,
    )

    assert parts["mean_extreme_weight"] >= 1.0
    assert parts["max_extreme_weight"] <= max_weight


def test_residual_physics_supports_residual_and_temporal_weighted_terms():
    pred = torch.zeros(1, 6, 2, 2)
    residual = torch.zeros_like(pred)
    current = torch.zeros(1, 2, 2)
    true_residual = torch.zeros_like(pred)
    true_residual[:, 3, :, :] = 2.0
    target = current.unsqueeze(1) + true_residual
    mask = torch.ones_like(target, dtype=torch.bool)
    height = torch.tensor([[250.0, 275.0]])

    parts = residual_physics_loss(
        pred,
        residual,
        target,
        mask,
        current,
        height,
        lambda_extreme=0.0,
        lambda_residual_weighted=0.5,
        lambda_temporal_weighted=0.5,
        lambda_amplitude=0.05,
        residual_weight_q_ref=1.0,
        temporal_weight_q_ref=1.0,
        y_mean=[0.0, 0.0],
        y_std=[1.0, 1.0],
    )

    assert torch.isfinite(parts["loss"])
    assert parts["residual_weighted"] > 0
    assert parts["temporal_weighted"] > 0
    assert parts["amplitude"] > 0


def test_residual_and_temporal_weights_use_physical_y_std():
    pred = torch.zeros(1, 6, 1, 2)
    residual = torch.zeros_like(pred)
    current = torch.zeros(1, 1, 2)
    target = torch.zeros_like(pred)
    target[:, 1:, :, :] = 0.5
    mask = torch.ones_like(target, dtype=torch.bool)
    height = torch.tensor([[250.0]])

    parts = residual_physics_loss(
        pred,
        residual,
        target,
        mask,
        current,
        height,
        lambda_residual_weighted=1.0,
        lambda_temporal_weighted=1.0,
        residual_weight_q_ref=1.0,
        temporal_weight_q_ref=1.0,
        y_mean=[0.0, 0.0],
        y_std=[4.0, 4.0],
    )

    assert parts["mean_residual_weight"] > 3.0
    assert parts["max_temporal_weight"] > 3.0


def test_residual_amplitude_loss_uses_only_positions_with_two_valid_times():
    pred = torch.zeros(1, 6, 2, 2)
    target = torch.zeros_like(pred)
    target[:, :, 0, :] = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    target[:, :, 1, :] = 100.0
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :, 0, :] = True
    mask[:, 0, 1, :] = True

    loss = residual_amplitude_loss(pred, target, mask, amplitude_eps=1e-4)

    assert torch.isfinite(loss)
    assert loss > 0


def test_fluctuation_losses_penalize_flat_prediction_and_accept_perfect_shape():
    true_residual = torch.tensor(
        [[[[0.0, 0.0]], [[1.0, -1.0]], [[-1.0, 1.0]], [[2.0, -2.0]], [[0.0, 0.0]], [[-2.0, 2.0]]]]
    )
    flat = torch.zeros_like(true_residual)
    mask = torch.ones_like(true_residual, dtype=torch.bool)

    assert masked_temporal_correlation_loss(flat, true_residual, mask) > 0.9
    assert temporal_gradient_correlation_loss(flat, true_residual, mask) > 0.9
    assert temporal_gradient_amplitude_loss(flat, true_residual, mask) > 0
    assert masked_temporal_correlation_loss(true_residual, true_residual, mask) < 1e-6
    assert temporal_gradient_correlation_loss(true_residual, true_residual, mask) < 1e-6
    assert temporal_gradient_amplitude_loss(true_residual, true_residual, mask) < 1e-6


def test_fluctuation_losses_are_mask_aware_and_finite_with_flat_targets():
    pred = torch.randn(2, 6, 2, 2)
    target = torch.zeros_like(pred)
    mask = torch.ones_like(pred, dtype=torch.bool)
    mask[:, :4, 0, :] = False

    residual_corr = masked_temporal_correlation_loss(pred, target, mask)
    temporal_corr = temporal_gradient_correlation_loss(pred, target, mask)
    gradient_amplitude = temporal_gradient_amplitude_loss(pred, target, mask)

    assert torch.isfinite(residual_corr)
    assert torch.isfinite(temporal_corr)
    assert torch.isfinite(gradient_amplitude)
    assert residual_corr == 0
    assert temporal_corr == 0


def test_fluctuation_focused_total_loss_backpropagates_from_flat_prediction():
    residual = torch.zeros(1, 6, 1, 2, requires_grad=True)
    current = torch.zeros(1, 1, 2)
    target = torch.tensor(
        [[[[0.0, 0.0]], [[1.0, -1.0]], [[-1.0, 1.0]], [[2.0, -2.0]], [[0.0, 0.0]], [[-2.0, 2.0]]]]
    )
    mask = torch.ones_like(target, dtype=torch.bool)
    parts = residual_physics_loss(
        current.unsqueeze(1) + residual,
        residual,
        target,
        mask,
        current,
        lambda_wind=0.0,
        lambda_extreme=0.0,
        lambda_temporal=0.0,
        lambda_roughness=0.0,
        lambda_vertical=0.0,
        lambda_consistency=0.0,
        lambda_gradient_amplitude=0.05,
        lambda_residual_corr=0.05,
        lambda_temporal_gradient_corr=0.05,
        y_mean=[0.0, 0.0],
        y_std=[1.0, 1.0],
    )

    parts["loss"].backward()

    assert torch.isfinite(parts["loss"])
    assert torch.isfinite(residual.grad).all()
    assert residual.grad.abs().sum() > 0


def test_masked_mse_loss_alias():
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([0.0, 0.0])
    mask = torch.tensor([True, False])

    loss = masked_mse_loss(pred, target, mask)

    assert torch.allclose(loss, torch.tensor(1.0))
