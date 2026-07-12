import torch

from scripts.train import default_loss_config, compute_loss_parts
from src.training.losses import (
    masked_mse_loss,
    residual_physics_loss,
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
        "residual",
        "extreme",
        "temporal",
        "roughness",
        "vertical",
        "consistency",
        "mean_extreme_weight",
        "max_extreme_weight",
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


def test_masked_mse_loss_alias():
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([0.0, 0.0])
    mask = torch.tensor([True, False])

    loss = masked_mse_loss(pred, target, mask)

    assert torch.allclose(loss, torch.tensor(1.0))
