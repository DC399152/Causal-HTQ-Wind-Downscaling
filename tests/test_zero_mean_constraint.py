import pytest

torch = pytest.importorskip("torch")

from src.training.losses import (
    masked_l1_loss,
    masked_mae,
    masked_mse,
    residual_physics_loss,
    temporal_gradient_loss,
    vertical_gradient_loss,
)
from src.models.htq_transformer import CausalHTQTransformer, HTQConfig


def test_masked_mse_and_mae_ignore_invalid_positions():
    pred = torch.tensor([[[[1.0], [100.0]]]])
    target = torch.tensor([[[[3.0], [0.0]]]])
    mask = torch.tensor([[[[True], [False]]]])

    assert masked_mse(pred, target, mask).item() == 4.0
    assert masked_mae(pred, target, mask).item() == 2.0


def test_causal_htq_transformer_default_does_not_enforce_zero_mean_residual():
    torch.manual_seed(2)
    model = CausalHTQTransformer()
    model.eval()
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)

    with torch.no_grad():
        out = model(x_hourly, x_mask)

    expected = x_hourly[:, -1].unsqueeze(1) + out["residual"]
    assert torch.allclose(out["pred"], expected, atol=1e-6)
    assert torch.isfinite(out["pred"]).all()
    assert torch.isfinite(out["residual"]).all()


def test_causal_htq_transformer_rejects_zero_mean_residual_enforcement():
    with pytest.raises(ValueError, match="enforce_zero_mean_residual"):
        CausalHTQTransformer(HTQConfig(enforce_zero_mean_residual=True))


def test_masked_l1_loss_ignores_invalid_positions():
    pred = torch.tensor([[[[1.0], [100.0]]]])
    target = torch.tensor([[[[3.0], [0.0]]]])
    mask = torch.tensor([[[[True], [False]]]])

    assert masked_l1_loss(pred, target, mask).item() == 2.0


def test_temporal_gradient_loss_uses_adjacent_valid_pairs():
    pred = torch.tensor([[[[0.0]], [[2.0]], [[5.0]]]])
    target = torch.tensor([[[[0.0]], [[1.0]], [[3.0]]]])
    mask = torch.tensor([[[[True]], [[True]], [[False]]]])

    assert temporal_gradient_loss(pred, target, mask).item() == 1.0


def test_vertical_gradient_loss_uses_adjacent_valid_height_pairs():
    pred = torch.tensor([[[[0.0], [2.0], [5.0]]]])
    target = torch.tensor([[[[0.0], [1.0], [3.0]]]])
    mask = torch.ones_like(pred, dtype=torch.bool)

    assert vertical_gradient_loss(pred, target, mask).item() == 1.0


def test_residual_physics_loss_has_no_zero_mean_or_legacy_terms():
    pred = torch.zeros(1, 3, 2, 2)
    residual = torch.zeros(1, 3, 2, 2)
    target = torch.ones(1, 3, 2, 2)
    mask = torch.ones_like(target, dtype=torch.bool)
    current = torch.zeros(1, 2, 2)
    height = torch.tensor([[250.0, 275.0]])

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

    assert set(parts) == {
        "loss",
        "wind",
        "extreme",
        "temporal",
        "roughness",
        "vertical",
        "consistency",
        "mean_extreme_weight",
        "max_extreme_weight",
    }
    assert "zero_mean" not in parts
    assert "l1" not in parts
    assert "weighted_l1" not in parts
    assert torch.isfinite(parts["loss"])
