import pytest

torch = pytest.importorskip("torch")

from src.training.losses import (
    htq_fluctuation_aware_loss,
    htq_reconstruction_loss,
    masked_l1_loss,
    masked_mae,
    masked_mse,
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


def test_htq_reconstruction_loss_weighted_sum():
    pred = torch.tensor([[[[0.0]], [[2.0]], [[5.0]]]])
    target = torch.tensor([[[[0.0]], [[1.0]], [[3.0]]]])
    mask = torch.ones_like(pred, dtype=torch.bool)

    parts = htq_reconstruction_loss(
        pred,
        target,
        mask,
        lambda_l1=1.0,
        lambda_temporal=0.2,
        lambda_vertical=0.05,
    )

    assert parts["l1"].item() == 1.0
    assert parts["temporal"].item() == 1.0
    assert parts["vertical"].item() == 0.0
    assert parts["loss"].item() == pytest.approx(1.2)
    assert "zero_mean" not in parts


def test_fluctuation_aware_loss_returns_weighted_terms():
    pred = torch.zeros(1, 2, 1, 2)
    target = torch.tensor([[[[1.0, 0.0]], [[3.0, 4.0]]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    current = torch.zeros(1, 1, 2)

    parts = htq_fluctuation_aware_loss(
        pred,
        target,
        mask,
        current,
        lambda_l1=1.0,
        lambda_weighted=0.5,
        lambda_temporal=0.2,
        lambda_vertical=0.05,
        alpha=1.0,
        gamma=1.0,
        q_ref=1.0,
        max_weight=3.0,
    )

    assert set(parts) == {"loss", "l1", "weighted_l1", "temporal", "vertical", "mean_weight", "max_weight"}
    assert parts["weighted_l1"].item() > 0.0
    assert parts["mean_weight"].item() >= 1.0
    assert parts["max_weight"].item() <= 3.0
    assert torch.isfinite(parts["loss"])
