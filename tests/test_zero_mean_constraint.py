import pytest

torch = pytest.importorskip("torch")

from src.training.losses import (
    htq_reconstruction_loss,
    masked_l1_loss,
    masked_mae,
    masked_mse,
    temporal_gradient_loss,
    vertical_gradient_loss,
    zero_mean_residual_penalty,
)
from src.models.htq_transformer import CausalHTQTransformer


def test_zero_mean_residual_penalty_is_zero_for_balanced_residual():
    residual = torch.tensor(
        [
            [
                [[1.0], [2.0]],
                [[-1.0], [-2.0]],
            ]
        ]
    )

    assert zero_mean_residual_penalty(residual).item() == 0.0


def test_masked_mse_and_mae_ignore_invalid_positions():
    pred = torch.tensor([[[[1.0], [100.0]]]])
    target = torch.tensor([[[[3.0], [0.0]]]])
    mask = torch.tensor([[[[True], [False]]]])

    assert masked_mse(pred, target, mask).item() == 4.0
    assert masked_mae(pred, target, mask).item() == 2.0


def test_zero_mean_residual_penalty_uses_masked_target_time_mean():
    residual = torch.tensor(
        [
            [
                [[1.0]],
                [[-1.0]],
                [[100.0]],
            ]
        ]
    )
    mask = torch.tensor(
        [
            [
                [[True]],
                [[True]],
                [[False]],
            ]
        ]
    )

    assert zero_mean_residual_penalty(residual, mask).item() == 0.0


def test_causal_htq_transformer_no_longer_enforces_zero_mean_residual():
    torch.manual_seed(2)
    model = CausalHTQTransformer()
    model.eval()
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)

    with torch.no_grad():
        out = model(x_hourly, x_mask)

    expected = x_hourly[:, -1].unsqueeze(1) + out["residual"]
    assert torch.allclose(out["pred"], expected, atol=1e-6)


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
