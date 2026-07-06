import pytest

torch = pytest.importorskip("torch")

from src.models.baselines import repeat_current_hour
from src.training.losses import masked_mae, masked_mse


def test_repeat_current_hour_shape_and_values():
    current = torch.randn(4, 6, 2)

    pred = repeat_current_hour(current, target_steps=6)

    assert pred.shape == (4, 6, 6, 2)
    assert torch.allclose(pred[:, 0], current)
    assert torch.allclose(pred[:, -1], current)


def test_repeat_current_hour_can_be_scored_with_masked_metrics():
    current = torch.zeros(1, 2, 1)
    target = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    mask = torch.tensor([[[[True], [False]], [[True], [False]]]])

    pred = repeat_current_hour(current, target_steps=2)

    assert masked_mae(pred, target, mask).item() == 2.0
    assert masked_mse(pred, target, mask).item() == 5.0
