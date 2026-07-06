import pytest

torch = pytest.importorskip("torch")

from src.training.losses import zero_mean_residual_penalty


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

