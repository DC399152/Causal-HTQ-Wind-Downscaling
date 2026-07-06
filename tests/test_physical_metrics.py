import math

import pytest

torch = pytest.importorskip("torch")

from src.training.metrics import (
    add_metric_sums,
    empty_physical_metric_sums,
    finalize_physical_metrics,
    physical_metric_sums,
    temporal_acc_sum,
)


def test_temporal_acc_is_one_for_matching_varying_series():
    values = torch.tensor([[[[1.0]], [[2.0]], [[4.0]], [[7.0]]]])
    mask = torch.ones_like(values, dtype=torch.bool)

    acc_sum, acc_count = temporal_acc_sum(values, values, mask)

    assert acc_count == 1.0
    assert acc_sum == pytest.approx(1.0)


def test_temporal_acc_skips_constant_prediction_series():
    pred = torch.ones(1, 4, 1, 1)
    target = torch.tensor([[[[1.0]], [[2.0]], [[4.0]], [[7.0]]]])
    mask = torch.ones_like(target, dtype=torch.bool)

    acc_sum, acc_count = temporal_acc_sum(pred, target, mask)

    assert acc_sum == 0.0
    assert acc_count == 0.0


def test_physical_metric_sums_include_speed_and_gradient_metrics():
    target = torch.tensor(
        [
            [
                [[3.0, 4.0]],
                [[6.0, 8.0]],
                [[9.0, 12.0]],
            ]
        ]
    )
    pred = target + 1.0
    mask = torch.ones_like(target, dtype=torch.bool)

    sums = empty_physical_metric_sums()
    add_metric_sums(sums, physical_metric_sums(pred, target, mask))
    metrics = finalize_physical_metrics(sums)

    assert metrics["MAE_ms"] == pytest.approx(1.0)
    assert metrics["RMSE_ms"] == pytest.approx(1.0)
    assert metrics["u_MAE_ms"] == pytest.approx(1.0)
    assert metrics["v_MAE_ms"] == pytest.approx(1.0)
    assert metrics["temporal_gradient_MAE_ms"] == pytest.approx(0.0)
    assert metrics["residual_ACC"] == pytest.approx(1.0)
    assert math.isnan(metrics["temporal_gradient_ACC"])
