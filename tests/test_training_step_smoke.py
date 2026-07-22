from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.data.dataset import WindDownscalingDataset
from src.models.htq_transformer import CausalHTQTransformer
from scripts.train import compute_loss_parts, default_loss_config


DATASET_DIR = Path("data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1")


@pytest.mark.skipif(not (DATASET_DIR / "dataset.npz").exists(), reason="generated dataset not found")
def test_one_step_backward_with_masked_loss_on_real_batch():
    dataset = WindDownscalingDataset(
        DATASET_DIR,
        split="train",
        return_metadata=False,
        normalize=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    model = CausalHTQTransformer()
    model.train()

    out = model(batch["x_hourly"], batch["x_mask"])
    loss_config = default_loss_config()
    loss_config.update({"y_mean": [0.0, 0.0], "y_std": [1.0, 1.0]})
    loss_parts = compute_loss_parts(out, batch, loss_config)
    loss = loss_parts["loss"]
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_parts["wind"])
    assert torch.isfinite(loss_parts["extreme"])
    assert torch.isfinite(loss_parts["temporal"])
    assert torch.isfinite(loss_parts["roughness"])
    assert torch.isfinite(loss_parts["vertical"])

    finite_grad_count = 0
    nonzero_grad_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all()
        finite_grad_count += 1
        if parameter.grad.abs().sum() > 0:
            nonzero_grad_count += 1

    assert finite_grad_count > 0
    assert nonzero_grad_count > 0


def test_default_training_loss_config_is_residual_physics_without_zero_mean():
    config = default_loss_config()
    assert config["type"] == "residual_physics"

    pred = torch.zeros(1, 3, 2, 2)
    residual = torch.zeros(1, 3, 2, 2)
    target = torch.ones(1, 3, 2, 2)
    mask = torch.ones_like(target, dtype=torch.bool)
    batch = {
        "y_10min": target,
        "y_mask": mask,
        "current_hourly_y_norm": torch.zeros(1, 2, 2),
        "height": torch.tensor([[250.0, 275.0]]),
    }
    config.update({"y_mean": [0.0, 0.0], "y_std": [1.0, 1.0]})

    parts = compute_loss_parts({"pred": pred, "residual": residual}, batch, config)

    assert "zero_mean" not in parts
    assert "weighted_l1" not in parts
    assert {
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
    } <= set(parts)
    assert config["lambda_gradient_amplitude"] == 0.0
    assert config["lambda_residual_corr"] == 0.0
    assert config["lambda_temporal_gradient_corr"] == 0.0
