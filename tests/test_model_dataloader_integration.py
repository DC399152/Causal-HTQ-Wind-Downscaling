from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.data.dataset import WindDownscalingDataset
from src.models.htq_transformer import CausalHTQTransformer


DATASET_DIR = Path("data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1")


@pytest.mark.skipif(not (DATASET_DIR / "dataset.npz").exists(), reason="generated dataset not found")
def test_causal_htq_transformer_accepts_real_dataloader_batch():
    dataset = WindDownscalingDataset(
        DATASET_DIR,
        split="train",
        return_metadata=False,
        normalize=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    model = CausalHTQTransformer()
    model.eval()

    with torch.no_grad():
        out = model(batch["x_hourly"], batch["x_mask"])

    assert batch["x_hourly"].shape == (4, 6, 6, 2)
    assert batch["x_mask"].shape == (4, 6, 6, 2)
    assert torch.allclose(batch["current_hourly"], batch["x_hourly"][:, -1])
    assert out["pred"].shape == (4, 6, 6, 2)
    assert out["residual"].shape == (4, 6, 6, 2)
    assert out["encoder_memory"].shape == (4, 36, 64)
    assert out["fusion_info"] is None
    expected = batch["current_hourly"].unsqueeze(1) + out["residual"]
    assert torch.allclose(out["pred"], expected, atol=1e-6)
