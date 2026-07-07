import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.dataset import WindDownscalingDataset, available_splits


DATASET_DIR = "data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1"


def test_dataset_loader_returns_training_tensors():
    dataset = WindDownscalingDataset(DATASET_DIR, split="train")

    assert len(dataset) > 0
    assert dataset.sample_shapes.input_context == (6, 6, 2)
    assert dataset.sample_shapes.target_10min == (6, 6, 2)

    item = dataset[0]

    assert item["x_hourly"].shape == (6, 6, 2)
    assert item["x_mask"].shape == (6, 6, 2)
    assert item["y_10min"].shape == (6, 6, 2)
    assert item["y_mask"].shape == (6, 6, 2)
    assert item["current_hourly"].shape == (6, 2)
    assert item["x_meteo"].shape == (6, 5, 2)
    assert item["meteo_mask"].shape == (6, 5, 2)
    assert item["x_hourly"].dtype == torch.float32
    assert item["x_mask"].dtype == torch.bool
    assert item["y_10min"].dtype == torch.float32
    assert item["y_mask"].dtype == torch.bool
    assert item["x_meteo"].dtype == torch.float32
    assert item["meteo_mask"].dtype == torch.bool
    assert torch.allclose(item["current_hourly"], item["x_hourly"][-1])
    assert item["meteo_mask"].all()
    assert item["split"] == "train"


def test_available_splits_matches_split_files():
    splits = available_splits(DATASET_DIR)

    assert {"train", "val", "test", "gap"}.issubset(splits)
    assert splits["train"] > 0
    assert splits["val"] > 0
    assert splits["test"] > 0
    assert splits["gap"] > 0


def test_dataset_loader_all_split_includes_gap_samples():
    all_dataset = WindDownscalingDataset(DATASET_DIR, split="all", return_metadata=False)
    splits = available_splits(DATASET_DIR)

    assert len(all_dataset) == sum(splits.values())
