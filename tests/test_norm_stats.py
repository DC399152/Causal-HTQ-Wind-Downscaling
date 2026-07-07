from pathlib import Path

import numpy as np
import pytest

from src.data.normalization import compute_norm_stats, save_norm_stats


DATASET_DIR = "data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1"


def test_compute_norm_stats_train_only_shapes_and_counts(tmp_path):
    stats = compute_norm_stats(DATASET_DIR, split="train")

    assert stats["computed_from_split"] == "train"
    assert stats["num_samples"] > 0
    assert len(stats["x_mean"]) == 2
    assert len(stats["x_std"]) == 2
    assert len(stats["y_mean"]) == 2
    assert len(stats["y_std"]) == 2
    assert all(v > 0 for v in stats["x_std"])
    assert all(v > 0 for v in stats["y_std"])
    assert all(v > 0 for v in stats["x_count"])
    assert all(v > 0 for v in stats["y_count"])
    assert len(stats["meteo_mean"]) == 2
    assert len(stats["meteo_std"]) == 2
    assert all(v > 0 for v in stats["meteo_std"])
    assert all(v > 0 for v in stats["meteo_count"])

    output = tmp_path / "norm_stats.json"
    save_norm_stats(stats, output)
    assert output.exists()


def test_compute_norm_stats_ignores_masked_values(tmp_path):
    dataset_dir = tmp_path / "dataset"
    split_dir = dataset_dir / "splits"
    split_dir.mkdir(parents=True)

    x = np.asarray([[[[1.0, 10.0], [999.0, 999.0]]]], dtype=np.float32)
    y = np.asarray([[[[2.0, 20.0], [999.0, 999.0]]]], dtype=np.float32)
    mask = np.asarray([[[[True, True], [False, False]]]], dtype=bool)
    np.savez_compressed(
        dataset_dir / "dataset.npz",
        x_hourly=x,
        x_mask=mask,
        y_10min=y,
        y_mask=mask,
        current_hourly=x[:, -1],
        station_id=np.asarray(["s0"], dtype=object),
        target_time_start=np.asarray(["2024-01-01T00:00"], dtype=object),
        target_times_10min=np.asarray([["2024-01-01T00:00"]], dtype=object),
        height_values=np.asarray([[250.0]], dtype=np.float32),
        source_file=np.asarray(["fake"], dtype=object),
        split=np.asarray(["train"], dtype=object),
    )
    (split_dir / "train.txt").write_text("0\n", encoding="utf-8")

    stats = compute_norm_stats(dataset_dir, split="train")

    assert stats["x_mean"] == [1.0, 10.0]
    assert stats["y_mean"] == [2.0, 20.0]
    assert stats["x_count"] == [1, 1]
    assert stats["y_count"] == [1, 1]


def test_dataset_normalize_uses_stats_when_torch_available(tmp_path):
    torch = pytest.importorskip("torch")

    from src.data.dataset import WindDownscalingDataset

    stats = compute_norm_stats(DATASET_DIR, split="train")
    output = tmp_path / "norm_stats.json"
    save_norm_stats(stats, output)
    dataset = WindDownscalingDataset(DATASET_DIR, split="train", normalize=True, norm_stats_path=output)
    item = dataset[0]

    assert item["x_hourly"].shape == (6, 6, 2)
    assert item["y_10min"].shape == (6, 6, 2)
    assert item["current_hourly"].shape == (6, 2)
    assert item["x_mask"].dtype == torch.bool
    assert item["y_mask"].dtype == torch.bool
    assert torch.all(item["x_hourly"][~item["x_mask"]] == 0)
    assert torch.all(item["y_10min"][~item["y_mask"]] == 0)
    assert item["x_meteo"].shape == (6, 5, 2)
    assert item["meteo_mask"].dtype == torch.bool
    assert torch.all(item["x_meteo"][~item["meteo_mask"]] == 0)
