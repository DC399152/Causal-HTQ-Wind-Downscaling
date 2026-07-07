import numpy as np
import pytest

from src.data.dataset import WindDownscalingDataset


def _write_split_files(dataset_dir, n=2):
    split_dir = dataset_dir / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("\n".join(str(i) for i in range(n)) + "\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("", encoding="utf-8")
    (split_dir / "test.txt").write_text("", encoding="utf-8")


def _write_dummy_dataset(dataset_dir, include_static: bool):
    dataset_dir.mkdir(parents=True)
    n = 2
    arrays = {
        "x_hourly": np.zeros((n, 6, 6, 2), dtype=np.float32),
        "x_mask": np.ones((n, 6, 6, 2), dtype=bool),
        "y_10min": np.zeros((n, 6, 6, 2), dtype=np.float32),
        "y_mask": np.ones((n, 6, 6, 2), dtype=bool),
        "current_hourly": np.zeros((n, 6, 2), dtype=np.float32),
        "station_id": np.asarray(["s0", "s1"], dtype=object),
        "target_time_start": np.asarray(["2024-01-01T00:00", "2024-01-01T01:00"], dtype=object),
        "target_times_10min": np.asarray(
            [
                ["2024-01-01T00:00"] * 6,
                ["2024-01-01T01:00"] * 6,
            ],
            dtype=object,
        ),
        "height_values": np.zeros((n, 6), dtype=np.float32),
        "source_file": np.asarray(["fake", "fake"], dtype=object),
        "split": np.asarray(["train", "train"], dtype=object),
    }
    if include_static:
        arrays["x_static"] = np.full((n, 17), 1.0 / 17.0, dtype=np.float32)
        arrays["static_feature_names"] = np.asarray([f"LCZ_{i}_frac_500m" for i in range(1, 18)], dtype=object)
        arrays["dominant_lcz"] = np.asarray([1, 2], dtype=np.float32)
    np.savez_compressed(dataset_dir / "dataset.npz", **arrays)
    _write_split_files(dataset_dir, n=n)


def test_dataset_returns_static_feature_vector(tmp_path):
    torch = pytest.importorskip("torch")
    dataset_dir = tmp_path / "with_static"
    _write_dummy_dataset(dataset_dir, include_static=True)

    dataset = WindDownscalingDataset(dataset_dir, split="train", return_metadata=True)
    item = dataset[0]

    assert item["x_static"].shape == (17,)
    assert item["x_static"].dtype == torch.float32
    assert item["static_feature_names"][0] == "LCZ_1_frac_500m"
    assert item["dominant_lcz"] == 1.0


def test_dataloader_batches_static_features(tmp_path):
    torch = pytest.importorskip("torch")
    dataset_dir = tmp_path / "with_static"
    _write_dummy_dataset(dataset_dir, include_static=True)

    dataset = WindDownscalingDataset(dataset_dir, split="train", return_metadata=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))

    assert batch["x_static"].shape == (2, 17)


def test_dataset_without_static_still_loads(tmp_path):
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "without_static"
    _write_dummy_dataset(dataset_dir, include_static=False)

    dataset = WindDownscalingDataset(dataset_dir, split="train", return_metadata=False)
    item = dataset[0]

    assert "x_static" not in item
    assert item["x_hourly"].shape == (6, 6, 2)
