from dataclasses import replace

import numpy as np

from src.data.dataset_builder import _balanced_block_split_labels
from src.data.preprocessing import parse_preprocessing_config, validate_config


BALANCED_CONFIG = "configs/preprocessing/paris_dufeng_1h_to_10min_12h_strict_agl_balanced_blocks.yaml"


def _dummy_arrays(num_hours: int = 24 * 30):
    times = np.datetime64("2024-01-01T00:00", "m") + np.arange(num_hours) * np.timedelta64(1, "h")
    current = np.zeros((num_hours, 1, 2), dtype=np.float32)
    y = np.zeros((num_hours, 6, 1, 2), dtype=np.float32)
    phase = np.arange(6, dtype=np.float32)[None, :, None]
    strength = (1.0 + (np.arange(num_hours) % 96) / 24.0).astype(np.float32)
    y[..., 0] = strength[:, None, None] * np.sin(phase)
    y[..., 1] = strength[:, None, None] * np.cos(phase)
    return {
        "x_hourly": np.zeros((num_hours, 12, 1, 2), dtype=np.float32),
        "x_mask": np.ones((num_hours, 12, 1, 2), dtype=bool),
        "y_10min": y,
        "y_mask": np.ones_like(y, dtype=bool),
        "current_hourly": current,
        "station_id": np.asarray(["A"] * num_hours, dtype=object),
        "source_group": np.asarray(["paris_nc"] * num_hours, dtype=object),
        "target_time_start": times.astype(str).astype(object),
    }


def test_balanced_blocks_are_deterministic_and_temporally_purged():
    config = parse_preprocessing_config(BALANCED_CONFIG)
    validate_config(config)
    config = replace(
        config,
        splits=replace(
            config.splits,
            block_duration_hours={"default": 48, "paris_nc": 48},
            purge_hours=12,
            search_trials=64,
        ),
    )
    arrays = _dummy_arrays()

    labels_a = _balanced_block_split_labels(arrays, config)
    labels_b = _balanced_block_split_labels(arrays, config)

    assert labels_a.tolist() == labels_b.tolist()
    assert {"train", "val", "test", "gap"}.issubset(set(labels_a.tolist()))

    times = np.asarray(arrays["target_time_start"], dtype="datetime64[m]")
    active = np.where(labels_a != "gap")[0]
    for left, right in zip(active[:-1], active[1:]):
        if labels_a[left] != labels_a[right]:
            assert times[right] - times[left] > np.timedelta64(12, "h")

    for block_start in range(0, len(labels_a), 48):
        block_labels = set(labels_a[block_start : block_start + 48].tolist()) - {"gap"}
        assert len(block_labels) <= 1

    non_gap = labels_a[labels_a != "gap"]
    fractions = {name: float(np.mean(non_gap == name)) for name in ("train", "val", "test")}
    assert abs(fractions["train"] - 0.8) < 0.15
    assert abs(fractions["val"] - 0.1) < 0.1
    assert abs(fractions["test"] - 0.1) < 0.1


def test_old_config_keeps_chronological_strategy():
    config = parse_preprocessing_config("configs/preprocessing/paris_dufeng_1h_to_10min_12h.yaml")
    assert config.splits.strategy == "chronological"
