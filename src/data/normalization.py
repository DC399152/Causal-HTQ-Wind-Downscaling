"""Normalization statistics for generated wind downscaling datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.dataset import DEFAULT_DATASET_DIR, load_metadata, load_split_indices


DEFAULT_NORM_STATS_NAME = "norm_stats.json"


def _masked_channel_stats(values: np.ndarray, mask: np.ndarray, eps: float) -> dict[str, list[float]]:
    """Compute per-channel mean/std using only valid positions.

    ``values`` and ``mask`` use semantic shapes ending in channel, e.g.
    [N, L, H, C] or [N, T_out, H, C].
    """

    if values.shape != mask.shape:
        raise ValueError(f"values and mask shapes differ: {values.shape} != {mask.shape}")
    channels = values.shape[-1]
    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []
    for channel in range(channels):
        channel_values = values[..., channel]
        channel_mask = mask[..., channel].astype(bool)
        valid_values = channel_values[channel_mask].astype(np.float64)
        count = int(valid_values.size)
        if count == 0:
            raise ValueError(f"No valid values for channel {channel}")
        mean = float(valid_values.mean())
        std = float(valid_values.std())
        means.append(mean)
        stds.append(max(std, eps))
        counts.append(count)
    return {"mean": means, "std": stds, "count": counts}


def compute_norm_stats(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    split: str = "train",
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Compute train-only, mask-aware per-channel normalization stats."""

    dataset_dir = Path(dataset_dir)
    dataset_path = dataset_dir / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    indices = load_split_indices(dataset_dir, split)
    if indices.size == 0:
        raise ValueError(f"Split {split!r} has no samples")

    with np.load(dataset_path, allow_pickle=True) as data:
        x = data["x_hourly"][indices]
        x_mask = data["x_mask"][indices]
        y = data["y_10min"][indices]
        y_mask = data["y_mask"][indices]

    x_stats = _masked_channel_stats(x, x_mask, eps)
    y_stats = _masked_channel_stats(y, y_mask, eps)
    metadata = load_metadata(dataset_dir)
    channel_names = metadata.get("channel_names") or [f"channel_{i}" for i in range(x.shape[-1])]

    return {
        "computed_from_split": split,
        "num_samples": int(indices.size),
        "channel_names": list(channel_names),
        "x_mean": x_stats["mean"],
        "x_std": x_stats["std"],
        "x_count": x_stats["count"],
        "y_mean": y_stats["mean"],
        "y_std": y_stats["std"],
        "y_count": y_stats["count"],
        "eps": float(eps),
        "normalization": "per_channel_zscore",
        "mask_convention": "True=valid, False=invalid",
    }


def save_norm_stats(stats: dict[str, Any], output_path: str | Path) -> None:
    """Save normalization stats as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_norm_stats(path: str | Path) -> dict[str, Any]:
    """Load normalization stats JSON."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def default_norm_stats_path(dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> Path:
    """Return default norm stats path for a dataset directory."""

    return Path(dataset_dir) / DEFAULT_NORM_STATS_NAME

