"""PyTorch dataset interface for generated Causal HTQ arrays."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATASET_DIR = Path("data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1")
DEFAULT_NORM_STATS_NAME = "norm_stats.json"
VALID_SPLITS = {"train", "val", "test", "gap", "all"}


@dataclass(frozen=True)
class SampleShapes:
    """Semantic sample shapes, excluding batch dimension."""

    input_context: tuple[int, int, int]
    target_10min: tuple[int, int, int]


def require_torch():
    """Import torch lazily so preprocessing remains NumPy-only."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to use WindDownscalingDataset.") from exc
    return torch


def load_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    """Load dataset metadata JSON when present."""

    path = Path(dataset_dir) / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split_indices(dataset_dir: str | Path, split: str) -> np.ndarray:
    """Load integer sample indices for a split file.

    ``split="all"`` returns all indices from ``dataset.npz``. The ``gap`` split
    is explicit and can be inspected, but training code should normally use
    train/val/test only.
    """

    dataset_dir = Path(dataset_dir)
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}. Expected one of {sorted(VALID_SPLITS)}")

    if split == "all":
        with np.load(dataset_dir / "dataset.npz", allow_pickle=True) as data:
            return np.arange(data["x_hourly"].shape[0], dtype=np.int64)

    split_path = dataset_dir / "splits" / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    lines = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()]
    return np.asarray([int(line) for line in lines if line], dtype=np.int64)


def _as_tensor(array: np.ndarray, dtype=None):
    torch = require_torch()
    tensor = torch.from_numpy(np.asarray(array))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def load_norm_stats(path: str | Path) -> dict[str, Any]:
    """Load normalization stats JSON."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_array(
    values: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply per-channel z-score and zero invalid positions.

    ``values`` and ``mask`` share a semantic shape ending in channel. ``mean``
    and ``std`` have shape [C].
    """

    normalized = (values.astype(np.float32) - mean) / std
    return np.where(mask.astype(bool), normalized, 0.0).astype(np.float32)


class WindDownscalingDataset:
    """Dataset wrapper for generated `.npz` artifacts.

    Returned tensor shapes for one sample:
    - ``x_hourly``: [L, H, C]
    - ``x_mask``: [L, H, C]
    - ``y_10min``: [T_out, H, C]
    - ``y_mask``: [T_out, H, C]
    - ``current_hourly``: [H, C]
    - ``height``: [H], height values used by this sample
    - optional ``hourly_height_values``: [H], actual hourly heights
    - optional ``target_height_values``: [H], actual target heights
    - optional ``current_hourly_y_norm``: [H, C], only when ``normalize=True``
    - optional ``x_meteo``: [L, P, C_m]
    - optional ``meteo_mask``: [L, P, C_m]
    - optional ``x_static``: [F_static]
    """

    def __init__(
        self,
        dataset_dir: str | Path = DEFAULT_DATASET_DIR,
        split: str = "train",
        *,
        return_metadata: bool = True,
        load_into_memory: bool = True,
        normalize: bool = False,
        norm_stats_path: str | Path | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.return_metadata = return_metadata
        self.load_into_memory = load_into_memory
        self.normalize = normalize
        self.dataset_path = self.dataset_dir / "dataset.npz"
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        self.metadata = load_metadata(self.dataset_dir)
        self.norm_stats_path = Path(norm_stats_path) if norm_stats_path else self.dataset_dir / DEFAULT_NORM_STATS_NAME
        self.norm_stats = load_norm_stats(self.norm_stats_path) if normalize else None
        self.indices = load_split_indices(self.dataset_dir, split)
        self._npz = None
        if load_into_memory:
            with np.load(self.dataset_path, allow_pickle=True) as data:
                self.arrays = {key: data[key] for key in data.files}
        else:
            self.arrays = None

        self._validate_required_keys()

    def _data(self):
        if self.arrays is not None:
            return self.arrays
        if self._npz is None:
            self._npz = np.load(self.dataset_path, allow_pickle=True)
        return self._npz

    def close(self) -> None:
        """Close lazy NPZ handle if one is open."""

        if self._npz is not None:
            self._npz.close()
            self._npz = None

    def _validate_required_keys(self) -> None:
        required = {
            "x_hourly",
            "x_mask",
            "y_10min",
            "y_mask",
            "current_hourly",
            "station_id",
            "target_time_start",
            "target_times_10min",
            "height_values",
            "source_file",
            "split",
        }
        data = self._data()
        missing = sorted(required - set(data.keys()))
        if missing:
            raise KeyError(f"Dataset is missing required keys: {missing}")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def sample_shapes(self) -> SampleShapes:
        data = self._data()
        return SampleShapes(
            input_context=tuple(int(v) for v in data["x_hourly"].shape[1:]),
            target_10min=tuple(int(v) for v in data["y_10min"].shape[1:]),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self._data()
        sample_index = int(self.indices[index])
        x_hourly = data["x_hourly"][sample_index]
        x_mask = data["x_mask"][sample_index].astype(bool)
        y_10min = data["y_10min"][sample_index]
        y_mask = data["y_mask"][sample_index].astype(bool)
        current_hourly = data["current_hourly"][sample_index]
        height = data["height_values"][sample_index].astype(np.float32)
        current_hourly_y_norm = None
        has_meteo = "x_meteo" in data and "meteo_mask" in data
        if has_meteo:
            x_meteo = data["x_meteo"][sample_index]
            meteo_mask = data["meteo_mask"][sample_index].astype(bool)
        has_static = "x_static" in data and data["x_static"].shape[-1] > 0
        if has_static:
            x_static = data["x_static"][sample_index].astype(np.float32)

        if self.normalize:
            if self.norm_stats is None:
                raise RuntimeError("normalize=True but norm_stats were not loaded")
            x_mean = np.asarray(self.norm_stats["x_mean"], dtype=np.float32)
            x_std = np.asarray(self.norm_stats["x_std"], dtype=np.float32)
            y_mean = np.asarray(self.norm_stats["y_mean"], dtype=np.float32)
            y_std = np.asarray(self.norm_stats["y_std"], dtype=np.float32)
            current_hourly_physical = current_hourly
            x_hourly = _normalize_array(x_hourly, x_mask, x_mean, x_std)
            y_10min = _normalize_array(y_10min, y_mask, y_mean, y_std)
            current_hourly_y_norm = _normalize_array(
                current_hourly_physical,
                x_mask[-1],
                y_mean,
                y_std,
            )
            current_hourly = _normalize_array(current_hourly, x_mask[-1], x_mean, x_std)
            if has_meteo:
                if "meteo_mean" not in self.norm_stats or "meteo_std" not in self.norm_stats:
                    raise KeyError(
                        "Dataset contains x_meteo but norm_stats.json is missing "
                        "meteo_mean/meteo_std. Re-run scripts/compute_norm_stats.py."
                    )
                meteo_mean = np.asarray(self.norm_stats["meteo_mean"], dtype=np.float32)
                meteo_std = np.asarray(self.norm_stats["meteo_std"], dtype=np.float32)
                x_meteo = _normalize_array(x_meteo, meteo_mask, meteo_mean, meteo_std)

        item: dict[str, Any] = {
            "x_hourly": _as_tensor(x_hourly, dtype=require_torch().float32),
            "x_mask": _as_tensor(x_mask, dtype=require_torch().bool),
            "y_10min": _as_tensor(y_10min, dtype=require_torch().float32),
            "y_mask": _as_tensor(y_mask, dtype=require_torch().bool),
            "current_hourly": _as_tensor(current_hourly, dtype=require_torch().float32),
            "height": _as_tensor(height, dtype=require_torch().float32),
            "sample_index": sample_index,
        }
        if current_hourly_y_norm is not None:
            item["current_hourly_y_norm"] = _as_tensor(
                current_hourly_y_norm,
                dtype=require_torch().float32,
            )
        if has_meteo:
            item["x_meteo"] = _as_tensor(x_meteo, dtype=require_torch().float32)
            item["meteo_mask"] = _as_tensor(meteo_mask, dtype=require_torch().bool)
        if has_static:
            item["x_static"] = _as_tensor(x_static, dtype=require_torch().float32)

        if self.return_metadata:
            metadata = {
                "station_id": str(data["station_id"][sample_index]),
                "target_time_start": str(data["target_time_start"][sample_index]),
                "target_times_10min": [
                    str(v) for v in data["target_times_10min"][sample_index]
                ],
                "height_values": _as_tensor(
                    data["height_values"][sample_index],
                    dtype=require_torch().float32,
                ),
                "source_file": str(data["source_file"][sample_index]),
                "split": str(data["split"][sample_index]),
            }
            if "station_lat" in data:
                metadata["station_lat"] = float(data["station_lat"][sample_index])
            if "station_lon" in data:
                metadata["station_lon"] = float(data["station_lon"][sample_index])
            if "context_times_hourly" in data:
                metadata["context_times_hourly"] = [
                    str(v) for v in data["context_times_hourly"][sample_index]
                ]
            if "hourly_height_values" in data:
                metadata["hourly_height_values"] = _as_tensor(
                    data["hourly_height_values"][sample_index],
                    dtype=require_torch().float32,
                )
            if "target_height_values" in data:
                metadata["target_height_values"] = _as_tensor(
                    data["target_height_values"][sample_index],
                    dtype=require_torch().float32,
                )
            if "hourly_source_files" in data:
                metadata["hourly_source_files"] = [
                    str(v) for v in data["hourly_source_files"][sample_index]
                ]
            if "target_source_files" in data:
                metadata["target_source_files"] = [
                    str(v) for v in data["target_source_files"][sample_index]
                ]
            if has_meteo and "meteo_pressure_levels" in data:
                metadata["meteo_pressure_levels"] = _as_tensor(
                    data["meteo_pressure_levels"],
                    dtype=require_torch().float32,
                )
            if has_meteo and "meteo_channel_names" in data:
                metadata["meteo_channel_names"] = [str(v) for v in data["meteo_channel_names"]]
            if has_static and "static_feature_names" in data:
                metadata["static_feature_names"] = [str(v) for v in data["static_feature_names"]]
            if has_static and "dominant_lcz" in data:
                metadata["dominant_lcz"] = float(data["dominant_lcz"][sample_index])
            item.update(metadata)

        return item


def available_splits(dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> dict[str, int]:
    """Return split sizes from split files."""

    dataset_dir = Path(dataset_dir)
    result: dict[str, int] = {}
    for split_path in sorted((dataset_dir / "splits").glob("*.txt")):
        indices = load_split_indices(dataset_dir, split_path.stem)
        result[split_path.stem] = int(indices.shape[0])
    return result


def iter_split_datasets(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    splits: Iterable[str] = ("train", "val", "test"),
) -> dict[str, WindDownscalingDataset]:
    """Create datasets for several splits."""

    return {split: WindDownscalingDataset(dataset_dir, split=split) for split in splits}
