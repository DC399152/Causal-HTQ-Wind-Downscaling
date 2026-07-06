"""Preprocessing configuration and raw-array utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class HeightSelectionConfig:
    """Configured vertical layer selection."""

    selected_heights_agl: tuple[float, ...]
    height_reference: str
    max_height_diff: float


@dataclass(frozen=True)
class QualityControlConfig:
    """Configured missing-value and sample filtering policy."""

    missing_value: float
    allow_missing: bool
    min_valid_ratio_x: float
    min_valid_ratio_x_per_hour: float
    min_valid_ratio_x_current_hour: float
    min_valid_ratio_y: float
    qc_policy: dict[str, Any]


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split settings."""

    train_ratio: float
    val_ratio: float
    test_ratio: float
    split_gap_hours: int
    split_by_unique_time: bool
    split_time_key: str


@dataclass(frozen=True)
class PreprocessingConfig:
    """Normalized preprocessing YAML config."""

    dataset_name: str
    raw_dir: Path
    raw_3600s_dir: Path
    raw_600s_dir: Path
    processed_dir: Path
    dataset_dir: Path
    timestamp_semantics: str
    raw_timestamp_semantics: str
    input_frequency_seconds: int
    target_frequency_seconds: int
    context_hours: int
    target_steps_per_hour: int
    target_offsets_minutes: tuple[int, ...]
    context_alignment: str
    variables: dict[str, Any]
    hourly_channels: tuple[str, ...]
    target_channels: tuple[str, ...]
    height: HeightSelectionConfig
    quality: QualityControlConfig
    splits: SplitConfig
    output_format: str


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _as_tuple(values: Sequence | None, default: Sequence) -> tuple:
    if values is None:
        return tuple(default)
    return tuple(values)


def parse_preprocessing_config(path: str | Path) -> PreprocessingConfig:
    """Parse preprocessing config into a typed object."""

    cfg = load_yaml_config(path)
    paths = cfg.get("paths", {})
    time = cfg.get("time", {})
    variables = dict(cfg.get("variables", {}))
    height_cfg = cfg.get("height_selection", {})
    qc = cfg.get("quality_control", {})
    split = cfg.get("splits", {})
    output = cfg.get("output", {})

    hourly_channels = _as_tuple(
        variables.get("hourly_channels"),
        cfg.get("features", {}).get("input_channels", ("u", "v")),
    )
    target_channels = _as_tuple(
        variables.get("target_channels"),
        cfg.get("features", {}).get("target_channels", ("u", "v")),
    )

    return PreprocessingConfig(
        dataset_name=cfg["dataset_name"],
        raw_dir=Path(paths.get("raw_dir", "data/raw")),
        raw_3600s_dir=Path(paths.get("raw_3600s_dir", paths.get("raw_dir", "data/raw"))),
        raw_600s_dir=Path(paths.get("raw_600s_dir", paths.get("raw_dir", "data/raw"))),
        processed_dir=Path(paths.get("processed_dir", "data/processed")),
        dataset_dir=Path(paths["dataset_dir"]),
        timestamp_semantics=str(time.get("timestamp_semantics", "start")),
        raw_timestamp_semantics=str(time.get("raw_timestamp_semantics", "start")),
        input_frequency_seconds=int(time.get("input_frequency_seconds", 3600)),
        target_frequency_seconds=int(time.get("target_frequency_seconds", 600)),
        context_hours=int(time.get("context_hours", 6)),
        target_steps_per_hour=int(time.get("target_steps_per_hour", 6)),
        target_offsets_minutes=tuple(
            int(v) for v in time.get("target_offsets_minutes", [0, 10, 20, 30, 40, 50])
        ),
        context_alignment=str(time.get("context_alignment", "causal_last")),
        variables=variables,
        hourly_channels=tuple(str(v) for v in hourly_channels),
        target_channels=tuple(str(v) for v in target_channels),
        height=HeightSelectionConfig(
            selected_heights_agl=tuple(
                float(v) for v in height_cfg.get("selected_heights_agl", [250, 275, 300, 325, 350, 375])
            ),
            height_reference=str(height_cfg.get("height_reference", "agl_rounded_station_altitude")),
            max_height_diff=float(height_cfg.get("max_height_diff", 0.1)),
        ),
        quality=QualityControlConfig(
            missing_value=float(qc.get("missing_value", -999.0)),
            allow_missing=bool(qc.get("allow_missing", True)),
            min_valid_ratio_x=float(qc.get("min_valid_ratio_x", 0.8)),
            min_valid_ratio_x_per_hour=float(qc.get("min_valid_ratio_x_per_hour", 0.8)),
            min_valid_ratio_x_current_hour=float(qc.get("min_valid_ratio_x_current_hour", 1.0)),
            min_valid_ratio_y=float(qc.get("min_valid_ratio_y", 0.8)),
            qc_policy=dict(qc.get("qc_policy", {})),
        ),
        splits=SplitConfig(
            train_ratio=float(split.get("train_ratio", 0.8)),
            val_ratio=float(split.get("val_ratio", 0.1)),
            test_ratio=float(split.get("test_ratio", 0.1)),
            split_gap_hours=int(split.get("split_gap_hours", 0)),
            split_by_unique_time=bool(split.get("split_by_unique_time", True)),
            split_time_key=str(split.get("split_time_key", "target_time_start")),
        ),
        output_format=str(output.get("format", "npz")),
    )


def validate_config(config: PreprocessingConfig) -> list[str]:
    """Return configuration warnings that require raw-data verification."""

    warnings: list[str] = []
    if config.timestamp_semantics != "start":
        raise ValueError("New preprocessing requires timestamp_semantics=start")
    if config.raw_timestamp_semantics not in {"start", "end"}:
        raise ValueError("raw_timestamp_semantics must be start or end")
    if config.context_alignment != "causal_last":
        raise ValueError("New preprocessing requires context_alignment=causal_last")
    if config.target_offsets_minutes != (0, 10, 20, 30, 40, 50):
        raise ValueError("New preprocessing requires target_offsets_minutes=[0,10,20,30,40,50]")
    if config.context_hours != 6 or config.target_steps_per_hour != 6:
        warnings.append("Expected L=6 and T_out=6 for v1 dataset")
    if len(config.hourly_channels) != len(config.target_channels):
        raise ValueError("hourly_channels and target_channels must have equal length")
    if len(config.height.selected_heights_agl) != 6:
        warnings.append("Expected H=6 selected height levels for v1 dataset")
    if config.quality.qc_policy.get("use_flags"):
        warnings.append("QC flag use is enabled; verify flag semantics with inspect_raw_nc.py")
    if config.splits.split_gap_hours < 0:
        raise ValueError("split_gap_hours must be non-negative")
    return warnings


def station_value(ds, name: str | None, station_index: int, default=None):
    """Read station metadata value, tolerating scalar or station-indexed variables."""

    if not name or name not in ds:
        return default
    values = np.asarray(ds[name].values)
    if values.ndim == 0:
        return values.item()
    return values[station_index].item()


def select_height_indices(
    raw_heights: np.ndarray,
    station_altitude: float,
    height_config: HeightSelectionConfig,
) -> dict[str, np.ndarray]:
    """Select nearest raw height layers for configured AGL target heights."""

    heights = np.asarray(raw_heights, dtype=float)
    station_ref = float(round(station_altitude)) if "rounded" in height_config.height_reference else float(station_altitude)
    selected_agl = np.asarray(height_config.selected_heights_agl, dtype=float)
    target_asl = station_ref + selected_agl
    indices = np.asarray([int(np.argmin(np.abs(heights - h))) for h in target_asl], dtype=np.int64)
    actual_asl = heights[indices]
    diff = np.abs(actual_asl - target_asl)
    if np.any(diff > height_config.max_height_diff):
        raise ValueError(
            "Height selection exceeds max_height_diff: "
            f"max={diff.max():.6g}, allowed={height_config.max_height_diff}"
        )
    return {
        "selected_heights_agl": selected_agl.astype(np.float32),
        "height_indices": indices,
        "target_heights_asl": target_asl.astype(np.float32),
        "actual_heights_asl": actual_asl.astype(np.float32),
        "actual_heights_agl": (actual_asl - station_ref).astype(np.float32),
    }
