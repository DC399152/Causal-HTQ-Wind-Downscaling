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
    instrument_height_agl_m: float = 0.0


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
    """Dataset split settings."""

    train_ratio: float
    val_ratio: float
    test_ratio: float
    split_gap_hours: int
    split_by_unique_time: bool
    split_time_key: str
    strategy: str = "chronological"
    seed: int = 42
    event_gap_hours: int = 24
    purge_hours: int = 24
    block_duration_hours: dict[str, int] | None = None
    balance_weights: dict[str, float] | None = None
    search_trials: int = 256


@dataclass(frozen=True)
class MeteoConfig:
    """Optional ERA5 meteorological auxiliary input settings."""

    enabled: bool
    source: str
    pressure_dir: Path | None
    interpolation: str
    out_of_bounds: str
    variables: dict[str, Any]
    channel_names: tuple[str, ...]
    expected_pressure_levels_hpa: tuple[float, ...]
    require_complete_context: bool = False


@dataclass(frozen=True)
class StaticFeatureConfig:
    """Optional station-level static feature settings."""

    use_lcz: bool
    lcz_feature_csv: Path | None
    feature_columns: tuple[str, ...]
    station_id_column: str
    dominant_lcz_column: str


@dataclass(frozen=True)
class DataAlignmentConfig:
    """Strict pairing and schema validation settings."""

    station_matching_mode: str
    max_hourly_target_height_diff_m: float
    height_schema_policy: str


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
    height_by_source: dict[str, HeightSelectionConfig]
    sources: dict[str, Any]
    meteo: MeteoConfig
    static_features: StaticFeatureConfig
    data_alignment: DataAlignmentConfig
    quality: QualityControlConfig
    splits: SplitConfig
    split_within_source: bool
    split_within_station: bool
    output_format: str
    expected_station_ids: tuple[str, ...] = ()


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
    if "raw_timestamp_semantics" not in time:
        raise ValueError("time.raw_timestamp_semantics must be explicitly set to start or end")
    variables = dict(cfg.get("variables", {}))
    height_cfg = cfg.get("height_selection", {})
    height_by_source_cfg = dict(height_cfg.get("by_source", {}))
    qc = cfg.get("quality_control", {})
    split = cfg.get("splits", {})
    output = cfg.get("output", {})
    meteo_cfg = cfg.get("meteo", {})
    static_cfg = cfg.get("static_features", {})
    alignment_cfg = cfg.get("data_alignment", {})

    hourly_channels = _as_tuple(
        variables.get("hourly_channels"),
        cfg.get("features", {}).get("input_channels", ("u", "v")),
    )
    target_channels = _as_tuple(
        variables.get("target_channels"),
        cfg.get("features", {}).get("target_channels", ("u", "v")),
    )

    default_height = HeightSelectionConfig(
        selected_heights_agl=tuple(
            float(v) for v in height_cfg.get("selected_heights_agl", [250, 275, 300, 325, 350, 375])
        ),
        height_reference=str(height_cfg.get("height_reference", "agl_rounded_station_altitude")),
        max_height_diff=float(height_cfg.get("max_height_diff", 0.1)),
        instrument_height_agl_m=float(height_cfg.get("instrument_height_agl_m", 0.0)),
    )
    parsed_height_by_source: dict[str, HeightSelectionConfig] = {}
    for source_name, source_height in height_by_source_cfg.items():
        source_height = dict(source_height or {})
        selected = source_height.get("selected_heights_agl", default_height.selected_heights_agl)
        parsed_height_by_source[str(source_name)] = HeightSelectionConfig(
            selected_heights_agl=tuple(float(v) for v in selected),
            height_reference=str(source_height.get("height_reference", default_height.height_reference)),
            max_height_diff=float(source_height.get("max_height_diff", default_height.max_height_diff)),
            instrument_height_agl_m=float(
                source_height.get("instrument_height_agl_m", default_height.instrument_height_agl_m)
            ),
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
        height=default_height,
        height_by_source=parsed_height_by_source,
        sources=dict(cfg.get("sources", {})),
        meteo=MeteoConfig(
            enabled=bool(meteo_cfg.get("enabled", False)),
            source=str(meteo_cfg.get("source", "era5")),
            pressure_dir=Path(meteo_cfg["pressure_dir"]) if meteo_cfg.get("pressure_dir") else None,
            interpolation=str(meteo_cfg.get("interpolation", "bilinear")),
            out_of_bounds=str(meteo_cfg.get("out_of_bounds", "nearest")),
            variables=dict(meteo_cfg.get("variables", {})),
            channel_names=tuple(str(v) for v in meteo_cfg.get("channel_names", ("temperature", "humidity"))),
            expected_pressure_levels_hpa=tuple(
                float(v) for v in meteo_cfg.get("expected_pressure_levels_hpa", ())
            ),
            require_complete_context=bool(meteo_cfg.get("require_complete_context", False)),
        ),
        static_features=StaticFeatureConfig(
            use_lcz=bool(static_cfg.get("use_lcz", False)),
            lcz_feature_csv=Path(static_cfg["lcz_feature_csv"]) if static_cfg.get("lcz_feature_csv") else None,
            feature_columns=tuple(str(v) for v in static_cfg.get("feature_columns", ())),
            station_id_column=str(static_cfg.get("station_id_column", "station_id")),
            dominant_lcz_column=str(static_cfg.get("dominant_lcz_column", "dominant_lcz_500m")),
        ),
        data_alignment=DataAlignmentConfig(
            station_matching_mode=str(alignment_cfg.get("station_matching_mode", "strict")),
            max_hourly_target_height_diff_m=float(
                alignment_cfg.get("max_hourly_target_height_diff_m", 2.0)
            ),
            height_schema_policy=str(alignment_cfg.get("height_schema_policy", "error")),
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
            strategy=str(split.get("strategy", "chronological")),
            seed=int(split.get("seed", 42)),
            event_gap_hours=int(split.get("event_gap_hours", 24)),
            purge_hours=int(split.get("purge_hours", split.get("split_gap_hours", 24))),
            block_duration_hours={
                str(name): int(hours)
                for name, hours in dict(split.get("block_duration_hours", {"default": 168})).items()
            },
            balance_weights={
                str(name): float(weight)
                for name, weight in dict(split.get("balance_weights", {})).items()
            },
            search_trials=int(split.get("search_trials", 256)),
        ),
        split_within_source=bool(split.get("split_within_source", False)),
        split_within_station=bool(split.get("split_within_station", False)),
        output_format=str(output.get("format", "npz")),
        expected_station_ids=tuple(str(v) for v in output.get("expected_station_ids", ())),
    )


def validate_config(config: PreprocessingConfig) -> list[str]:
    """Return configuration warnings that require raw-data verification."""

    warnings: list[str] = []
    if config.timestamp_semantics != "start":
        raise ValueError("New preprocessing requires timestamp_semantics=start")
    if config.raw_timestamp_semantics not in {"start", "end"}:
        raise ValueError("raw_timestamp_semantics must be start or end")
    if config.data_alignment.station_matching_mode not in {"strict", "intersection"}:
        raise ValueError("data_alignment.station_matching_mode must be strict or intersection")
    if config.data_alignment.height_schema_policy not in {"error"}:
        raise ValueError("Only data_alignment.height_schema_policy=error is currently supported")
    if config.data_alignment.max_hourly_target_height_diff_m < 0:
        raise ValueError("data_alignment.max_hourly_target_height_diff_m must be non-negative")
    allowed_height_references = {
        "agl",
        "agl_rounded_station_altitude",
        "ground_agl_from_asl",
        "instrument_relative_to_ground_agl",
    }
    for source_name, height in {"default": config.height, **config.height_by_source}.items():
        if height.height_reference not in allowed_height_references:
            raise ValueError(
                f"Unsupported height_reference for {source_name}: {height.height_reference}"
            )
        if height.max_height_diff < 0:
            raise ValueError(f"max_height_diff must be non-negative for {source_name}")
        if height.instrument_height_agl_m < 0:
            raise ValueError(f"instrument_height_agl_m must be non-negative for {source_name}")
        selected = np.asarray(height.selected_heights_agl, dtype=float)
        if not np.isfinite(selected).all() or np.any(np.diff(selected) <= 0):
            raise ValueError(f"selected_heights_agl must be finite and strictly increasing for {source_name}")
    if config.context_alignment != "causal_last":
        raise ValueError("New preprocessing requires context_alignment=causal_last")
    if config.target_offsets_minutes != (0, 10, 20, 30, 40, 50):
        raise ValueError("New preprocessing requires target_offsets_minutes=[0,10,20,30,40,50]")
    if config.context_hours not in {6, 12}:
        warnings.append("Expected context_hours to be 6 or 12 for planned datasets")
    if config.target_steps_per_hour != 6:
        warnings.append("Expected T_out=6 for 1h-to-10min datasets")
    if len(config.hourly_channels) != len(config.target_channels):
        raise ValueError("hourly_channels and target_channels must have equal length")
    if len(config.height.selected_heights_agl) != 6:
        warnings.append("Expected H=6 selected height levels for v1 dataset")
    if config.quality.qc_policy.get("use_flags"):
        warnings.append("QC flag use is enabled; verify flag semantics with inspect_raw_nc.py")
    if config.splits.split_gap_hours < 0:
        raise ValueError("split_gap_hours must be non-negative")
    if config.splits.strategy not in {"chronological", "balanced_blocks"}:
        raise ValueError("splits.strategy must be chronological or balanced_blocks")
    if config.splits.event_gap_hours < 0 or config.splits.purge_hours < 0:
        raise ValueError("event_gap_hours and purge_hours must be non-negative")
    if config.splits.search_trials < 1:
        raise ValueError("search_trials must be positive")
    if not np.isclose(
        config.splits.train_ratio + config.splits.val_ratio + config.splits.test_ratio,
        1.0,
    ):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")
    if any(hours <= 0 for hours in (config.splits.block_duration_hours or {}).values()):
        raise ValueError("All block_duration_hours values must be positive")
    if config.splits.strategy == "balanced_blocks" and any(
        config.splits.purge_hours >= hours
        for hours in (config.splits.block_duration_hours or {}).values()
    ):
        raise ValueError("purge_hours must be smaller than every balanced block duration")
    if any(weight < 0 for weight in (config.splits.balance_weights or {}).values()):
        raise ValueError("All balance_weights values must be non-negative")
    if config.meteo.enabled:
        if config.meteo.source != "era5":
            raise ValueError("Only meteo.source=era5 is supported")
        if config.meteo.pressure_dir is None:
            raise ValueError("meteo.pressure_dir is required when meteo.enabled=true")
        if config.meteo.interpolation not in {"bilinear", "nearest"}:
            raise ValueError("meteo.interpolation must be bilinear or nearest")
        if config.meteo.out_of_bounds not in {"nearest", "error"}:
            raise ValueError("meteo.out_of_bounds must be nearest or error")
        required = {"time", "pressure_level", "latitude", "longitude", "temperature", "humidity"}
        missing = sorted(required - set(config.meteo.variables))
        if missing:
            raise ValueError(f"meteo.variables is missing required keys: {missing}")
    if config.static_features.use_lcz:
        if config.static_features.lcz_feature_csv is None:
            raise ValueError("static_features.lcz_feature_csv is required when use_lcz=true")
        if not config.static_features.feature_columns:
            raise ValueError("static_features.feature_columns is required when use_lcz=true")
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
    station_height: float = 0.0,
) -> dict[str, np.ndarray]:
    """Select nearest raw height layers for configured AGL target heights."""

    heights = np.asarray(raw_heights, dtype=float)
    if height_config.height_reference == "ground_agl_from_asl":
        if not np.isfinite(station_altitude) or not np.isfinite(station_height):
            raise ValueError("Strict ground AGL selection requires finite station_altitude and station_height")
        station_ref = float(station_altitude) - float(station_height)
    else:
        # Legacy modes define AGL relative to the measurement station itself.
        station_ref = (
            float(round(station_altitude))
            if "rounded" in height_config.height_reference
            else float(station_altitude)
        )
    selected_agl = np.asarray(height_config.selected_heights_agl, dtype=float)
    target_asl = station_ref + selected_agl
    indices = np.asarray([int(np.argmin(np.abs(heights - h))) for h in target_asl], dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Selected AGL heights map to duplicate raw height layers")
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
