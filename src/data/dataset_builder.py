"""Dataset builder for aligned hourly-to-10min wind profile samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd

from src.data.alignment import (
    AlignmentSpec,
    build_time_index,
    context_times_for_hour,
    target_times_for_hour,
    time_key,
    times_to_strings,
    validate_sample_alignment,
)
from src.data.era5_reader import Era5StationData, StationLocation, load_era5_for_stations
from src.data.masks import fill_invalid, valid_numeric_mask, valid_ratio
from src.data.preprocessing import (
    PreprocessingConfig,
    select_height_indices,
    station_value,
    validate_config,
)
from src.data.raw_reader import RawFilePair, dataset_time_values, open_dataset, pair_nc_files, require_variables
from src.data.static_features import StationStaticFeatures, load_station_static_features


@dataclass(frozen=True)
class BuildSummary:
    """Summary returned by the dataset builder."""

    dataset_name: str
    dataset_dir: Path
    status: str
    num_pairs: int
    num_samples: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class SampleAccumulator:
    """In-memory sample store before writing the final NPZ dataset."""

    x_hourly: list[np.ndarray] = field(default_factory=list)
    x_mask: list[np.ndarray] = field(default_factory=list)
    y_10min: list[np.ndarray] = field(default_factory=list)
    y_mask: list[np.ndarray] = field(default_factory=list)
    current_hourly: list[np.ndarray] = field(default_factory=list)
    x_meteo: list[np.ndarray] = field(default_factory=list)
    meteo_mask: list[np.ndarray] = field(default_factory=list)
    x_static: list[np.ndarray] = field(default_factory=list)
    dominant_lcz: list[float] = field(default_factory=list)
    station_id: list[str] = field(default_factory=list)
    station_lat: list[float] = field(default_factory=list)
    station_lon: list[float] = field(default_factory=list)
    target_time_start: list[str] = field(default_factory=list)
    target_times_10min: list[list[str]] = field(default_factory=list)
    context_times_hourly: list[list[str]] = field(default_factory=list)
    height_values: list[np.ndarray] = field(default_factory=list)
    hourly_height_values: list[np.ndarray] = field(default_factory=list)
    target_height_values: list[np.ndarray] = field(default_factory=list)
    source_file: list[str] = field(default_factory=list)
    hourly_source_files: list[list[str]] = field(default_factory=list)
    target_source_files: list[list[str]] = field(default_factory=list)
    source_group: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.x_hourly)


@dataclass
class StationSeries:
    """Global time-indexed data for one station across all raw file pairs."""

    station_id: str
    height_values: np.ndarray
    hourly_height_values: np.ndarray
    target_height_values: np.ndarray
    latitude: float | None = None
    longitude: float | None = None
    hourly: dict[np.datetime64, np.ndarray] = field(default_factory=dict)
    target: dict[np.datetime64, np.ndarray] = field(default_factory=dict)
    hourly_source: dict[np.datetime64, str] = field(default_factory=dict)
    target_source: dict[np.datetime64, str] = field(default_factory=dict)
    source_group: str = "paris_nc"


def _required_hourly_variables(config: PreprocessingConfig) -> list[str]:
    vars_cfg = config.variables
    names = [
        vars_cfg.get("time"),
        vars_cfg.get("height"),
        vars_cfg.get("station"),
        vars_cfg.get("station_lat"),
        vars_cfg.get("station_lon"),
        vars_cfg.get("station_altitude"),
        vars_cfg.get("station_height"),
        *config.hourly_channels,
        *config.target_channels,
    ]
    return [str(name) for name in names if name]


def _required_target_variables(config: PreprocessingConfig) -> list[str]:
    vars_cfg = config.variables
    names = [
        vars_cfg.get("time"),
        vars_cfg.get("height"),
        vars_cfg.get("station"),
        *config.target_channels,
    ]
    return [str(name) for name in names if name]


def _stack_channels(ds, channels: tuple[str, ...], config: PreprocessingConfig, source: Path) -> np.ndarray:
    """Stack configured variables into [station, time, height, channel]."""

    time_name = str(config.variables["time"])
    height_name = str(config.variables["height"])
    station_name = config.variables.get("station")
    arrays: list[np.ndarray] = []

    for channel in channels:
        if channel not in ds:
            raise KeyError(f"{source} is missing channel variable: {channel}")
        da = ds[channel]
        required_dims = [time_name, height_name]
        if station_name and station_name in da.dims:
            order = [station_name, time_name, height_name]
            values = da.transpose(*order).values
        else:
            order = required_dims
            values = da.transpose(*order).values[None, ...]
        arrays.append(np.asarray(values, dtype=np.float32))

    return np.stack(arrays, axis=-1)


def _station_ids(ds, config: PreprocessingConfig, station_count: int) -> list[str]:
    station_name = config.variables.get("station")
    if station_name and station_name in ds:
        values = np.asarray(ds[station_name].values).reshape(-1)
        if values.size == station_count:
            return [
                v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                for v in values
            ]
    return [str(i) for i in range(station_count)]


def _normalize_station_id(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip()


def _station_enabled_for_source(config: PreprocessingConfig, source_name: str, station_id: str) -> bool:
    source_cfg = dict((config.sources or {}).get(source_name, {}) or {})
    included = {str(v) for v in source_cfg.get("include_station_ids", ())}
    excluded = {str(v) for v in source_cfg.get("exclude_station_ids", ())}
    return (not included or station_id in included) and station_id not in excluded


def _station_ids_for_dataset(ds, config: PreprocessingConfig, station_count: int, source) -> list[str]:
    ids = [_normalize_station_id(v) for v in _station_ids(ds, config, station_count)]
    duplicates = sorted({station_id for station_id in ids if ids.count(station_id) > 1})
    if duplicates:
        raise ValueError(f"{source} contains duplicate station_id(s): {duplicates}")
    if any(not station_id for station_id in ids):
        raise ValueError(f"{source} contains empty station_id values")
    return ids


def _matched_target_station_indices(
    hourly_ids: list[str],
    target_ids: list[str],
    hourly_path: Path,
    target_path: Path,
    config: PreprocessingConfig,
) -> dict[str, int]:
    hourly_set = set(hourly_ids)
    target_set = set(target_ids)
    only_hourly = sorted(hourly_set - target_set)
    only_target = sorted(target_set - hourly_set)
    mode = config.data_alignment.station_matching_mode
    if mode == "strict" and (only_hourly or only_target):
        raise ValueError(
            "Station ID mismatch between hourly and target files. "
            f"hourly={hourly_path}; target={target_path}; "
            f"only_hourly={only_hourly}; only_target={only_target}"
        )
    if only_hourly or only_target:
        # intersection mode is intentionally noisy; no station is dropped silently.
        print(
            "WARNING: station intersection mode dropping unmatched stations: "
            f"hourly={hourly_path}; target={target_path}; "
            f"only_hourly={only_hourly}; only_target={only_target}"
        )
    return {station_id: idx for idx, station_id in enumerate(target_ids)}


def _height_values_for_station(ds, config: PreprocessingConfig, station_index: int) -> np.ndarray:
    height_name = str(config.variables["height"])
    heights = np.asarray(ds[height_name].values)
    if heights.ndim == 1:
        return heights.astype(float)
    if heights.ndim == 2:
        return heights[station_index].astype(float)
    raise ValueError(f"Unsupported height variable shape for {height_name}: {heights.shape}")


def _representative_heights(hourly_actual_agl: np.ndarray, target_actual_agl: np.ndarray) -> np.ndarray:
    return (0.5 * (hourly_actual_agl.astype(np.float32) + target_actual_agl.astype(np.float32))).astype(np.float32)


def _validate_hourly_target_height_match(
    station_id: str,
    hourly_source: Path | str,
    target_source: Path | str,
    requested_heights: np.ndarray,
    hourly_actual_agl: np.ndarray,
    target_actual_agl: np.ndarray,
    config: PreprocessingConfig,
) -> None:
    diff = np.abs(hourly_actual_agl.astype(float) - target_actual_agl.astype(float))
    tolerance = float(config.data_alignment.max_hourly_target_height_diff_m)
    if np.any(diff > tolerance):
        raise ValueError(
            "Hourly/target height mismatch exceeds tolerance. "
            f"station_id={station_id}; hourly={hourly_source}; target={target_source}; "
            f"requested_heights_agl={requested_heights.astype(float).tolist()}; "
            f"hourly_actual_agl={hourly_actual_agl.astype(float).tolist()}; "
            f"target_actual_agl={target_actual_agl.astype(float).tolist()}; "
            f"diff_m={diff.tolist()}; allowed_m={tolerance}"
        )


def _validate_station_height_schema(
    series: StationSeries,
    new_height_values: np.ndarray,
    new_hourly_height_values: np.ndarray,
    new_target_height_values: np.ndarray,
    station_id: str,
    source: str,
    config: PreprocessingConfig,
) -> None:
    tolerance = float(config.data_alignment.max_hourly_target_height_diff_m)
    checks = (
        ("representative", series.height_values, new_height_values),
        ("hourly", series.hourly_height_values, new_hourly_height_values),
        ("target", series.target_height_values, new_target_height_values),
    )
    for label, old, new in checks:
        diff = np.abs(np.asarray(old, dtype=float) - np.asarray(new, dtype=float))
        if np.any(diff > tolerance):
            raise ValueError(
                "Height schema changed across files. "
                f"station_id={station_id}; schema={label}; source={source}; "
                f"old_heights={np.asarray(old, dtype=float).tolist()}; "
                f"new_heights={np.asarray(new, dtype=float).tolist()}; "
                f"max_diff_m={float(diff.max())}; allowed_m={tolerance}"
            )


def _passes_sample_filters(
    x_mask: np.ndarray,
    y_mask: np.ndarray,
    config: PreprocessingConfig,
) -> bool:
    """Apply configured validity-ratio filters.

    Shapes:
    - x_mask: [L, H, C]
    - y_mask: [T_out, H, C]
    """

    q = config.quality
    if valid_ratio(x_mask) < q.min_valid_ratio_x:
        return False
    if valid_ratio(x_mask[-1]) < q.min_valid_ratio_x_current_hour:
        return False
    for hour_mask in x_mask:
        if valid_ratio(hour_mask) < q.min_valid_ratio_x_per_hour:
            return False
    if valid_ratio(y_mask) < q.min_valid_ratio_y:
        return False
    return True


def _height_config_for_source(config: PreprocessingConfig, source_name: str):
    return config.height_by_source.get(source_name, config.height)


def _append_sample(
    acc: SampleAccumulator,
    x: np.ndarray,
    x_mask: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    station_id: str,
    station_lat: float | None,
    station_lon: float | None,
    hour_start,
    target_times,
    context_times,
    height_values: np.ndarray,
    hourly_height_values: np.ndarray,
    target_height_values: np.ndarray,
    source_file: str,
    hourly_source_files: list[str],
    target_source_files: list[str],
    source_group: str,
    config: PreprocessingConfig,
    x_meteo: np.ndarray | None = None,
    meteo_mask: np.ndarray | None = None,
    x_static: np.ndarray | None = None,
    dominant_lcz: float | None = None,
) -> None:
    """Append one sample with semantic shapes.

    - x: [L, H, C]
    - y: [T_out, H, C]
    """

    missing_value = config.quality.missing_value
    acc.x_hourly.append(fill_invalid(x, x_mask, missing_value))
    acc.x_mask.append(x_mask.astype(bool))
    acc.y_10min.append(fill_invalid(y, y_mask, missing_value))
    acc.y_mask.append(y_mask.astype(bool))
    acc.current_hourly.append(fill_invalid(x[-1], x_mask[-1], missing_value))
    if x_meteo is not None and meteo_mask is not None:
        acc.x_meteo.append(fill_invalid(x_meteo, meteo_mask, missing_value))
        acc.meteo_mask.append(meteo_mask.astype(bool))
    if x_static is not None:
        acc.x_static.append(np.asarray(x_static, dtype=np.float32))
        acc.dominant_lcz.append(float(dominant_lcz) if dominant_lcz is not None else np.nan)
    acc.station_id.append(station_id)
    acc.station_lat.append(float(station_lat) if station_lat is not None else np.nan)
    acc.station_lon.append(float(station_lon) if station_lon is not None else np.nan)
    acc.target_time_start.append(str(time_key(hour_start)))
    acc.target_times_10min.append(times_to_strings(target_times))
    acc.context_times_hourly.append(times_to_strings(context_times))
    acc.height_values.append(np.asarray(height_values, dtype=np.float32))
    acc.hourly_height_values.append(np.asarray(hourly_height_values, dtype=np.float32))
    acc.target_height_values.append(np.asarray(target_height_values, dtype=np.float32))
    acc.source_file.append(source_file)
    acc.hourly_source_files.append([str(v) for v in hourly_source_files])
    acc.target_source_files.append([str(v) for v in target_source_files])
    acc.source_group.append(str(source_group))


def _build_samples_from_pair(
    pair: RawFilePair,
    config: PreprocessingConfig,
    acc: SampleAccumulator,
) -> list[str]:
    warnings: list[str] = []
    hourly_ds = open_dataset(pair.hourly_path)
    target_ds = open_dataset(pair.target_path)
    require_variables(hourly_ds, _required_hourly_variables(config), pair.hourly_path)
    require_variables(target_ds, _required_target_variables(config), pair.target_path)

    time_name = str(config.variables["time"])
    hourly_times = _to_start_times(
        dataset_time_values(hourly_ds, time_name),
        config.raw_timestamp_semantics,
        config.input_frequency_seconds,
    )
    target_times = _to_start_times(
        dataset_time_values(target_ds, time_name),
        config.raw_timestamp_semantics,
        config.target_frequency_seconds,
    )
    hourly_index = build_time_index(hourly_times)
    target_index = build_time_index(target_times)
    spec = AlignmentSpec(
        context_hours=config.context_hours,
        target_steps=config.target_steps_per_hour,
        target_step_minutes=config.target_frequency_seconds // 60,
        input_step_minutes=config.input_frequency_seconds // 60,
    )

    # raw arrays are [station, time, raw_height, channel]
    hourly_values = _stack_channels(hourly_ds, config.hourly_channels, config, pair.hourly_path)
    target_values = _stack_channels(target_ds, config.target_channels, config, pair.target_path)
    station_count = hourly_values.shape[0]
    hourly_station_ids = _station_ids_for_dataset(hourly_ds, config, station_count, pair.hourly_path)
    target_station_ids = _station_ids_for_dataset(target_ds, config, target_values.shape[0], pair.target_path)
    target_index_by_station = _matched_target_station_indices(
        hourly_station_ids,
        target_station_ids,
        pair.hourly_path,
        pair.target_path,
        config,
    )
    station_alt_name = config.variables.get("station_altitude")
    station_height_name = config.variables.get("station_height")

    for station_idx, station_id in enumerate(hourly_station_ids):
        if not _station_enabled_for_source(config, "paris_nc", station_id):
            continue
        if station_id not in target_index_by_station:
            continue
        target_station_idx = target_index_by_station[station_id]
        station_altitude = station_value(hourly_ds, station_alt_name, station_idx, default=0.0)
        target_station_altitude = station_value(target_ds, station_alt_name, target_station_idx, default=station_altitude)
        station_height = station_value(hourly_ds, station_height_name, station_idx, default=0.0)
        target_station_height = station_value(target_ds, station_height_name, target_station_idx, default=station_height)
        try:
            height_config = _height_config_for_source(config, "paris_nc")
            hourly_raw_heights = _height_values_for_station(hourly_ds, config, station_idx)
            target_raw_heights = _height_values_for_station(target_ds, config, target_station_idx)
            hourly_height_meta = select_height_indices(
                hourly_raw_heights,
                station_altitude,
                height_config,
                station_height=station_height,
            )
            target_height_meta = select_height_indices(
                target_raw_heights,
                target_station_altitude,
                height_config,
                station_height=target_station_height,
            )
            _validate_hourly_target_height_match(
                station_id,
                pair.hourly_path,
                pair.target_path,
                hourly_height_meta["selected_heights_agl"],
                hourly_height_meta["actual_heights_agl"],
                target_height_meta["actual_heights_agl"],
                config,
            )
        except Exception as exc:
            raise ValueError(f"Station {station_id} in {pair.prefix} failed height alignment: {exc}") from exc

        hourly_hidx = hourly_height_meta["height_indices"]
        target_hidx = target_height_meta["height_indices"]
        actual_agl = _representative_heights(
            hourly_height_meta["actual_heights_agl"],
            target_height_meta["actual_heights_agl"],
        )
        station_hourly = np.take(hourly_values[station_idx], hourly_hidx, axis=1)
        station_target = np.take(target_values[target_station_idx], target_hidx, axis=1)

        for hour_start in hourly_times:
            ok, errors = validate_sample_alignment(hour_start, hourly_times, target_times, spec)
            if not ok:
                continue
            context_times = context_times_for_hour(hour_start, spec)
            wanted_target_times = target_times_for_hour(hour_start, spec)
            context_idx = [hourly_index[time_key(t)] for t in context_times]
            target_idx = [target_index[time_key(t)] for t in wanted_target_times]

            # x: [L=6, H=6, C=2], y: [T_out=6, H=6, C=2]
            x = station_hourly[context_idx, :, :]
            y = station_target[target_idx, :, :]
            x_mask = valid_numeric_mask(x, config.quality.missing_value)
            y_mask = valid_numeric_mask(y, config.quality.missing_value)

            if not _passes_sample_filters(x_mask, y_mask, config):
                continue

            _append_sample(
                acc,
                x,
                x_mask,
                y,
                y_mask,
                station_id,
                None,
                None,
                hour_start,
                wanted_target_times,
                context_times,
                actual_agl,
                hourly_height_meta["actual_heights_agl"],
                target_height_meta["actual_heights_agl"],
                f"{pair.hourly_path.name}|{pair.target_path.name}",
                [pair.hourly_path.name for _ in context_times],
                [pair.target_path.name for _ in wanted_target_times],
                "paris_nc",
                config,
            )

    return warnings


def _load_pair_into_station_series(
    pair: RawFilePair,
    config: PreprocessingConfig,
    series_by_station: dict[str, StationSeries],
) -> list[str]:
    """Load one raw pair into global station-indexed time series.

    Stored arrays:
    - hourly[time]: [H=6, C=2]
    - target[time]: [H=6, C=2]
    """

    warnings: list[str] = []
    hourly_ds = open_dataset(pair.hourly_path)
    target_ds = open_dataset(pair.target_path)
    require_variables(hourly_ds, _required_hourly_variables(config), pair.hourly_path)
    require_variables(target_ds, _required_target_variables(config), pair.target_path)

    time_name = str(config.variables["time"])
    hourly_times = _to_start_times(
        dataset_time_values(hourly_ds, time_name),
        config.raw_timestamp_semantics,
        config.input_frequency_seconds,
    )
    target_times = _to_start_times(
        dataset_time_values(target_ds, time_name),
        config.raw_timestamp_semantics,
        config.target_frequency_seconds,
    )

    # raw arrays are [station, time, raw_height, channel]
    hourly_values = _stack_channels(hourly_ds, config.hourly_channels, config, pair.hourly_path)
    target_values = _stack_channels(target_ds, config.target_channels, config, pair.target_path)
    station_count = hourly_values.shape[0]
    hourly_station_ids = _station_ids_for_dataset(hourly_ds, config, station_count, pair.hourly_path)
    target_station_ids = _station_ids_for_dataset(target_ds, config, target_values.shape[0], pair.target_path)
    target_index_by_station = _matched_target_station_indices(
        hourly_station_ids,
        target_station_ids,
        pair.hourly_path,
        pair.target_path,
        config,
    )
    station_alt_name = config.variables.get("station_altitude")
    station_height_name = config.variables.get("station_height")
    station_lat_name = config.variables.get("station_lat")
    station_lon_name = config.variables.get("station_lon")
    source = f"{pair.hourly_path.name}|{pair.target_path.name}"

    for station_idx, station_id in enumerate(hourly_station_ids):
        if not _station_enabled_for_source(config, "paris_nc", station_id):
            continue
        if station_id not in target_index_by_station:
            continue
        target_station_idx = target_index_by_station[station_id]
        station_altitude = station_value(hourly_ds, station_alt_name, station_idx, default=0.0)
        target_station_altitude = station_value(target_ds, station_alt_name, target_station_idx, default=station_altitude)
        station_height = station_value(hourly_ds, station_height_name, station_idx, default=0.0)
        target_station_height = station_value(target_ds, station_height_name, target_station_idx, default=station_height)
        station_lat = station_value(hourly_ds, station_lat_name, station_idx, default=np.nan)
        station_lon = station_value(hourly_ds, station_lon_name, station_idx, default=np.nan)
        try:
            height_config = _height_config_for_source(config, "paris_nc")
            hourly_raw_heights = _height_values_for_station(hourly_ds, config, station_idx)
            target_raw_heights = _height_values_for_station(target_ds, config, target_station_idx)
            hourly_height_meta = select_height_indices(
                hourly_raw_heights,
                station_altitude,
                height_config,
                station_height=station_height,
            )
            target_height_meta = select_height_indices(
                target_raw_heights,
                target_station_altitude,
                height_config,
                station_height=target_station_height,
            )
            _validate_hourly_target_height_match(
                station_id,
                pair.hourly_path,
                pair.target_path,
                hourly_height_meta["selected_heights_agl"],
                hourly_height_meta["actual_heights_agl"],
                target_height_meta["actual_heights_agl"],
                config,
            )
        except Exception as exc:
            raise ValueError(f"Station {station_id} in {pair.prefix} failed height alignment: {exc}") from exc

        hourly_hidx = hourly_height_meta["height_indices"]
        target_hidx = target_height_meta["height_indices"]
        hourly_actual_agl = hourly_height_meta["actual_heights_agl"].astype(np.float32)
        target_actual_agl = target_height_meta["actual_heights_agl"].astype(np.float32)
        actual_agl = _representative_heights(hourly_actual_agl, target_actual_agl)
        station_hourly = np.take(hourly_values[station_idx], hourly_hidx, axis=1)
        station_target = np.take(target_values[target_station_idx], target_hidx, axis=1)

        if station_id not in series_by_station:
            series_by_station[station_id] = StationSeries(
                station_id,
                actual_agl,
                hourly_actual_agl,
                target_actual_agl,
                float(station_lat) if np.isfinite(station_lat) else None,
                float(station_lon) if np.isfinite(station_lon) else None,
                source_group="paris_nc",
            )
        series = series_by_station[station_id]
        _validate_station_height_schema(
            series,
            actual_agl,
            hourly_actual_agl,
            target_actual_agl,
            station_id,
            source,
            config,
        )

        for idx, timestamp in enumerate(hourly_times):
            key = time_key(timestamp)
            if key in series.hourly:
                warnings.append(f"Duplicate hourly timestamp for station {station_id}: {key}; keeping first value.")
                continue
            series.hourly[key] = station_hourly[idx]
            series.hourly_source[key] = source

        for idx, timestamp in enumerate(target_times):
            key = time_key(timestamp)
            if key in series.target:
                warnings.append(f"Duplicate 10min timestamp for station {station_id}: {key}; keeping first value.")
                continue
            series.target[key] = station_target[idx]
            series.target_source[key] = source

    return warnings


def _parse_bool_series(values) -> np.ndarray:
    """Parse bool-like CSV columns into a boolean numpy array."""

    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def _load_standard_csv_source(
    source_name: str,
    source_cfg: dict[str, Any],
    config: PreprocessingConfig,
    series_by_station: dict[str, StationSeries],
) -> list[str]:
    """Load a standard long-table source into global station series.

    Expected CSV schema:
    station_id,time_start,height,u,v,u_mask,v_mask,latitude,longitude,source_file,source_frequency
    """

    warnings: list[str] = []
    hourly_path = Path(source_cfg["hourly_csv"])
    target_path = Path(source_cfg["target_csv"])
    if not hourly_path.exists():
        raise FileNotFoundError(f"Standard hourly CSV not found: {hourly_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Standard target CSV not found: {target_path}")

    hourly = pd.read_csv(hourly_path)
    target = pd.read_csv(target_path)
    required = {
        "station_id",
        "time_start",
        "height",
        "u",
        "v",
        "u_mask",
        "v_mask",
        "latitude",
        "longitude",
        "source_file",
    }
    for label, frame in (("hourly", hourly), ("target", target)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{label} standard CSV is missing columns: {missing}")

    height_config = _height_config_for_source(config, source_name)
    if not height_config.selected_heights_agl:
        warnings.append(
            f"Standard source {source_name} has no selected_heights_agl; skipping source until config is completed."
        )
        return warnings

    for frame in (hourly, target):
        frame["time_start"] = pd.to_datetime(frame["time_start"], errors="coerce")
        frame["height"] = pd.to_numeric(frame["height"], errors="coerce")
        frame["u"] = pd.to_numeric(frame["u"], errors="coerce")
        frame["v"] = pd.to_numeric(frame["v"], errors="coerce")
        frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
        frame["u_mask_bool"] = _parse_bool_series(frame["u_mask"])
        frame["v_mask_bool"] = _parse_bool_series(frame["v_mask"])

    source_label = str(source_cfg.get("source_label", source_name))
    selected = np.asarray(height_config.selected_heights_agl, dtype=float)

    include_station_ids = {str(v) for v in source_cfg.get("include_station_ids", [])}
    exclude_station_ids = {str(v) for v in source_cfg.get("exclude_station_ids", [])}
    station_ids = sorted(set(hourly["station_id"].dropna().astype(str)) | set(target["station_id"].dropna().astype(str)))
    if include_station_ids:
        station_ids = [station_id for station_id in station_ids if station_id in include_station_ids]
    if exclude_station_ids:
        station_ids = [station_id for station_id in station_ids if station_id not in exclude_station_ids]
    if not station_ids:
        warnings.append(f"Standard source {source_name} has no stations after include/exclude filtering.")

    for station_id in station_ids:
        station_hourly_raw = hourly[hourly["station_id"].astype(str) == station_id].copy()
        station_target_raw = target[target["station_id"].astype(str) == station_id].copy()
        if station_hourly_raw.empty or station_target_raw.empty:
            warnings.append(f"Standard source {source_name} station {station_id} missing hourly or target records.")
            continue

        hourly_raw_heights = np.asarray(sorted(station_hourly_raw["height"].dropna().unique()), dtype=float)
        target_raw_heights = np.asarray(sorted(station_target_raw["height"].dropna().unique()), dtype=float)
        if hourly_raw_heights.size == 0:
            warnings.append(f"Standard source {source_name} station {station_id} has no valid heights.")
            continue
        if target_raw_heights.size == 0:
            warnings.append(f"Standard source {source_name} station {station_id} has no valid target heights.")
            continue
        height_offset = (
            height_config.instrument_height_agl_m
            if height_config.height_reference == "instrument_relative_to_ground_agl"
            else 0.0
        )
        hourly_heights_agl = hourly_raw_heights + height_offset
        target_heights_agl = target_raw_heights + height_offset
        hourly_indices = np.asarray(
            [int(np.argmin(np.abs(hourly_heights_agl - h))) for h in selected],
            dtype=np.int64,
        )
        target_indices = np.asarray(
            [int(np.argmin(np.abs(target_heights_agl - h))) for h in selected],
            dtype=np.int64,
        )
        hourly_selected_raw = hourly_raw_heights[hourly_indices].astype(np.float32)
        target_selected_raw = target_raw_heights[target_indices].astype(np.float32)
        hourly_actual = hourly_heights_agl[hourly_indices].astype(np.float32)
        target_actual = target_heights_agl[target_indices].astype(np.float32)
        diff = np.abs(hourly_actual.astype(float) - selected)
        target_diff = np.abs(target_actual.astype(float) - selected)
        if np.any(diff > height_config.max_height_diff) or np.any(target_diff > height_config.max_height_diff):
            raise ValueError(
                f"Skipping standard source {source_name} station {station_id}: "
                f"height diff max={max(float(diff.max()), float(target_diff.max())):.6g} exceeds "
                f"{height_config.max_height_diff}."
            )
        if len(np.unique(hourly_indices)) != len(hourly_indices) or len(np.unique(target_indices)) != len(target_indices):
            raise ValueError(
                f"Skipping standard source {source_name} station {station_id}: selected heights map to duplicate raw layers."
            )
        try:
            _validate_hourly_target_height_match(
                station_id,
                hourly_path,
                target_path,
                selected.astype(np.float32),
                hourly_actual,
                target_actual,
                config,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        actual = _representative_heights(hourly_actual, target_actual)

        lat = float(station_hourly_raw["latitude"].median())
        lon = float(station_hourly_raw["longitude"].median())
        if station_id not in series_by_station:
            series_by_station[station_id] = StationSeries(
                station_id=station_id,
                height_values=actual,
                hourly_height_values=hourly_actual,
                target_height_values=target_actual,
                latitude=lat if np.isfinite(lat) else None,
                longitude=lon if np.isfinite(lon) else None,
                source_group=source_label,
            )
        series = series_by_station[station_id]
        _validate_station_height_schema(
            series,
            actual,
            hourly_actual,
            target_actual,
            station_id,
            f"{hourly_path}|{target_path}",
            config,
        )

        def add_records(frame, selected_actual, target_dict, source_dict, expected_step: str) -> None:
            subset = frame[frame["height"].isin(selected_actual.astype(float))].copy()
            for time_start, time_group in subset.groupby("time_start", sort=True):
                if pd.isna(time_start):
                    continue
                by_height = time_group.drop_duplicates("height", keep="first").set_index("height")
                values = np.full((len(selected_actual), 2), config.quality.missing_value, dtype=np.float32)
                masks = np.zeros((len(selected_actual), 2), dtype=bool)
                for h_i, height in enumerate(selected_actual.astype(float)):
                    if height not in by_height.index:
                        continue
                    row = by_height.loc[height]
                    u = float(row["u"])
                    v = float(row["v"])
                    u_valid = bool(row["u_mask_bool"]) and np.isfinite(u) and u != config.quality.missing_value
                    v_valid = bool(row["v_mask_bool"]) and np.isfinite(v) and v != config.quality.missing_value
                    values[h_i, 0] = u if u_valid else config.quality.missing_value
                    values[h_i, 1] = v if v_valid else config.quality.missing_value
                    masks[h_i, 0] = u_valid
                    masks[h_i, 1] = v_valid
                key = time_key(np.datetime64(time_start.to_datetime64(), "m"))
                if key in target_dict:
                    warnings.append(f"Duplicate {expected_step} timestamp for station {station_id}: {key}; keeping first value.")
                    continue
                target_dict[key] = values
                source_files = sorted(set(str(v) for v in time_group["source_file"].dropna().unique()))
                source_dict[key] = f"{source_name}:{';'.join(source_files)}"

        add_records(station_hourly_raw, hourly_selected_raw, series.hourly, series.hourly_source, "hourly")
        add_records(station_target_raw, target_selected_raw, series.target, series.target_source, "10min")

    return warnings


def _build_samples_from_global_series(
    series_by_station: dict[str, StationSeries],
    config: PreprocessingConfig,
    acc: SampleAccumulator,
    era5_data: Era5StationData | None = None,
    static_data: StationStaticFeatures | None = None,
) -> list[str]:
    """Build sliding-window samples from globally merged station series."""

    warnings: list[str] = []
    spec = AlignmentSpec(
        context_hours=config.context_hours,
        target_steps=config.target_steps_per_hour,
        target_step_minutes=config.target_frequency_seconds // 60,
        input_step_minutes=config.input_frequency_seconds // 60,
    )

    for station_id, series in sorted(series_by_station.items()):
        hourly_times = sorted(series.hourly)
        target_times_available = set(series.target)

        for hour_start in hourly_times:
            context_times = [time_key(t) for t in context_times_for_hour(hour_start, spec)]
            target_times = [time_key(t) for t in target_times_for_hour(hour_start, spec)]

            if any(t not in series.hourly for t in context_times):
                continue
            if any(t not in target_times_available for t in target_times):
                continue

            # x: [L=6, H=6, C=2], y: [T_out=6, H=6, C=2]
            x = np.stack([series.hourly[t] for t in context_times]).astype(np.float32)
            y = np.stack([series.target[t] for t in target_times]).astype(np.float32)
            x_mask = valid_numeric_mask(x, config.quality.missing_value)
            y_mask = valid_numeric_mask(y, config.quality.missing_value)
            x_meteo = None
            meteo_mask = None
            if era5_data is not None:
                x_meteo, meteo_mask = era5_data.sample_context(
                    station_id,
                    context_times,
                    config.quality.missing_value,
                )
            x_static = None
            dominant_lcz = None
            if static_data is not None:
                x_static = static_data.features_for_station(station_id)
                dominant_lcz = static_data.dominant_lcz_for_station(station_id)
                if not np.all(np.isfinite(x_static)):
                    warnings.append(f"Static features for station {station_id} contain NaN/Inf values.")

            if not _passes_sample_filters(x_mask, y_mask, config):
                continue

            source_files = sorted(
                {
                    *(series.hourly_source[t] for t in context_times),
                    *(series.target_source[t] for t in target_times),
                }
            )
            hourly_source_files = [series.hourly_source[t] for t in context_times]
            target_source_files = [series.target_source[t] for t in target_times]
            _append_sample(
                acc,
                x,
                x_mask,
                y,
                y_mask,
                station_id,
                series.latitude,
                series.longitude,
                hour_start,
                target_times,
                context_times,
                series.height_values,
                series.hourly_height_values,
                series.target_height_values,
                ";".join(source_files),
                hourly_source_files,
                target_source_files,
                series.source_group,
                config,
                x_meteo=x_meteo,
                meteo_mask=meteo_mask,
                x_static=x_static,
                dominant_lcz=dominant_lcz,
            )

    if not series_by_station:
        warnings.append("No station series were loaded from raw files.")
    return warnings


def _to_start_times(times: np.ndarray, raw_timestamp_semantics: str, interval_seconds: int) -> np.ndarray:
    """Convert raw timestamps to interval-start timestamps."""

    times = np.asarray(times, dtype="datetime64[m]")
    if raw_timestamp_semantics == "start":
        return times
    if raw_timestamp_semantics == "end":
        return times - np.timedelta64(interval_seconds, "s")
    raise ValueError(f"Unsupported raw_timestamp_semantics: {raw_timestamp_semantics}")


def _empty_arrays(config: PreprocessingConfig) -> dict[str, np.ndarray]:
    h = len(config.height.selected_heights_agl)
    c = len(config.hourly_channels)
    l = config.context_hours
    t_out = config.target_steps_per_hour
    return {
        "x_hourly": np.empty((0, l, h, c), dtype=np.float32),
        "x_mask": np.empty((0, l, h, c), dtype=bool),
        "y_10min": np.empty((0, t_out, h, c), dtype=np.float32),
        "y_mask": np.empty((0, t_out, h, c), dtype=bool),
        "current_hourly": np.empty((0, h, c), dtype=np.float32),
        "x_meteo": np.empty((0, l, 0, 0), dtype=np.float32),
        "meteo_mask": np.empty((0, l, 0, 0), dtype=bool),
        "meteo_pressure_levels": np.empty((0,), dtype=np.float32),
        "meteo_channel_names": np.empty((0,), dtype=object),
        "x_static": np.empty((0, 0), dtype=np.float32),
        "static_feature_names": np.empty((0,), dtype=object),
        "dominant_lcz": np.empty((0,), dtype=np.float32),
        "station_id": np.empty((0,), dtype=object),
        "station_lat": np.empty((0,), dtype=np.float32),
        "station_lon": np.empty((0,), dtype=np.float32),
        "target_time_start": np.empty((0,), dtype=object),
        "target_times_10min": np.empty((0, t_out), dtype=object),
        "context_times_hourly": np.empty((0, l), dtype=object),
        "height_values": np.empty((0, h), dtype=np.float32),
        "hourly_height_values": np.empty((0, h), dtype=np.float32),
        "target_height_values": np.empty((0, h), dtype=np.float32),
        "source_file": np.empty((0,), dtype=object),
        "hourly_source_files": np.empty((0, l), dtype=object),
        "target_source_files": np.empty((0, t_out), dtype=object),
        "source_group": np.empty((0,), dtype=object),
        "split": np.empty((0,), dtype=object),
    }


def _split_labels_for_times(target_time_start: list[str], config: PreprocessingConfig) -> np.ndarray:
    """Create chronological split labels with an optional temporal embargo.

    Unique target times are split as:
    train | gap | val | gap | test

    Gap samples remain in the dataset with ``split == "gap"`` but are not
    written to train/val/test split files.
    """

    n = len(target_time_start)
    labels = np.full((n,), "gap", dtype=object)
    if n == 0:
        return labels
    unique_times = sorted(set(target_time_start))
    n_unique = len(unique_times)
    unique_dt = np.asarray(unique_times, dtype="datetime64[m]")
    gap = np.timedelta64(config.splits.split_gap_hours, "h")

    train_count = int(n_unique * config.splits.train_ratio)
    val_count = int(n_unique * config.splits.val_ratio)
    train_times = set(unique_times[:train_count])

    if train_count > 0:
        val_start_after = unique_dt[train_count - 1] + gap
        val_candidates = [t for t, dt in zip(unique_times, unique_dt) if dt > val_start_after]
    else:
        val_candidates = unique_times

    val_times = set(val_candidates[:val_count])
    if val_times:
        last_val = np.asarray(sorted(val_times), dtype="datetime64[m]")[-1]
        test_start_after = last_val + gap
        test_times = {
            t for t, dt in zip(unique_times, unique_dt) if dt > test_start_after
        }
    else:
        test_times = set(unique_times) - train_times

    for i, t in enumerate(target_time_start):
        if t in train_times:
            labels[i] = "train"
        elif t in val_times:
            labels[i] = "val"
        elif t in test_times:
            labels[i] = "test"
    return labels


def _split_labels(
    target_time_start: list[str],
    source_group: list[str],
    station_id: list[str],
    config: PreprocessingConfig,
) -> np.ndarray:
    """Create split labels globally, by source, or by station."""

    if not config.split_within_source and not config.split_within_station:
        return _split_labels_for_times(target_time_start, config)

    labels = np.full((len(target_time_start),), "gap", dtype=object)
    if config.split_within_station:
        groups = np.asarray(
            [f"{source}\0{station}" for source, station in zip(source_group, station_id)],
            dtype=object,
        )
    else:
        groups = np.asarray(source_group, dtype=object)
    times = np.asarray(target_time_start, dtype=object)
    for group in sorted(set(groups.tolist())):
        indices = np.where(groups == group)[0]
        group_labels = _split_labels_for_times(times[indices].tolist(), config)
        labels[indices] = group_labels
    return labels


def _sample_balance_metrics(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return physical-space sample summaries used only for balanced splitting."""

    y = arrays["y_10min"].astype(float)
    y_mask = arrays["y_mask"].astype(bool)
    current = arrays["current_hourly"].astype(float)
    current_mask = arrays["x_mask"][:, -1].astype(bool)
    vector_valid = y_mask[..., 0] & y_mask[..., 1]
    current_valid = current_mask[..., 0] & current_mask[..., 1]

    def masked_sample_mean(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        axes = tuple(range(1, values.ndim))
        numerator = np.where(valid, values, 0.0).sum(axis=axes)
        denominator = valid.sum(axis=axes)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    wind_speed = np.hypot(y[..., 0], y[..., 1])
    residual = y - current[:, None, :, :]
    residual_valid = vector_valid & current_valid[:, None, :]
    residual_mag = np.hypot(residual[..., 0], residual[..., 1])
    residual_delta = residual[:, 1:] - residual[:, :-1]
    delta_valid = residual_valid[:, 1:] & residual_valid[:, :-1]
    gradient_mag = np.hypot(residual_delta[..., 0], residual_delta[..., 1])
    return {
        "wind_speed": masked_sample_mean(wind_speed, vector_valid),
        "residual_magnitude": masked_sample_mean(residual_mag, residual_valid),
        "temporal_gradient": masked_sample_mean(gradient_mag, delta_valid),
        "valid_mask_rate": y_mask.reshape(y_mask.shape[0], -1).mean(axis=1),
    }


def _balanced_block_split_labels(
    arrays: dict[str, np.ndarray],
    config: PreprocessingConfig,
) -> np.ndarray:
    """Assign station-local time blocks while matching marginal distributions."""

    n = int(arrays["x_hourly"].shape[0])
    labels = np.full((n,), "gap", dtype=object)
    if n == 0:
        return labels

    ratios = np.asarray(
        [config.splits.train_ratio, config.splits.val_ratio, config.splits.test_ratio],
        dtype=float,
    )
    split_names = np.asarray(["train", "val", "test"], dtype=object)
    times = np.asarray(arrays["target_time_start"], dtype="datetime64[m]")
    stations = arrays["station_id"].astype(str)
    sources = arrays["source_group"].astype(str)
    metrics = _sample_balance_metrics(arrays)
    durations = config.splits.block_duration_hours or {"default": 168}
    default_duration = int(durations.get("default", 168))
    event_gap = np.timedelta64(config.splits.event_gap_hours, "h")
    purge = np.timedelta64(config.splits.purge_hours, "h")
    configured_weights = config.splits.balance_weights or {}

    groups = np.asarray([f"{source}\0{station}" for source, station in zip(sources, stations)], dtype=object)
    for group in sorted(set(groups.tolist())):
        group_indices = np.where(groups == group)[0]
        group_indices = group_indices[np.argsort(times[group_indices])]
        source = sources[group_indices[0]]
        duration = np.timedelta64(int(durations.get(source, default_duration)), "h")

        blocks: list[dict[str, object]] = []
        event_id = 0
        current: list[int] = []
        block_start = times[group_indices[0]]
        previous_time = block_start
        for idx in group_indices:
            timestamp = times[idx]
            new_event = bool(current) and timestamp - previous_time > event_gap
            new_block = bool(current) and timestamp - block_start >= duration
            if new_event or new_block:
                blocks.append(
                    {
                        "indices": np.asarray(current, dtype=np.int64),
                        "start": block_start,
                        "event": event_id,
                    }
                )
                if new_event:
                    event_id += 1
                current = []
                block_start = timestamp
            current.append(int(idx))
            previous_time = timestamp
        if current:
            blocks.append(
                {
                    "indices": np.asarray(current, dtype=np.int64),
                    "start": block_start,
                    "event": event_id,
                }
            )
        if len(blocks) < 3:
            raise ValueError(f"Balanced block split requires at least 3 blocks for {group!r}; got {len(blocks)}")

        station_values = {name: values[group_indices] for name, values in metrics.items()}
        binned: dict[str, np.ndarray] = {}
        for name in ("wind_speed", "residual_magnitude", "temporal_gradient"):
            thresholds = np.quantile(station_values[name], [1.0 / 3.0, 2.0 / 3.0])
            binned[name] = np.searchsorted(thresholds, station_values[name], side="right")
        local_position = {int(idx): pos for pos, idx in enumerate(group_indices)}
        months = times[group_indices].astype("datetime64[M]").astype(int) % 12
        seasons = np.asarray([(month % 12) // 3 for month in months], dtype=np.int64)

        vectors: list[np.ndarray] = []
        for block in blocks:
            positions = np.asarray([local_position[int(idx)] for idx in block["indices"]], dtype=np.int64)
            vectors.append(
                np.concatenate(
                    [
                        np.asarray([len(positions)], dtype=float),
                        np.bincount(months[positions], minlength=12).astype(float),
                        np.bincount(seasons[positions], minlength=4).astype(float),
                        np.bincount(binned["wind_speed"][positions], minlength=3).astype(float),
                        np.bincount(binned["residual_magnitude"][positions], minlength=3).astype(float),
                        np.bincount(binned["temporal_gradient"][positions], minlength=3).astype(float),
                        np.asarray([station_values["valid_mask_rate"][positions].sum()], dtype=float),
                    ]
                )
            )
        block_vectors = np.stack(vectors)
        feature_weights = np.concatenate(
            [
                np.full(1, configured_weights.get("sample_count", 2.0)),
                np.full(12, configured_weights.get("month", 0.5)),
                np.full(4, configured_weights.get("season", 1.0)),
                np.full(3, configured_weights.get("wind_speed", 1.0)),
                np.full(3, configured_weights.get("residual_magnitude", 1.5)),
                np.full(3, configured_weights.get("temporal_gradient", 1.5)),
                np.full(1, configured_weights.get("valid_mask_rate", 0.5)),
            ]
        )
        target = ratios[:, None] * block_vectors.sum(axis=0, keepdims=True)
        target_samples = ratios * block_vectors[:, 0].sum()

        group_seed = config.splits.seed + sum((i + 1) * ord(char) for i, char in enumerate(group))
        rng = np.random.default_rng(group_seed)
        best_assignment = None
        best_score = float("inf")
        for _ in range(config.splits.search_trials):
            assignment = np.full((len(blocks),), -1, dtype=np.int64)
            assigned_samples = np.zeros(3, dtype=float)
            block_counts = np.zeros(3, dtype=np.int64)
            for block_index in rng.permutation(len(blocks)):
                remaining = len(blocks) - int((assignment >= 0).sum())
                empty = np.where(block_counts == 0)[0]
                candidates = empty if len(empty) >= remaining else np.arange(3)
                deficits = np.divide(
                    target_samples[candidates] - assigned_samples[candidates],
                    np.maximum(target_samples[candidates], 1.0),
                )
                split_index = int(candidates[np.argmax(deficits + rng.random(len(candidates)) * 1e-9)])
                assignment[block_index] = split_index
                assigned_samples[split_index] += block_vectors[block_index, 0]
                block_counts[split_index] += 1

            actual = np.zeros_like(target)
            for block_index, split_index in enumerate(assignment):
                actual[split_index] += block_vectors[block_index]
            normalized_error = (actual - target) / np.maximum(target, 1.0)
            score = float(np.mean(feature_weights[None, :] * normalized_error**2))
            if score < best_score:
                best_score = score
                best_assignment = assignment.copy()

        assert best_assignment is not None
        for block, split_index in zip(blocks, best_assignment):
            labels[block["indices"]] = split_names[split_index]

        for previous, current_block, previous_split, current_split in zip(
            blocks[:-1],
            blocks[1:],
            best_assignment[:-1],
            best_assignment[1:],
        ):
            if previous["event"] != current_block["event"] or previous_split == current_split:
                continue
            cutoff = current_block["start"] + purge
            current_indices = current_block["indices"]
            labels[current_indices[times[current_indices] < cutoff]] = "gap"

        for split_name in split_names:
            if not np.any(labels[group_indices] == split_name):
                raise ValueError(f"Balanced block split left {group!r} without {split_name} samples")
    return labels


def _arrays_from_accumulator(acc: SampleAccumulator, config: PreprocessingConfig) -> dict[str, np.ndarray]:
    if len(acc) == 0:
        return _empty_arrays(config)
    arrays = {
        # x_hourly: [N, L=6, H=6, C=2]
        "x_hourly": np.stack(acc.x_hourly).astype(np.float32),
        "x_mask": np.stack(acc.x_mask).astype(bool),
        # y_10min: [N, T_out=6, H=6, C=2]
        "y_10min": np.stack(acc.y_10min).astype(np.float32),
        "y_mask": np.stack(acc.y_mask).astype(bool),
        "current_hourly": np.stack(acc.current_hourly).astype(np.float32),
        "station_id": np.asarray(acc.station_id, dtype=object),
        "station_lat": np.asarray(acc.station_lat, dtype=np.float32),
        "station_lon": np.asarray(acc.station_lon, dtype=np.float32),
        "target_time_start": np.asarray(acc.target_time_start, dtype=object),
        "target_times_10min": np.asarray(acc.target_times_10min, dtype=object),
        "context_times_hourly": np.asarray(acc.context_times_hourly, dtype=object),
        "height_values": np.stack(acc.height_values).astype(np.float32),
        "hourly_height_values": np.stack(acc.hourly_height_values).astype(np.float32),
        "target_height_values": np.stack(acc.target_height_values).astype(np.float32),
        "source_file": np.asarray(acc.source_file, dtype=object),
        "hourly_source_files": np.asarray(acc.hourly_source_files, dtype=object),
        "target_source_files": np.asarray(acc.target_source_files, dtype=object),
        "source_group": np.asarray(acc.source_group, dtype=object),
    }
    if acc.x_meteo:
        arrays["x_meteo"] = np.stack(acc.x_meteo).astype(np.float32)
        arrays["meteo_mask"] = np.stack(acc.meteo_mask).astype(bool)
    if acc.x_static:
        arrays["x_static"] = np.stack(acc.x_static).astype(np.float32)
        arrays["static_feature_names"] = np.asarray(config.static_features.feature_columns, dtype=object)
        arrays["dominant_lcz"] = np.asarray(acc.dominant_lcz, dtype=np.float32)
    if config.splits.strategy == "balanced_blocks":
        arrays["split"] = _balanced_block_split_labels(arrays, config)
    else:
        arrays["split"] = _split_labels(
            acc.target_time_start,
            acc.source_group,
            acc.station_id,
            config,
        )
    return arrays


def _validate_arrays_before_write(arrays: dict[str, np.ndarray], config: PreprocessingConfig) -> None:
    """Validate global dataset invariants before writing dataset.npz."""

    n = int(arrays["x_hourly"].shape[0])
    for key, value in arrays.items():
        if key in {"meteo_pressure_levels", "meteo_channel_names", "static_feature_names"}:
            continue
        if value.shape and int(value.shape[0]) != n:
            raise ValueError(f"Dataset field {key!r} has first dimension {value.shape[0]}, expected {n}")

    expected_x = (n, config.context_hours, len(config.height.selected_heights_agl), len(config.hourly_channels))
    expected_y = (n, config.target_steps_per_hour, len(config.height.selected_heights_agl), len(config.target_channels))
    if arrays["x_hourly"].shape != expected_x:
        raise ValueError(f"x_hourly shape {arrays['x_hourly'].shape} does not match expected {expected_x}")
    if arrays["y_10min"].shape != expected_y:
        raise ValueError(f"y_10min shape {arrays['y_10min'].shape} does not match expected {expected_y}")

    h_shape = (n, len(config.height.selected_heights_agl))
    for key in ("height_values", "hourly_height_values", "target_height_values"):
        if arrays[key].shape != h_shape:
            raise ValueError(f"{key} shape {arrays[key].shape} does not match expected {h_shape}")
        if not np.all(np.diff(arrays[key], axis=1) > 0):
            raise ValueError(f"{key} must be strictly increasing for every sample")

    height_diff = np.abs(arrays["hourly_height_values"].astype(float) - arrays["target_height_values"].astype(float))
    tolerance = float(config.data_alignment.max_hourly_target_height_diff_m)
    if height_diff.size and np.nanmax(height_diff) > tolerance:
        raise ValueError(
            f"hourly_height_values and target_height_values differ by more than {tolerance} m; "
            f"max_diff={float(np.nanmax(height_diff))}"
        )

    for value_key, mask_key in (("x_hourly", "x_mask"), ("y_10min", "y_mask")):
        values = arrays[value_key]
        mask = arrays[mask_key].astype(bool)
        if not np.isfinite(values[mask]).all():
            raise ValueError(f"{value_key} contains NaN/Inf at valid mask positions")

    station_ids = [str(v).strip() for v in arrays["station_id"]]
    if any(not station_id for station_id in station_ids):
        raise ValueError("station_id contains empty values")
    if config.expected_station_ids:
        expected = set(config.expected_station_ids)
        actual = set(station_ids)
        if actual != expected:
            raise ValueError(
                "Dataset station set does not match output.expected_station_ids: "
                f"missing={sorted(expected - actual)}; unexpected={sorted(actual - expected)}"
            )
    sample_keys = list(zip(station_ids, [str(v) for v in arrays["target_time_start"]]))
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("Duplicate station_id + target_time_start samples detected")

    spec = AlignmentSpec(
        context_hours=config.context_hours,
        target_steps=config.target_steps_per_hour,
        target_step_minutes=config.target_frequency_seconds // 60,
        input_step_minutes=config.input_frequency_seconds // 60,
    )
    for idx in range(n):
        start = np.datetime64(arrays["target_time_start"][idx], "m")
        context = [np.datetime64(t, "m") for t in arrays["context_times_hourly"][idx]]
        targets = [np.datetime64(t, "m") for t in arrays["target_times_10min"][idx]]
        if [time_key(t) for t in context] != [time_key(t) for t in context_times_for_hour(start, spec)]:
            raise ValueError(f"context_times_hourly mismatch at sample {idx}")
        if [time_key(t) for t in targets] != [time_key(t) for t in target_times_for_hour(start, spec)]:
            raise ValueError(f"target_times_10min mismatch at sample {idx}")
        if targets and not all(start <= t < start + np.timedelta64(1, "h") for t in targets):
            raise ValueError(f"target_times_10min outside target hour at sample {idx}")
        if not np.allclose(arrays["current_hourly"][idx], arrays["x_hourly"][idx, -1], equal_nan=True):
            raise ValueError(f"current_hourly does not match x_hourly[-1] at sample {idx}")

    if config.splits.strategy == "balanced_blocks":
        times = np.asarray(arrays["target_time_start"], dtype="datetime64[m]")
        groups = np.asarray(
            [
                f"{source}\0{station}"
                for source, station in zip(arrays["source_group"].astype(str), arrays["station_id"].astype(str))
            ],
            dtype=object,
        )
        purge = np.timedelta64(config.splits.purge_hours, "h")
        for group in set(groups.tolist()):
            active = np.where((groups == group) & (arrays["split"] != "gap"))[0]
            active = active[np.argsort(times[active])]
            for left, right in zip(active[:-1], active[1:]):
                if arrays["split"][left] != arrays["split"][right] and times[right] - times[left] <= purge:
                    raise ValueError(f"Balanced split purge violated for {group!r}")


def _write_dataset(
    arrays: dict[str, np.ndarray],
    pairs: list[RawFilePair],
    config: PreprocessingConfig,
    warnings: list[str],
) -> None:
    _validate_arrays_before_write(arrays, config)
    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.dataset_dir / "dataset.npz", **arrays)

    split_dir = config.dataset_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    for split_name in ("train", "val", "test", "gap"):
        indices = np.where(arrays["split"] == split_name)[0]
        with (split_dir / f"{split_name}.txt").open("w", encoding="utf-8") as f:
            for idx in indices:
                f.write(f"{int(idx)}\n")

    split_labels, split_counts = np.unique(arrays["split"], return_counts=True)

    metadata = {
        "dataset_name": config.dataset_name,
        "timestamp_semantics": config.timestamp_semantics,
        "context_alignment": config.context_alignment,
        "target_offsets_minutes": list(config.target_offsets_minutes),
        "raw_timestamp_semantics": config.raw_timestamp_semantics,
        "data_alignment": {
            "station_matching_mode": config.data_alignment.station_matching_mode,
            "max_hourly_target_height_diff_m": config.data_alignment.max_hourly_target_height_diff_m,
            "height_schema_policy": config.data_alignment.height_schema_policy,
            "height_values_definition": "0.5 * (hourly_height_values + target_height_values)",
            "source_file_definition": "summary set of hourly and target source files for the sample",
        },
        "split_policy": {
            "strategy": config.splits.strategy,
            "split_time_key": config.splits.split_time_key,
            "train_ratio": config.splits.train_ratio,
            "val_ratio": config.splits.val_ratio,
            "test_ratio": config.splits.test_ratio,
            "split_gap_hours": config.splits.split_gap_hours,
            "split_within_source": config.split_within_source,
            "split_within_station": config.split_within_station,
            "seed": config.splits.seed,
            "event_gap_hours": config.splits.event_gap_hours,
            "purge_hours": config.splits.purge_hours,
            "block_duration_hours": config.splits.block_duration_hours,
            "balance_weights": config.splits.balance_weights,
            "search_trials": config.splits.search_trials,
            "gap_label": "gap",
        },
        "split_counts": {
            str(label): int(count) for label, count in zip(split_labels, split_counts)
        },
        "station_selection": {
            "expected_station_ids": list(config.expected_station_ids),
            "actual_station_ids": sorted({str(v) for v in arrays["station_id"]}),
        },
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "channel_names": list(config.hourly_channels),
        "meteo": {
            "enabled": bool(config.meteo.enabled),
            "source": config.meteo.source,
            "pressure_dir": str(config.meteo.pressure_dir) if config.meteo.pressure_dir else None,
            "interpolation": config.meteo.interpolation,
            "out_of_bounds": config.meteo.out_of_bounds,
            "channel_names": list(config.meteo.channel_names),
            "pressure_levels_hpa": arrays.get("meteo_pressure_levels", np.asarray([])).tolist(),
        },
        "static_features": {
            "use_lcz": bool(config.static_features.use_lcz),
            "lcz_feature_csv": (
                str(config.static_features.lcz_feature_csv)
                if config.static_features.lcz_feature_csv is not None
                else None
            ),
            "feature_columns": list(config.static_features.feature_columns),
            "has_x_static": "x_static" in arrays and arrays["x_static"].shape[-1] > 0,
        },
        "selected_heights_agl": list(config.height.selected_heights_agl),
        "height_selection_by_source": {
            name: {
                "selected_heights_agl": list(height.selected_heights_agl),
                "height_reference": height.height_reference,
                "max_height_diff": height.max_height_diff,
                "instrument_height_agl_m": height.instrument_height_agl_m,
            }
            for name, height in config.height_by_source.items()
        },
        "sources": config.sources,
        "missing_value": config.quality.missing_value,
        "raw_pairs": [
            {
                "prefix": pair.prefix,
                "hourly_path": str(pair.hourly_path),
                "target_path": str(pair.target_path),
            }
            for pair in pairs
        ],
        "warnings": warnings,
    }
    with (config.dataset_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def build_dataset(config: PreprocessingConfig, dry_run: bool = False) -> BuildSummary:
    """Build dataset artifacts from paired raw NetCDF files.

    Output arrays:
    - x_hourly: [N, L=6, H=6, C=2]
    - x_mask: [N, L=6, H=6, C=2]
    - y_10min: [N, T_out=6, H=6, C=2]
    - y_mask: [N, T_out=6, H=6, C=2]
    - current_hourly: [N, H=6, C=2]
    """

    warnings = validate_config(config)
    source_cfg = config.sources or {}
    paris_cfg = dict(source_cfg.get("paris_nc", {}))
    paris_enabled = bool(paris_cfg.get("enabled", True))
    pairs = pair_nc_files(config.raw_3600s_dir, config.raw_600s_dir) if paris_enabled else []
    acc = SampleAccumulator()
    series_by_station: dict[str, StationSeries] = {}
    era5_data: Era5StationData | None = None
    static_data: StationStaticFeatures | None = None

    if paris_enabled:
        if not pairs:
            warnings.append("Paris NC source enabled but no paired *_3600s.nc / *_600s.nc files found.")
        for pair in pairs:
            try:
                warnings.extend(_load_pair_into_station_series(pair, config, series_by_station))
            except Exception as exc:
                raise RuntimeError(f"Failed to load raw pair {pair.prefix}: {exc}") from exc

    standard_sources = dict(source_cfg.get("standard_csv", {}))
    for source_name, standard_cfg in standard_sources.items():
        standard_cfg = dict(standard_cfg or {})
        if not bool(standard_cfg.get("enabled", False)):
            continue
        try:
            warnings.extend(_load_standard_csv_source(source_name, standard_cfg, config, series_by_station))
        except Exception as exc:
            raise RuntimeError(f"Failed to load standard CSV source {source_name}: {exc}") from exc

    if not series_by_station:
        warnings.append("No station series were loaded from enabled sources; writing empty dataset.")

    if config.meteo.enabled and series_by_station:
        station_locations = {
            station_id: StationLocation(station_id, series.latitude, series.longitude)
            for station_id, series in series_by_station.items()
            if series.latitude is not None and series.longitude is not None
        }
        missing_locations = sorted(set(series_by_station) - set(station_locations))
        if missing_locations:
            warnings.append(
                "ERA5 enabled but station lat/lon missing for: "
                + ", ".join(missing_locations)
            )
        if station_locations:
            era5_data = load_era5_for_stations(config.meteo, station_locations)
            for station_id, method in sorted(era5_data.interpolation_method_by_station.items()):
                if method == "nearest":
                    warnings.append(f"ERA5 nearest interpolation used for station {station_id}.")

    if config.static_features.use_lcz and series_by_station:
        static_data = load_station_static_features(
            config.static_features.lcz_feature_csv,
            config.static_features.feature_columns,
            station_id_column=config.static_features.station_id_column,
            dominant_lcz_column=config.static_features.dominant_lcz_column,
        )
        missing_static = sorted(set(series_by_station) - set(static_data.values_by_station))
        if missing_static:
            raise KeyError(
                "LCZ static features missing for station_id(s): "
                + ", ".join(missing_static)
            )

    warnings.extend(
        _build_samples_from_global_series(
            series_by_station,
            config,
            acc,
            era5_data,
            static_data,
        )
    )

    arrays = _arrays_from_accumulator(acc, config)
    if era5_data is not None:
        arrays["meteo_pressure_levels"] = era5_data.pressure_levels.astype(np.float32)
        arrays["meteo_channel_names"] = np.asarray(era5_data.channel_names, dtype=object)
    if not dry_run:
        _write_dataset(arrays, pairs, config, warnings)

    status = "dry_run" if dry_run else "written"
    if len(acc) == 0:
        status = f"{status}_empty"
    return BuildSummary(
        dataset_name=config.dataset_name,
        dataset_dir=config.dataset_dir,
        status=status,
        num_pairs=len(pairs),
        num_samples=len(acc),
        warnings=tuple(warnings),
    )
