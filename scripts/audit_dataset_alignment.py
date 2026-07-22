"""Audit raw file pairing and generated dataset alignment invariants."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.alignment import AlignmentSpec, context_times_for_hour, target_times_for_hour, time_key
from src.data.dataset_builder import (
    _height_config_for_source,
    _height_values_for_station,
    _matched_target_station_indices,
    _station_ids_for_dataset,
    _to_start_times,
    _validate_hourly_target_height_match,
    _validate_station_height_schema,
    StationSeries,
)
from src.data.preprocessing import parse_preprocessing_config, select_height_indices
from src.data.raw_reader import dataset_time_values, open_dataset, pair_nc_files


def audit_raw_config(config_path: str | Path) -> list[str]:
    config = parse_preprocessing_config(config_path)
    errors: list[str] = []
    schema_by_station: dict[str, StationSeries] = {}
    pairs = pair_nc_files(config.raw_3600s_dir, config.raw_600s_dir)
    for pair in pairs:
        try:
            hourly_ds = open_dataset(pair.hourly_path)
            target_ds = open_dataset(pair.target_path)
            hourly_count = int(hourly_ds.sizes.get(config.variables["station"], 1))
            target_count = int(target_ds.sizes.get(config.variables["station"], 1))
            hourly_ids = _station_ids_for_dataset(hourly_ds, config, hourly_count, pair.hourly_path)
            target_ids = _station_ids_for_dataset(target_ds, config, target_count, pair.target_path)
            target_index_by_station = _matched_target_station_indices(
                hourly_ids,
                target_ids,
                pair.hourly_path,
                pair.target_path,
                config,
            )
            _check_time_axis(hourly_ds, config, pair.hourly_path, config.input_frequency_seconds)
            _check_time_axis(target_ds, config, pair.target_path, config.target_frequency_seconds)

            station_alt_name = config.variables.get("station_altitude")
            station_height_name = config.variables.get("station_height")
            height_cfg = _height_config_for_source(config, "paris_nc")
            for hourly_idx, station_id in enumerate(hourly_ids):
                if station_id not in target_index_by_station:
                    continue
                target_idx = target_index_by_station[station_id]
                station_alt = _station_value(hourly_ds, station_alt_name, hourly_idx, 0.0)
                target_alt = _station_value(target_ds, station_alt_name, target_idx, station_alt)
                station_height = _station_value(hourly_ds, station_height_name, hourly_idx, 0.0)
                target_station_height = _station_value(
                    target_ds,
                    station_height_name,
                    target_idx,
                    station_height,
                )
                hourly_meta = select_height_indices(
                    _height_values_for_station(hourly_ds, config, hourly_idx),
                    station_alt,
                    height_cfg,
                    station_height=station_height,
                )
                target_meta = select_height_indices(
                    _height_values_for_station(target_ds, config, target_idx),
                    target_alt,
                    height_cfg,
                    station_height=target_station_height,
                )
                _validate_hourly_target_height_match(
                    station_id,
                    pair.hourly_path,
                    pair.target_path,
                    hourly_meta["selected_heights_agl"],
                    hourly_meta["actual_heights_agl"],
                    target_meta["actual_heights_agl"],
                    config,
                )
                representative = 0.5 * (
                    hourly_meta["actual_heights_agl"] + target_meta["actual_heights_agl"]
                )
                current = StationSeries(
                    station_id,
                    representative.astype(np.float32),
                    hourly_meta["actual_heights_agl"].astype(np.float32),
                    target_meta["actual_heights_agl"].astype(np.float32),
                )
                if station_id in schema_by_station:
                    _validate_station_height_schema(
                        schema_by_station[station_id],
                        current.height_values,
                        current.hourly_height_values,
                        current.target_height_values,
                        station_id,
                        f"{pair.hourly_path}|{pair.target_path}",
                        config,
                    )
                else:
                    schema_by_station[station_id] = current
        except Exception as exc:  # audit should collect all pair failures
            errors.append(str(exc))
    print(f"raw_pairs: {len(pairs)}")
    print(f"raw_audit_errors: {len(errors)}")
    for error in errors[:20]:
        print(f"ERROR: {error}")
    return errors


def audit_dataset(dataset_dir: str | Path) -> list[str]:
    dataset_dir = Path(dataset_dir)
    errors: list[str] = []
    with np.load(dataset_dir / "dataset.npz", allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    n = int(arrays["x_hourly"].shape[0])
    print(f"samples: {n}")
    for key, value in arrays.items():
        if key in {"meteo_pressure_levels", "meteo_channel_names", "static_feature_names"}:
            continue
        if value.shape and int(value.shape[0]) != n:
            errors.append(f"{key} first dimension {value.shape[0]} != {n}")
    if "context_times_hourly" not in arrays:
        errors.append("missing context_times_hourly")
    for key in ("hourly_height_values", "target_height_values"):
        if key not in arrays:
            errors.append(f"missing {key}")
    if "hourly_source_files" not in arrays or "target_source_files" not in arrays:
        errors.append("missing per-step source file fields")

    sample_keys = list(zip(map(str, arrays["station_id"]), map(str, arrays["target_time_start"])))
    duplicate_count = len(sample_keys) - len(set(sample_keys))
    if duplicate_count:
        errors.append(f"duplicate station_id + target_time_start count={duplicate_count}")

    if "context_times_hourly" in arrays:
        l = arrays["x_hourly"].shape[1]
        t_out = arrays["y_10min"].shape[1]
        spec = AlignmentSpec(context_hours=l, target_steps=t_out)
        for idx in range(n):
            start = np.datetime64(arrays["target_time_start"][idx], "m")
            context = [np.datetime64(v, "m") for v in arrays["context_times_hourly"][idx]]
            targets = [np.datetime64(v, "m") for v in arrays["target_times_10min"][idx]]
            if [time_key(v) for v in context] != [time_key(v) for v in context_times_for_hour(start, spec)]:
                errors.append(f"context time mismatch at sample {idx}")
                break
            if [time_key(v) for v in targets] != [time_key(v) for v in target_times_for_hour(start, spec)]:
                errors.append(f"target time mismatch at sample {idx}")
                break

    for values_key, mask_key in (("x_hourly", "x_mask"), ("y_10min", "y_mask")):
        if not np.isfinite(arrays[values_key][arrays[mask_key].astype(bool)]).all():
            errors.append(f"{values_key} contains NaN/Inf at valid positions")

    if "hourly_height_values" in arrays and "target_height_values" in arrays:
        diff = np.abs(arrays["hourly_height_values"].astype(float) - arrays["target_height_values"].astype(float))
        print(f"hourly_target_height_diff_max_m: {float(np.nanmax(diff)) if diff.size else 0.0}")

    stations, counts = np.unique(arrays["station_id"], return_counts=True)
    print("samples_by_station:")
    for station, count in zip(stations, counts):
        print(f"  {station}: {int(count)}")
    print(f"dataset_audit_errors: {len(errors)}")
    for error in errors[:20]:
        print(f"ERROR: {error}")
    return errors


def _check_time_axis(ds, config, path: Path, expected_seconds: int) -> None:
    times = _to_start_times(
        dataset_time_values(ds, str(config.variables["time"])),
        config.raw_timestamp_semantics,
        expected_seconds,
    )
    if len(times) != len(np.unique(times)):
        raise ValueError(f"{path} contains duplicate timestamps after semantic conversion")
    if len(times) > 1:
        diffs = np.diff(np.asarray(times, dtype="datetime64[s]")).astype("timedelta64[s]").astype(int)
        if np.any(diffs <= 0):
            raise ValueError(f"{path} timestamps are not strictly increasing")
        if np.any(diffs != expected_seconds):
            raise ValueError(f"{path} timestamp interval mismatch; expected {expected_seconds}s")


def _station_value(ds, name: str | None, station_index: int, default):
    if not name or name not in ds:
        return default
    values = np.asarray(ds[name].values)
    if values.ndim == 0:
        return values.item()
    return values[station_index].item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Audit raw paired files using this preprocessing config")
    parser.add_argument("--dataset-dir", help="Audit an existing generated dataset directory")
    args = parser.parse_args()
    errors: list[str] = []
    if args.config:
        errors.extend(audit_raw_config(args.config))
    if args.dataset_dir:
        errors.extend(audit_dataset(args.dataset_dir))
    if not args.config and not args.dataset_dir:
        parser.error("Provide --config and/or --dataset-dir")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
