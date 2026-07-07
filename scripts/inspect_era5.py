"""Inspect ERA5 temperature/humidity NetCDF files."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import parse_preprocessing_config


def _dt(value: int) -> str:
    return datetime.fromtimestamp(int(value), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def inspect_era5(config_path: str | Path) -> None:
    config = parse_preprocessing_config(config_path)
    if not config.meteo.enabled:
        print("meteo.enabled: false")
        return
    if config.meteo.pressure_dir is None:
        raise ValueError("meteo.pressure_dir is not configured")

    files = sorted(Path(config.meteo.pressure_dir).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 files found in {config.meteo.pressure_dir}")

    names = config.meteo.variables
    time_name = str(names["time"])
    level_name = str(names["pressure_level"])
    lat_name = str(names["latitude"])
    lon_name = str(names["longitude"])
    temp_name = str(names["temperature"])
    humidity_name = str(names["humidity"])

    print(f"era5_pressure_dir: {config.meteo.pressure_dir}")
    print(f"files: {len(files)}")
    all_times: list[np.ndarray] = []
    total_nan = {temp_name: 0, humidity_name: 0}
    total_count = {temp_name: 0, humidity_name: 0}

    first_dims_printed = False
    pressure_levels = None
    latitudes = None
    longitudes = None
    for path in files:
        with h5py.File(path, "r") as f:
            times = np.asarray(f[time_name][:], dtype=np.int64)
            all_times.append(times)
            if not first_dims_printed:
                print("dimensions:")
                for key, value in f.items():
                    if isinstance(value, h5py.Dataset):
                        print(f"  {key}: shape={value.shape} dtype={value.dtype}")
                print("variables:")
                for key in f.keys():
                    print(f"  {key}")
                pressure_levels = np.asarray(f[level_name][:], dtype=float)
                latitudes = np.asarray(f[lat_name][:], dtype=float)
                longitudes = np.asarray(f[lon_name][:], dtype=float)
                print(f"pressure_levels_hpa: {pressure_levels.tolist()}")
                print(f"latitude_range: ({float(latitudes.min())}, {float(latitudes.max())}) values={latitudes.tolist()}")
                print(f"longitude_range: ({float(longitudes.min())}, {float(longitudes.max())}) values={longitudes.tolist()}")
                print(f"{temp_name}_shape: {f[temp_name].shape}")
                print(f"{humidity_name}_shape: {f[humidity_name].shape}")
                first_dims_printed = True

            for var in (temp_name, humidity_name):
                values = np.asarray(f[var][:])
                total_nan[var] += int(np.isnan(values).sum())
                total_count[var] += int(values.size)

    concat = np.concatenate(all_times)
    concat.sort()
    diffs = np.diff(concat)
    unique_diffs, counts = np.unique(diffs, return_counts=True)
    print(f"time_range: {_dt(concat[0])} -> {_dt(concat[-1])}")
    print(f"time_count: {concat.size}")
    print(f"time_step_seconds: {dict(zip(unique_diffs.tolist(), counts.tolist()))}")
    print("nan_counts:")
    for var in (temp_name, humidity_name):
        print(f"  {var}: nan={total_nan[var]} total={total_count[var]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/preprocessing/paris_1h_to_10min_6h_causal_start_v1.yaml",
    )
    args = parser.parse_args()
    inspect_era5(args.config)


if __name__ == "__main__":
    main()
