"""ERA5 pressure-level temperature/humidity reader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from src.data.alignment import time_key
from src.data.meteo_interpolation import interpolation_weights, interpolate_field
from src.data.preprocessing import MeteoConfig


@dataclass(frozen=True)
class StationLocation:
    """Station latitude/longitude metadata."""

    station_id: str
    latitude: float
    longitude: float


@dataclass
class Era5StationData:
    """ERA5 data interpolated to station locations."""

    pressure_levels: np.ndarray
    channel_names: tuple[str, ...]
    values_by_station: dict[str, dict[np.datetime64, np.ndarray]]
    mask_by_station: dict[str, dict[np.datetime64, np.ndarray]]
    interpolation_method_by_station: dict[str, str]

    def sample_context(
        self,
        station_id: str,
        context_times: list[np.datetime64],
        missing_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return x_meteo [L, P, C_m] and meteo_mask [L, P, C_m]."""

        p_count = int(self.pressure_levels.shape[0])
        c_count = len(self.channel_names)
        values = np.full((len(context_times), p_count, c_count), missing_value, dtype=np.float32)
        mask = np.zeros((len(context_times), p_count, c_count), dtype=bool)
        station_values = self.values_by_station.get(station_id, {})
        station_masks = self.mask_by_station.get(station_id, {})
        for idx, timestamp in enumerate(context_times):
            key = time_key(timestamp)
            if key in station_values:
                values[idx] = station_values[key]
                mask[idx] = station_masks[key]
        return values, mask


def load_era5_for_stations(
    config: MeteoConfig,
    station_locations: Mapping[str, StationLocation],
) -> Era5StationData:
    """Load ERA5 pressure variables and interpolate them to each station."""

    if config.pressure_dir is None:
        raise ValueError("meteo pressure_dir is required")
    files = sorted(Path(config.pressure_dir).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 pressure files found in {config.pressure_dir}")

    var_time = str(config.variables["time"])
    var_level = str(config.variables["pressure_level"])
    var_lat = str(config.variables["latitude"])
    var_lon = str(config.variables["longitude"])
    var_temp = str(config.variables["temperature"])
    var_humidity = str(config.variables["humidity"])
    channel_vars = (var_temp, var_humidity)

    values_by_station: dict[str, dict[np.datetime64, np.ndarray]] = {
        station_id: {} for station_id in station_locations
    }
    mask_by_station: dict[str, dict[np.datetime64, np.ndarray]] = {
        station_id: {} for station_id in station_locations
    }
    interpolation_method_by_station: dict[str, str] = {}
    pressure_levels: np.ndarray | None = None

    for path in files:
        with h5py.File(path, "r") as f:
            for name in (var_time, var_level, var_lat, var_lon, *channel_vars):
                if name not in f:
                    raise KeyError(f"{path} is missing ERA5 variable {name!r}")

            times = _decode_epoch_seconds(f[var_time][:])
            levels = np.asarray(f[var_level][:], dtype=np.float32)
            latitudes = np.asarray(f[var_lat][:], dtype=np.float64)
            longitudes = np.asarray(f[var_lon][:], dtype=np.float64)

            if pressure_levels is None:
                pressure_levels = levels
            elif not np.allclose(pressure_levels, levels):
                raise ValueError(f"Pressure levels changed in {path}")

            fields = [np.asarray(f[var][:], dtype=np.float32) for var in channel_vars]
            for field, var in zip(fields, channel_vars):
                if field.ndim != 4:
                    raise ValueError(f"Expected {var} shape [time, pressure, lat, lon], got {field.shape}")

            for station_id, loc in station_locations.items():
                weights = interpolation_weights(
                    latitudes,
                    longitudes,
                    loc.latitude,
                    loc.longitude,
                    method=config.interpolation,
                    out_of_bounds=config.out_of_bounds,
                )
                interpolation_method_by_station.setdefault(station_id, weights.method_used)
                interpolated = [interpolate_field(field, weights) for field in fields]
                stacked = np.stack(interpolated, axis=-1).astype(np.float32)
                valid = np.isfinite(stacked)
                for idx, timestamp in enumerate(times):
                    key = time_key(timestamp)
                    values_by_station[station_id][key] = stacked[idx]
                    mask_by_station[station_id][key] = valid[idx]

    if pressure_levels is None:
        raise RuntimeError("No ERA5 pressure levels were loaded")
    if config.expected_pressure_levels_hpa and not np.allclose(
        pressure_levels,
        np.asarray(config.expected_pressure_levels_hpa, dtype=np.float32),
    ):
        raise ValueError(
            f"ERA5 pressure levels {pressure_levels.tolist()} do not match "
            f"expected {list(config.expected_pressure_levels_hpa)}"
        )

    return Era5StationData(
        pressure_levels=pressure_levels.astype(np.float32),
        channel_names=config.channel_names,
        values_by_station=values_by_station,
        mask_by_station=mask_by_station,
        interpolation_method_by_station=interpolation_method_by_station,
    )


def _decode_epoch_seconds(values) -> np.ndarray:
    seconds = np.asarray(values, dtype=np.int64)
    return seconds.astype("datetime64[s]").astype("datetime64[m]")
