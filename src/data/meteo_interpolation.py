"""Spatial interpolation utilities for gridded meteorological data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterpolationWeights:
    """Interpolation indices and weights for one station location."""

    lat_indices: tuple[int, int]
    lon_indices: tuple[int, int]
    weights: tuple[float, float, float, float]
    method_used: str


def interpolation_weights(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    station_lat: float,
    station_lon: float,
    method: str = "bilinear",
    out_of_bounds: str = "nearest",
) -> InterpolationWeights:
    """Return bilinear or nearest-neighbor interpolation weights.

    ``latitudes`` may be ascending or descending. ``longitudes`` are expected
    in the same convention as station_lon.
    """

    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    if method not in {"bilinear", "nearest"}:
        raise ValueError("method must be bilinear or nearest")
    if out_of_bounds not in {"nearest", "error"}:
        raise ValueError("out_of_bounds must be nearest or error")

    inside = _inside_range(latitudes, station_lat) and _inside_range(longitudes, station_lon)
    if method == "nearest" or not inside:
        if not inside and out_of_bounds == "error":
            raise ValueError(
                f"Station lat/lon ({station_lat}, {station_lon}) is outside ERA5 grid "
                f"lat=({latitudes.min()}, {latitudes.max()}), lon=({longitudes.min()}, {longitudes.max()})"
            )
        lat_idx = int(np.argmin(np.abs(latitudes - station_lat)))
        lon_idx = int(np.argmin(np.abs(longitudes - station_lon)))
        return InterpolationWeights(
            lat_indices=(lat_idx, lat_idx),
            lon_indices=(lon_idx, lon_idx),
            weights=(1.0, 0.0, 0.0, 0.0),
            method_used="nearest",
        )

    lat0, lat1, lat_frac = _bracket(latitudes, station_lat)
    lon0, lon1, lon_frac = _bracket(longitudes, station_lon)
    w00 = (1.0 - lat_frac) * (1.0 - lon_frac)
    w01 = (1.0 - lat_frac) * lon_frac
    w10 = lat_frac * (1.0 - lon_frac)
    w11 = lat_frac * lon_frac
    return InterpolationWeights(
        lat_indices=(lat0, lat1),
        lon_indices=(lon0, lon1),
        weights=(float(w00), float(w01), float(w10), float(w11)),
        method_used="bilinear",
    )


def interpolate_field(field: np.ndarray, weights: InterpolationWeights) -> np.ndarray:
    """Interpolate a field with trailing dimensions [..., latitude, longitude].

    For ERA5 pressure variables, ``field`` is [time, pressure, lat, lon].
    The returned array is [time, pressure].
    """

    lat0, lat1 = weights.lat_indices
    lon0, lon1 = weights.lon_indices
    w00, w01, w10, w11 = weights.weights
    return (
        field[..., lat0, lon0] * w00
        + field[..., lat0, lon1] * w01
        + field[..., lat1, lon0] * w10
        + field[..., lat1, lon1] * w11
    )


def _inside_range(values: np.ndarray, point: float) -> bool:
    return float(values.min()) <= float(point) <= float(values.max())


def _bracket(values: np.ndarray, point: float) -> tuple[int, int, float]:
    """Return bracketing indices and fractional position in array order."""

    values = np.asarray(values, dtype=float)
    ascending = values[0] <= values[-1]
    ordered = values if ascending else values[::-1]
    right = int(np.searchsorted(ordered, point, side="right"))
    right = min(max(right, 1), len(ordered) - 1)
    left = right - 1
    lower = ordered[left]
    upper = ordered[right]
    frac = 0.0 if upper == lower else (float(point) - lower) / (upper - lower)
    if ascending:
        return left, right, float(frac)
    original_left = len(values) - 1 - left
    original_right = len(values) - 1 - right
    return original_left, original_right, float(frac)
