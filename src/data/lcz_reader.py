"""LCZ raster extraction utilities.

LCZ is a categorical raster. This module only supports area/count based
extraction, never bilinear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np


@dataclass(frozen=True)
class LCZExtractionResult:
    """LCZ fraction result for one station and one buffer radius."""

    x_raster_crs: float
    y_raster_crs: float
    fractions: np.ndarray
    dominant_lcz: int
    valid_pixel_count: int
    fraction_sum: float


def validate_lcz_extraction_method(method: str) -> None:
    """Validate that LCZ extraction uses categorical-safe methods."""

    if method != "buffer_fraction":
        raise ValueError("LCZ categorical raster only supports extraction.method=buffer_fraction")


def fraction_vector_from_lcz_values(
    values: np.ndarray,
    class_values: list[int] | tuple[int, ...],
    nodata_values: list[int] | tuple[int, ...] = (),
) -> tuple[np.ndarray, int, int, float]:
    """Count LCZ classes and return fractions.

    Values may be a masked array. Fractions are ordered by ``class_values``.
    If no valid pixels exist, fractions are all zeros and dominant_lcz is 0.
    """

    if np.ma.isMaskedArray(values):
        flat = np.asarray(values.compressed()).reshape(-1)
    else:
        flat = np.asarray(values).reshape(-1)

    class_values_arr = np.asarray(class_values, dtype=np.int64)
    nodata = set(int(v) for v in nodata_values)
    valid = np.asarray(
        [v for v in flat if np.isfinite(v) and int(v) not in nodata and int(v) in set(class_values_arr)],
        dtype=np.int64,
    )
    valid_count = int(valid.size)
    fractions = np.zeros((len(class_values_arr),), dtype=np.float32)
    if valid_count == 0:
        return fractions, 0, 0, 0.0

    for i, class_value in enumerate(class_values_arr):
        fractions[i] = float(np.count_nonzero(valid == class_value)) / float(valid_count)
    dominant_lcz = int(class_values_arr[int(np.argmax(fractions))])
    fraction_sum = float(np.sum(fractions))
    return fractions, dominant_lcz, valid_count, fraction_sum


def transform_lonlat_to_raster_crs(
    longitude: float,
    latitude: float,
    *,
    station_crs: str = "EPSG:4326",
    raster_crs: str = "EPSG:3035",
) -> tuple[float, float]:
    """Transform station lon/lat into the raster CRS."""

    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError("pyproj is required for LCZ station coordinate transforms.") from exc

    transformer = Transformer.from_crs(station_crs, raster_crs, always_xy=True)
    x, y = transformer.transform(float(longitude), float(latitude))
    return float(x), float(y)


def extract_lcz_buffer_fraction(
    raster_path: str | Path,
    *,
    longitude: float,
    latitude: float,
    radius_m: float,
    class_values: list[int] | tuple[int, ...],
    nodata_values: list[int] | tuple[int, ...],
    station_crs: str = "EPSG:4326",
    raster_crs: str = "EPSG:3035",
) -> LCZExtractionResult:
    """Extract LCZ class fractions inside a station buffer.

    The buffer is built in the raster CRS, so ``radius_m`` is interpreted in
    meters for EPSG:3035. Categorical LCZ values are counted directly.
    """

    try:
        import rasterio
        from rasterio.mask import mask
        from shapely.geometry import Point, mapping
    except ImportError as exc:
        raise RuntimeError("rasterio and shapely are required for LCZ raster extraction.") from exc

    x, y = transform_lonlat_to_raster_crs(
        longitude,
        latitude,
        station_crs=station_crs,
        raster_crs=raster_crs,
    )
    point_buffer = Point(x, y).buffer(float(radius_m))

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"LCZ raster has no CRS: {raster_path}")
        if str(src.crs).upper() != raster_crs.upper():
            raise ValueError(f"LCZ raster CRS mismatch: got {src.crs}, expected {raster_crs}")
        try:
            clipped, _ = mask(src, [mapping(point_buffer)], crop=True, filled=False, indexes=1)
        except ValueError:
            warnings.warn(
                f"Station ({latitude}, {longitude}) buffer is outside LCZ raster extent; writing zero fractions.",
                RuntimeWarning,
                stacklevel=2,
            )
            zeros = np.zeros((len(class_values),), dtype=np.float32)
            return LCZExtractionResult(x, y, zeros, 0, 0, 0.0)

    fractions, dominant_lcz, valid_count, fraction_sum = fraction_vector_from_lcz_values(
        clipped,
        class_values,
        nodata_values,
    )
    return LCZExtractionResult(x, y, fractions, dominant_lcz, valid_count, fraction_sum)
