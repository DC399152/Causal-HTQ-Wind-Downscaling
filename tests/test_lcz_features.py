import numpy as np
import pytest

from src.data.lcz_reader import (
    fraction_vector_from_lcz_values,
    transform_lonlat_to_raster_crs,
    validate_lcz_extraction_method,
)


def test_lcz_fraction_vector_shape_sum_and_dominant_class():
    values = np.asarray(
        [
            [1, 1, 2, 2],
            [2, 3, 3, 3],
            [0, 255, 3, 3],
        ],
        dtype=np.uint8,
    )

    fractions, dominant_lcz, valid_count, fraction_sum = fraction_vector_from_lcz_values(
        values,
        tuple(range(1, 18)),
        nodata_values=(0, 255),
    )

    assert fractions.shape == (17,)
    assert valid_count == 10
    assert np.isclose(fraction_sum, 1.0)
    assert 1 <= dominant_lcz <= 17
    assert dominant_lcz == 3


def test_lcz_extraction_rejects_bilinear_interpolation():
    validate_lcz_extraction_method("buffer_fraction")
    with pytest.raises(ValueError, match="buffer_fraction"):
        validate_lcz_extraction_method("bilinear")


def test_epsg_4326_to_3035_transform_runs():
    pytest.importorskip("pyproj")

    x, y = transform_lonlat_to_raster_crs(
        longitude=2.3522,
        latitude=48.8566,
        station_crs="EPSG:4326",
        raster_crs="EPSG:3035",
    )

    assert np.isfinite(x)
    assert np.isfinite(y)
