from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from src.data.dataset_builder import (
    _load_pair_into_station_series,
    _validate_arrays_before_write,
)
from src.data.preprocessing import DataAlignmentConfig, HeightSelectionConfig, parse_preprocessing_config
from src.data.raw_reader import RawFilePair


def _config(selected=(25.0,), *, height_tol=1.0, pair_tol=2.0):
    cfg = parse_preprocessing_config("configs/preprocessing/paris_1h_to_10min_6h_causal_start_v1.yaml")
    height = HeightSelectionConfig(tuple(float(v) for v in selected), "agl", height_tol)
    return replace(
        cfg,
        height=height,
        height_by_source={"paris_nc": height},
        data_alignment=DataAlignmentConfig("strict", pair_tol, "error"),
        meteo=replace(cfg.meteo, enabled=False),
        static_features=replace(cfg.static_features, use_lcz=False),
    )


def _dataset(stations, heights, values_by_station, *, start="2024-01-01T00:00"):
    times = np.asarray([np.datetime64(start, "m")])
    heights = np.asarray(heights, dtype=np.float32)
    values = np.zeros((len(stations), len(times), len(heights)), dtype=np.float32)
    for s_idx, station in enumerate(stations):
        station_values = values_by_station.get(station, {})
        for h_idx, height in enumerate(heights):
            values[s_idx, 0, h_idx] = float(station_values.get(float(height), s_idx * 100 + h_idx))
    return xr.Dataset(
        {
            "u": (("station", "time", "altitude"), values),
            "v": (("station", "time", "altitude"), values + 0.5),
            "station_altitude": (("station",), np.zeros((len(stations),), dtype=np.float32)),
            "station_height": (("station",), np.zeros((len(stations),), dtype=np.float32)),
            "station_lat": (("station",), np.arange(len(stations), dtype=np.float32)),
            "station_lon": (("station",), np.arange(len(stations), dtype=np.float32)),
        },
        coords={
            "station": np.asarray(stations, dtype=object),
            "time": times,
            "altitude": heights,
        },
    )


def _load(monkeypatch, hourly_ds, target_ds, cfg):
    pair = RawFilePair(Path("hourly_3600s.nc"), Path("target_600s.nc"), "pair")

    def fake_open(path):
        return hourly_ds if str(path).startswith("hourly") else target_ds

    monkeypatch.setattr("src.data.dataset_builder.open_dataset", fake_open)
    series = {}
    _load_pair_into_station_series(pair, cfg, series)
    return series


def test_station_order_mismatch_is_reordered_by_station_id(monkeypatch):
    cfg = _config(selected=(25.0,))
    hourly = _dataset(["A", "B", "C"], [25.0], {"A": {25.0: 1}, "B": {25.0: 2}, "C": {25.0: 3}})
    target = _dataset(["B", "A", "C"], [25.0], {"A": {25.0: 10}, "B": {25.0: 20}, "C": {25.0: 30}})

    series = _load(monkeypatch, hourly, target, cfg)

    timestamp = np.datetime64("2024-01-01T00:00", "m")
    assert float(series["A"].hourly[timestamp][0, 0]) == 1.0
    assert float(series["A"].target[timestamp][0, 0]) == 10.0
    assert float(series["B"].target[timestamp][0, 0]) == 20.0


def test_station_set_mismatch_raises_in_strict_mode(monkeypatch):
    cfg = _config(selected=(25.0,))
    hourly = _dataset(["A", "B", "C"], [25.0], {})
    target = _dataset(["A", "C", "D"], [25.0], {})

    with pytest.raises(ValueError, match="only_hourly=.*B.*only_target=.*D"):
        _load(monkeypatch, hourly, target, cfg)


def test_duplicate_station_id_raises(monkeypatch):
    cfg = _config(selected=(25.0,))
    hourly = _dataset(["A", "A"], [25.0], {})
    target = _dataset(["A", "B"], [25.0], {})

    with pytest.raises(ValueError, match="duplicate station_id"):
        _load(monkeypatch, hourly, target, cfg)


def test_height_order_mismatch_uses_physical_height_not_same_index(monkeypatch):
    cfg = _config(selected=(25.0, 75.0, 100.0), height_tol=1.0)
    hourly = _dataset(["A"], [25.0, 75.0, 100.0, 150.0], {})
    target = _dataset(
        ["A"],
        [100.0, 25.0, 150.0, 75.0],
        {"A": {25.0: 25, 75.0: 75, 100.0: 100, 150.0: 150}},
    )

    series = _load(monkeypatch, hourly, target, cfg)

    timestamp = np.datetime64("2024-01-01T00:00", "m")
    assert series["A"].target[timestamp][:, 0].tolist() == [25.0, 75.0, 100.0]


def test_slight_hourly_target_height_difference_is_recorded(monkeypatch):
    cfg = _config(selected=(25.0, 75.0, 100.0), height_tol=1.0, pair_tol=1.0)
    hourly = _dataset(["A"], [25.0, 75.0, 100.0], {})
    target = _dataset(["A"], [24.8, 75.2, 99.5], {})

    series = _load(monkeypatch, hourly, target, cfg)

    assert np.allclose(series["A"].hourly_height_values, [25.0, 75.0, 100.0])
    assert np.allclose(series["A"].target_height_values, [24.8, 75.2, 99.5])
    assert np.allclose(series["A"].height_values, [24.9, 75.1, 99.75])


def test_hourly_target_height_difference_over_limit_raises(monkeypatch):
    cfg = _config(selected=(75.0,), height_tol=30.0, pair_tol=2.0)
    hourly = _dataset(["A"], [75.0], {})
    target = _dataset(["A"], [100.0], {})

    with pytest.raises(ValueError, match="Hourly/target height mismatch exceeds tolerance"):
        _load(monkeypatch, hourly, target, cfg)


def test_cross_file_height_schema_change_raises(monkeypatch):
    cfg = _config(selected=(75.0,), height_tol=30.0, pair_tol=2.0)
    first_hourly = _dataset(["A"], [75.0], {}, start="2024-01-01T00:00")
    first_target = _dataset(["A"], [75.0], {}, start="2024-01-01T00:00")
    second_hourly = _dataset(["A"], [100.0], {}, start="2024-01-01T01:00")
    second_target = _dataset(["A"], [100.0], {}, start="2024-01-01T01:00")

    pair1 = RawFilePair(Path("hourly_one_3600s.nc"), Path("target_one_600s.nc"), "one")
    pair2 = RawFilePair(Path("hourly_two_3600s.nc"), Path("target_two_600s.nc"), "two")
    sources = {
        pair1.hourly_path: first_hourly,
        pair1.target_path: first_target,
        pair2.hourly_path: second_hourly,
        pair2.target_path: second_target,
    }

    monkeypatch.setattr("src.data.dataset_builder.open_dataset", lambda path: sources[path])
    series = {}
    _load_pair_into_station_series(pair1, cfg, series)
    with pytest.raises(ValueError, match="Height schema changed across files"):
        _load_pair_into_station_series(pair2, cfg, series)


def test_global_validation_rejects_duplicate_station_target_time():
    cfg = _config(selected=(25.0,))
    arrays = {
        "x_hourly": np.zeros((2, 6, 1, 2), dtype=np.float32),
        "x_mask": np.ones((2, 6, 1, 2), dtype=bool),
        "y_10min": np.zeros((2, 6, 1, 2), dtype=np.float32),
        "y_mask": np.ones((2, 6, 1, 2), dtype=bool),
        "current_hourly": np.zeros((2, 1, 2), dtype=np.float32),
        "station_id": np.asarray(["A", "A"], dtype=object),
        "target_time_start": np.asarray(["2024-01-01T00:00", "2024-01-01T00:00"], dtype=object),
        "target_times_10min": np.asarray([["2024-01-01T00:00"] * 6] * 2, dtype=object),
        "context_times_hourly": np.asarray([["2023-12-31T19:00"] * 6] * 2, dtype=object),
        "height_values": np.ones((2, 1), dtype=np.float32) * 25,
        "hourly_height_values": np.ones((2, 1), dtype=np.float32) * 25,
        "target_height_values": np.ones((2, 1), dtype=np.float32) * 25,
        "source_file": np.asarray(["x", "x"], dtype=object),
        "hourly_source_files": np.asarray([["x"] * 6] * 2, dtype=object),
        "target_source_files": np.asarray([["y"] * 6] * 2, dtype=object),
        "source_group": np.asarray(["g", "g"], dtype=object),
        "split": np.asarray(["train", "train"], dtype=object),
    }

    with pytest.raises(ValueError, match="Duplicate station_id"):
        _validate_arrays_before_write(arrays, replace(cfg, context_hours=6),)
