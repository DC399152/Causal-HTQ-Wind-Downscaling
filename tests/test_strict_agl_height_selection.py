from dataclasses import replace

import numpy as np
import pandas as pd

from src.data.dataset_builder import _load_standard_csv_source, _station_enabled_for_source
from src.data.preprocessing import (
    HeightSelectionConfig,
    parse_preprocessing_config,
    select_height_indices,
    validate_config,
)


STRICT_CONFIG = "configs/preprocessing/paris_dufeng_1h_to_10min_12h_strict_agl.yaml"


def test_strict_agl_config_explicitly_declares_final_station_set():
    config = parse_preprocessing_config(STRICT_CONFIG)

    assert config.expected_station_ids == (
        "PAARBO",
        "PACHEM",
        "PALUPD",
        "PASIRT",
        "dufeng_site_a",
    )
    assert _station_enabled_for_source(config, "paris_nc", "PAARBO")
    assert not _station_enabled_for_source(config, "paris_nc", "PAJUSS")
    assert not _station_enabled_for_source(config, "paris_nc", "PAROIS")


def test_paris_strict_agl_uses_street_level_reference_and_preserves_legacy_mode():
    raw_asl = np.arange(0.0, 501.0, 25.0)
    selected = (175.0, 200.0, 225.0, 250.0, 275.0, 300.0)
    strict = HeightSelectionConfig(selected, "ground_agl_from_asl", 12.5)

    result = select_height_indices(
        raw_asl,
        station_altitude=98.0,
        height_config=strict,
        station_height=52.0,
    )

    assert result["target_heights_asl"].tolist() == [221.0, 246.0, 271.0, 296.0, 321.0, 346.0]
    assert result["actual_heights_agl"].tolist() == [179.0, 204.0, 229.0, 254.0, 279.0, 304.0]

    legacy = HeightSelectionConfig((250.0,), "agl_rounded_station_altitude", 12.5)
    legacy_result = select_height_indices(
        raw_asl,
        station_altitude=98.0,
        height_config=legacy,
        station_height=52.0,
    )
    assert legacy_result["target_heights_asl"].tolist() == [348.0]
    assert legacy_result["actual_heights_agl"].tolist() == [252.0]


def test_dufeng_strict_agl_adds_instrument_height_but_reads_raw_csv_layers(tmp_path):
    rows = []
    for height in range(10, 301, 10):
        rows.append(
            {
                "station_id": "dufeng_site_a",
                "time_start": "2026-04-18 16:00:00",
                "height": float(height),
                "u": float(height),
                "v": -float(height),
                "u_mask": True,
                "v_mask": True,
                "latitude": 39.9591,
                "longitude": 116.352,
                "source_file": "dummy.xlsx",
                "source_frequency": "test",
            }
        )
    hourly_path = tmp_path / "hourly.csv"
    target_path = tmp_path / "target.csv"
    pd.DataFrame(rows).to_csv(hourly_path, index=False)
    pd.DataFrame(rows).to_csv(target_path, index=False)

    config = parse_preprocessing_config(STRICT_CONFIG)
    validate_config(config)
    config = replace(
        config,
        meteo=replace(config.meteo, enabled=False),
        static_features=replace(config.static_features, use_lcz=False),
    )
    series_by_station = {}
    _load_standard_csv_source(
        "dufeng_standard_csv",
        {
            "hourly_csv": str(hourly_path),
            "target_csv": str(target_path),
            "source_label": "dufeng",
            "include_station_ids": ["dufeng_site_a"],
        },
        config,
        series_by_station,
    )

    series = series_by_station["dufeng_site_a"]
    assert series.height_values.tolist() == [170.0, 200.0, 220.0, 250.0, 270.0, 300.0]
    hourly_values = next(iter(series.hourly.values()))
    assert hourly_values[:, 0].tolist() == [160.0, 190.0, 210.0, 240.0, 260.0, 290.0]
