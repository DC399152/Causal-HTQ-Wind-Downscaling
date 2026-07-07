"""Station-level static feature loading for dataset construction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StationStaticFeatures:
    """Static features indexed by station id.

    Values are station-level vectors, for example LCZ buffer fractions [17].
    """

    feature_columns: tuple[str, ...]
    values_by_station: dict[str, np.ndarray]
    dominant_lcz_by_station: dict[str, float]

    def features_for_station(self, station_id: str) -> np.ndarray:
        if station_id not in self.values_by_station:
            raise KeyError(f"Static features missing for station_id={station_id!r}")
        return self.values_by_station[station_id]

    def dominant_lcz_for_station(self, station_id: str) -> float:
        return self.dominant_lcz_by_station.get(station_id, np.nan)


def load_station_static_features(
    csv_path: str | Path,
    feature_columns: tuple[str, ...] | list[str],
    *,
    station_id_column: str = "station_id",
    dominant_lcz_column: str = "dominant_lcz_500m",
) -> StationStaticFeatures:
    """Load station-level static features from CSV.

    The returned feature vectors are float32 and keep the CSV column order.
    LCZ fractions are already 0-1 proportions and are not normalized here.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Static feature CSV not found: {path}")

    columns = tuple(str(v) for v in feature_columns)
    values_by_station: dict[str, np.ndarray] = {}
    dominant_by_station: dict[str, float] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Static feature CSV has no header: {path}")
        missing = [c for c in (station_id_column, *columns) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Static feature CSV is missing required columns: {missing}")

        for row in reader:
            station_id = str(row[station_id_column])
            values = np.asarray([float(row[col]) for col in columns], dtype=np.float32)
            values_by_station[station_id] = values
            if dominant_lcz_column in row and row[dominant_lcz_column] not in {"", None}:
                dominant_by_station[station_id] = float(row[dominant_lcz_column])

    if not values_by_station:
        raise ValueError(f"Static feature CSV contains no station rows: {path}")

    return StationStaticFeatures(
        feature_columns=columns,
        values_by_station=values_by_station,
        dominant_lcz_by_station=dominant_by_station,
    )
