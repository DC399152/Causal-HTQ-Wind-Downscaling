"""Extract station-level LCZ buffer fraction features from a categorical GeoTIFF."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import warnings

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.lcz_reader import extract_lcz_buffer_fraction, validate_lcz_extraction_method


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _read_stations(path: Path, id_column: str, lat_column: str, lon_column: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Station metadata CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Station metadata CSV has no header: {path}")
        missing = [c for c in (id_column, lat_column, lon_column) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Station metadata CSV missing columns: {missing}")
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/preprocessing/lcz_features.yaml",
        help="LCZ feature extraction YAML config.",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)["lcz"]
    extraction = cfg["extraction"]
    station_cfg = cfg["station_metadata"]
    output_cfg = cfg["output"]

    validate_lcz_extraction_method(str(extraction.get("method", "buffer_fraction")))
    radii_m = [float(v) for v in extraction.get("buffer_radii_m", [500])]
    class_values = [int(v) for v in cfg["class_values"]]
    nodata_values = [int(v) for v in extraction.get("nodata_values", [])]
    require_sum = bool(extraction.get("require_fraction_sum_close_to_one", True))

    raster_path = Path(cfg["raster_path"])
    if not raster_path.exists():
        raise FileNotFoundError(f"LCZ raster not found: {raster_path}")

    station_rows = _read_stations(
        Path(station_cfg["path"]),
        str(station_cfg.get("id_column", "station_id")),
        str(station_cfg.get("latitude_column", "latitude")),
        str(station_cfg.get("longitude_column", "longitude")),
    )

    output_path = Path(output_cfg["csv_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    radius_labels = [f"{int(radius)}m" for radius in radii_m]
    fieldnames = [
        "station_id",
        "latitude",
        "longitude",
        "x_3035",
        "y_3035",
    ]
    for label in radius_labels:
        fieldnames.extend(
            [
                f"dominant_lcz_{label}",
                f"valid_pixel_count_{label}",
                f"fraction_sum_{label}",
            ]
        )
        fieldnames.extend([f"LCZ_{class_value}_frac_{label}" for class_value in class_values])

    id_col = str(station_cfg.get("id_column", "station_id"))
    lat_col = str(station_cfg.get("latitude_column", "latitude"))
    lon_col = str(station_cfg.get("longitude_column", "longitude"))

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for station in station_rows:
            station_id = str(station[id_col])
            latitude = float(station[lat_col])
            longitude = float(station[lon_col])
            row: dict[str, object] = {
                "station_id": station_id,
                "latitude": latitude,
                "longitude": longitude,
            }

            first_xy_written = False
            for radius, label in zip(radii_m, radius_labels):
                result = extract_lcz_buffer_fraction(
                    raster_path,
                    longitude=longitude,
                    latitude=latitude,
                    radius_m=radius,
                    class_values=class_values,
                    nodata_values=nodata_values,
                    station_crs=str(cfg.get("station_crs", "EPSG:4326")),
                    raster_crs=str(cfg.get("raster_crs", "EPSG:3035")),
                )
                if not first_xy_written:
                    row["x_3035"] = result.x_raster_crs
                    row["y_3035"] = result.y_raster_crs
                    first_xy_written = True

                row[f"dominant_lcz_{label}"] = result.dominant_lcz
                row[f"valid_pixel_count_{label}"] = result.valid_pixel_count
                row[f"fraction_sum_{label}"] = result.fraction_sum
                for class_value, fraction in zip(class_values, result.fractions):
                    row[f"LCZ_{class_value}_frac_{label}"] = float(fraction)

                if result.valid_pixel_count == 0:
                    warnings.warn(
                        f"Station {station_id} has no valid LCZ pixels inside {label} buffer.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                elif require_sum and not np.isclose(result.fraction_sum, 1.0, atol=1e-4):
                    warnings.warn(
                        f"Station {station_id} {label} LCZ fractions sum to {result.fraction_sum:.6f}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            writer.writerow(row)

    print(f"wrote: {output_path}")
    print(f"stations: {len(station_rows)}")
    print(f"radii_m: {[int(v) for v in radii_m]}")


if __name__ == "__main__":
    main()
