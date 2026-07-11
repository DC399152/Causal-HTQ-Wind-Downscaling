"""Inspect raw Dufeng station Excel files and write quality audit CSVs.

This script is read-only for raw data. It scans daily Excel workbooks under
data/raw/dufeng and writes two audit files under data/processed/dufeng:

- dufeng_file_inventory.csv: one row per source file
- dufeng_location_audit.csv: per-file location summaries and distance checks

The output is meant to decide which files can safely enter the standardization
step. It does not generate model-ready samples.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to inspect Dufeng Excel files.") from exc
    return pd


def _parse_coord(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    sign = -1.0 if any(mark in text.upper() for mark in ("S", "W")) else 1.0
    for token in ("°", "N", "S", "E", "W", "n", "s", "e", "w"):
        text = text.replace(token, "")
    text = text.strip()
    try:
        return sign * float(text)
    except ValueError:
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def _infer_frequency_minutes(times) -> float | None:
    pd = _require_pandas()
    clean = pd.Series(times).dropna().sort_values()
    if len(clean) < 2:
        return None
    diffs = clean.diff().dropna().dt.total_seconds() / 60.0
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    return float(diffs.median())


def _frequency_label(freq_minutes: float | None, filename: str) -> str:
    lower = filename.lower()
    if "1min" in lower or "1 min" in lower:
        return "1min"
    if "10min" in lower or "10 min" in lower:
        return "10min"
    if freq_minutes is None:
        return "unknown"
    if abs(freq_minutes - 1.0) <= 0.1:
        return "1min"
    if abs(freq_minutes - 10.0) <= 0.5:
        return "10min"
    return f"{freq_minutes:g}min"


def _expected_rows_for_frequency(label: str) -> int | None:
    if label == "1min":
        return 1440
    if label == "10min":
        return 144
    return None


def _read_excel(path: Path):
    pd = _require_pandas()
    return pd.read_excel(path, sheet_name=0, engine="openpyxl")


def _summarize_file(path: Path) -> dict[str, Any]:
    pd = _require_pandas()
    df = _read_excel(path)

    time_col = "time" if "time" in df.columns else None
    lat_col = "latitude" if "latitude" in df.columns else None
    lon_col = "longitude" if "longitude" in df.columns else None
    station_col = "ID_system" if "ID_system" in df.columns else None

    if time_col is not None:
        times = pd.to_datetime(df[time_col], errors="coerce")
    else:
        times = pd.Series([pd.NaT] * len(df))

    freq_minutes = _infer_frequency_minutes(times)
    freq_label = _frequency_label(freq_minutes, path.name)
    expected_full_day_rows = _expected_rows_for_frequency(freq_label)
    n_rows = int(len(df))

    unique_times = int(times.dropna().nunique())
    duplicated_times = max(0, int(times.notna().sum()) - unique_times)
    if freq_minutes and times.notna().any():
        expected_range_rows = int(round(((times.max() - times.min()).total_seconds() / 60.0) / freq_minutes)) + 1
        missing_steps_in_range = max(0, expected_range_rows - unique_times)
    else:
        expected_range_rows = None
        missing_steps_in_range = None

    lat = df[lat_col].map(_parse_coord) if lat_col is not None else pd.Series(dtype=float)
    lon = df[lon_col].map(_parse_coord) if lon_col is not None else pd.Series(dtype=float)
    station_ids = sorted(str(v) for v in df[station_col].dropna().unique()) if station_col is not None else []

    has_ws_mean = any(str(c).startswith("WS_") and str(c).endswith("_MEAN") for c in df.columns)
    has_wd_mean = any(str(c).startswith("WD_") and str(c).endswith("_MEAN") for c in df.columns)

    complete_day = None
    if expected_full_day_rows is not None:
        complete_day = n_rows >= expected_full_day_rows

    warnings: list[str] = []
    if "缺" in path.name:
        warnings.append("filename_marks_missing")
    if expected_full_day_rows is not None and n_rows < expected_full_day_rows:
        warnings.append("fewer_than_expected_full_day_rows")
    if duplicated_times > 0:
        warnings.append("duplicated_timestamps")
    if missing_steps_in_range and missing_steps_in_range > 0:
        warnings.append("missing_steps_in_time_range")
    if not has_ws_mean or not has_wd_mean:
        warnings.append("missing_ws_or_wd_mean_columns")
    if times.isna().any():
        warnings.append("unparsed_timestamps")
    if lat.dropna().empty or lon.dropna().empty:
        warnings.append("missing_location")

    return {
        "source_file": path.name,
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "n_rows": n_rows,
        "n_columns": int(len(df.columns)),
        "station_ids": ";".join(station_ids),
        "time_start": "" if times.dropna().empty else str(times.min()),
        "time_end": "" if times.dropna().empty else str(times.max()),
        "unique_times": unique_times,
        "duplicated_times": duplicated_times,
        "freq_minutes_median": freq_minutes,
        "frequency_label": freq_label,
        "expected_full_day_rows": expected_full_day_rows,
        "expected_rows_in_time_range": expected_range_rows,
        "missing_steps_in_time_range": missing_steps_in_range,
        "complete_day_by_row_count": complete_day,
        "latitude_median": None if lat.dropna().empty else float(lat.median()),
        "longitude_median": None if lon.dropna().empty else float(lon.median()),
        "latitude_min": None if lat.dropna().empty else float(lat.min()),
        "latitude_max": None if lat.dropna().empty else float(lat.max()),
        "longitude_min": None if lon.dropna().empty else float(lon.min()),
        "longitude_max": None if lon.dropna().empty else float(lon.max()),
        "has_ws_mean_columns": has_ws_mean,
        "has_wd_mean_columns": has_wd_mean,
        "warning_flags": ";".join(warnings),
    }


def _make_location_audit(inventory):
    pd = _require_pandas()
    loc = inventory.dropna(subset=["latitude_median", "longitude_median"]).copy()
    if loc.empty:
        return inventory.assign(distance_from_global_median_km=None, location_cluster_id=None)

    ref_lat = float(loc["latitude_median"].median())
    ref_lon = float(loc["longitude_median"].median())
    distances = []
    clusters = []
    for _, row in inventory.iterrows():
        lat = row.get("latitude_median")
        lon = row.get("longitude_median")
        if pd.isna(lat) or pd.isna(lon):
            distances.append(None)
            clusters.append("missing_location")
            continue
        distance = _haversine_km(ref_lat, ref_lon, float(lat), float(lon))
        distances.append(distance)
        clusters.append(f"lat{round(float(lat), 3):.3f}_lon{round(float(lon), 3):.3f}")

    out = inventory[
        [
            "source_file",
            "station_ids",
            "time_start",
            "time_end",
            "frequency_label",
            "n_rows",
            "latitude_median",
            "longitude_median",
            "latitude_min",
            "latitude_max",
            "longitude_min",
            "longitude_max",
            "warning_flags",
        ]
    ].copy()
    out["reference_latitude_median"] = ref_lat
    out["reference_longitude_median"] = ref_lon
    out["distance_from_global_median_km"] = distances
    out["location_cluster_id"] = clusters
    out["possible_location_shift"] = [
        bool(d is not None and d > 5.0) for d in distances
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/dufeng")
    parser.add_argument("--processed-dir", default="data/processed/dufeng")
    parser.add_argument("--glob", default="*.xlsx")
    parser.add_argument("--inventory-name", default="dufeng_file_inventory.csv")
    parser.add_argument("--location-name", default="dufeng_location_audit.csv")
    args = parser.parse_args()

    pd = _require_pandas()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No Excel files found under {raw_dir} with glob {args.glob!r}")

    rows = []
    for path in files:
        print(f"inspect: {path}")
        try:
            rows.append(_summarize_file(path))
        except Exception as exc:  # Keep one bad workbook from hiding the rest.
            rows.append(
                {
                    "source_file": path.name,
                    "path": str(path),
                    "size_bytes": int(path.stat().st_size),
                    "n_rows": None,
                    "n_columns": None,
                    "warning_flags": f"read_failed:{type(exc).__name__}:{exc}",
                }
            )

    inventory = pd.DataFrame(rows)
    location_audit = _make_location_audit(inventory)

    inventory_path = processed_dir / args.inventory_name
    location_path = processed_dir / args.location_name
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    location_audit.to_csv(location_path, index=False, encoding="utf-8-sig")

    print(f"wrote: {inventory_path}")
    print(f"wrote: {location_path}")
    print(f"files: {len(inventory)}")
    if "warning_flags" in inventory:
        warned = inventory["warning_flags"].fillna("").astype(str).ne("").sum()
        print(f"files_with_warnings: {int(warned)}")
    if "possible_location_shift" in location_audit:
        shifted = location_audit["possible_location_shift"].fillna(False).sum()
        print(f"possible_location_shift_files: {int(shifted)}")


if __name__ == "__main__":
    main()
