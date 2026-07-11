"""Standardize Dufeng Excel wind profiles into canonical 10min and 1h CSVs.

Input:
- data/raw/dufeng/*.xlsx
- data/processed/dufeng/dufeng_file_inventory.csv
- data/processed/dufeng/dufeng_location_audit.csv

Output:
- data/processed/dufeng/dufeng_10min_standard.csv
- data/processed/dufeng/dufeng_1h_standard.csv
- data/processed/dufeng/dufeng_standardization_report.json

The output schema is intentionally shared with the rest of the project:
station_id, time_start, height, u, v, u_mask, v_mask, latitude, longitude,
source_file, source_frequency.

Timestamp convention: interval start.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MISSING_VALUE = -999.0


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to standardize Dufeng Excel files.") from exc
    return pd


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to standardize Dufeng Excel files.") from exc
    return np


def _parse_coord(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return None if math.isnan(value) else value

    text = str(value).strip()
    if not text:
        return None
    sign = -1.0 if any(mark in text.upper() for mark in ("S", "W")) else 1.0
    for token in ("°", "掳", "N", "S", "E", "W", "n", "s", "e", "w"):
        text = text.replace(token, "")
    text = text.strip()
    try:
        return sign * float(text)
    except ValueError:
        return None


def _load_location_map(location_audit_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    pd = _require_pandas()
    audit = pd.read_csv(location_audit_path, encoding="utf-8-sig")
    required = {"source_file", "location_cluster_id", "latitude_median", "longitude_median"}
    missing = sorted(required - set(audit.columns))
    if missing:
        raise KeyError(f"Location audit is missing columns: {missing}")

    valid = audit.dropna(subset=["location_cluster_id", "latitude_median", "longitude_median"]).copy()
    clusters = (
        valid.groupby("location_cluster_id", sort=False)
        .agg(latitude=("latitude_median", "median"), longitude=("longitude_median", "median"), files=("source_file", "count"))
        .reset_index()
        .sort_values(["latitude", "longitude"], ascending=[False, False])
        .reset_index(drop=True)
    )
    station_names = ["dufeng_site_a", "dufeng_site_b"]
    cluster_to_station: dict[str, str] = {}
    station_coords: dict[str, dict[str, float]] = {}
    for idx, row in clusters.iterrows():
        station_id = station_names[idx] if idx < len(station_names) else f"dufeng_site_{idx + 1}"
        cluster_to_station[str(row["location_cluster_id"])] = station_id
        station_coords[station_id] = {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "num_source_files": int(row["files"]),
        }

    file_map: dict[str, dict[str, Any]] = {}
    for _, row in audit.iterrows():
        cluster_id = str(row.get("location_cluster_id", ""))
        station_id = cluster_to_station.get(cluster_id)
        if station_id is None:
            continue
        file_map[str(row["source_file"])] = {
            "station_id": station_id,
            "latitude": station_coords[station_id]["latitude"],
            "longitude": station_coords[station_id]["longitude"],
            "location_cluster_id": cluster_id,
        }
    return file_map, station_coords


def _extract_heights(columns) -> list[int]:
    heights = []
    names = set(str(c) for c in columns)
    for name in names:
        if not (name.startswith("WS_") and name.endswith("_MEAN")):
            continue
        middle = name[len("WS_") : -len("_MEAN")]
        try:
            height = int(middle)
        except ValueError:
            continue
        if f"WD_{height}_MEAN" in names:
            heights.append(height)
    return sorted(set(heights))


def _wind_speed_direction_to_uv(ws, wd):
    np = _require_numpy()
    theta = np.deg2rad(wd)
    u = -ws * np.sin(theta)
    v = -ws * np.cos(theta)
    return u, v


def _read_excel(path: Path):
    pd = _require_pandas()
    return pd.read_excel(path, sheet_name=0, engine="openpyxl")


def _wide_excel_to_long(path: Path, frequency_label: str, file_meta: dict[str, Any]):
    pd = _require_pandas()
    np = _require_numpy()
    raw = _read_excel(path)
    if "time" not in raw.columns:
        raise KeyError(f"{path.name} has no 'time' column")

    times = pd.to_datetime(raw["time"], errors="coerce")
    heights = _extract_heights(raw.columns)
    if not heights:
        raise ValueError(f"{path.name} has no matched WS_*_MEAN / WD_*_MEAN height columns")

    rows = []
    for height in heights:
        ws = pd.to_numeric(raw[f"WS_{height}_MEAN"], errors="coerce")
        wd = pd.to_numeric(raw[f"WD_{height}_MEAN"], errors="coerce")
        valid = times.notna() & ws.notna() & wd.notna() & (ws >= 0.0) & (wd >= 0.0) & (wd <= 360.0)
        u, v = _wind_speed_direction_to_uv(ws.astype(float), wd.astype(float))

        frame = pd.DataFrame(
            {
                "station_id": file_meta["station_id"],
                "time_start": times,
                "height": float(height),
                "u": np.where(valid, u, MISSING_VALUE),
                "v": np.where(valid, v, MISSING_VALUE),
                "u_mask": valid.to_numpy(dtype=bool),
                "v_mask": valid.to_numpy(dtype=bool),
                "latitude": file_meta["latitude"],
                "longitude": file_meta["longitude"],
                "source_file": path.name,
                "source_frequency": frequency_label,
            }
        )
        rows.append(frame)

    out = pd.concat(rows, ignore_index=True)
    out = out[out["time_start"].notna()].copy()
    return out.sort_values(["station_id", "time_start", "height"]).reset_index(drop=True)


def _aggregate_component(group, value_col: str, mask_col: str, expected_count: int, min_valid_fraction: float):
    pd = _require_pandas()
    valid_values = pd.to_numeric(group[value_col], errors="coerce").where(group[mask_col].astype(bool))
    valid_count = int(valid_values.notna().sum())
    valid_fraction = min(valid_count / float(expected_count), 1.0)
    is_valid = valid_fraction >= min_valid_fraction
    value = float(valid_values.mean()) if is_valid and valid_count > 0 else MISSING_VALUE
    return value, bool(is_valid), valid_count, valid_fraction


def _aggregate_to_rule(long_df, rule: str, expected_count: int, min_valid_fraction: float, source_frequency: str):
    pd = _require_pandas()
    if long_df.empty:
        return long_df.copy()

    work = long_df.copy()
    work["time_start"] = pd.to_datetime(work["time_start"], errors="coerce")
    work["time_bin"] = work["time_start"].dt.floor(rule)

    rows = []
    group_cols = ["station_id", "time_bin", "height"]
    for (station_id, time_bin, height), group in work.groupby(group_cols, sort=True):
        u, u_valid, u_count, u_fraction = _aggregate_component(group, "u", "u_mask", expected_count, min_valid_fraction)
        v, v_valid, v_count, v_fraction = _aggregate_component(group, "v", "v_mask", expected_count, min_valid_fraction)
        rows.append(
            {
                "station_id": station_id,
                "time_start": time_bin,
                "height": height,
                "u": u,
                "v": v,
                "u_mask": u_valid,
                "v_mask": v_valid,
                "latitude": float(group["latitude"].median()),
                "longitude": float(group["longitude"].median()),
                "source_file": ";".join(sorted(set(str(v) for v in group["source_file"].dropna().unique()))),
                "source_frequency": source_frequency,
                "u_valid_count": u_count,
                "v_valid_count": v_count,
                "u_valid_fraction": u_fraction,
                "v_valid_fraction": v_fraction,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["station_id", "time_start", "height"]).reset_index(drop=True)


def _deduplicate_10min(tenmin):
    pd = _require_pandas()
    if tenmin.empty:
        return tenmin
    work = tenmin.copy()
    work["source_priority"] = work["source_frequency"].map({"10min": 0, "1min_to_10min": 1}).fillna(2)
    work = work.sort_values(["station_id", "time_start", "height", "source_priority"])

    rows = []
    for key, group in work.groupby(["station_id", "time_start", "height"], sort=True):
        best = group.iloc[0].copy()
        if len(group) > 1:
            valid = group[(group["u_mask"].astype(bool)) & (group["v_mask"].astype(bool))]
            if not valid.empty:
                best = valid.sort_values("source_priority").iloc[0].copy()
            best["source_file"] = ";".join(sorted(set(str(v) for v in group["source_file"].dropna().unique())))
            best["source_frequency"] = "deduplicated_10min"
        rows.append(best.drop(labels=["source_priority"], errors="ignore"))
    return pd.DataFrame(rows).sort_values(["station_id", "time_start", "height"]).reset_index(drop=True)


def _standard_columns(frame):
    columns = [
        "station_id",
        "time_start",
        "height",
        "u",
        "v",
        "u_mask",
        "v_mask",
        "latitude",
        "longitude",
        "source_file",
        "source_frequency",
    ]
    return frame[columns].copy()


def _make_report(tenmin, hourly, station_coords: dict[str, dict[str, float]], args) -> dict[str, Any]:
    def summarize(frame):
        if frame.empty:
            return {"rows": 0}
        out = {"rows": int(len(frame))}
        for station_id, group in frame.groupby("station_id", sort=True):
            valid_uv = group["u_mask"].astype(bool) & group["v_mask"].astype(bool)
            out[station_id] = {
                "rows": int(len(group)),
                "time_start": str(group["time_start"].min()),
                "time_end": str(group["time_start"].max()),
                "num_times": int(group["time_start"].nunique()),
                "num_heights": int(group["height"].nunique()),
                "valid_uv_ratio": float(valid_uv.mean()) if len(valid_uv) else None,
                "latitude": float(group["latitude"].median()),
                "longitude": float(group["longitude"].median()),
            }
        return out

    return {
        "timestamp_semantics": "interval_start",
        "wind_direction_convention": "meteorological_from_direction_degrees",
        "uv_formula": "u = -ws * sin(wd_rad); v = -ws * cos(wd_rad)",
        "source_raw_dir": args.raw_dir,
        "processed_dir": args.processed_dir,
        "min_valid_fraction_10min_from_1min": args.min_valid_fraction_10min,
        "min_valid_fraction_1h_from_10min": args.min_valid_fraction_1h,
        "station_coords": station_coords,
        "tenmin": summarize(tenmin),
        "hourly": summarize(hourly),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/dufeng")
    parser.add_argument("--processed-dir", default="data/processed/dufeng")
    parser.add_argument("--inventory", default="data/processed/dufeng/dufeng_file_inventory.csv")
    parser.add_argument("--location-audit", default="data/processed/dufeng/dufeng_location_audit.csv")
    parser.add_argument("--min-valid-fraction-10min", type=float, default=0.8)
    parser.add_argument("--min-valid-fraction-1h", type=float, default=0.8)
    parser.add_argument("--tenmin-output", default="dufeng_10min_standard.csv")
    parser.add_argument("--hourly-output", default="dufeng_1h_standard.csv")
    parser.add_argument("--report-output", default="dufeng_standardization_report.json")
    args = parser.parse_args()

    pd = _require_pandas()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_csv(args.inventory, encoding="utf-8-sig")
    file_meta, station_coords = _load_location_map(Path(args.location_audit))

    direct_10min = []
    from_1min = []
    errors = []
    for _, row in inventory.iterrows():
        source_file = str(row["source_file"])
        path = raw_dir / source_file
        if not path.exists():
            errors.append({"source_file": source_file, "error": "missing_file"})
            continue
        if source_file not in file_meta:
            errors.append({"source_file": source_file, "error": "missing_location_metadata"})
            continue

        frequency = str(row["frequency_label"])
        print(f"standardize: {source_file} ({frequency})")
        try:
            long = _wide_excel_to_long(path, frequency, file_meta[source_file])
            if frequency == "1min":
                aggregated = _aggregate_to_rule(
                    long,
                    rule="10min",
                    expected_count=10,
                    min_valid_fraction=args.min_valid_fraction_10min,
                    source_frequency="1min_to_10min",
                )
                from_1min.append(_standard_columns(aggregated))
            elif frequency == "10min":
                direct_10min.append(_standard_columns(long))
            else:
                errors.append({"source_file": source_file, "error": f"unsupported_frequency:{frequency}"})
        except Exception as exc:
            errors.append({"source_file": source_file, "error": f"{type(exc).__name__}: {exc}"})

    pieces = direct_10min + from_1min
    if not pieces:
        raise RuntimeError("No Dufeng records could be standardized.")

    tenmin = pd.concat(pieces, ignore_index=True)
    tenmin["time_start"] = pd.to_datetime(tenmin["time_start"], errors="coerce")
    tenmin = _deduplicate_10min(tenmin)
    tenmin = _standard_columns(tenmin)
    tenmin["u"] = tenmin["u"].where(tenmin["u_mask"].astype(bool), MISSING_VALUE)
    tenmin["v"] = tenmin["v"].where(tenmin["v_mask"].astype(bool), MISSING_VALUE)

    hourly_full = _aggregate_to_rule(
        tenmin,
        rule="1h",
        expected_count=6,
        min_valid_fraction=args.min_valid_fraction_1h,
        source_frequency="10min_to_1h",
    )
    hourly = _standard_columns(hourly_full)
    hourly["u"] = hourly["u"].where(hourly["u_mask"].astype(bool), MISSING_VALUE)
    hourly["v"] = hourly["v"].where(hourly["v_mask"].astype(bool), MISSING_VALUE)

    tenmin_path = processed_dir / args.tenmin_output
    hourly_path = processed_dir / args.hourly_output
    report_path = processed_dir / args.report_output
    tenmin.to_csv(tenmin_path, index=False, encoding="utf-8-sig")
    hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")

    report = _make_report(tenmin, hourly, station_coords, args)
    report["errors"] = errors
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"wrote: {tenmin_path}")
    print(f"wrote: {hourly_path}")
    print(f"wrote: {report_path}")
    print(f"tenmin_rows: {len(tenmin)}")
    print(f"hourly_rows: {len(hourly)}")
    print(f"errors: {len(errors)}")


if __name__ == "__main__":
    main()
