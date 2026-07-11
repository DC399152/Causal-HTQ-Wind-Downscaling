"""Standardize new-station CSV wind profiles into canonical CSV files.

Output schema:
- station_id
- time_start
- height
- u
- v
- u_mask
- v_mask
- latitude
- longitude
- source_file

The first implementation is config-driven and conservative. It assumes each
row is one timestamp-height observation. If your raw CSV is wide by height,
inspect it first and add a reshape step before using this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read new-station CSV config files.") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to standardize new-station CSV files.") from exc
    return pd


def _valid_numeric(series, missing_values):
    pd = _require_pandas()
    values = pd.to_numeric(series.replace(missing_values, pd.NA), errors="coerce")
    return values, values.notna()


def _read_raw_files(cfg: dict[str, Any]):
    pd = _require_pandas()
    paths = cfg["paths"]
    input_cfg = cfg.get("input", {})
    csv_dir = Path(paths["raw_csv_dir"])
    files = sorted(csv_dir.glob(input_cfg.get("file_glob", "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {csv_dir}")
    frames = []
    for path in files:
        frame = pd.read_csv(
            path,
            encoding=input_cfg.get("encoding", "utf-8"),
            sep=input_cfg.get("delimiter", ","),
        )
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _standardize_observation_rows(raw, cfg: dict[str, Any]):
    pd = _require_pandas()
    station_cfg = cfg["station"]
    col_cfg = cfg["columns"]
    qc_cfg = cfg.get("quality_control", {})
    missing_values = qc_cfg.get("missing_values", [-999, -999.0])

    required = [col_cfg["time"], col_cfg["height"], col_cfg["u"], col_cfg["v"]]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise KeyError(f"Raw CSV is missing configured columns: {missing}")

    time_start = pd.to_datetime(raw[col_cfg["time"]], errors="coerce")
    if time_start.isna().any():
        raise ValueError(f"Failed to parse {int(time_start.isna().sum())} timestamps")

    height, height_mask = _valid_numeric(raw[col_cfg["height"]], missing_values)
    u, u_mask = _valid_numeric(raw[col_cfg["u"]], missing_values)
    v, v_mask = _valid_numeric(raw[col_cfg["v"]], missing_values)
    valid_height = height_mask

    out = pd.DataFrame(
        {
            "station_id": station_cfg["station_id"],
            "time_start": time_start,
            "height": height,
            "u": u,
            "v": v,
            "u_mask": u_mask & valid_height,
            "v_mask": v_mask & valid_height,
            "latitude": float(station_cfg["latitude"]),
            "longitude": float(station_cfg["longitude"]),
            "source_file": raw["source_file"],
        }
    )
    out = out[valid_height].copy()
    out["u"] = out["u"].where(out["u_mask"], -999.0)
    out["v"] = out["v"].where(out["v_mask"], -999.0)
    return out.sort_values(["time_start", "height"]).reset_index(drop=True)


def _aggregate(standard, rule: str, min_valid_fraction: float):
    pd = _require_pandas()
    rows = []
    for (station_id, height), group in standard.groupby(["station_id", "height"], sort=False):
        group = group.set_index("time_start").sort_index()
        for column in ("u", "v"):
            valid_col = f"{column}_mask"
            values = pd.to_numeric(group[column], errors="coerce").where(group[valid_col])
            mean = values.resample(rule, label="left", closed="left").mean()
            count = values.resample(rule, label="left", closed="left").count()
            total = group[valid_col].resample(rule, label="left", closed="left").count()
            valid = (count / total.clip(lower=1)) >= min_valid_fraction
            part = pd.DataFrame({column: mean.where(valid, -999.0), valid_col: valid})
            part["station_id"] = station_id
            part["height"] = height
            part["time_start"] = part.index
            rows.append(part.reset_index(drop=True))

    if not rows:
        return standard.iloc[0:0].copy()

    merged = None
    for frame in rows:
        key_cols = ["station_id", "time_start", "height"]
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=key_cols, how="outer")
    merged["latitude"] = float(standard["latitude"].iloc[0])
    merged["longitude"] = float(standard["longitude"].iloc[0])
    merged["source_file"] = "aggregated_from_new_station_csv"
    return merged.sort_values(["time_start", "height"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    station_id = cfg["station"]["station_id"]
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_raw_files(cfg)
    standard = _standardize_observation_rows(raw, cfg)

    aggregation = cfg.get("aggregation", {})
    qc = cfg.get("quality_control", {})
    outputs = cfg.get("output", {})

    if aggregation.get("make_10min", True):
        tenmin = _aggregate(
            standard,
            aggregation.get("tenmin_rule", "10min"),
            float(qc.get("min_valid_fraction_10min", 0.5)),
        )
        tenmin_path = processed_dir / outputs.get("tenmin_filename", "{station_id}_10min_standard.csv").format(station_id=station_id)
        tenmin.to_csv(tenmin_path, index=False)
        print(f"wrote: {tenmin_path}")
    else:
        tenmin = None

    if aggregation.get("make_1h", True):
        hourly = _aggregate(
            standard,
            aggregation.get("hourly_rule", "1h"),
            float(qc.get("min_valid_fraction_1h", 0.5)),
        )
        hourly_path = processed_dir / outputs.get("hourly_filename", "{station_id}_1h_standard.csv").format(station_id=station_id)
        hourly.to_csv(hourly_path, index=False)
        print(f"wrote: {hourly_path}")
    else:
        hourly = None

    report = {
        "station_id": station_id,
        "raw_rows": int(len(raw)),
        "standard_rows": int(len(standard)),
        "time_range": [
            str(standard["time_start"].min()) if len(standard) else None,
            str(standard["time_start"].max()) if len(standard) else None,
        ],
        "height_values": sorted(float(v) for v in standard["height"].dropna().unique()),
        "tenmin_rows": int(len(tenmin)) if tenmin is not None else None,
        "hourly_rows": int(len(hourly)) if hourly is not None else None,
    }
    report_path = processed_dir / outputs.get("report_filename", "{station_id}_processing_report.json").format(station_id=station_id)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote: {report_path}")


if __name__ == "__main__":
    main()
