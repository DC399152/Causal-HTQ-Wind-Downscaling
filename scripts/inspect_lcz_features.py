"""Inspect generated station-level LCZ feature CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/processed/static/station_lcz_features_500m.csv",
    )
    parser.add_argument("--radius-label", default="500m")
    args = parser.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        raise FileNotFoundError(f"LCZ feature CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"station_count: 0")
        return

    label = args.radius_label
    frac_cols = [f"LCZ_{i}_frac_{label}" for i in range(1, 18)]
    missing = [col for col in frac_cols if col not in rows[0]]
    if missing:
        raise ValueError(f"CSV missing LCZ fraction columns: {missing}")

    fractions = np.asarray([[float(row[col]) for col in frac_cols] for row in rows], dtype=np.float32)
    sums = np.sum(fractions, axis=1)
    dominant_col = f"dominant_lcz_{label}"
    valid_col = f"valid_pixel_count_{label}"
    fraction_sum_col = f"fraction_sum_{label}"

    print(f"csv_path: {path}")
    print(f"station_count: {len(rows)}")
    print("stations:")
    for row, frac_sum in zip(rows, sums):
        station_id = row.get("station_id", "")
        print(
            f"  {station_id}: dominant_lcz={row.get(dominant_col)} "
            f"fraction_sum={row.get(fraction_sum_col, float(frac_sum))} "
            f"valid_pixel_count={row.get(valid_col)} "
            f"computed_fraction_sum={float(frac_sum):.6f}"
        )

    print("LCZ fraction stats:")
    for col, values in zip(frac_cols, fractions.T):
        print(
            f"  {col}: min={float(np.nanmin(values)):.6f} "
            f"max={float(np.nanmax(values)):.6f} mean={float(np.nanmean(values)):.6f}"
        )

    close = np.isclose(sums, 1.0, atol=1e-4)
    print(f"fraction_sum_close_to_one: {int(np.count_nonzero(close))}/{len(close)}")
    if not np.all(close):
        bad_ids = [rows[i].get("station_id", str(i)) for i, ok in enumerate(close) if not ok]
        print(f"warning: fraction sums not close to 1 for station_id(s): {bad_ids}")


if __name__ == "__main__":
    main()
