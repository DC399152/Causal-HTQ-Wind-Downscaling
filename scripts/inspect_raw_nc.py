"""Inspect a raw NetCDF file.

Prints dimensions, coordinates, variables, time ranges, variable shapes,
NaN/missing counts, and variable attributes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.raw_reader import summarize_netcdf


def _print_attrs(attrs: dict, indent: str = "    ") -> None:
    if not attrs:
        print(f"{indent}attrs: {{}}")
        return
    print(f"{indent}attrs:")
    for key, value in attrs.items():
        print(f"{indent}  {key}: {value}")


def inspect_raw_nc(path: str | Path, missing_value: float | None = -999.0) -> None:
    summary = summarize_netcdf(path, missing_value=missing_value)

    print(f"file: {summary['path']}")
    print("\ndimensions:")
    for name, size in summary["dimensions"].items():
        print(f"  {name}: {size}")

    print("\ncoordinates:")
    for name, item in summary["coordinates"].items():
        print(f"  {name}: dims={item['dims']} shape={item['shape']} dtype={item['dtype']}")
        _print_attrs(item["attrs"])

    print("\nvariables:")
    for name, item in summary["variables"].items():
        print(
            f"  {name}: dims={item['dims']} shape={item['shape']} dtype={item['dtype']} "
            f"nan_count={item['nan_count']} missing_value_count={item['missing_value_count']}"
        )
        _print_attrs(item["attrs"])

    print("\ntime ranges:")
    if not summary["time_ranges"]:
        print("  WARNING: no obvious time variable found")
    for name, item in summary["time_ranges"].items():
        print(
            f"  {name}: start={item['start']} end={item['end']} "
            f"count={item['count']} dtype={item.get('dtype')}"
        )
        _print_attrs(item.get("attrs", {}))

    print("\nglobal attrs:")
    _print_attrs(summary["attrs"], indent="  ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a raw .nc file")
    parser.add_argument("--missing-value", type=float, default=-999.0)
    args = parser.parse_args()
    inspect_raw_nc(args.path, missing_value=args.missing_value)


if __name__ == "__main__":
    main()

