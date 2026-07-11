"""Inspect raw CSV files for a new wind-profile station.

This script is intentionally read-only. Use it before filling the column
mapping in configs/preprocessing/new_station/*.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_csv_sample(path: Path, nrows: int, encoding: str, delimiter: str):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to inspect new-station CSV files.") from exc
    return pd.read_csv(path, nrows=nrows, encoding=encoding, sep=delimiter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_dir", help="Directory containing raw CSV files for one new station.")
    parser.add_argument("--glob", default="*.csv")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--nrows", type=int, default=5)
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    files = sorted(csv_dir.glob(args.glob))
    print(f"csv_dir: {csv_dir}")
    print(f"file_glob: {args.glob}")
    print(f"num_files: {len(files)}")
    if not files:
        raise FileNotFoundError(f"No CSV files found under {csv_dir} with glob {args.glob!r}")

    for path in files[:10]:
        print(f"\nfile: {path}")
        print(f"size_bytes: {path.stat().st_size}")
        df = _read_csv_sample(path, args.nrows, args.encoding, args.delimiter)
        print(f"columns: {list(df.columns)}")
        print("dtypes:")
        for name, dtype in df.dtypes.items():
            print(f"  {name}: {dtype}")
        print("head:")
        print(df.to_string(index=False))

    if len(files) > 10:
        print(f"\n... {len(files) - 10} more files not printed")


if __name__ == "__main__":
    main()
