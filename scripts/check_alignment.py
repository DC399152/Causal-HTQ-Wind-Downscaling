"""Check generated dataset alignment and hourly consistency."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.alignment import TARGET_OFFSETS_MINUTES


def _load_dataset(dataset_dir: Path) -> dict[str, np.ndarray]:
    path = dataset_dir / "dataset.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _expected_target_times(start: str) -> list[str]:
    base = np.datetime64(start, "m")
    return [str(base + np.timedelta64(offset, "m")) for offset in TARGET_OFFSETS_MINUTES]


def _masked_target_mean(y: np.ndarray, mask: np.ndarray, missing_value: float) -> np.ndarray:
    """Mean over target time using valid mask.

    y: [N, T_out, H, C]
    mask: [N, T_out, H, C]
    returns: [N, H, C]
    """

    values = np.where(mask, y, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=1)
    return np.where(np.isfinite(mean), mean, missing_value)


def check_alignment(dataset_dir: str | Path, num_examples: int = 5, seed: int = 42) -> None:
    dataset_dir = Path(dataset_dir)
    ds = _load_dataset(dataset_dir)

    required = [
        "x_hourly",
        "x_mask",
        "y_10min",
        "y_mask",
        "current_hourly",
        "station_id",
        "target_time_start",
        "target_times_10min",
        "height_values",
        "source_file",
        "split",
    ]
    missing = [key for key in required if key not in ds]
    if missing:
        raise KeyError(f"Dataset is missing required keys: {missing}")

    x = ds["x_hourly"]
    x_mask = ds["x_mask"]
    y = ds["y_10min"]
    y_mask = ds["y_mask"]
    current = ds["current_hourly"]
    n = x.shape[0]

    print(f"dataset_dir: {dataset_dir}")
    print(f"N: {n}")
    print(f"x_hourly shape: {x.shape}  # [N, L, H, C]")
    print(f"y_10min shape: {y.shape}  # [N, T_out, H, C]")
    print(f"current_hourly shape: {current.shape}  # [N, H, C]")

    if n == 0:
        print("WARNING: dataset contains no samples; alignment checks are structurally complete only.")
        return

    current_equal = np.allclose(current, x[:, -1], equal_nan=True)
    max_current_diff = float(np.nanmax(np.abs(current - x[:, -1])))
    print(f"current_hourly equals x_hourly[:, -1]: {current_equal}")
    print(f"max |current_hourly - x_hourly[:, -1]|: {max_current_diff:.6g}")

    bad_target_rows = []
    for i, start in enumerate(ds["target_time_start"]):
        expected = _expected_target_times(str(start))
        actual = [str(t) for t in ds["target_times_10min"][i]]
        if actual != expected:
            bad_target_rows.append((i, expected, actual))
    print(f"target_times_10min offset check failures: {len(bad_target_rows)}")
    if bad_target_rows:
        idx, expected, actual = bad_target_rows[0]
        print(f"  first failure sample={idx}")
        print(f"  expected={expected}")
        print(f"  actual={actual}")

    valid_positions = x_mask[:, -1] & y_mask.any(axis=1)
    missing_value = -999.0
    y_mean = _masked_target_mean(y, y_mask, missing_value)
    diff = y_mean - current
    valid_diff = diff[valid_positions & np.isfinite(diff)]
    print(f"x valid ratio: {float(x_mask.mean()):.6f}")
    print(f"y valid ratio: {float(y_mask.mean()):.6f}")
    print(f"current-hour x valid ratio: {float(x_mask[:, -1].mean()):.6f}")

    if valid_diff.size == 0:
        print("WARNING: no valid positions for masked mean(y_10min) vs current_hourly comparison.")
    else:
        abs_diff = np.abs(valid_diff)
        print("masked mean(y_10min over target time) - current_hourly statistics:")
        print(f"  count: {valid_diff.size}")
        print(f"  mean: {float(valid_diff.mean()):.6g}")
        print(f"  mean_abs: {float(abs_diff.mean()):.6g}")
        print(f"  max_abs: {float(abs_diff.max()):.6g}")
        print(f"  p95_abs: {float(np.percentile(abs_diff, 95)):.6g}")
        if not np.allclose(valid_diff, 0.0, atol=1e-6):
            print(
                "WARNING: mean(y_10min) and current_hourly are not exactly equal under valid masks. "
                "This can happen with missing values, QC filtering, or raw aggregation differences."
            )

    rng = random.Random(seed)
    sample_indices = sorted(rng.sample(range(n), k=min(num_examples, n)))
    print("sample examples:")
    for idx in sample_indices:
        print(f"  sample_index: {idx}")
        print(f"    station_id: {ds['station_id'][idx]}")
        print(f"    target_time_start: {ds['target_time_start'][idx]}")
        print(f"    target_times_10min: {[str(t) for t in ds['target_times_10min'][idx]]}")
        print(f"    height_values: {ds['height_values'][idx].tolist()}")
        print(f"    split: {ds['split'][idx]}")
        print(f"    source_file: {ds['source_file'][idx]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1",
    )
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    check_alignment(args.dataset_dir, num_examples=args.num_examples, seed=args.seed)


if __name__ == "__main__":
    main()

