"""Compute train-only, mask-aware normalization statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.normalization import compute_norm_stats, default_norm_stats_path, save_norm_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--output", default=None, help="Defaults to <dataset_dir>/norm_stats.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output = Path(args.output) if args.output else default_norm_stats_path(dataset_dir)
    stats = compute_norm_stats(dataset_dir, split=args.split, eps=args.eps)
    save_norm_stats(stats, output)

    print(f"output: {output}")
    print(f"computed_from_split: {stats['computed_from_split']}")
    print(f"num_samples: {stats['num_samples']}")
    print(f"channel_names: {stats['channel_names']}")
    print(f"x_mean: {stats['x_mean']}")
    print(f"x_std: {stats['x_std']}")
    print(f"y_mean: {stats['y_mean']}")
    print(f"y_std: {stats['y_std']}")


if __name__ == "__main__":
    main()
