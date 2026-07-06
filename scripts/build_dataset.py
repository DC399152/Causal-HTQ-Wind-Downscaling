"""Build the Phase 2 preprocessing dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_builder import build_dataset
from src.data.preprocessing import parse_preprocessing_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/preprocessing/paris_1h_to_10min_6h_causal_start_v1.yaml",
        help="Preprocessing YAML config path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and scan without writing files")
    args = parser.parse_args()

    config = parse_preprocessing_config(args.config)
    summary = build_dataset(config, dry_run=args.dry_run)

    print(f"dataset_name: {summary.dataset_name}")
    print(f"dataset_dir: {summary.dataset_dir}")
    print(f"status: {summary.status}")
    print(f"raw_pairs: {summary.num_pairs}")
    print(f"samples: {summary.num_samples}")
    if summary.warnings:
        max_warnings = 20
        print(f"warnings ({len(summary.warnings)} total, showing first {min(max_warnings, len(summary.warnings))}):")
        for warning in summary.warnings[:max_warnings]:
            print(f"  - {warning}")
        if len(summary.warnings) > max_warnings:
            print("  - ...")


if __name__ == "__main__":
    main()
