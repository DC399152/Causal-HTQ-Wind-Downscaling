"""Inspect generated dataset artifacts and array shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_path = dataset_dir / "dataset.npz"
    metadata_path = dataset_dir / "metadata.json"
    print(f"dataset_dir: {dataset_dir}")
    print(f"exists: {dataset_dir.exists()}")
    print(f"dataset_file: {dataset_path}")
    print(f"dataset_file_exists: {dataset_path.exists()}")

    if dataset_path.exists():
        with np.load(dataset_path, allow_pickle=True) as data:
            print("arrays:")
            for key in data.files:
                arr = data[key]
                print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")
            if "split" in data:
                labels, counts = np.unique(data["split"], return_counts=True)
                print("splits:")
                for label, count in zip(labels, counts):
                    print(f"  {label}: {count}")
            if "x_meteo" in data:
                print("meteo:")
                print(f"  x_meteo shape: {data['x_meteo'].shape}  # [N, L, P, C_m]")
                print(f"  meteo_mask shape: {data['meteo_mask'].shape}")
                print(f"  meteo valid ratio: {float(data['meteo_mask'].mean()):.6f}")
                if "meteo_pressure_levels" in data:
                    print(f"  pressure_levels_hpa: {data['meteo_pressure_levels'].tolist()}")
                if "meteo_channel_names" in data:
                    print(f"  channel_names: {[str(v) for v in data['meteo_channel_names']]}")
            if "x_static" in data and data["x_static"].shape[-1] > 0:
                x_static = data["x_static"]
                print("static:")
                print(f"  x_static shape: {x_static.shape}  # [N, F_static]")
                if "static_feature_names" in data:
                    names = [str(v) for v in data["static_feature_names"]]
                    print(f"  static_feature_names: {names}")
                print(
                    "  x_static stats: "
                    f"min={float(np.nanmin(x_static)):.6f} "
                    f"max={float(np.nanmax(x_static)):.6f} "
                    f"mean={float(np.nanmean(x_static)):.6f}"
                )
                if "static_feature_names" in data:
                    names = [str(v) for v in data["static_feature_names"]]
                    nan_counts = np.isnan(x_static).sum(axis=0)
                    print("  static NaN counts:")
                    for name, count in zip(names, nan_counts):
                        print(f"    {name}: {int(count)}")
                if "dominant_lcz" in data:
                    labels, counts = np.unique(data["dominant_lcz"], return_counts=True)
                    print("  dominant_lcz distribution:")
                    for label, count in zip(labels, counts):
                        print(f"    {label}: {int(count)}")

    if metadata_path.exists():
        print("metadata:")
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        for key in ("dataset_name", "timestamp_semantics", "context_alignment", "target_offsets_minutes"):
            print(f"  {key}: {metadata.get(key)}")
        if metadata.get("split_policy"):
            print(f"  split_policy: {metadata.get('split_policy')}")
        if metadata.get("split_counts"):
            print(f"  split_counts: {metadata.get('split_counts')}")
        if metadata.get("meteo"):
            print(f"  meteo: {metadata.get('meteo')}")
        if metadata.get("static_features"):
            print(f"  static_features: {metadata.get('static_features')}")
        warnings = metadata.get("warnings", [])
        if warnings:
            max_warnings = 20
            print(f"  warnings ({len(warnings)} total, showing first {min(max_warnings, len(warnings))}):")
            for warning in warnings[:max_warnings]:
                print(f"    - {warning}")
            if len(warnings) > max_warnings:
                print("    - ...")


if __name__ == "__main__":
    main()
