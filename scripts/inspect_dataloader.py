"""Inspect PyTorch DataLoader batches for the generated dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import WindDownscalingDataset, require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "gap", "all"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    try:
        torch = require_torch()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = WindDownscalingDataset(args.dataset_dir, split=args.split, normalize=args.normalize)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))

    print(f"dataset_dir: {args.dataset_dir}")
    print(f"split: {args.split}")
    print(f"normalize: {args.normalize}")
    print(f"dataset_len: {len(dataset)}")
    print("batch tensors:")
    for key in ("x_hourly", "x_mask", "y_10min", "y_mask", "current_hourly"):
        value = batch[key]
        print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")

    invalid_x_count = int((~batch["x_mask"]).sum().item())
    invalid_y_count = int((~batch["y_mask"]).sum().item())
    print(f"invalid_x_count: {invalid_x_count}")
    print(f"invalid_y_count: {invalid_y_count}")
    if args.normalize:
        x_invalid_nonzero = int((batch["x_hourly"][~batch["x_mask"]] != 0).sum().item())
        y_invalid_nonzero = int((batch["y_10min"][~batch["y_mask"]] != 0).sum().item())
        print(f"x_invalid_nonzero_after_normalize: {x_invalid_nonzero}")
        print(f"y_invalid_nonzero_after_normalize: {y_invalid_nonzero}")

    if "split" in batch:
        split_values = sorted(set(str(v) for v in batch["split"]))
        print(f"batch_split_values: {split_values}")


if __name__ == "__main__":
    main()
