"""Evaluate repeat-current-hour baseline with masked metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import WindDownscalingDataset, require_torch
from src.models.baselines import repeat_current_hour
from src.training.metrics import (
    add_metric_sums,
    empty_physical_metric_sums,
    finalize_physical_metrics,
    physical_metric_sums,
)


def evaluate_split(dataset_dir: str, split: str, batch_size: int, normalize: bool) -> dict[str, float]:
    torch = require_torch()
    dataset = WindDownscalingDataset(
        dataset_dir,
        split=split,
        normalize=normalize,
        return_metadata=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    metric_sums = empty_physical_metric_sums()
    for batch in loader:
        pred = repeat_current_hour(batch["current_hourly"], target_steps=batch["y_10min"].shape[1])
        add_metric_sums(metric_sums, physical_metric_sums(pred, batch["y_10min"], batch["y_mask"]))

    metrics = {
        "samples": float(len(dataset)),
    }
    metrics.update(finalize_physical_metrics(metric_sums))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    results = {
        "dataset_dir": args.dataset_dir,
        "baseline": "repeat_current_hour",
        "normalize": args.normalize,
        "metric_space": "normalized" if args.normalize else "physical_m_per_s",
        "splits": {},
    }

    print(f"dataset_dir: {args.dataset_dir}")
    print(f"baseline: repeat_current_hour")
    print(f"normalize: {args.normalize}")
    print(f"metric_space: {results['metric_space']}")
    try:
        for split in args.splits:
            metrics = evaluate_split(args.dataset_dir, split, args.batch_size, args.normalize)
            results["splits"][split] = metrics
            print(f"{split}:")
            print(f"  samples: {int(metrics['samples'])}")
            print(f"  valid_target_values: {int(metrics['valid_target_values'])}")
            print(f"  MAE_ms: {metrics['MAE_ms']:.6g}")
            print(f"  RMSE_ms: {metrics['RMSE_ms']:.6g}")
            print(f"  u_MAE_ms: {metrics['u_MAE_ms']:.6g}")
            print(f"  v_MAE_ms: {metrics['v_MAE_ms']:.6g}")
            print(f"  speed_MAE_ms: {metrics['speed_MAE_ms']:.6g}")
            print(f"  global_flattened_ACC: {metrics['global_flattened_ACC']:.6g}")
            print(f"  residual_ACC: {metrics['residual_ACC']:.6g}")
            print(f"  temporal_gradient_MAE_ms: {metrics['temporal_gradient_MAE_ms']:.6g}")
            print(f"  temporal_gradient_ACC: {metrics['temporal_gradient_ACC']:.6g}")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
