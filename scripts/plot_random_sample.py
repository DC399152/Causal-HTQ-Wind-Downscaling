"""Plot one random truth / HTQ prediction / repeat-baseline sample."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import load_checkpoint, model_config_from_checkpoint
from scripts.train import DEFAULT_DATASET_DIR
from src.data.dataset import WindDownscalingDataset, load_norm_stats, require_torch
from src.models.baselines import repeat_current_hour
from src.models.model_factory import build_model
from src.training.utils import get_device, x_denormalize, y_denormalize
from src.visualization.plot_samples import plot_sample_timeseries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/htq_first_full/best.pt")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "gap", "all"])
    parser.add_argument("--sample-index", type=int, default=None, help="Dataset-local index inside the selected split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="runs/htq_first_full/random_sample_test.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = require_torch()
    device = get_device(args.device)
    norm_stats = load_norm_stats(Path(args.dataset_dir) / "norm_stats.json")

    checkpoint = load_checkpoint(args.checkpoint, device)
    model_config = model_config_from_checkpoint(checkpoint)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = WindDownscalingDataset(
        args.dataset_dir,
        split=args.split,
        normalize=True,
        return_metadata=True,
    )
    if len(dataset) == 0:
        raise ValueError(f"Split {args.split!r} has no samples")

    if args.sample_index is None:
        rng = random.Random(args.seed)
        local_index = rng.randrange(len(dataset))
    else:
        local_index = args.sample_index
    if local_index < 0 or local_index >= len(dataset):
        raise IndexError(f"sample-index {local_index} is outside split length {len(dataset)}")

    item = dataset[local_index]
    x_hourly = item["x_hourly"].unsqueeze(0).to(device)
    x_mask = item["x_mask"].unsqueeze(0).to(device)
    x_meteo = item["x_meteo"].unsqueeze(0).to(device) if "x_meteo" in item else None
    meteo_mask = item["meteo_mask"].unsqueeze(0).to(device) if "meteo_mask" in item else None
    x_static = item["x_static"].unsqueeze(0).to(device) if "x_static" in item else None
    current_hourly_reference = (
        item["current_hourly_y_norm"].unsqueeze(0).to(device)
        if "current_hourly_y_norm" in item
        else None
    )

    with torch.no_grad():
        out = model(
            x_hourly,
            x_mask,
            x_meteo=x_meteo,
            meteo_mask=meteo_mask,
            x_static=x_static,
            current_hourly_reference=current_hourly_reference,
            height_values=item["height_values"].unsqueeze(0).to(device),
        )
        pred_ms = y_denormalize(out["pred"], norm_stats)[0].cpu()
        target_ms = y_denormalize(item["y_10min"].unsqueeze(0).to(device), norm_stats)[0].cpu()
        current_ms = x_denormalize(item["current_hourly"].unsqueeze(0).to(device), norm_stats)[0].cpu()
        repeat_ms = repeat_current_hour(current_ms.unsqueeze(0), target_steps=target_ms.shape[0])[0].cpu()

    title = (
        f"{args.split} local_index={local_index}, sample_index={item['sample_index']}, "
        f"station={item.get('station_id', 'unknown')}, T={item.get('target_time_start', 'unknown')}"
    )
    plot_sample_timeseries(
        target=target_ms,
        pred=pred_ms,
        repeat=repeat_ms,
        y_mask=item["y_mask"],
        height_values=[float(v) for v in item["height_values"]],
        output_path=args.output,
        title=title,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}")
    print(f"local_index: {local_index}")
    print(f"sample_index: {item['sample_index']}")
    print(f"station_id: {item.get('station_id', 'unknown')}")
    print(f"target_time_start: {item.get('target_time_start', 'unknown')}")
    print(f"target_times_10min: {item.get('target_times_10min', [])}")
    print(f"height_values: {[float(v) for v in item['height_values']]}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
