"""Evaluate a saved Causal HTQ-Transformer checkpoint.

Metrics are computed with the same convention as ``scripts/train.py``:
normalized-space masked MSE plus denormalized physical m/s MAE and RMSE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import DEFAULT_DATASET_DIR, default_loss_weights, evaluate, make_loader
from src.data.dataset import load_norm_stats, require_torch
from src.models.htq_transformer import CausalHTQTransformer, HTQConfig
from src.training.utils import get_device


def load_checkpoint(path: str | Path, device):
    """Load a checkpoint onto the selected device."""

    torch = require_torch()
    return torch.load(Path(path), map_location=device)


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> HTQConfig:
    """Reconstruct HTQConfig from checkpoint metadata."""

    config = checkpoint.get("model_config")
    if not config:
        raise KeyError("Checkpoint is missing model_config")
    config = dict(config)
    # Checkpoints created before context-conditioned queries used fixed target
    # queries and do not contain the extra context projection parameters.
    config.setdefault("query_builder_type", "fixed")
    return HTQConfig(**config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/htq_minimal/best.pt")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = require_torch()
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model_config = model_config_from_checkpoint(checkpoint)
    model = CausalHTQTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loss_weights = checkpoint.get("loss_weights") or default_loss_weights()

    norm_stats = load_norm_stats(Path(args.dataset_dir) / "norm_stats.json")
    results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "device": str(device),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "loss_weights": loss_weights,
        "loss_space": "normalized",
        "metric_space": "physical_m_per_s",
        "splits": {},
    }

    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset_dir: {args.dataset_dir}")
    print(f"device: {device}")
    print(f"checkpoint_epoch: {checkpoint.get('epoch')}")
    print(
        "loss: normalized weighted L1 "
        f"(lambda_l1={loss_weights['lambda_l1']}, "
        f"lambda_temporal={loss_weights['lambda_temporal']}, "
        f"lambda_vertical={loss_weights['lambda_vertical']})"
    )
    print("metrics: denormalized physical m/s")

    with torch.no_grad():
        for split in args.splits:
            loader = make_loader(
                args.dataset_dir,
                split,
                args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            metrics = evaluate(
                model,
                loader,
                norm_stats,
                device,
                loss_weights=loss_weights,
                limit_batches=args.limit_batches,
            )
            results["splits"][split] = metrics
            print(
                f"{split}: "
                f"loss_norm_total={metrics['loss_norm_total']:.6g} "
                f"loss_norm_l1={metrics['loss_norm_l1']:.6g} "
                f"loss_norm_temporal={metrics['loss_norm_temporal']:.6g} "
                f"loss_norm_vertical={metrics['loss_norm_vertical']:.6g} "
                f"MAE_ms={metrics['MAE_ms']:.6g} "
                f"RMSE_ms={metrics['RMSE_ms']:.6g} "
                f"u_MAE_ms={metrics['u_MAE_ms']:.6g} "
                f"v_MAE_ms={metrics['v_MAE_ms']:.6g} "
                f"speed_MAE_ms={metrics['speed_MAE_ms']:.6g} "
                f"residual_ACC={metrics['residual_ACC']:.6g} "
                f"temporal_gradient_MAE_ms={metrics['temporal_gradient_MAE_ms']:.6g} "
                f"temporal_gradient_ACC={metrics['temporal_gradient_ACC']:.6g} "
                f"valid_target_values={int(metrics['valid_target_values'])}"
            )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
