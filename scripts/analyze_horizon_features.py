"""Diagnose whether target horizons collapse before or after the residual head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import load_checkpoint, model_config_from_checkpoint
from scripts.train import make_loader, model_forward, move_batch
from src.data.dataset import load_norm_stats, require_torch
from src.models.model_factory import architecture_from_config, build_model
from src.training.utils import get_device


def _masked_temporal_std(values, mask, eps: float = 1e-8):
    torch = require_torch()
    valid = mask.to(dtype=values.dtype)
    count = valid.sum(dim=1)
    mean = (values * valid).sum(dim=1) / count.clamp_min(1.0)
    variance = (
        ((values - mean[:, None]) ** 2 * valid).sum(dim=1)
        / count.clamp_min(1.0)
    )
    usable = count >= 2
    std = torch.sqrt(variance.clamp_min(0.0) + eps)
    return std[usable], usable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = require_torch()
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model_config = model_config_from_checkpoint(checkpoint)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = make_loader(
        args.dataset_dir,
        args.split,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    norm_stats = load_norm_stats(Path(args.dataset_dir) / "norm_stats.json")
    y_std = torch.as_tensor(
        norm_stats["y_std"],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 1, -1)
    target_steps = model_config.target_steps
    pair_mask = torch.triu(
        torch.ones(target_steps, target_steps, dtype=torch.bool, device=device),
        diagonal=1,
    )

    cosine_matrix_sum = torch.zeros(
        target_steps,
        target_steps,
        dtype=torch.float64,
        device=device,
    )
    cosine_matrix_count = 0
    feature_variance_sum = 0.0
    feature_variance_count = 0
    pred_std_sum = 0.0
    pred_std_count = 0
    target_std_sum = 0.0
    target_std_count = 0
    pairwise_distance_sum = 0.0
    pairwise_distance_count = 0
    sample_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.limit_batches is not None and batch_index >= args.limit_batches:
                break
            batch = move_batch(batch, device)
            output = model_forward(model, batch, return_features=True)
            features = output["target_features"]
            if features is None:
                raise RuntimeError("Model did not return target_features")
            residual = output["residual"] * y_std
            target_residual = (
                batch["y_10min"]
                - batch["current_hourly_y_norm"].unsqueeze(1)
            ) * y_std
            valid = batch["y_mask"].to(dtype=torch.bool)

            normalized = torch.nn.functional.normalize(features, dim=-1)
            by_height = normalized.permute(0, 2, 1, 3)
            cosine = torch.matmul(by_height, by_height.transpose(-1, -2))
            cosine_matrix_sum += cosine.sum(dim=(0, 1), dtype=torch.float64)
            cosine_matrix_count += cosine.shape[0] * cosine.shape[1]

            feature_variance = features.var(dim=1, unbiased=False)
            feature_variance_sum += float(feature_variance.sum())
            feature_variance_count += feature_variance.numel()

            pred_std, _ = _masked_temporal_std(residual, valid)
            target_std, _ = _masked_temporal_std(target_residual, valid)
            pred_std_sum += float(pred_std.sum())
            pred_std_count += pred_std.numel()
            target_std_sum += float(target_std.sum())
            target_std_count += target_std.numel()

            pair_difference = (
                residual[:, :, None] - residual[:, None, :]
            ).abs()
            pair_valid = valid[:, :, None] & valid[:, None, :]
            selected = pair_valid & pair_mask.view(
                1,
                target_steps,
                target_steps,
                1,
                1,
            )
            pairwise_distance_sum += float(pair_difference[selected].sum())
            pairwise_distance_count += int(selected.sum())
            sample_count += int(features.shape[0])

    if sample_count == 0:
        raise ValueError(f"No samples were evaluated for split {args.split!r}")
    cosine_matrix = cosine_matrix_sum / max(cosine_matrix_count, 1)
    average_pairwise_cosine = float(cosine_matrix[pair_mask].mean())
    predicted_std = pred_std_sum / max(pred_std_count, 1)
    target_std = target_std_sum / max(target_std_count, 1)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "architecture": architecture_from_config(model_config),
        "output_head_type": model.output_head_config.type,
        "sample_count": sample_count,
        "horizon_feature_average_pairwise_cosine_similarity": (
            average_pairwise_cosine
        ),
        "horizon_feature_cosine_similarity_matrix": cosine_matrix.cpu().tolist(),
        "horizon_feature_temporal_variance": (
            feature_variance_sum / max(feature_variance_count, 1)
        ),
        "predicted_residual_temporal_std_ms": predicted_std,
        "predicted_residual_pairwise_horizon_distance_ms": (
            pairwise_distance_sum / max(pairwise_distance_count, 1)
        ),
        "target_residual_temporal_std_ms": target_std,
        "residual_amplitude_ratio": (
            predicted_std / target_std if target_std > 0.0 else None
        ),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
