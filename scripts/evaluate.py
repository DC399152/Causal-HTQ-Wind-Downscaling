"""Evaluate a saved Causal HTQ-Transformer checkpoint.

Metrics are computed with the same convention as ``scripts/train.py``:
normalized-space masked MSE plus denormalized physical m/s MAE and RMSE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import (
    DEFAULT_DATASET_DIR,
    attach_norm_stats_to_loss_config,
    default_loss_config,
    evaluate,
    make_loader,
    model_forward,
    move_batch,
)
from src.data.dataset import WindDownscalingDataset, load_norm_stats, require_torch
from src.models.baselines import repeat_current_hour
from src.models.model_factory import (
    ModelConfig,
    architecture_from_config,
    build_model,
    model_config_from_dict,
)
from src.training.utils import get_device, x_denormalize, y_denormalize


def load_checkpoint(path: str | Path, device):
    """Load a checkpoint onto the selected device."""

    torch = require_torch()
    return torch.load(Path(path), map_location=device)


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> ModelConfig:
    """Reconstruct the correct model config from checkpoint metadata."""

    config = checkpoint.get("model_config")
    if not config:
        raise KeyError("Checkpoint is missing model_config")
    config = dict(config)
    config.setdefault("architecture", checkpoint.get("architecture", "htq_encoder_decoder"))
    # Checkpoints created before context-conditioned queries used fixed target
    # queries and do not contain the extra context projection parameters.
    if config["architecture"] == "htq_encoder_decoder":
        config.setdefault("query_builder_type", "fixed")
    return model_config_from_dict(config)


def loss_config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Load residual-physics loss config from checkpoint metadata."""

    if "loss_config" in checkpoint:
        config = dict(default_loss_config())
        config.update(checkpoint["loss_config"])
        if config.get("type") != "residual_physics":
            raise ValueError(
                f"Checkpoint loss.type={config.get('type')!r} is no longer supported; "
                "only residual_physics is supported."
            )
        return config
    raise KeyError("Checkpoint is missing residual_physics loss_config")


def dataset_fingerprint(dataset_dir: str | Path) -> str:
    """Return the checkpoint-compatible fingerprint for dataset.npz."""

    dataset_path = Path(dataset_dir) / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    digest = hashlib.sha256()
    with dataset_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def resolve_evaluation_dataset_dir(
    checkpoint: dict[str, Any],
    requested_dataset_dir: str | Path | None,
    *,
    allow_mismatch: bool = False,
) -> Path:
    """Resolve the dataset and reject accidental cross-dataset evaluation."""

    metadata = checkpoint.get("dataset_metadata") or {}
    checkpoint_dataset_path = metadata.get("dataset_path")
    checkpoint_dataset_dir = (
        Path(checkpoint_dataset_path).parent
        if checkpoint_dataset_path
        else None
    )

    if requested_dataset_dir is None:
        dataset_dir = checkpoint_dataset_dir or Path(DEFAULT_DATASET_DIR)
    else:
        dataset_dir = Path(requested_dataset_dir)

    expected_fingerprint = (
        checkpoint.get("dataset_fingerprint")
        or metadata.get("dataset_fingerprint")
    )
    if expected_fingerprint and not allow_mismatch:
        actual_fingerprint = dataset_fingerprint(dataset_dir)
        if actual_fingerprint != expected_fingerprint:
            expected_location = (
                f" Expected training dataset: {checkpoint_dataset_dir}."
                if checkpoint_dataset_dir is not None
                else ""
            )
            raise ValueError(
                "Evaluation dataset does not match the checkpoint training dataset. "
                f"Checkpoint fingerprint={expected_fingerprint}, "
                f"evaluation fingerprint={actual_fingerprint}."
                f"{expected_location} "
                "Use the matching dataset, or pass --allow-dataset-mismatch only "
                "for an intentional cross-dataset evaluation."
            )
    elif (
        checkpoint_dataset_dir is not None
        and requested_dataset_dir is not None
        and dataset_dir.resolve() != checkpoint_dataset_dir.resolve()
        and not allow_mismatch
    ):
        raise ValueError(
            "Evaluation dataset path does not match the checkpoint training dataset: "
            f"{dataset_dir} != {checkpoint_dataset_dir}. "
            "Use the matching dataset, or pass --allow-dataset-mismatch only "
            "for an intentional cross-dataset evaluation."
        )

    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/htq_minimal/best.pt")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Dataset to evaluate. Defaults to the training dataset recorded in "
            "the checkpoint, or the legacy project default for old checkpoints."
        ),
    )
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="Allow intentional evaluation on a dataset different from training.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output", dest="output_json", default=None, help="Alias for --output-json")
    parser.add_argument("--make-plots", action="store_true", help="Write training curves and sample visualizations")
    parser.add_argument("--plots-only", action="store_true", help="Generate plots without recomputing or rewriting metrics")
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--num-random-plots", type=int, default=3)
    parser.add_argument("--num-high-error-plots", type=int, default=3)
    parser.add_argument("--num-high-fluctuation-plots", type=int, default=3)
    parser.add_argument("--plot-split", default=None, help="Split used for sample plots; defaults to the last evaluated split")
    parser.add_argument("--plot-seed", type=int, default=42)
    parser.add_argument("--sample-gallery-size", type=int, default=100)
    parser.add_argument("--sample-gallery-split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = require_torch()
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    dataset_dir = resolve_evaluation_dataset_dir(
        checkpoint,
        args.dataset_dir,
        allow_mismatch=args.allow_dataset_mismatch,
    )
    args.dataset_dir = str(dataset_dir)
    model_config = model_config_from_checkpoint(checkpoint)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loss_config = loss_config_from_checkpoint(checkpoint)

    norm_stats = load_norm_stats(dataset_dir / "norm_stats.json")
    loss_config = attach_norm_stats_to_loss_config(loss_config, norm_stats)
    results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "device": str(device),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "architecture": architecture_from_config(model_config),
        "output_head_type": model.output_head_config.type,
        "output_head_config": model.output_head_config.__dict__,
        "loss_config": loss_config,
        "loss_weights": {
            "lambda_wind": float(loss_config["lambda_wind"]),
            "lambda_extreme": float(loss_config["lambda_extreme"]),
            "lambda_residual_weighted": float(loss_config["lambda_residual_weighted"]),
            "lambda_temporal": float(loss_config["lambda_temporal"]),
            "lambda_temporal_weighted": float(loss_config["lambda_temporal_weighted"]),
            "lambda_roughness": float(loss_config["lambda_roughness"]),
            "lambda_amplitude": float(loss_config["lambda_amplitude"]),
            "lambda_gradient_amplitude": float(loss_config["lambda_gradient_amplitude"]),
            "lambda_residual_corr": float(loss_config["lambda_residual_corr"]),
            "lambda_temporal_gradient_corr": float(loss_config["lambda_temporal_gradient_corr"]),
            "lambda_vertical": float(loss_config["lambda_vertical"]),
            "lambda_consistency": float(loss_config["lambda_consistency"]),
        },
        "loss_space": "normalized",
        "metric_space": "physical_m_per_s",
        "splits": {},
    }

    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset_dir: {args.dataset_dir}")
    print(f"device: {device}")
    print(f"checkpoint_epoch: {checkpoint.get('epoch')}")
    print(f"architecture: {architecture_from_config(model_config)}")
    print(f"output_head_type: {model.output_head_config.type}")
    print(
        f"loss: {loss_config['type']} normalized loss "
        f"(lambda_wind={loss_config['lambda_wind']}, "
        f"lambda_extreme={loss_config['lambda_extreme']}, "
        f"lambda_residual_weighted={loss_config['lambda_residual_weighted']}, "
        f"lambda_temporal={loss_config['lambda_temporal']}, "
        f"lambda_temporal_weighted={loss_config['lambda_temporal_weighted']}, "
        f"lambda_roughness={loss_config['lambda_roughness']}, "
        f"lambda_amplitude={loss_config['lambda_amplitude']}, "
        f"lambda_gradient_amplitude={loss_config['lambda_gradient_amplitude']}, "
        f"lambda_residual_corr={loss_config['lambda_residual_corr']}, "
        f"lambda_temporal_gradient_corr={loss_config['lambda_temporal_gradient_corr']}, "
        f"lambda_vertical={loss_config['lambda_vertical']})"
    )
    print("metrics: denormalized physical m/s")

    output_path = Path(args.output_json) if args.output_json else Path(args.checkpoint).resolve().parent / "eval.json"
    if args.plots_only:
        figures_dir = Path(args.figures_dir) if args.figures_dir else output_path.parent / "figures"
        written = make_plots(
            args=args,
            model=model,
            norm_stats=norm_stats,
            device=device,
            figures_dir=figures_dir,
            sample_gallery_dir=output_path.parent / "sample_gallery",
        )
        print(f"wrote {len(written)} plot files")
        return

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
                loss_config=loss_config,
                limit_batches=args.limit_batches,
            )
            results["splits"][split] = metrics
            print(
                f"{split}: "
                f"loss_norm_total={metrics['loss_norm_total']:.6g} "
                f"loss_norm_wind={metrics['loss_norm_wind']:.6g} "
                f"loss_norm_extreme={metrics['loss_norm_extreme']:.6g} "
                f"loss_norm_temporal={metrics['loss_norm_temporal']:.6g} "
                f"loss_norm_roughness={metrics['loss_norm_roughness']:.6g} "
                f"loss_norm_vertical={metrics['loss_norm_vertical']:.6g} "
                f"MAE_ms={metrics['MAE_ms']:.6g} "
                f"RMSE_ms={metrics['RMSE_ms']:.6g} "
                f"u_MAE_ms={metrics['u_MAE_ms']:.6g} "
                f"v_MAE_ms={metrics['v_MAE_ms']:.6g} "
                f"speed_MAE_ms={metrics['speed_MAE_ms']:.6g} "
                f"global_flattened_ACC={metrics['global_flattened_ACC']:.6g} "
                f"residual_ACC={metrics['residual_ACC']:.6g} "
                f"temporal_gradient_MAE_ms={metrics['temporal_gradient_MAE_ms']:.6g} "
                f"temporal_gradient_ACC={metrics['temporal_gradient_ACC']:.6g} "
                f"valid_target_values={int(metrics['valid_target_values'])}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote: {output_path}")

    if args.make_plots:
        figures_dir = Path(args.figures_dir) if args.figures_dir else output_path.parent / "figures"
        try:
            written = make_plots(
                args=args,
                model=model,
                norm_stats=norm_stats,
                device=device,
                figures_dir=figures_dir,
                sample_gallery_dir=output_path.parent / "sample_gallery",
            )
            for path in written:
                print(f"wrote: {path}")
        except Exception as exc:
            print(f"warning: plot generation failed: {exc}")


def make_plots(
    *,
    args: argparse.Namespace,
    model,
    norm_stats: dict[str, Any],
    device,
    figures_dir: Path,
    sample_gallery_dir: Path,
) -> list[Path]:
    """Generate training curves and an unbiased test-sample gallery."""

    # Windows/conda can load both libomp and libiomp5md when torch and
    # matplotlib/numpy meet. Keep metrics usable and allow plotting to proceed.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from src.visualization.plot_training_curves import plot_training_curves

    written: list[Path] = []
    metrics_path = Path(args.checkpoint).resolve().parent / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            written.extend(plot_training_curves(json.load(f), figures_dir))
    else:
        print(f"warning: metrics.json not found, skipped loss curves: {metrics_path}")

    written.extend(
        plot_sample_gallery(
            model=model,
            dataset_dir=args.dataset_dir,
            split=args.sample_gallery_split,
            norm_stats=norm_stats,
            device=device,
            output_dir=sample_gallery_dir,
            seed=args.plot_seed,
            num_samples=args.sample_gallery_size,
        )
    )
    return written


def plot_sample_gallery(
    *,
    model,
    dataset_dir: str | Path,
    split: str,
    norm_stats: dict[str, Any],
    device,
    output_dir: Path,
    seed: int,
    num_samples: int,
) -> list[Path]:
    """Plot deterministic random samples without error/fluctuation selection."""

    torch = require_torch()
    dataset = WindDownscalingDataset(
        dataset_dir,
        split=split,
        normalize=True,
        return_metadata=True,
    )
    count = min(max(num_samples, 0), len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "split": split,
        "seed": seed,
        "requested_samples": num_samples,
        "selected_samples": [],
    }
    written: list[Path] = []
    for rank, local_index in enumerate(indices, start=1):
        item = dataset[int(local_index)]
        output_path = output_dir / f"sample_{rank:03d}_local_{int(local_index):06d}.png"
        plot_one_sample(
            model=model,
            dataset=dataset,
            local_index=int(local_index),
            split=split,
            norm_stats=norm_stats,
            device=device,
            output_path=None,
            context_output_path=output_path,
            label="gallery",
            item=item,
        )
        manifest["selected_samples"].append(
            {
                "rank": rank,
                "local_index": int(local_index),
                "sample_index": int(item["sample_index"]),
                "station_id": str(item.get("station_id", "unknown")),
                "target_time_start": str(item.get("target_time_start", "unknown")),
                "file": output_path.name,
            }
        )
        written.append(output_path)

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    written.append(manifest_path)
    return written


def plot_representative_samples(
    *,
    model,
    dataset_dir: str | Path,
    split: str,
    norm_stats: dict[str, Any],
    device,
    figures_dir: Path,
    seed: int,
    num_random: int,
    num_high_error: int,
    num_high_fluctuation: int,
    batch_size: int,
    num_workers: int,
    limit_batches: int | None,
) -> list[Path]:
    """Select random/high-error/high-fluctuation samples and plot them."""

    torch = require_torch()
    dataset = WindDownscalingDataset(dataset_dir, split=split, normalize=True, return_metadata=False)
    if len(dataset) == 0:
        return []

    rng = torch.Generator()
    rng.manual_seed(seed)
    random_count = min(max(num_random, 0), len(dataset))
    random_indices = torch.randperm(len(dataset), generator=rng)[:random_count].tolist()

    scored = score_samples(
        model=model,
        dataset=dataset,
        norm_stats=norm_stats,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        limit_batches=limit_batches,
    )
    high_error_indices = [idx for idx, _ in sorted(scored["error"], key=lambda item: item[1], reverse=True)[: max(num_high_error, 0)]]
    high_fluctuation_indices = [
        idx
        for idx, _ in sorted(scored["fluctuation"], key=lambda item: item[1], reverse=True)[: max(num_high_fluctuation, 0)]
    ]

    selected: list[tuple[str, int]] = []
    selected.extend(("random", int(idx)) for idx in random_indices)
    selected.extend(("high_error", int(idx)) for idx in high_error_indices)
    selected.extend(("high_fluctuation", int(idx)) for idx in high_fluctuation_indices)

    written: list[Path] = []
    seen: set[tuple[str, int]] = set()
    for label, local_index in selected:
        key = (label, local_index)
        if key in seen:
            continue
        seen.add(key)
        rank = sum(1 for prev_label, _ in seen if prev_label == label)
        filename = f"sample_{label}_{rank:03d}.png"
        output_path = figures_dir / filename
        context_output_path = figures_dir.parent / "figure_12" / filename
        plot_one_sample(
            model=model,
            dataset=dataset,
            local_index=local_index,
            split=split,
            norm_stats=norm_stats,
            device=device,
            output_path=output_path,
            context_output_path=context_output_path,
            label=label,
        )
        written.extend((output_path, context_output_path))
    return written


def score_samples(
    *,
    model,
    dataset: WindDownscalingDataset,
    norm_stats: dict[str, Any],
    device,
    batch_size: int,
    num_workers: int,
    limit_batches: int | None,
) -> dict[str, list[tuple[int, float]]]:
    """Return per-sample error and true residual fluctuation scores."""

    torch = require_torch()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    errors: list[tuple[int, float]] = []
    fluctuations: list[tuple[int, float]] = []
    local_offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            batch_size_actual = int(batch["y_10min"].shape[0])
            batch = move_batch(batch, device)
            out = model_forward(model, batch)
            pred_ms = y_denormalize(out["pred"], norm_stats)
            target_ms = y_denormalize(batch["y_10min"], norm_stats)
            current_ms = x_denormalize(batch["current_hourly"], norm_stats)
            valid = batch["y_mask"].to(dtype=pred_ms.dtype)

            abs_error = ((pred_ms - target_ms).abs() * valid).flatten(start_dim=1)
            valid_count = valid.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
            sample_error = abs_error.sum(dim=1) / valid_count

            residual = target_ms - current_ms.unsqueeze(1)
            residual_mag = residual.pow(2).sum(dim=-1).sqrt()
            both_valid = (batch["y_mask"][..., 0] & batch["y_mask"][..., 1]).to(dtype=residual_mag.dtype)
            sample_fluct = (residual_mag * both_valid).flatten(start_dim=1).sum(dim=1) / both_valid.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)

            for i in range(batch_size_actual):
                local_index = local_offset + i
                errors.append((local_index, float(sample_error[i].detach().cpu().item())))
                fluctuations.append((local_index, float(sample_fluct[i].detach().cpu().item())))
            local_offset += batch_size_actual
    return {"error": errors, "fluctuation": fluctuations}


def plot_one_sample(
    *,
    model,
    dataset: WindDownscalingDataset,
    local_index: int,
    split: str,
    norm_stats: dict[str, Any],
    device,
    output_path: Path | None,
    context_output_path: Path | None,
    label: str,
    item=None,
) -> None:
    """Run model on one dataset item and save truth/pred/repeat plot."""

    torch = require_torch()
    if item is None:
        item = WindDownscalingDataset(
            dataset.dataset_dir,
            split=split,
            normalize=True,
            return_metadata=True,
        )[local_index]
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in item.items()
        if hasattr(value, "unsqueeze") and key not in {"height_values"}
    }
    with torch.no_grad():
        out = model_forward(model, batch)
        pred_ms = y_denormalize(out["pred"], norm_stats)[0].cpu()
        target_ms = y_denormalize(batch["y_10min"], norm_stats)[0].cpu()
        context_ms = x_denormalize(batch["x_hourly"], norm_stats)[0].cpu()
        current_ms = x_denormalize(batch["current_hourly"], norm_stats)[0].cpu()
        repeat_ms = repeat_current_hour(current_ms.unsqueeze(0), target_steps=target_ms.shape[0])[0].cpu()

    title = (
        f"{label}, {split} local_index={local_index}, sample_index={item['sample_index']}, "
        f"station={item.get('station_id', 'unknown')}, T={item.get('target_time_start', 'unknown')}"
    )
    from src.visualization.plot_samples import plot_sample_timeseries, plot_sample_with_hourly_context

    if output_path is not None:
        plot_sample_timeseries(
            target=target_ms,
            pred=pred_ms,
            repeat=repeat_ms,
            y_mask=item["y_mask"],
            height_values=[float(v) for v in item["height_values"]],
            output_path=output_path,
            title=title,
        )
    if context_output_path is not None:
        plot_sample_with_hourly_context(
            context=context_ms,
            target=target_ms,
            pred=pred_ms,
            repeat=repeat_ms,
            x_mask=item["x_mask"],
            y_mask=item["y_mask"],
            height_values=[float(v) for v in item["height_values"]],
            output_path=context_output_path,
            title=title,
        )


if __name__ == "__main__":
    main()
