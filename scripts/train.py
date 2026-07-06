"""Minimal Causal HTQ-Transformer training entry point.

Training loss is computed in normalized space. Validation and test metrics are
computed after denormalizing predictions and targets to physical m/s units.
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

from src.data.dataset import WindDownscalingDataset, load_norm_stats, require_torch
from src.models.htq_transformer import CausalHTQTransformer, HTQConfig
from src.training.losses import htq_reconstruction_loss
from src.training.metrics import (
    add_metric_sums,
    empty_physical_metric_sums,
    finalize_physical_metrics,
    physical_metric_sums,
)
from src.training.utils import get_device, set_seed, y_denormalize


DEFAULT_CONFIG = "configs/htq/htq_paris_1h_to_10min_6h_causal_start_v1.yaml"
DEFAULT_DATASET_DIR = "data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1"
DEFAULT_RUN_DIR = "runs/htq_minimal"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with a clear dependency error."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read training config files.") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model_config(config: dict[str, Any], args: argparse.Namespace) -> HTQConfig:
    """Map YAML keys to the HTQConfig dataclass."""

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    return HTQConfig(
        d_model=args.d_model or int(model_cfg.get("d_model", 64)),
        nhead=args.nhead or int(model_cfg.get("n_heads", model_cfg.get("nhead", 4))),
        num_encoder_layers=args.encoder_layers
        or int(model_cfg.get("encoder_layers", model_cfg.get("num_encoder_layers", 2))),
        num_decoder_layers=args.decoder_layers
        or int(model_cfg.get("decoder_layers", model_cfg.get("num_decoder_layers", 2))),
        dim_feedforward=args.dim_feedforward or int(model_cfg.get("dim_feedforward", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        context_hours=int(data_cfg.get("context_hours", 6)),
        target_steps=int(data_cfg.get("target_steps", 6)),
        height_levels=int(model_cfg.get("height_levels", 6)),
        input_channels=int(model_cfg.get("input_channels", 2)),
        output_channels=int(model_cfg.get("output_channels", 2)),
    )


def make_loader(
    dataset_dir: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    """Create a normalized DataLoader for one split."""

    torch = require_torch()
    dataset = WindDownscalingDataset(
        dataset_dir,
        split=split,
        normalize=True,
        return_metadata=False,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch(batch: dict[str, Any], device):
    """Move tensor batch values to the selected device."""

    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_weights: dict[str, float],
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Train for one epoch using normalized-space weighted HTQ losses."""

    model.train()
    total = {"loss": 0.0, "l1": 0.0, "temporal": 0.0, "vertical": 0.0}
    total_batches = 0
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["x_hourly"], batch["x_mask"])
        loss_parts = htq_reconstruction_loss(
            out["pred"],
            batch["y_10min"],
            batch["y_mask"],
            lambda_l1=loss_weights["lambda_l1"],
            lambda_temporal=loss_weights["lambda_temporal"],
            lambda_vertical=loss_weights["lambda_vertical"],
        )
        loss = loss_parts["loss"]
        loss.backward()
        optimizer.step()

        for key in total:
            total[key] += float(loss_parts[key].detach().item())
        total_batches += 1

    if total_batches == 0:
        raise ValueError("No training batches were processed")
    return {
        "train_loss_norm_total": total["loss"] / total_batches,
        "train_loss_norm_l1": total["l1"] / total_batches,
        "train_loss_norm_temporal": total["temporal"] / total_batches,
        "train_loss_norm_vertical": total["vertical"] / total_batches,
    }


def evaluate(
    model,
    loader,
    norm_stats: dict[str, Any],
    device,
    loss_weights: dict[str, float] | None = None,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate normalized weighted losses and physical-unit m/s metrics."""

    torch = require_torch()
    model.eval()
    loss_weights = loss_weights or default_loss_weights()
    total = {"loss": 0.0, "l1": 0.0, "temporal": 0.0, "vertical": 0.0}
    total_batches = 0
    metric_sums = empty_physical_metric_sums()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            batch = move_batch(batch, device)
            out = model(batch["x_hourly"], batch["x_mask"])
            loss_parts = htq_reconstruction_loss(
                out["pred"],
                batch["y_10min"],
                batch["y_mask"],
                lambda_l1=loss_weights["lambda_l1"],
                lambda_temporal=loss_weights["lambda_temporal"],
                lambda_vertical=loss_weights["lambda_vertical"],
            )

            pred_ms = y_denormalize(out["pred"], norm_stats)
            target_ms = y_denormalize(batch["y_10min"], norm_stats)

            for key in total:
                total[key] += float(loss_parts[key].item())
            total_batches += 1
            add_metric_sums(metric_sums, physical_metric_sums(pred_ms, target_ms, batch["y_mask"]))

    if total_batches == 0:
        raise ValueError("No evaluation batches were processed")

    metrics = {
        "loss_norm_total": total["loss"] / total_batches,
        "loss_norm_l1": total["l1"] / total_batches,
        "loss_norm_temporal": total["temporal"] / total_batches,
        "loss_norm_vertical": total["vertical"] / total_batches,
    }
    metrics.update(finalize_physical_metrics(metric_sums))
    return metrics


def default_loss_weights() -> dict[str, float]:
    """Default normalized-space HTQ training loss weights."""

    return {
        "lambda_l1": 1.0,
        "lambda_temporal": 0.2,
        "lambda_vertical": 0.05,
    }


def loss_weights_from_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    """Read loss weights from config and allow CLI overrides."""

    loss_cfg = config.get("loss", {})
    defaults = default_loss_weights()
    return {
        "lambda_l1": args.lambda_l1
        if args.lambda_l1 is not None
        else float(loss_cfg.get("lambda_l1", defaults["lambda_l1"])),
        "lambda_temporal": args.lambda_temporal
        if args.lambda_temporal is not None
        else float(loss_cfg.get("lambda_temporal", defaults["lambda_temporal"])),
        "lambda_vertical": args.lambda_vertical
        if args.lambda_vertical is not None
        else float(loss_cfg.get("lambda_vertical", defaults["lambda_vertical"])),
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict[str, float],
    model_config: HTQConfig,
    loss_weights: dict[str, float],
) -> None:
    """Save a compact PyTorch checkpoint."""

    torch = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_config": model_config.__dict__,
            "loss_weights": loss_weights,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--lambda-l1", type=float, default=None)
    parser.add_argument("--lambda-temporal", type=float, default=None)
    parser.add_argument("--lambda-vertical", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 32))
    max_epochs = args.max_epochs or int(train_cfg.get("max_epochs") or 1)
    learning_rate = args.learning_rate or float(train_cfg.get("learning_rate") or 1e-4)
    dataset_dir = args.dataset_dir or data_cfg.get("dataset_dir", DEFAULT_DATASET_DIR)
    run_dir = Path(args.run_dir)
    loss_weights = loss_weights_from_config(config, args)

    set_seed(seed)
    torch = require_torch()
    device = get_device(args.device)
    norm_stats = load_norm_stats(Path(dataset_dir) / "norm_stats.json")

    model_config = build_model_config(config, args)
    model = CausalHTQTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_loader = make_loader(dataset_dir, "train", batch_size, True, args.num_workers)
    val_loader = make_loader(dataset_dir, "val", batch_size, False, args.num_workers)
    test_loader = make_loader(dataset_dir, "test", batch_size, False, args.num_workers)

    print(f"dataset_dir: {dataset_dir}")
    print(f"run_dir: {run_dir}")
    print(f"device: {device}")
    print(f"epochs: {max_epochs}")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(
        "loss: normalized weighted L1 "
        f"(lambda_l1={loss_weights['lambda_l1']}, "
        f"lambda_temporal={loss_weights['lambda_temporal']}, "
        f"lambda_vertical={loss_weights['lambda_vertical']})"
    )
    print("val/test metrics: denormalized physical m/s")

    best_val_mae = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights,
            limit_batches=args.limit_train_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            norm_stats,
            device,
            loss_weights=loss_weights,
            limit_batches=args.limit_eval_batches,
        )
        row = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"epoch {epoch:03d} "
            f"train_loss_norm_total={row['train_loss_norm_total']:.6g} "
            f"val_loss_norm_total={row['val_loss_norm_total']:.6g} "
            f"val_MAE_ms={row['val_MAE_ms']:.6g} "
            f"val_RMSE_ms={row['val_RMSE_ms']:.6g} "
            f"val_residual_ACC={row['val_residual_ACC']:.6g}"
        )

        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, row, model_config, loss_weights)
        if val_metrics["MAE_ms"] < best_val_mae:
            best_val_mae = val_metrics["MAE_ms"]
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, row, model_config, loss_weights)

    test_metrics = evaluate(
        model,
        test_loader,
        norm_stats,
        device,
        loss_weights=loss_weights,
        limit_batches=args.limit_eval_batches,
    )
    print(
        "test "
        f"loss_norm_total={test_metrics['loss_norm_total']:.6g} "
        f"MAE_ms={test_metrics['MAE_ms']:.6g} "
        f"RMSE_ms={test_metrics['RMSE_ms']:.6g} "
        f"residual_ACC={test_metrics['residual_ACC']:.6g}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": str(args.config),
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "seed": seed,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "loss_weights": loss_weights,
        "model_config": model_config.__dict__,
        "history": history,
        "test": test_metrics,
        "loss_space": "normalized",
        "loss_type": "lambda_l1*masked_l1 + lambda_temporal*temporal_gradient_l1 + lambda_vertical*vertical_gradient_l1",
        "metric_space": "physical_m_per_s",
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
