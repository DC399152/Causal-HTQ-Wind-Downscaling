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
from src.training.losses import htq_fluctuation_aware_loss, htq_reconstruction_loss
from src.training.metrics import (
    add_metric_sums,
    empty_physical_metric_sums,
    finalize_physical_metrics,
    physical_metric_sums,
)
from src.training.utils import get_device, set_seed, y_denormalize


DEFAULT_CONFIG = "configs/model/htq_meteo.yaml"
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
    multimodal_cfg = config.get("multimodal", {})
    meteo_cfg = config.get("meteo", {})
    static_cfg = config.get("static", {})
    fusion_cfg = config.get("fusion", {})
    query_cfg = config.get("query_builder", {})
    pressure_levels = tuple(int(v) for v in meteo_cfg.get("pressure_levels_hpa", [1000, 975, 950, 925, 900]))
    trend_scales = tuple(int(v) for v in query_cfg.get("trend_scales", [1, 3, 5]))
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
        enforce_zero_mean_residual=bool(model_cfg.get("enforce_zero_mean_residual", False)),
        use_meteo=bool(multimodal_cfg.get("use_meteo", model_cfg.get("use_meteo", False))),
        use_static=bool(multimodal_cfg.get("use_static", model_cfg.get("use_static", False))),
        meteo_context_hours=int(meteo_cfg.get("context_hours", data_cfg.get("context_hours", 6))),
        meteo_pressure_levels_hpa=pressure_levels,
        num_meteo_channels=int(meteo_cfg.get("num_meteo_channels", 2)),
        meteo_use_delta=bool(meteo_cfg.get("use_delta", True)),
        meteo_use_mask_channels=bool(meteo_cfg.get("use_mask_channels", False)),
        fusion_nhead=int(fusion_cfg.get("nhead", model_cfg.get("n_heads", model_cfg.get("nhead", 4)))),
        fusion_dropout=float(fusion_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        fusion_gate_init_bias=float(fusion_cfg.get("gate_init_bias", -2.0)),
        static_input_dim=int(static_cfg.get("input_dim", 17)),
        static_n_tokens=int(static_cfg.get("n_static_tokens", 1)),
        static_hidden_dim=int(static_cfg.get("hidden_dim", 128)),
        static_dropout=float(static_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        query_builder_type=str(query_cfg.get("type", model_cfg.get("query_builder_type", "context_conditioned"))),
        query_use_context_projection=bool(query_cfg.get("use_context_projection", True)),
        query_use_context_layernorm=bool(query_cfg.get("use_context_layernorm", True)),
        query_use_temporal_context=bool(query_cfg.get("use_temporal_context", False)),
        query_use_multiscale_trend=bool(
            query_cfg.get("use_multiscale_trend", query_cfg.get("use_trend_context", False))
        ),
        query_trend_scales=trend_scales,
        query_use_trend_context=bool(query_cfg.get("use_trend_context", False)),
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


def model_forward(model, batch: dict[str, Any]):
    """Run HTQ with optional multimodal inputs when present in the batch."""

    return model(
        x_hourly=batch["x_hourly"],
        x_mask=batch["x_mask"],
        x_meteo=batch.get("x_meteo"),
        meteo_mask=batch.get("meteo_mask"),
        x_static=batch.get("x_static"),
        current_hourly_reference=batch.get("current_hourly_y_norm"),
    )


def compute_loss_parts(pred, batch: dict[str, Any], loss_config: dict[str, float | str]):
    """Compute the configured normalized-space training loss."""

    loss_type = str(loss_config.get("type", "standard"))
    common = {
        "lambda_l1": float(loss_config["lambda_l1"]),
        "lambda_temporal": float(loss_config["lambda_temporal"]),
        "lambda_vertical": float(loss_config["lambda_vertical"]),
    }
    if loss_type == "standard":
        return htq_reconstruction_loss(
            pred,
            batch["y_10min"],
            batch["y_mask"],
            **common,
        )
    if loss_type == "fluctuation_aware":
        return htq_fluctuation_aware_loss(
            pred,
            batch["y_10min"],
            batch["y_mask"],
            batch.get("current_hourly_y_norm"),
            lambda_weighted=float(loss_config["lambda_weighted"]),
            alpha=float(loss_config["alpha"]),
            gamma=float(loss_config["gamma"]),
            q_ref=float(loss_config["q_ref"]),
            max_weight=float(loss_config["max_weight"]),
            **common,
        )
    raise ValueError(f"Unknown loss.type {loss_type!r}; expected 'standard' or 'fluctuation_aware'")


def _add_loss_totals(total: dict[str, float], loss_parts: dict[str, Any]) -> None:
    for key, value in loss_parts.items():
        if key not in total:
            total[key] = 0.0
        if hasattr(value, "detach"):
            total[key] += float(value.detach().item())
        else:
            total[key] += float(value)


def _average_loss_totals(prefix: str, total: dict[str, float], total_batches: int) -> dict[str, float]:
    return {f"{prefix}_loss_norm_{key}": value / total_batches for key, value in total.items()}


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_config: dict[str, float | str],
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Train for one epoch using normalized-space weighted HTQ losses."""

    model.train()
    total: dict[str, float] = {}
    total_batches = 0
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model_forward(model, batch)
        loss_parts = compute_loss_parts(out["pred"], batch, loss_config)
        loss = loss_parts["loss"]
        loss.backward()
        optimizer.step()

        _add_loss_totals(total, loss_parts)
        total_batches += 1

    if total_batches == 0:
        raise ValueError("No training batches were processed")
    averaged = _average_loss_totals("train", total, total_batches)
    averaged["train_loss_norm_total"] = averaged.pop("train_loss_norm_loss")
    return averaged


def evaluate(
    model,
    loader,
    norm_stats: dict[str, Any],
    device,
    loss_config: dict[str, float | str] | None = None,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate normalized weighted losses and physical-unit m/s metrics."""

    torch = require_torch()
    model.eval()
    loss_config = loss_config or default_loss_config()
    total: dict[str, float] = {}
    total_batches = 0
    metric_sums = empty_physical_metric_sums()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            batch = move_batch(batch, device)
            out = model_forward(model, batch)
            loss_parts = compute_loss_parts(out["pred"], batch, loss_config)

            pred_ms = y_denormalize(out["pred"], norm_stats)
            target_ms = y_denormalize(batch["y_10min"], norm_stats)

            _add_loss_totals(total, loss_parts)
            total_batches += 1
            add_metric_sums(metric_sums, physical_metric_sums(pred_ms, target_ms, batch["y_mask"]))

    if total_batches == 0:
        raise ValueError("No evaluation batches were processed")

    metrics = _average_loss_totals("", total, total_batches)
    metrics = {key.removeprefix("_"): value for key, value in metrics.items()}
    metrics["loss_norm_total"] = metrics.pop("loss_norm_loss")
    metrics.update(finalize_physical_metrics(metric_sums))
    return metrics


def default_loss_config() -> dict[str, float | str]:
    """Default normalized-space HTQ training loss configuration."""

    return {
        "type": "standard",
        "lambda_l1": 1.0,
        "lambda_weighted": 0.0,
        "lambda_temporal": 0.2,
        "lambda_vertical": 0.05,
        "alpha": 1.0,
        "gamma": 1.0,
        "q_ref": 1.0,
        "max_weight": 5.0,
    }


def default_loss_weights() -> dict[str, float]:
    """Backward-compatible standard loss weights."""

    defaults = default_loss_config()
    return {
        "lambda_l1": float(defaults["lambda_l1"]),
        "lambda_temporal": float(defaults["lambda_temporal"]),
        "lambda_vertical": float(defaults["lambda_vertical"]),
    }


def loss_config_from_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, float | str]:
    """Read loss config and allow CLI overrides for standard weights."""

    loss_cfg = config.get("loss", {})
    defaults = default_loss_config()
    return {
        "type": str(loss_cfg.get("type", defaults["type"])),
        "lambda_l1": args.lambda_l1
        if args.lambda_l1 is not None
        else float(loss_cfg.get("lambda_l1", defaults["lambda_l1"])),
        "lambda_weighted": float(loss_cfg.get("lambda_weighted", defaults["lambda_weighted"])),
        "lambda_temporal": args.lambda_temporal
        if args.lambda_temporal is not None
        else float(loss_cfg.get("lambda_temporal", defaults["lambda_temporal"])),
        "lambda_vertical": args.lambda_vertical
        if args.lambda_vertical is not None
        else float(loss_cfg.get("lambda_vertical", defaults["lambda_vertical"])),
        "alpha": float(loss_cfg.get("alpha", defaults["alpha"])),
        "gamma": float(loss_cfg.get("gamma", defaults["gamma"])),
        "q_ref": float(loss_cfg.get("q_ref", defaults["q_ref"])),
        "max_weight": float(loss_cfg.get("max_weight", defaults["max_weight"])),
    }


def early_stopping_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Read early-stopping settings.

    Early stopping monitors validation MAE in physical m/s by default. Lower is
    better, matching the checkpoint selection criterion used by this script.
    """

    training_cfg = config.get("training", {})
    early_cfg = training_cfg.get("early_stopping", {})
    patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else early_cfg.get("patience")
    )
    return {
        "enabled": patience is not None and int(patience) > 0,
        "patience": int(patience) if patience is not None else None,
        "min_delta": args.early_stopping_min_delta
        if args.early_stopping_min_delta is not None
        else float(early_cfg.get("min_delta", 0.0)),
        "monitor": args.early_stopping_monitor or early_cfg.get("monitor", "MAE_ms"),
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict[str, float],
    model_config: HTQConfig,
    loss_config: dict[str, float | str],
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
            "loss_weights": {
                "lambda_l1": float(loss_config["lambda_l1"]),
                "lambda_temporal": float(loss_config["lambda_temporal"]),
                "lambda_vertical": float(loss_config["lambda_vertical"]),
            },
            "loss_config": loss_config,
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
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop if validation monitor does not improve for this many epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=None,
        help="Minimum validation improvement required to reset patience.",
    )
    parser.add_argument(
        "--early-stopping-monitor",
        default=None,
        help="Validation metric to monitor, e.g. MAE_ms or RMSE_ms. Lower is better.",
    )
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
    loss_config = loss_config_from_config(config, args)
    early_stopping = early_stopping_config(config, args)

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
        f"loss: {loss_config['type']} normalized loss "
        f"(lambda_l1={loss_config['lambda_l1']}, "
        f"lambda_weighted={loss_config['lambda_weighted']}, "
        f"lambda_temporal={loss_config['lambda_temporal']}, "
        f"lambda_vertical={loss_config['lambda_vertical']})"
    )
    print("val/test metrics: denormalized physical m/s")
    if early_stopping["enabled"]:
        print(
            "early stopping: "
            f"monitor=val_{early_stopping['monitor']} "
            f"patience={early_stopping['patience']} "
            f"min_delta={early_stopping['min_delta']}"
        )

    best_monitor = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_config,
            limit_batches=args.limit_train_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            norm_stats,
            device,
            loss_config=loss_config,
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

        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, row, model_config, loss_config)
        monitor_name = str(early_stopping["monitor"])
        if monitor_name not in val_metrics:
            raise KeyError(f"Validation metric {monitor_name!r} is not available")
        monitor_value = float(val_metrics[monitor_name])
        improved = monitor_value < best_monitor - float(early_stopping["min_delta"])
        if improved:
            best_monitor = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, row, model_config, loss_config)
        else:
            epochs_without_improvement += 1

        row["early_stopping_monitor"] = monitor_name
        row["early_stopping_monitor_value"] = monitor_value
        row["best_epoch"] = best_epoch
        row["epochs_without_improvement"] = epochs_without_improvement

        if (
            early_stopping["enabled"]
            and epochs_without_improvement >= int(early_stopping["patience"])
        ):
            stopped_early = True
            stop_reason = (
                f"val_{monitor_name} did not improve by "
                f"{early_stopping['min_delta']} for {early_stopping['patience']} epochs"
            )
            print(f"early stopping at epoch {epoch:03d}: {stop_reason}")
            break

    test_metrics = evaluate(
        model,
        test_loader,
        norm_stats,
        device,
        loss_config=loss_config,
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
        "loss_config": loss_config,
        "loss_weights": {
            "lambda_l1": float(loss_config["lambda_l1"]),
            "lambda_temporal": float(loss_config["lambda_temporal"]),
            "lambda_vertical": float(loss_config["lambda_vertical"]),
        },
        "model_config": model_config.__dict__,
        "early_stopping": {
            **early_stopping,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "best_epoch": best_epoch,
            "best_monitor_value": best_monitor,
        },
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
