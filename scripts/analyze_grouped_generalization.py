"""Compare split distributions and grouped model performance in physical space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import load_checkpoint, model_config_from_checkpoint
from scripts.train import make_loader, model_forward, move_batch
from src.data.dataset import load_norm_stats, load_split_indices, require_torch
from src.models.model_factory import build_model
from src.training.utils import get_device, y_denormalize


DEFAULT_DATASET = Path("data/datasets/paris_dufeng_1h_to_10min_12h")
DEFAULT_CHECKPOINT = Path("runs/htq_meteo_lcz_version7/best_mae.pt")
DEFAULT_JSON = Path("outputs/diagnostics/grouped_generalization_report.json")
DEFAULT_MD = Path("outputs/diagnostics/grouped_generalization_summary.md")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _stats(values: np.ndarray) -> dict[str, float | None]:
    values = _finite(values)
    if values.size == 0:
        return {key: None for key in ("mean", "std", "q50", "q90", "q95")}
    q50, q90, q95 = np.quantile(values, [0.5, 0.9, 0.95])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q50": float(q50),
        "q90": float(q90),
        "q95": float(q95),
    }


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(values)[np.asarray(mask, dtype=bool)]


def _sample_vector_rms(values: np.ndarray, both_mask: np.ndarray) -> np.ndarray:
    squared_mag = np.sum(np.asarray(values, dtype=np.float64) ** 2, axis=-1)
    valid = np.asarray(both_mask, dtype=bool)
    count = valid.sum(axis=tuple(range(1, valid.ndim)))
    total = np.where(valid, squared_mag, 0.0).sum(axis=tuple(range(1, valid.ndim)))
    return np.sqrt(total / np.maximum(count, 1))


def _temporal_corr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> tuple[float, int]:
    valid = np.asarray(mask, dtype=bool)
    valid_f = valid.astype(np.float64)
    count = valid_f.sum(axis=1)
    safe_count = np.maximum(count, 1.0)
    pred_mean = (pred * valid_f).sum(axis=1) / safe_count
    target_mean = (target * valid_f).sum(axis=1) / safe_count
    pred_centered = (pred - pred_mean[:, None]) * valid_f
    target_centered = (target - target_mean[:, None]) * valid_f
    numerator = (pred_centered * target_centered).sum(axis=1)
    pred_energy = (pred_centered**2).sum(axis=1)
    target_energy = (target_centered**2).sum(axis=1)
    denominator = np.sqrt(pred_energy * target_energy)
    usable = (count >= 2) & (pred_energy > eps) & (target_energy > eps) & (denominator > eps)
    if not np.any(usable):
        return float("nan"), 0
    return float(np.mean(numerator[usable] / denominator[usable])), int(usable.sum())


def _model_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    valid = np.asarray(mask, dtype=bool)
    error = pred - target
    error_values = _masked(error, valid)
    if error_values.size == 0:
        return {"sample_count": int(pred.shape[0]), "MAE_ms": None}

    both = valid[..., 0] & valid[..., 1]
    pred_speed = np.sqrt(np.sum(pred**2, axis=-1))
    target_speed = np.sqrt(np.sum(target**2, axis=-1))
    pred_residual = pred - current[:, None]
    true_residual = target - current[:, None]
    pred_residual_values = _masked(pred_residual, valid)
    true_residual_values = _masked(true_residual, valid)

    dy_pred = pred[:, 1:] - pred[:, :-1]
    dy_target = target[:, 1:] - target[:, :-1]
    dy_mask = valid[:, 1:] & valid[:, :-1]
    dy_error_values = _masked(dy_pred - dy_target, dy_mask)
    pred_dy_values = _masked(dy_pred, dy_mask)
    true_dy_values = _masked(dy_target, dy_mask)
    residual_acc, residual_series = _temporal_corr(pred_residual, true_residual, valid)
    gradient_acc, gradient_series = _temporal_corr(dy_pred, dy_target, dy_mask)

    true_res_std = float(np.std(true_residual_values))
    true_dy_rms = float(np.sqrt(np.mean(true_dy_values**2))) if true_dy_values.size else float("nan")
    result = {
        "sample_count": int(pred.shape[0]),
        "MAE_ms": float(np.mean(np.abs(error_values))),
        "RMSE_ms": float(np.sqrt(np.mean(error_values**2))),
        "speed_MAE_ms": float(np.mean(np.abs(_masked(pred_speed - target_speed, both)))),
        "residual_ACC": residual_acc,
        "temporal_gradient_MAE_ms": float(np.mean(np.abs(dy_error_values))) if dy_error_values.size else None,
        "temporal_gradient_ACC": gradient_acc,
        "residual_std_ratio": float(np.std(pred_residual_values) / true_res_std) if true_res_std > 1e-8 else None,
        "temporal_gradient_rms_ratio": (
            float(np.sqrt(np.mean(pred_dy_values**2)) / true_dy_rms)
            if pred_dy_values.size and true_dy_rms > 1e-8
            else None
        ),
        "valid_target_values": int(valid.sum()),
        "valid_residual_acc_series": residual_series,
        "valid_temporal_gradient_acc_series": gradient_series,
    }
    return result


def _distribution(
    target: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray,
    target_times: np.ndarray,
) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool)
    both = valid[..., 0] & valid[..., 1]
    speed = np.sqrt(np.sum(target**2, axis=-1))
    residual = target - current[:, None]
    residual_mag = np.sqrt(np.sum(residual**2, axis=-1))
    dy = residual[:, 1:] - residual[:, :-1]
    dy_both = both[:, 1:] & both[:, :-1]
    dy_mag = np.sqrt(np.sum(dy**2, axis=-1))
    times = np.asarray(target_times).astype("datetime64[m]")
    return {
        "sample_count": int(target.shape[0]),
        "valid_ratio": float(valid.mean()),
        "target_speed_ms": _stats(_masked(speed, both)),
        "residual_magnitude_ms": _stats(_masked(residual_mag, both)),
        "temporal_gradient_magnitude_ms": _stats(_masked(dy_mag, dy_both)),
        "time_start": str(times.min()) if times.size else None,
        "time_end": str(times.max()) if times.size else None,
    }


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF_winter"
    if month in (3, 4, 5):
        return "MAM_spring"
    if month in (6, 7, 8):
        return "JJA_summer"
    return "SON_autumn"


def _group_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(labels).astype(str)
    return {label: np.flatnonzero(labels == label) for label in sorted(np.unique(labels))}


def _group_report(
    labels: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for label, idx in _group_indices(labels).items():
        result[label] = {
            "distribution": _distribution(target[idx], current[idx], mask[idx], times[idx]),
            "metrics": _model_metrics(pred[idx], target[idx], current[idx], mask[idx]),
        }
    return result


def _strength_labels(values: np.ndarray, q50: float, q90: float) -> np.ndarray:
    return np.where(values <= q50, "low_le_q50", np.where(values <= q90, "mid_q50_q90", "high_gt_q90"))


def _infer_predictions(
    dataset_dir: Path,
    checkpoint_path: Path,
    splits: list[str],
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> tuple[dict[str, np.ndarray], str, int | None]:
    torch = require_torch()
    device = get_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(model_config_from_checkpoint(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm_stats = load_norm_stats(dataset_dir / "norm_stats.json")
    predictions: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for split in splits:
            chunks = []
            loader = make_loader(dataset_dir, split, batch_size, shuffle=False, num_workers=num_workers)
            for batch in loader:
                batch = move_batch(batch, device)
                output = model_forward(model, batch)
                chunks.append(y_denormalize(output["pred"], norm_stats).cpu().numpy())
            predictions[split] = np.concatenate(chunks, axis=0)
            print(f"inferred {split}: {predictions[split].shape[0]} samples")
    return predictions, str(device), checkpoint.get("epoch")


def _continuity(times: np.ndarray) -> dict[str, Any]:
    times = np.sort(np.asarray(times).astype("datetime64[m]"))
    if times.size < 2:
        return {"gaps_over_24h": 0, "max_gap_hours": 0.0}
    gaps_hours = np.diff(times).astype("timedelta64[m]").astype(np.float64) / 60.0
    return {
        "gaps_over_24h": int(np.sum(gaps_hours > 24.0)),
        "max_gap_hours": float(gaps_hours.max()),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None or not np.isfinite(value) else f"{float(value):.{digits}f}"


def _metric_table(title: str, grouped: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", "", "| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split, groups in grouped.items():
        for name, values in groups.items():
            metric = values["metrics"]
            lines.append(
                f"| {split} | {name} | {metric['sample_count']} | {_fmt(metric.get('MAE_ms'))} | "
                f"{_fmt(metric.get('residual_ACC'))} | {_fmt(metric.get('temporal_gradient_ACC'))} | "
                f"{_fmt(metric.get('residual_std_ratio'))} | {_fmt(metric.get('temporal_gradient_rms_ratio'))} |"
            )
    lines.append("")
    return lines


def _write_summary(report: dict[str, Any], path: Path) -> None:
    overall = report["overall"]
    lines = [
        "# Grouped Generalization Diagnostic",
        "",
        f"Checkpoint: `{report['checkpoint']}` (epoch {report['checkpoint_epoch']})",
        "",
        "Strength bins use train-set sample-level RMS thresholds, so train/val/test are compared against the same scale.",
        "",
        "## Overall",
        "",
        "| split | n | speed mean | residual mean | gradient mean | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, values in overall.items():
        dist = values["distribution"]
        metric = values["metrics"]
        lines.append(
            f"| {split} | {metric['sample_count']} | {_fmt(dist['target_speed_ms']['mean'])} | "
            f"{_fmt(dist['residual_magnitude_ms']['mean'])} | {_fmt(dist['temporal_gradient_magnitude_ms']['mean'])} | "
            f"{_fmt(metric['MAE_ms'])} | {_fmt(metric['residual_ACC'])} | {_fmt(metric['temporal_gradient_ACC'])} | "
            f"{_fmt(metric['residual_std_ratio'])} | {_fmt(metric['temporal_gradient_rms_ratio'])} |"
        )
    lines.append("")
    lines.extend(_metric_table("By Station", report["by_station"]))
    lines.extend(_metric_table("By Season", report["by_season"]))
    lines.extend(_metric_table("By Calendar Month", report["by_calendar_month"]))
    lines.extend(_metric_table("By Residual Strength", report["by_residual_strength"]))
    lines.extend(_metric_table("By Temporal-Gradient Strength", report["by_gradient_strength"]))

    train_m = overall["train"]["metrics"]
    val_m = overall["val"]["metrics"]
    val_stations = report["by_station"]["val"]
    worst_station = max(val_stations, key=lambda key: val_stations[key]["metrics"]["MAE_ms"])
    weakest_acc_station = min(
        val_stations,
        key=lambda key: val_stations[key]["metrics"]["residual_ACC"]
        if val_stations[key]["metrics"]["residual_ACC"] is not None
        else float("inf"),
    )
    high_res = report["by_residual_strength"]["val"].get("high_gt_q90", {})
    high_grad = report["by_gradient_strength"]["val"].get("high_gt_q90", {})
    train_dist = overall["train"]["distribution"]
    val_dist = overall["val"]["distribution"]
    speed_shift = 100.0 * (val_dist["target_speed_ms"]["mean"] / train_dist["target_speed_ms"]["mean"] - 1.0)
    residual_shift = 100.0 * (
        val_dist["residual_magnitude_ms"]["mean"] / train_dist["residual_magnitude_ms"]["mean"] - 1.0
    )
    gradient_shift = 100.0 * (
        val_dist["temporal_gradient_magnitude_ms"]["mean"]
        / train_dist["temporal_gradient_magnitude_ms"]["mean"]
        - 1.0
    )
    val_seasons = ", ".join(
        f"{name}={values['metrics']['sample_count']}" for name, values in report["by_season"]["val"].items()
    )
    lines.extend([
        "## Findings",
        "",
        f"- Overall train-to-val MAE gap is {_fmt(val_m['MAE_ms'] - train_m['MAE_ms'])} m/s; residual ACC changes from {_fmt(train_m['residual_ACC'])} to {_fmt(val_m['residual_ACC'])}.",
        f"- Validation mean wind speed shifts by {speed_shift:+.1f}% from train, while mean residual magnitude shifts by {residual_shift:+.1f}% and temporal-gradient magnitude by {gradient_shift:+.1f}%.",
        f"- Validation seasonal composition is {val_seasons}; not every season is represented.",
        f"- Highest validation MAE station is `{worst_station}` ({_fmt(val_stations[worst_station]['metrics']['MAE_ms'])} m/s). Lowest validation residual ACC station is `{weakest_acc_station}` ({_fmt(val_stations[weakest_acc_station]['metrics']['residual_ACC'])}).",
        f"- Validation high-residual group: MAE {_fmt(high_res.get('metrics', {}).get('MAE_ms'))}, residual ACC {_fmt(high_res.get('metrics', {}).get('residual_ACC'))}.",
        f"- Validation high-gradient group: gradient ACC {_fmt(high_grad.get('metrics', {}).get('temporal_gradient_ACC'))}, gradient RMS ratio {_fmt(high_grad.get('metrics', {}).get('temporal_gradient_rms_ratio'))}.",
        f"- Even on train, residual std ratio is {_fmt(train_m['residual_std_ratio'])} and gradient RMS ratio is {_fmt(train_m['temporal_gradient_rms_ratio'])}. The model is already strongly under-dispersed before considering validation shift.",
        "- A ratio below 1 means predicted residual/gradient variability is too small; a ratio above 1 means it is over-amplified.",
        "",
        "The JSON report also contains calendar-month and station-by-month tables for detailed auditing.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    predictions, device, checkpoint_epoch = _infer_predictions(
        args.dataset_dir,
        args.checkpoint,
        args.splits,
        args.batch_size,
        args.num_workers,
        args.device,
    )
    with np.load(args.dataset_dir / "dataset.npz", allow_pickle=True) as data:
        arrays = {key: data[key] for key in ("y_10min", "y_mask", "current_hourly", "station_id", "target_time_start")}

    split_data: dict[str, dict[str, np.ndarray]] = {}
    train_residual_strength = None
    train_gradient_strength = None
    for split in args.splits:
        idx = load_split_indices(args.dataset_dir, split)
        target = arrays["y_10min"][idx].astype(np.float64)
        mask = arrays["y_mask"][idx].astype(bool)
        current = arrays["current_hourly"][idx].astype(np.float64)
        residual = target - current[:, None]
        both = mask[..., 0] & mask[..., 1]
        dy = residual[:, 1:] - residual[:, :-1]
        dy_both = both[:, 1:] & both[:, :-1]
        residual_strength = _sample_vector_rms(residual, both)
        gradient_strength = _sample_vector_rms(dy, dy_both)
        if split == "train":
            train_residual_strength = residual_strength
            train_gradient_strength = gradient_strength
        times = arrays["target_time_start"][idx].astype("datetime64[m]")
        station = arrays["station_id"][idx].astype(str)
        month = np.array([str(value)[5:7] for value in times])
        year_month = np.array([str(value)[:7] for value in times])
        seasons = np.array([_season(int(value)) for value in month])
        split_data[split] = {
            "pred": predictions[split],
            "target": target,
            "mask": mask,
            "current": current,
            "times": times,
            "station": station,
            "month": month,
            "year_month": year_month,
            "season": seasons,
            "residual_strength": residual_strength,
            "gradient_strength": gradient_strength,
        }

    if train_residual_strength is None or train_gradient_strength is None:
        raise ValueError("The train split is required to define common strength thresholds")
    residual_q50, residual_q90 = np.quantile(train_residual_strength, [0.5, 0.9])
    gradient_q50, gradient_q90 = np.quantile(train_gradient_strength, [0.5, 0.9])

    report: dict[str, Any] = {
        "dataset_dir": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "device": device,
        "strength_thresholds_ms": {
            "residual_sample_rms_q50": float(residual_q50),
            "residual_sample_rms_q90": float(residual_q90),
            "gradient_sample_rms_q50": float(gradient_q50),
            "gradient_sample_rms_q90": float(gradient_q90),
        },
        "overall": {},
        "by_station": {},
        "by_calendar_month": {},
        "by_year_month": {},
        "by_season": {},
        "by_station_calendar_month": {},
        "by_residual_strength": {},
        "by_gradient_strength": {},
        "continuity": {},
    }
    for split, values in split_data.items():
        pred, target, current, mask, times = (values[key] for key in ("pred", "target", "current", "mask", "times"))
        report["overall"][split] = {
            "distribution": _distribution(target, current, mask, times),
            "metrics": _model_metrics(pred, target, current, mask),
        }
        report["by_station"][split] = _group_report(values["station"], pred, target, current, mask, times)
        report["by_calendar_month"][split] = _group_report(values["month"], pred, target, current, mask, times)
        report["by_year_month"][split] = _group_report(values["year_month"], pred, target, current, mask, times)
        report["by_season"][split] = _group_report(values["season"], pred, target, current, mask, times)
        station_month = np.char.add(np.char.add(values["station"], "|"), values["month"])
        report["by_station_calendar_month"][split] = _group_report(station_month, pred, target, current, mask, times)
        residual_labels = _strength_labels(values["residual_strength"], residual_q50, residual_q90)
        gradient_labels = _strength_labels(values["gradient_strength"], gradient_q50, gradient_q90)
        report["by_residual_strength"][split] = _group_report(residual_labels, pred, target, current, mask, times)
        report["by_gradient_strength"][split] = _group_report(gradient_labels, pred, target, current, mask, times)
        report["continuity"][split] = {
            station: _continuity(times[idx]) for station, idx in _group_indices(values["station"]).items()
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    _write_summary(report, args.output_md)
    print(f"wrote: {args.output_json}")
    print(f"wrote: {args.output_md}")


if __name__ == "__main__":
    main()
