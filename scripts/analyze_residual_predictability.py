"""Diagnose whether hourly context can predict 10min residuals.

This script does not touch model/training code. It reads generated dataset
arrays in physical m/s space and writes:

- outputs/diagnostics/residual_predictability_report.json
- outputs/diagnostics/residual_predictability_summary.md
"""

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

from src.data.dataset import load_split_indices


DEFAULT_DATASET_DIR = Path("data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1")
DEFAULT_JSON = Path("outputs/diagnostics/residual_predictability_report.json")
DEFAULT_MD = Path("outputs/diagnostics/residual_predictability_summary.md")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _safe_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)
    if int(valid.sum()) < 3:
        return float("nan")
    x = x[valid]
    y = y[valid]
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= eps:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(values)[np.asarray(mask, dtype=bool)]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: float("nan") for k in ("q50", "q75", "q90", "q95")}
    q50, q75, q90, q95 = np.quantile(values, [0.5, 0.75, 0.9, 0.95])
    return {"q50": float(q50), "q75": float(q75), "q90": float(q90), "q95": float(q95)}


def _residual_stats(residual: np.ndarray, residual_mag: np.ndarray, mask: np.ndarray, station_id: np.ndarray) -> dict[str, Any]:
    both_mask = mask[..., 0] & mask[..., 1]
    residual_mag_values = _masked_values(residual_mag, both_mask)
    channel_values = _masked_values(residual, mask)
    stats = {
        "residual_mag": {
            "mean": float(np.mean(residual_mag_values)),
            "std": float(np.std(residual_mag_values)),
            **_quantiles(residual_mag_values),
        },
        "residual_channel_values": {
            "mean": float(np.mean(channel_values)),
            "std": float(np.std(channel_values)),
        },
        "per_height_residual_std": [],
        "per_target_bin_residual_std": [],
        "per_station_residual_std": {},
    }
    for h in range(residual.shape[2]):
        stats["per_height_residual_std"].append(float(np.std(_masked_values(residual[:, :, h, :], mask[:, :, h, :]))))
    for t in range(residual.shape[1]):
        stats["per_target_bin_residual_std"].append(float(np.std(_masked_values(residual[:, t, :, :], mask[:, t, :, :]))))
    for sid in sorted(set(str(v) for v in station_id)):
        idx = np.asarray([str(v) == sid for v in station_id])
        stats["per_station_residual_std"][sid] = float(np.std(_masked_values(residual[idx], mask[idx])))
    return stats


def _trend_correlations(
    x_hourly: np.ndarray,
    current_hourly: np.ndarray,
    residual: np.ndarray,
    residual_mag: np.ndarray,
    y_mask: np.ndarray,
) -> dict[str, Any]:
    trend_1h = x_hourly[:, -1] - x_hourly[:, -2]  # [N, H, C]
    trend_3h = x_hourly[:, -1] - x_hourly[:, -4]
    current_speed = np.sqrt(np.sum(current_hourly * current_hourly, axis=-1))  # [N, H]
    vertical_shear = current_hourly[:, -1, :] - current_hourly[:, 0, :]  # [N, C]

    predictors = {
        "trend_1h": trend_1h[:, None, :, :],
        "trend_3h": trend_3h[:, None, :, :],
        "current_speed": current_speed[:, None, :, None],
        "vertical_shear": vertical_shear[:, None, None, :],
    }
    result: dict[str, Any] = {}
    both_mask = y_mask[..., 0] & y_mask[..., 1]
    for name, pred in predictors.items():
        pred_b = np.broadcast_to(pred, residual.shape if pred.shape[-1] == 2 else residual_mag[..., None].shape)
        if pred_b.shape[-1] == 1:
            pred_scalar = pred_b[..., 0]
            result[name] = {
                "overall_vs_residual_mag": _safe_corr(pred_scalar, residual_mag, both_mask),
                "per_height_vs_residual_mag": [
                    _safe_corr(pred_scalar[:, :, h], residual_mag[:, :, h], both_mask[:, :, h])
                    for h in range(residual.shape[2])
                ],
                "per_target_bin_vs_residual_mag": [
                    _safe_corr(pred_scalar[:, t, :], residual_mag[:, t, :], both_mask[:, t, :])
                    for t in range(residual.shape[1])
                ],
            }
        else:
            result[name] = {
                "overall_u": _safe_corr(pred_b[..., 0], residual[..., 0], y_mask[..., 0]),
                "overall_v": _safe_corr(pred_b[..., 1], residual[..., 1], y_mask[..., 1]),
                "per_height_u": [
                    _safe_corr(pred_b[:, :, h, 0], residual[:, :, h, 0], y_mask[:, :, h, 0])
                    for h in range(residual.shape[2])
                ],
                "per_height_v": [
                    _safe_corr(pred_b[:, :, h, 1], residual[:, :, h, 1], y_mask[:, :, h, 1])
                    for h in range(residual.shape[2])
                ],
                "per_target_bin_u": [
                    _safe_corr(pred_b[:, t, :, 0], residual[:, t, :, 0], y_mask[:, t, :, 0])
                    for t in range(residual.shape[1])
                ],
                "per_target_bin_v": [
                    _safe_corr(pred_b[:, t, :, 1], residual[:, t, :, 1], y_mask[:, t, :, 1])
                    for t in range(residual.shape[1])
                ],
            }
    return result


def _temporal_acc(pred_residual: np.ndarray, true_residual: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> float:
    values = []
    for c in range(true_residual.shape[-1]):
        for h in range(true_residual.shape[2]):
            p = pred_residual[:, :, h, c]
            y = true_residual[:, :, h, c]
            m = mask[:, :, h, c]
            for i in range(true_residual.shape[0]):
                valid = m[i] & np.isfinite(p[i]) & np.isfinite(y[i])
                if int(valid.sum()) < 2:
                    continue
                pc = p[i, valid] - p[i, valid].mean()
                yc = y[i, valid] - y[i, valid].mean()
                denom = np.sqrt(np.sum(pc * pc) * np.sum(yc * yc))
                if denom > eps:
                    values.append(float(np.sum(pc * yc) / denom))
    return float(np.mean(values)) if values else float("nan")


def _metrics_from_residual_prediction(
    pred_residual: np.ndarray,
    true_residual: np.ndarray,
    y_mask: np.ndarray,
    high_mask: np.ndarray | None = None,
) -> dict[str, float]:
    mask = y_mask if high_mask is None else (y_mask & high_mask[..., None])
    err = pred_residual - true_residual
    valid_err = _masked_values(err, mask)
    if valid_err.size == 0:
        return {
            "MAE_ms": float("nan"),
            "RMSE_ms": float("nan"),
            "residual_MAE": float("nan"),
            "residual_ACC": float("nan"),
            "temporal_gradient_MAE_ms": float("nan"),
            "temporal_gradient_ACC": float("nan"),
            "residual_std_ratio": float("nan"),
        }
    mae = float(np.mean(np.abs(valid_err)))
    rmse = float(np.sqrt(np.mean(valid_err * valid_err)))
    dy_pred = pred_residual[:, 1:] - pred_residual[:, :-1]
    dy_true = true_residual[:, 1:] - true_residual[:, :-1]
    dy_mask = mask[:, 1:] & mask[:, :-1]
    dy_err = _masked_values(dy_pred - dy_true, dy_mask)
    pred_values = _masked_values(pred_residual, mask)
    true_values = _masked_values(true_residual, mask)
    pred_std = float(np.std(pred_values)) if pred_values.size else float("nan")
    true_std = float(np.std(true_values)) if true_values.size else float("nan")
    return {
        "MAE_ms": mae,
        "RMSE_ms": rmse,
        "residual_MAE": mae,
        "residual_ACC": _temporal_acc(pred_residual, true_residual, mask),
        "temporal_gradient_MAE_ms": float(np.mean(np.abs(dy_err))) if dy_err.size else float("nan"),
        "temporal_gradient_ACC": _temporal_acc(dy_pred, dy_true, dy_mask),
        "residual_std_ratio": pred_std / true_std if true_std > 1e-8 else float("nan"),
    }


def _high_fluctuation_metrics(pred_residual: np.ndarray, true_residual: np.ndarray, residual_mag: np.ndarray, y_mask: np.ndarray) -> dict[str, Any]:
    both_mask = y_mask[..., 0] & y_mask[..., 1]
    mag_values = _masked_values(residual_mag, both_mask)
    q50, q90 = np.quantile(mag_values, [0.5, 0.9])
    subsets = {
        "low_le_q50": residual_mag <= q50,
        "mid_q50_to_q90": (residual_mag > q50) & (residual_mag <= q90),
        "high_gt_q90": residual_mag > q90,
    }
    return {
        name: _metrics_from_residual_prediction(pred_residual, true_residual, y_mask, subset_mask)
        for name, subset_mask in subsets.items()
    }


def _template_baseline(
    train_station: np.ndarray,
    train_residual: np.ndarray,
    train_mask: np.ndarray,
    eval_station: np.ndarray,
    eval_residual: np.ndarray,
    eval_mask: np.ndarray,
    eval_residual_mag: np.ndarray,
) -> dict[str, Any]:
    global_template = _masked_mean(train_residual, train_mask, axis=0)
    station_templates = {}
    for sid in sorted(set(str(v) for v in train_station)):
        idx = np.asarray([str(v) == sid for v in train_station])
        station_templates[sid] = _masked_mean(train_residual[idx], train_mask[idx], axis=0)
    pred = np.empty_like(eval_residual)
    for i, sid in enumerate(eval_station):
        pred[i] = station_templates.get(str(sid), global_template)
    return {
        "metrics": _metrics_from_residual_prediction(pred, eval_residual, eval_mask),
        "high_fluctuation": _high_fluctuation_metrics(pred, eval_residual, eval_residual_mag, eval_mask),
    }


def _masked_mean(values: np.ndarray, mask: np.ndarray, axis=0, eps: float = 1e-8) -> np.ndarray:
    weights = mask.astype(np.float32)
    return (values * weights).sum(axis=axis) / np.maximum(weights.sum(axis=axis), eps)


def _raw_knn_features(x_hourly: np.ndarray, x_mask: np.ndarray) -> np.ndarray:
    valid = x_mask.astype(bool)
    values = np.where(valid, x_hourly, 0.0).astype(np.float32)
    return values.reshape(values.shape[0], -1)


def _standardize_train_eval_features(
    train_features: np.ndarray,
    eval_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (
        ((train_features - mean) / std).astype(np.float32),
        ((eval_features - mean) / std).astype(np.float32),
    )


def _topk_neighbor_indices(
    train_features: np.ndarray,
    eval_features: np.ndarray,
    *,
    max_k: int,
    chunk_size: int,
    train_station: np.ndarray | None = None,
    eval_station: np.ndarray | None = None,
    same_station_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-K train indices and distances for each eval sample."""

    max_k = min(max_k, train_features.shape[0])
    top_indices = np.full((eval_features.shape[0], max_k), -1, dtype=np.int64)
    top_distances = np.full((eval_features.shape[0], max_k), np.inf, dtype=np.float32)

    if same_station_only:
        if train_station is None or eval_station is None:
            raise ValueError("station arrays are required for same_station_only=True")
        for sid in sorted(set(str(v) for v in eval_station)):
            eval_local = np.where(np.asarray([str(v) == sid for v in eval_station]))[0]
            train_local = np.where(np.asarray([str(v) == sid for v in train_station]))[0]
            if train_local.size == 0:
                continue
            local_k = min(max_k, train_local.size)
            local_idx, local_dist = _topk_neighbor_indices(
                train_features[train_local],
                eval_features[eval_local],
                max_k=local_k,
                chunk_size=chunk_size,
                same_station_only=False,
            )
            top_indices[np.ix_(eval_local, np.arange(local_k))] = train_local[local_idx]
            top_distances[np.ix_(eval_local, np.arange(local_k))] = local_dist
        return top_indices, top_distances

    train_norm = np.sum(train_features * train_features, axis=1)[None, :]
    for start in range(0, eval_features.shape[0], chunk_size):
        end = min(start + chunk_size, eval_features.shape[0])
        chunk = eval_features[start:end]
        distances = np.sum(chunk * chunk, axis=1, keepdims=True) + train_norm - 2.0 * chunk @ train_features.T
        local_k = min(max_k, train_features.shape[0])
        idx = np.argpartition(distances, kth=local_k - 1, axis=1)[:, :local_k]
        dist = np.take_along_axis(distances, idx, axis=1)
        order = np.argsort(dist, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        dist = np.take_along_axis(dist, order, axis=1)
        top_indices[start:end, :local_k] = idx
        top_distances[start:end, :local_k] = dist.astype(np.float32)
    return top_indices, top_distances


def _knn_baselines(
    train_residual: np.ndarray,
    train_mask: np.ndarray,
    eval_residual: np.ndarray,
    eval_mask: np.ndarray,
    eval_residual_mag: np.ndarray,
    top_indices: np.ndarray,
    *,
    ks: list[int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    pred_sums = {k: np.zeros_like(eval_residual, dtype=np.float32) for k in ks}
    train_residual_filled = np.where(train_mask, train_residual, 0.0).astype(np.float32)
    train_mask_f = train_mask.astype(np.float32)
    global_template = _masked_mean(train_residual, train_mask, axis=0)
    for start in range(eval_residual.shape[0]):
        for k in ks:
            neighbor_idx = top_indices[start, :k]
            neighbor_idx = neighbor_idx[neighbor_idx >= 0]
            if neighbor_idx.size == 0:
                pred_sums[k][start] = global_template
                continue
            residual_sum = train_residual_filled[neighbor_idx].sum(axis=0)
            mask_sum = train_mask_f[neighbor_idx].sum(axis=0)
            pred = residual_sum / np.maximum(mask_sum, 1e-8)
            pred = np.where(mask_sum > 0, pred, global_template)
            pred_sums[k][start] = pred.astype(np.float32)
    for k in ks:
        pred = pred_sums[k]
        results[f"k{k}"] = {
            "metrics": _metrics_from_residual_prediction(pred, eval_residual, eval_mask),
            "high_fluctuation": _high_fluctuation_metrics(pred, eval_residual, eval_residual_mag, eval_mask),
        }
    return results


def _candidate_residual_mae(
    candidate_residual: np.ndarray,
    true_residual: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return masked residual MAE for candidate residuals [K,T,H,C]."""

    candidate_error = np.abs(candidate_residual - true_residual[None, :, :, :])
    valid = mask.astype(np.float32)[None, :, :, :]
    return (candidate_error * valid).sum(axis=(1, 2, 3)) / (valid.sum(axis=(1, 2, 3)) + eps)


def _knn_oracle_best_of_k_baselines(
    train_residual: np.ndarray,
    eval_residual: np.ndarray,
    eval_mask: np.ndarray,
    eval_residual_mag: np.ndarray,
    top_indices: np.ndarray,
    top_distances: np.ndarray,
    *,
    ks: list[int],
) -> dict[str, Any]:
    """Oracle best-of-K residual baseline using validation target for selection."""

    results: dict[str, Any] = {}
    n_eval = eval_residual.shape[0]
    for k in ks:
        pred = np.zeros_like(eval_residual, dtype=np.float32)
        selected_indices = np.full((n_eval,), -1, dtype=np.int64)
        selected_distances = np.full((n_eval,), np.nan, dtype=np.float32)
        selected_mae = np.full((n_eval,), np.nan, dtype=np.float32)
        for i in range(n_eval):
            candidate_idx = top_indices[i, :k]
            candidate_dist = top_distances[i, :k]
            valid = candidate_idx >= 0
            candidate_idx = candidate_idx[valid]
            candidate_dist = candidate_dist[valid]
            if candidate_idx.size == 0:
                continue
            candidate_residual = train_residual[candidate_idx]
            candidate_mae = _candidate_residual_mae(candidate_residual, eval_residual[i], eval_mask[i])
            best_local = int(np.argmin(candidate_mae))
            selected_indices[i] = int(candidate_idx[best_local])
            selected_distances[i] = float(candidate_dist[best_local])
            selected_mae[i] = float(candidate_mae[best_local])
            pred[i] = candidate_residual[best_local]
        finite_dist = selected_distances[np.isfinite(selected_distances)]
        finite_mae = selected_mae[np.isfinite(selected_mae)]
        results[f"k{k}"] = {
            "metrics": _metrics_from_residual_prediction(pred, eval_residual, eval_mask),
            "high_fluctuation": _high_fluctuation_metrics(pred, eval_residual, eval_residual_mag, eval_mask),
            "mean_selected_neighbor_distance": float(np.mean(finite_dist)) if finite_dist.size else float("nan"),
            "median_selected_neighbor_distance": float(np.median(finite_dist)) if finite_dist.size else float("nan"),
            "mean_selected_candidate_mae": float(np.mean(finite_mae)) if finite_mae.size else float("nan"),
            "median_selected_candidate_mae": float(np.median(finite_mae)) if finite_mae.size else float("nan"),
            "selected_neighbor_index": selected_indices,
            "selected_neighbor_distance": selected_distances,
            "selected_candidate_mae": selected_mae,
        }
    return results


def _sample_indices(indices: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or max_samples >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=max_samples, replace=False)
    return np.sort(selected)


def _write_summary(report: dict[str, Any], path: Path) -> None:
    residual_mag = report["true_residual_statistics"]["residual_mag"]
    trend = report["hourly_trend_correlation"]
    repeat = report["repeat_current_baseline"]["metrics"]
    template = report["template_baseline"]["metrics"]
    knn = report["knn_baseline"]
    oracle = report.get("knn_oracle_baseline", {})
    oracle_same_station = report.get("knn_oracle_same_station_baseline", {})
    best_knn_name, best_knn = min(knn.items(), key=lambda kv: kv[1]["metrics"]["MAE_ms"])
    best_knn_metrics = best_knn["metrics"]
    oracle_k20 = oracle.get("k20", {}).get("metrics", {})
    oracle_best_name, oracle_best = (None, None)
    if oracle:
        oracle_best_name, oracle_best = min(oracle.items(), key=lambda kv: kv[1]["metrics"]["MAE_ms"])
    max_trend_corr = _max_abs_corr(trend)
    near_zero = residual_mag["q75"] < 0.5
    weak_trend = max_trend_corr < 0.15
    knn_gain = repeat["MAE_ms"] - best_knn_metrics["MAE_ms"]
    model_space = "weak" if knn_gain < 0.02 else "present"

    lines = [
        "# Residual Predictability Diagnostics",
        "",
        "## Main Findings",
        "",
        f"- Residual magnitude median/q75/q90/q95: {residual_mag['q50']:.4f}, {residual_mag['q75']:.4f}, {residual_mag['q90']:.4f}, {residual_mag['q95']:.4f} m/s.",
        f"- Residuals are {'mostly close to zero' if near_zero else 'not mostly close to zero'} by the q75<0.5 m/s heuristic.",
        f"- Strongest absolute hourly-context correlation found: {max_trend_corr:.4f}. This is {'weak' if weak_trend else 'moderate/strong'} by a 0.15 threshold.",
        f"- Repeat-current residual MAE: {repeat['MAE_ms']:.4f} m/s.",
        f"- Station residual template MAE: {template['MAE_ms']:.4f} m/s, residual_ACC: {template['residual_ACC']:.4f}.",
        f"- Best KNN baseline: {best_knn_name}, MAE: {best_knn_metrics['MAE_ms']:.4f} m/s, residual_ACC: {best_knn_metrics['residual_ACC']:.4f}, temporal_gradient_ACC: {best_knn_metrics['temporal_gradient_ACC']:.4f}.",
        f"- KNN improvement over repeat-current MAE: {knn_gain:.4f} m/s, suggesting hourly-context predictability is {model_space}.",
        "",
        "## Interpretation",
        "",
    ]
    if weak_trend and knn_gain < 0.02:
        lines.append("- Hourly context appears to contain weak information about 10min residuals. If the trained model also has low ACC, the bottleneck may be input predictability, not only architecture.")
    elif knn_gain >= 0.02:
        lines.append("- KNN beats repeat-current by a visible margin, so there is exploitable context information. If HTQ underperforms KNN on residual/gradient ACC, model/loss choices still have room.")
    else:
        lines.append("- Diagnostics are mixed. Hourly predictors have limited linear signal, but non-linear/local analog behavior may still help.")
    lines.extend([
        "",
        "## Baseline Metrics",
        "",
        "| baseline | MAE_ms | RMSE_ms | residual_ACC | temporal_gradient_ACC | residual_std_ratio |",
        "|---|---:|---:|---:|---:|---:|",
        _metric_row("repeat_current", repeat),
        _metric_row("template", template),
    ])
    for name, data in knn.items():
        lines.append(_metric_row(name, data["metrics"]))
    if oracle:
        lines.extend([
            "",
            "## KNN Oracle Best-of-K Upper-Bound Diagnostic",
            "",
            "This baseline uses the validation target to choose the best residual among K nearest train candidates. It is not a deployable model; it is an optimistic upper-bound diagnostic for the current hourly-context feature space.",
            "",
            "If oracle residual_ACC is still low, even similar hourly contexts do not reliably match the true 10min residual phase. If oracle is far above the Transformer, train-set residual modes exist but the model/loss has not fully learned them. If larger K helps a lot, stronger representation or retrieval features may be useful.",
            "",
            "| baseline | MAE_ms | RMSE_ms | residual_ACC | temporal_gradient_ACC | residual_std_ratio |",
            "|---|---:|---:|---:|---:|---:|",
            _metric_row("repeat_current", repeat),
            _metric_row("template", template),
            _metric_row(f"knn_mean_{best_knn_name}", best_knn_metrics),
        ])
        for name, data in oracle.items():
            lines.append(_metric_row(f"knn_oracle_best_of_{name[1:]}", data["metrics"]))
        if oracle_same_station:
            lines.extend([
                "",
                "Same-station-only oracle:",
                "",
                "| baseline | MAE_ms | RMSE_ms | residual_ACC | temporal_gradient_ACC | residual_std_ratio |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for name, data in oracle_same_station.items():
                lines.append(_metric_row(f"knn_oracle_same_station_best_of_{name[1:]}", data["metrics"]))
        lines.extend(["", "### Automatic Oracle Interpretation", ""])
        if oracle_k20:
            acc20 = oracle_k20.get("residual_ACC", float("nan"))
            grad20 = oracle_k20.get("temporal_gradient_ACC", float("nan"))
            if np.isfinite(acc20) and acc20 <= 0.25 and (not np.isfinite(grad20) or abs(grad20) < 0.05):
                lines.append("- Case A: best-of-20 oracle residual_ACC is <= 0.25 and temporal_gradient_ACC is near 0. Current hourly context weakly constrains 10min residual phase, so deterministic predictability appears low.")
            else:
                lines.append("- Case A not triggered: best-of-20 oracle has non-trivial phase signal or gradient correlation.")
            if np.isfinite(acc20):
                lines.append("- Case B requires comparing oracle residual_ACC to the current Transformer residual_ACC. If oracle exceeds the model by >0.15, model/loss likely has substantial room.")
        if "k20" in oracle and ("k50" in oracle or "k100" in oracle):
            acc20 = oracle["k20"]["metrics"]["residual_ACC"]
            later = [
                oracle[k]["metrics"]["residual_ACC"]
                for k in ("k50", "k100")
                if k in oracle and np.isfinite(oracle[k]["metrics"]["residual_ACC"])
            ]
            if later and max(later) - acc20 > 0.05:
                lines.append("- Case C: oracle improves clearly beyond K=20. Residual modes may exist but require better retrieval/features.")
            else:
                lines.append("- Case C not triggered: larger K does not substantially raise residual_ACC over K=20.")
        if oracle_best is not None:
            oracle_mae_gain = repeat["MAE_ms"] - oracle_best["metrics"]["MAE_ms"]
            oracle_acc = oracle_best["metrics"]["residual_ACC"]
            if oracle_mae_gain > 0.05 and (not np.isfinite(oracle_acc) or oracle_acc < 0.15):
                lines.append("- Case D: oracle improves MAE but residual_ACC remains low, suggesting amplitude is easier than temporal phase.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_row(name: str, metrics: dict[str, float]) -> str:
    return (
        f"| {name} | {metrics['MAE_ms']:.4f} | {metrics['RMSE_ms']:.4f} | "
        f"{metrics['residual_ACC']:.4f} | {metrics['temporal_gradient_ACC']:.4f} | "
        f"{metrics['residual_std_ratio']:.4f} |"
    )


def _max_abs_corr(table: dict[str, Any]) -> float:
    values = []
    def walk(v):
        if isinstance(v, dict):
            for vv in v.values():
                walk(vv)
        elif isinstance(v, list):
            for vv in v:
                walk(vv)
        elif isinstance(v, (float, int)) and np.isfinite(v):
            values.append(abs(float(v)))
    walk(table)
    return max(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--json-output", default=str(DEFAULT_JSON))
    parser.add_argument("--summary-output", default=str(DEFAULT_MD))
    parser.add_argument("--knn-k", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--oracle-k", nargs="+", type=int, default=[5, 10, 20, 50, 100])
    parser.add_argument("--knn-chunk-size", type=int, default=256)
    parser.add_argument("--knn-feature-type", default="hourly_context_flatten", choices=["hourly_context_flatten"])
    parser.add_argument("--no-standardize-features", action="store_true")
    parser.add_argument("--skip-same-station-oracle", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    with np.load(dataset_dir / "dataset.npz", allow_pickle=True) as data:
        train_idx = _sample_indices(load_split_indices(dataset_dir, args.train_split), args.max_train_samples, args.seed)
        eval_idx = _sample_indices(load_split_indices(dataset_dir, args.eval_split), args.max_eval_samples, args.seed + 1)
        x_train = data["x_hourly"][train_idx].astype(np.float32)
        x_eval = data["x_hourly"][eval_idx].astype(np.float32)
        x_mask_train = data["x_mask"][train_idx].astype(bool)
        x_mask_eval = data["x_mask"][eval_idx].astype(bool)
        y_train = data["y_10min"][train_idx].astype(np.float32)
        y_eval = data["y_10min"][eval_idx].astype(np.float32)
        current_train = data["current_hourly"][train_idx].astype(np.float32)
        current_eval = data["current_hourly"][eval_idx].astype(np.float32)
        mask_train = data["y_mask"][train_idx].astype(bool)
        mask_eval = data["y_mask"][eval_idx].astype(bool)
        station_train = data["station_id"][train_idx]
        station_eval = data["station_id"][eval_idx]

    train_residual = y_train - current_train[:, None, :, :]
    eval_residual = y_eval - current_eval[:, None, :, :]
    eval_residual_mag = np.sqrt(np.sum(eval_residual * eval_residual, axis=-1))
    train_residual_mag = np.sqrt(np.sum(train_residual * train_residual, axis=-1))

    repeat_residual = np.zeros_like(eval_residual, dtype=np.float32)
    repeat = {
        "metrics": _metrics_from_residual_prediction(repeat_residual, eval_residual, mask_eval),
        "high_fluctuation": _high_fluctuation_metrics(repeat_residual, eval_residual, eval_residual_mag, mask_eval),
    }
    template = _template_baseline(station_train, train_residual, mask_train, station_eval, eval_residual, mask_eval, eval_residual_mag)
    train_features_raw = _raw_knn_features(x_train, x_mask_train)
    eval_features_raw = _raw_knn_features(x_eval, x_mask_eval)
    if args.no_standardize_features:
        train_features, eval_features = train_features_raw.astype(np.float32), eval_features_raw.astype(np.float32)
    else:
        train_features, eval_features = _standardize_train_eval_features(train_features_raw, eval_features_raw)
    mean_ks = sorted(k for k in set(args.knn_k) if k <= train_features.shape[0])
    oracle_ks = sorted(k for k in set(args.oracle_k) if k <= train_features.shape[0])
    max_k = max([*mean_ks, *oracle_ks])
    top_indices, top_distances = _topk_neighbor_indices(
        train_features,
        eval_features,
        max_k=max_k,
        chunk_size=args.knn_chunk_size,
    )
    knn = _knn_baselines(
        train_residual,
        mask_train,
        eval_residual,
        mask_eval,
        eval_residual_mag,
        top_indices,
        ks=mean_ks,
    )
    oracle = _knn_oracle_best_of_k_baselines(
        train_residual,
        eval_residual,
        mask_eval,
        eval_residual_mag,
        top_indices,
        top_distances,
        ks=oracle_ks,
    )
    oracle_same_station = {}
    if not args.skip_same_station_oracle:
        top_indices_station, top_distances_station = _topk_neighbor_indices(
            train_features,
            eval_features,
            max_k=max(oracle_ks),
            chunk_size=args.knn_chunk_size,
            train_station=station_train,
            eval_station=station_eval,
            same_station_only=True,
        )
        oracle_same_station = _knn_oracle_best_of_k_baselines(
            train_residual,
            eval_residual,
            mask_eval,
            eval_residual_mag,
            top_indices_station,
            top_distances_station,
            ks=oracle_ks,
        )

    report = {
        "dataset_dir": str(dataset_dir),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "train_samples": int(len(train_idx)),
        "eval_samples": int(len(eval_idx)),
        "true_residual_statistics": _residual_stats(eval_residual, eval_residual_mag, mask_eval, station_eval),
        "train_true_residual_statistics": _residual_stats(train_residual, train_residual_mag, mask_train, station_train),
        "hourly_trend_correlation": _trend_correlations(x_eval, current_eval, eval_residual, eval_residual_mag, mask_eval),
        "repeat_current_baseline": repeat,
        "template_baseline": template,
        "knn_baseline": knn,
        "knn_config": {
            "feature_type": args.knn_feature_type,
            "standardize_features": not args.no_standardize_features,
            "same_station_only_default": False,
            "knn_mean_k": mean_ks,
            "oracle_k": oracle_ks,
        },
        "knn_oracle_baseline": oracle,
        "knn_oracle_same_station_baseline": oracle_same_station,
    }

    json_path = Path(args.json_output)
    md_path = Path(args.summary_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    _write_summary(report, md_path)
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")


if __name__ == "__main__":
    main()
