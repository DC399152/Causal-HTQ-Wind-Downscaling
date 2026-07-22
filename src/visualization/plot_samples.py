"""Plot sample-level 10min reconstruction trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def plot_sample_timeseries(
    *,
    target,
    pred,
    repeat,
    y_mask,
    height_values: Sequence[float],
    output_path: str | Path,
    title: str,
    target_offsets_minutes: Sequence[int] = (0, 10, 20, 30, 40, 50),
) -> None:
    """Plot truth, HTQ prediction, and repeat baseline for one sample.

    Shapes:
    - target: [T_out, H, 2], physical m/s
    - pred: [T_out, H, 2], physical m/s
    - repeat: [T_out, H, 2], physical m/s
    - y_mask: [T_out, H, 2], True=valid target
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required for plotting samples") from exc

    target_np = _to_numpy(target).astype(float)
    pred_np = _to_numpy(pred).astype(float)
    repeat_np = _to_numpy(repeat).astype(float)
    mask_np = _to_numpy(y_mask).astype(bool)
    heights = list(height_values)

    if target_np.shape != pred_np.shape or target_np.shape != repeat_np.shape:
        raise ValueError("target, pred, and repeat must share shape [T_out, H, 2]")
    if target_np.shape != mask_np.shape:
        raise ValueError("y_mask must have shape [T_out, H, 2]")
    if target_np.ndim != 3 or target_np.shape[-1] != 2:
        raise ValueError("Expected arrays with shape [T_out, H, 2]")

    num_heights = target_np.shape[1]
    if len(heights) != num_heights:
        raise ValueError("height_values length must match H")

    channel_names = ("u", "v")
    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_heights,
        figsize=(3.2 * num_heights, 6.4),
        sharex=True,
        constrained_layout=True,
    )
    offsets = np.asarray(target_offsets_minutes)

    for channel_idx, channel_name in enumerate(channel_names):
        for height_idx, height in enumerate(heights):
            ax = axes[channel_idx, height_idx]
            truth = target_np[:, height_idx, channel_idx].copy()
            truth[~mask_np[:, height_idx, channel_idx]] = np.nan
            ax.plot(offsets, truth, marker="o", linewidth=2.0, label="truth")
            ax.plot(offsets, pred_np[:, height_idx, channel_idx], marker="s", linewidth=1.7, label="HTQ pred")
            ax.plot(offsets, repeat_np[:, height_idx, channel_idx], linestyle="--", linewidth=1.5, label="repeat")
            ax.set_title(f"{channel_name}, {height:g} m")
            ax.grid(True, alpha=0.3)
            if height_idx == 0:
                ax.set_ylabel("m/s")
            if channel_idx == 1:
                ax.set_xlabel("minute offset")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sample_with_hourly_context(
    *,
    context,
    target,
    pred,
    repeat,
    x_mask,
    y_mask,
    height_values: Sequence[float],
    output_path: str | Path,
    title: str,
    target_offsets_minutes: Sequence[int] = (0, 10, 20, 30, 40, 50),
) -> None:
    """Plot hourly context followed by target-hour truth and predictions."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required for plotting samples") from exc

    context_np = _to_numpy(context).astype(float)
    target_np = _to_numpy(target).astype(float)
    pred_np = _to_numpy(pred).astype(float)
    repeat_np = _to_numpy(repeat).astype(float)
    x_mask_np = _to_numpy(x_mask).astype(bool)
    y_mask_np = _to_numpy(y_mask).astype(bool)
    heights = list(height_values)

    if context_np.ndim != 3 or context_np.shape[-1] != 2:
        raise ValueError("context must have shape [L, H, 2]")
    if target_np.shape != pred_np.shape or target_np.shape != repeat_np.shape:
        raise ValueError("target, pred, and repeat must share shape [T_out, H, 2]")
    if context_np.shape != x_mask_np.shape or target_np.shape != y_mask_np.shape:
        raise ValueError("x_mask and y_mask must match context and target shapes")
    if context_np.shape[1:] != target_np.shape[1:] or len(heights) != target_np.shape[1]:
        raise ValueError("context and target must share height/channel dimensions")

    if len(target_offsets_minutes) != target_np.shape[0]:
        raise ValueError("target_offsets_minutes length must match T_out")
    # Display spacing is intentionally categorical: each hourly context point
    # and each target 10-minute point gets one horizontal unit.
    context_steps = np.arange(context_np.shape[0], dtype=float)
    target_steps = np.arange(context_np.shape[0], context_np.shape[0] + target_np.shape[0], dtype=float)
    boundary = float(context_np.shape[0]) - 0.5
    last_context_tick = max(context_np.shape[0] - 2, 0)
    context_tick_indices = np.unique(
        np.linspace(0, last_context_tick, min(5, last_context_tick + 1), dtype=int)
    )
    target_tick_indices = np.unique(
        np.linspace(0, target_np.shape[0] - 1, min(4, target_np.shape[0]), dtype=int)
    )
    tick_positions = np.concatenate((context_steps[context_tick_indices], target_steps[target_tick_indices]))
    tick_labels = [f"T{idx - (context_np.shape[0] - 1):+d}h" for idx in context_tick_indices] + [
        f"T+{int(target_offsets_minutes[idx])}m" for idx in target_tick_indices
    ]
    channel_names = ("u", "v")
    fig, axes = plt.subplots(
        nrows=2,
        ncols=target_np.shape[1],
        figsize=(3.4 * target_np.shape[1], 6.4),
        sharex=True,
        constrained_layout=True,
    )

    for channel_idx, channel_name in enumerate(channel_names):
        for height_idx, height in enumerate(heights):
            ax = axes[channel_idx, height_idx]
            hourly = context_np[:, height_idx, channel_idx].copy()
            truth = target_np[:, height_idx, channel_idx].copy()
            hourly[~x_mask_np[:, height_idx, channel_idx]] = np.nan
            truth[~y_mask_np[:, height_idx, channel_idx]] = np.nan
            ax.plot(context_steps, hourly, color="black", marker="o", markersize=3, linewidth=1.5, label="hourly context")
            ax.plot(target_steps, truth, marker="o", linewidth=2.0, label="truth")
            ax.plot(target_steps, pred_np[:, height_idx, channel_idx], marker="s", linewidth=1.7, label="HTQ pred")
            ax.plot(target_steps, repeat_np[:, height_idx, channel_idx], linestyle="--", linewidth=1.5, label="repeat")
            ax.axvline(boundary, color="0.5", linestyle=":", linewidth=1.0)
            ax.set_xticks(tick_positions, tick_labels, rotation=35, ha="right")
            ax.set_title(f"{channel_name}, {height:g} m")
            ax.grid(True, alpha=0.3)
            if height_idx == 0:
                ax.set_ylabel("m/s")
            if channel_idx == 1:
                ax.set_xlabel("context hours | target minutes")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    fig.suptitle(title, y=1.02)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    import numpy as np

    return np.asarray(value)
