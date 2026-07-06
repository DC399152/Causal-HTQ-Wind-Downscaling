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


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    import numpy as np

    return np.asarray(value)
