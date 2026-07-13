"""Plot training history curves from a run metrics.json file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def plot_training_curves(metrics: dict[str, Any], output_dir: str | Path) -> list[Path]:
    """Save compact training/validation loss curves.

    The training script stores one row per epoch under ``history``. This helper
    plots the total loss and then every other train/val loss part that appears
    in the history.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for training curve plots") from exc

    history = metrics.get("history", [])
    if not history:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(history)]
    written: list[Path] = []

    total_path = output_dir / "training_loss_curve.png"
    _plot_keys(
        plt,
        history,
        epochs,
        ["train_loss_norm_total", "val_loss_norm_total"],
        total_path,
        title="Normalized total loss",
        ylabel="loss",
    )
    written.append(total_path)

    part_keys = sorted(
        key
        for row in history
        for key in row
        if key.startswith(("train_loss_norm_", "val_loss_norm_"))
        and not key.endswith("_total")
    )
    if part_keys:
        parts_path = output_dir / "training_loss_parts.png"
        _plot_keys(
            plt,
            history,
            epochs,
            part_keys,
            parts_path,
            title="Normalized loss parts",
            ylabel="loss",
        )
        written.append(parts_path)

    return written


def _plot_keys(plt, history: Sequence[dict[str, Any]], epochs: Sequence[int], keys: Sequence[str], path: Path, *, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for key in keys:
        values = [row.get(key) for row in history]
        if all(value is None for value in values):
            continue
        ax.plot(epochs, values, marker="o", linewidth=1.5, markersize=3, label=key)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
