"""Training utility placeholders."""

from __future__ import annotations

from typing import Any


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds when available."""

    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(requested: str = "auto"):
    """Return a torch device from ``auto``, ``cpu``, or a concrete CUDA spec."""

    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def y_denormalize(values, norm_stats: dict[str, Any]):
    """Convert target-normalized [B, T_out, H, C] tensors back to physical units."""

    return _denormalize(values, norm_stats["y_mean"], norm_stats["y_std"])


def x_denormalize(values, norm_stats: dict[str, Any]):
    """Convert input-normalized tensors back to physical units."""

    return _denormalize(values, norm_stats["x_mean"], norm_stats["x_std"])


def _denormalize(values, mean_values, std_values):
    import torch

    mean = torch.as_tensor(mean_values, dtype=values.dtype, device=values.device)
    std = torch.as_tensor(std_values, dtype=values.dtype, device=values.device)
    return values * std.view(*([1] * (values.ndim - 1)), -1) + mean.view(*([1] * (values.ndim - 1)), -1)
