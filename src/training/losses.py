"""Loss interfaces for future HTQ training."""

from __future__ import annotations


def zero_mean_residual_penalty(residual):
    """Return mean squared target-time residual mean."""

    mean_residual = residual.mean(dim=1)
    return (mean_residual * mean_residual).mean()

