"""Metric interfaces for wind profile reconstruction."""

from __future__ import annotations


def mean_absolute_error(pred, target):
    """Return MAE for tensors with matching semantic shapes."""

    return (pred - target).abs().mean()

