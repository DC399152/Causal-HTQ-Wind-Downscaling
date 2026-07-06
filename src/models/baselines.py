"""Simple non-learned baselines for reconstruction checks."""

from __future__ import annotations


def repeat_current_hour(current_hourly, target_steps: int = 6):
    """Repeat current hourly profile across target 10min steps.

    Parameters
    ----------
    current_hourly:
        [B, H, C] tensor.
    target_steps:
        Number of 10min output steps. Paris v1 uses 6.

    Returns
    -------
    Tensor with shape [B, T_out, H, C].
    """

    if current_hourly.ndim != 3:
        raise ValueError("current_hourly must have shape [B, H, C]")
    return current_hourly.unsqueeze(1).repeat(1, target_steps, 1, 1)

