"""Small model utility modules."""

from __future__ import annotations


def require_torch():
    """Import torch lazily for optional Phase 0 model checks."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model execution.") from exc
    return torch

