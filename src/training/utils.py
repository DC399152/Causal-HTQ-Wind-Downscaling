"""Training utility placeholders."""

from __future__ import annotations


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
    except ImportError:
        pass

