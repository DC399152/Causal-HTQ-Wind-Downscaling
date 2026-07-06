"""Mask helpers for missing data and valid samples.

Mask convention across the repository:

```
True  = valid
False = invalid
```
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def valid_numeric_mask(array: np.ndarray, missing_value: float | None = -999.0) -> np.ndarray:
    """Return validity mask for numeric values.

    The returned mask has the same shape as ``array``. Values are valid when
    finite and not equal to the configured missing sentinel.
    """

    mask = np.isfinite(array)
    if missing_value is not None:
        mask &= array != missing_value
    return mask


def finite_mask(array: np.ndarray) -> np.ndarray:
    """Return a boolean mask marking finite values."""

    return np.isfinite(array)


def fill_invalid(
    array: np.ndarray,
    valid_mask: np.ndarray,
    missing_value: float = -999.0,
) -> np.ndarray:
    """Return a copy with invalid positions filled by ``missing_value``."""

    out = np.asarray(array, dtype=np.float32).copy()
    out[~valid_mask] = missing_value
    return out


def missing_fraction(array: np.ndarray, missing_value: float | None = -999.0) -> float:
    """Return the fraction of invalid values in an array."""

    if array.size == 0:
        return 0.0
    return float((~valid_numeric_mask(array, missing_value)).sum() / array.size)


def valid_ratio(mask: np.ndarray) -> float:
    """Return fraction of ``True`` values in a validity mask."""

    if mask.size == 0:
        return 0.0
    return float(mask.sum() / mask.size)


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """Combine masks with logical AND."""

    if not masks:
        raise ValueError("At least one mask is required")
    combined = np.asarray(masks[0], dtype=bool).copy()
    for mask in masks[1:]:
        combined &= np.asarray(mask, dtype=bool)
    return combined


def flag_valid_mask(
    flag_array: np.ndarray,
    threshold: float,
    invalid_when: str = "greater_than",
) -> np.ndarray:
    """Convert a QC flag array into a validity mask.

    Parameters
    ----------
    flag_array:
        QC flag values with dimensions compatible with the data array.
    threshold:
        Configured rejection threshold.
    invalid_when:
        One of ``greater_than``, ``greater_equal``, ``equal``, or
        ``nonzero``.
    """

    values = np.asarray(flag_array)
    if invalid_when == "greater_than":
        invalid = values > threshold
    elif invalid_when == "greater_equal":
        invalid = values >= threshold
    elif invalid_when == "equal":
        invalid = values == threshold
    elif invalid_when == "nonzero":
        invalid = values != 0
    else:
        raise ValueError(f"Unknown flag invalid_when policy: {invalid_when}")
    return ~invalid


def configured_flag_masks(
    flags: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Mapping[str, object]],
) -> list[np.ndarray]:
    """Build validity masks from configured QC flag policies."""

    masks: list[np.ndarray] = []
    for name, values in flags.items():
        policy = thresholds.get(name)
        if not policy:
            continue
        threshold = float(policy.get("threshold", 0.0))
        invalid_when = str(policy.get("invalid_when", "greater_than"))
        masks.append(flag_valid_mask(values, threshold, invalid_when))
    return masks
