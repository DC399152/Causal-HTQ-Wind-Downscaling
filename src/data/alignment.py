"""Timestamp alignment helpers for hour-end reconstruction.

All timestamps are interval starts. For an hour start ``T``:

```
X_T -> Y_T, Y_T+10, ..., Y_T+50
```
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

import numpy as np


TEN_MINUTES = timedelta(minutes=10)
ONE_HOUR = timedelta(hours=1)
TARGET_OFFSETS_MINUTES = (0, 10, 20, 30, 40, 50)


@dataclass(frozen=True)
class AlignmentSpec:
    """Dataset timing specification."""

    context_hours: int = 6
    target_steps: int = 6
    target_step_minutes: int = 10
    input_step_minutes: int = 60


def _add_minutes(value, minutes: int):
    if isinstance(value, np.datetime64):
        return value + np.timedelta64(minutes, "m")
    return value + timedelta(minutes=minutes)


def _diff_minutes(later, earlier) -> int:
    if isinstance(later, np.datetime64) or isinstance(earlier, np.datetime64):
        delta = np.datetime64(later) - np.datetime64(earlier)
        return int(delta / np.timedelta64(1, "m"))
    return int((later - earlier).total_seconds() // 60)


def normalize_time_array(values: Sequence) -> np.ndarray:
    """Return times normalized to minute-resolution ``datetime64``."""

    return np.asarray(values, dtype="datetime64[m]")


def time_key(value) -> np.datetime64:
    """Return a hashable minute-resolution key for timestamp lookup."""

    return np.datetime64(value, "m")


def times_to_strings(times: Sequence) -> list[str]:
    """Convert timestamps to stable ISO-like strings."""

    return [str(np.datetime64(t, "m")) for t in times]


def target_times_for_hour(hour_start, spec: AlignmentSpec | None = None) -> list:
    """Return target start timestamps ``Y_T ... Y_T+50``."""

    spec = spec or AlignmentSpec()
    return [
        _add_minutes(hour_start, i * spec.target_step_minutes)
        for i in range(spec.target_steps)
    ]


def context_times_for_hour(hour_start, spec: AlignmentSpec | None = None) -> list:
    """Return causal context timestamps ``X_{T-L+1:T}``."""

    spec = spec or AlignmentSpec()
    first_offset = -(spec.context_hours - 1) * spec.input_step_minutes
    return [
        _add_minutes(hour_start, first_offset + i * spec.input_step_minutes)
        for i in range(spec.context_hours)
    ]


def is_strictly_consecutive_minutes(times: Sequence, step_minutes: int) -> bool:
    """Check that timestamps are exactly ``step_minutes`` apart."""

    if len(times) < 2:
        return True
    return all(
        _diff_minutes(times[i + 1], times[i]) == step_minutes
        for i in range(len(times) - 1)
    )


def is_strictly_consecutive(times: Sequence, step: timedelta) -> bool:
    """Check that timestamps are separated by exactly ``step``."""

    return is_strictly_consecutive_minutes(times, int(step.total_seconds() // 60))


def has_required_times(required: Iterable, available: Iterable) -> bool:
    """Return true when every required timestamp is present."""

    available_set = {time_key(t) for t in available}
    return all(time_key(t) in available_set for t in required)


def build_time_index(times: Sequence) -> dict[np.datetime64, int]:
    """Map minute-resolution timestamps to integer positions."""

    return {time_key(t): i for i, t in enumerate(times)}


def validate_target_offsets(
    hour_start,
    target_times: Sequence,
    spec: AlignmentSpec | None = None,
) -> tuple[bool, list[str]]:
    """Validate ``[T, T+10, ..., T+50]`` target timestamps."""

    spec = spec or AlignmentSpec()
    expected = target_times_for_hour(hour_start, spec)
    errors: list[str] = []
    if len(target_times) != spec.target_steps:
        errors.append(f"expected {spec.target_steps} target times, got {len(target_times)}")
    elif [time_key(t) for t in target_times] != [time_key(t) for t in expected]:
        errors.append("target timestamps do not match [T, T+10, ..., T+50]")
    if not is_strictly_consecutive_minutes(target_times, spec.target_step_minutes):
        errors.append("target timestamps are not consecutive 10min starts")
    return len(errors) == 0, errors


def validate_sample_alignment(
    hour_start,
    available_hourly: Iterable,
    available_10min: Iterable,
    spec: AlignmentSpec | None = None,
) -> tuple[bool, list[str]]:
    """Validate context and target availability for one sample hour."""

    spec = spec or AlignmentSpec()
    context = context_times_for_hour(hour_start, spec)
    targets = target_times_for_hour(hour_start, spec)
    errors: list[str] = []

    if not has_required_times(context, available_hourly):
        errors.append("missing one or more causal hourly context timestamps")
    if not has_required_times(targets, available_10min):
        errors.append("missing one or more 10min target timestamps")
    if not is_strictly_consecutive_minutes(context, spec.input_step_minutes):
        errors.append("context timestamps are not consecutive hourly starts")
    if not is_strictly_consecutive_minutes(targets, spec.target_step_minutes):
        errors.append("target timestamps are not consecutive 10min starts")
    if time_key(context[-1]) != time_key(hour_start):
        errors.append("last context timestamp is not current X_T")
    if time_key(targets[0]) != time_key(hour_start):
        errors.append("first target timestamp is not T")

    return len(errors) == 0, errors

