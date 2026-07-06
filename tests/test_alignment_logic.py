from datetime import datetime, timedelta

from src.data.alignment import (
    AlignmentSpec,
    context_times_for_hour,
    target_times_for_hour,
    validate_target_offsets,
)


def test_hourly_maps_to_six_10min_start_times():
    start = datetime(2024, 1, 1, 12)
    targets = target_times_for_hour(start, AlignmentSpec())

    assert targets == [start + i * timedelta(minutes=10) for i in range(6)]
    ok, errors = validate_target_offsets(start, targets)
    assert ok
    assert errors == []


def test_causal_context_includes_current_hour_and_no_future():
    start = datetime(2024, 1, 1, 12)
    context = context_times_for_hour(start, AlignmentSpec(context_hours=6))

    assert context[0] == start - timedelta(hours=5)
    assert context[-1] == start
    assert all(t <= start for t in context)


def test_old_end_aligned_targets_are_rejected():
    start = datetime(2024, 1, 1, 12)
    old_targets = [start + timedelta(minutes=m) for m in [-50, -40, -30, -20, -10, 0]]

    ok, errors = validate_target_offsets(start, old_targets)

    assert not ok
    assert "target timestamps do not match [T, T+10, ..., T+50]" in errors
