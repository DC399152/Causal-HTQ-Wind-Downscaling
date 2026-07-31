from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.evaluate import (
    dataset_fingerprint,
    resolve_evaluation_dataset_dir,
)


def _write_dataset(directory: Path, content: bytes) -> str:
    directory.mkdir(parents=True)
    (directory / "dataset.npz").write_bytes(content)
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def test_evaluation_uses_checkpoint_dataset_by_default(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "training_dataset"
    fingerprint = _write_dataset(dataset_dir, b"training")
    checkpoint = {
        "dataset_fingerprint": fingerprint,
        "dataset_metadata": {
            "dataset_path": str(dataset_dir / "dataset.npz"),
            "dataset_fingerprint": fingerprint,
        },
    }

    assert resolve_evaluation_dataset_dir(checkpoint, None) == dataset_dir
    assert dataset_fingerprint(dataset_dir) == fingerprint


def test_evaluation_rejects_mismatched_dataset(tmp_path: Path) -> None:
    training_dir = tmp_path / "training_dataset"
    evaluation_dir = tmp_path / "other_dataset"
    fingerprint = _write_dataset(training_dir, b"training")
    _write_dataset(evaluation_dir, b"other")
    checkpoint = {
        "dataset_fingerprint": fingerprint,
        "dataset_metadata": {
            "dataset_path": str(training_dir / "dataset.npz"),
            "dataset_fingerprint": fingerprint,
        },
    }

    with pytest.raises(ValueError, match="does not match"):
        resolve_evaluation_dataset_dir(checkpoint, evaluation_dir)


def test_evaluation_mismatch_requires_explicit_override(tmp_path: Path) -> None:
    training_dir = tmp_path / "training_dataset"
    evaluation_dir = tmp_path / "other_dataset"
    fingerprint = _write_dataset(training_dir, b"training")
    _write_dataset(evaluation_dir, b"other")
    checkpoint = {
        "dataset_fingerprint": fingerprint,
        "dataset_metadata": {
            "dataset_path": str(training_dir / "dataset.npz"),
            "dataset_fingerprint": fingerprint,
        },
    }

    assert (
        resolve_evaluation_dataset_dir(
            checkpoint,
            evaluation_dir,
            allow_mismatch=True,
        )
        == evaluation_dir
    )


def test_legacy_checkpoint_keeps_default_dataset_behavior() -> None:
    assert resolve_evaluation_dataset_dir({}, None) == Path(
        "data/datasets/ds_paris_1h_to_10min_6h_causal_start_v1"
    )
