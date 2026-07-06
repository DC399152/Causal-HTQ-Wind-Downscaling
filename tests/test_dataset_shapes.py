from dataclasses import replace

import numpy as np

from src.data.dataset import SampleShapes
from src.data.dataset_builder import build_dataset
from src.data.preprocessing import parse_preprocessing_config


def test_semantic_dataset_shapes():
    shapes = SampleShapes(input_context=(6, 40, 2), target_10min=(6, 40, 2))

    assert shapes.input_context == (6, 40, 2)
    assert shapes.target_10min == (6, 40, 2)


def test_empty_dataset_builder_writes_required_shape_contract(tmp_path):
    config = parse_preprocessing_config(
        "configs/preprocessing/paris_1h_to_10min_6h_causal_start_v1.yaml"
    )
    config = replace(
        config,
        raw_3600s_dir=tmp_path / "raw",
        raw_600s_dir=tmp_path / "raw",
        dataset_dir=tmp_path / "dataset",
    )

    summary = build_dataset(config, dry_run=False)

    assert summary.status == "written_empty"
    with np.load(config.dataset_dir / "dataset.npz", allow_pickle=True) as data:
        assert data["x_hourly"].shape == (0, 6, 6, 2)
        assert data["x_mask"].shape == (0, 6, 6, 2)
        assert data["y_10min"].shape == (0, 6, 6, 2)
        assert data["y_mask"].shape == (0, 6, 6, 2)
        assert data["current_hourly"].shape == (0, 6, 2)
