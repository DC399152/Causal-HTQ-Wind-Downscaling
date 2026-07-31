import json

import pytest

pytest.importorskip("torch")

from scripts import evaluate


def test_sample_gallery_selects_unique_seeded_samples_and_writes_manifest(
    tmp_path,
    monkeypatch,
):
    class FakeDataset:
        def __init__(self, dataset_dir, split, normalize, return_metadata):
            self.dataset_dir = dataset_dir

        def __len__(self):
            return 8

        def __getitem__(self, index):
            return {
                "sample_index": index + 100,
                "station_id": f"station_{index % 2}",
                "target_time_start": f"time_{index}",
            }

    def fake_plot_one_sample(**kwargs):
        kwargs["context_output_path"].write_text("plot", encoding="utf-8")

    monkeypatch.setattr(evaluate, "WindDownscalingDataset", FakeDataset)
    monkeypatch.setattr(evaluate, "plot_one_sample", fake_plot_one_sample)

    written = evaluate.plot_sample_gallery(
        model=object(),
        dataset_dir="dataset",
        split="test",
        norm_stats={},
        device="cpu",
        output_dir=tmp_path / "sample_gallery",
        seed=42,
        num_samples=5,
    )

    manifest = json.loads((tmp_path / "sample_gallery" / "manifest.json").read_text())
    local_indices = [sample["local_index"] for sample in manifest["selected_samples"]]
    assert manifest["split"] == "test"
    assert len(local_indices) == len(set(local_indices)) == 5
    assert len(written) == 6
    assert all(sample["file"].startswith("sample_") for sample in manifest["selected_samples"])
