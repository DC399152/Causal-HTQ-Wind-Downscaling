import numpy as np

from src.visualization.plot_samples import plot_sample_with_hourly_context


def test_plot_sample_with_hourly_context(tmp_path):
    output = tmp_path / "figure_12" / "sample.png"
    plot_sample_with_hourly_context(
        context=np.zeros((12, 6, 2), dtype=np.float32),
        target=np.zeros((6, 6, 2), dtype=np.float32),
        pred=np.ones((6, 6, 2), dtype=np.float32),
        repeat=np.zeros((6, 6, 2), dtype=np.float32),
        x_mask=np.ones((12, 6, 2), dtype=bool),
        y_mask=np.ones((6, 6, 2), dtype=bool),
        height_values=[175, 200, 225, 250, 275, 300],
        output_path=output,
        title="test",
    )
    assert output.exists()
    assert output.stat().st_size > 0
