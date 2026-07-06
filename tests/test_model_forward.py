import pytest

torch = pytest.importorskip("torch")

from src.models.htq_transformer import CausalHTQTransformer, HTQConfig


def test_phase0_model_forward_shape_and_zero_residual():
    model = CausalHTQTransformer(HTQConfig(context_hours=6, target_steps=6, input_channels=2, output_channels=2))
    x = torch.randn(3, 6, 8, 2)

    out = model(x)

    assert out["pred"].shape == (3, 6, 8, 2)
    assert out["residual"].shape == (3, 6, 8, 2)
    assert torch.allclose(out["residual"].mean(dim=1), torch.zeros(3, 8, 2))
