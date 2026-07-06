import pytest

torch = pytest.importorskip("torch")

from src.models.htq_transformer import CausalHTQTransformer


def test_causal_htq_transformer_minimal_forward_shapes():
    torch.manual_seed(0)
    model = CausalHTQTransformer()
    model.eval()
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)

    with torch.no_grad():
        out = model(x_hourly, x_mask)

    assert set(out) == {"pred", "residual", "encoder_memory"}
    assert out["pred"].shape == (2, 6, 6, 2)
    assert out["residual"].shape == (2, 6, 6, 2)
    assert out["encoder_memory"].shape == (2, 36, 64)


def test_causal_htq_transformer_prediction_is_current_hourly_plus_residual():
    torch.manual_seed(1)
    model = CausalHTQTransformer()
    model.eval()
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)

    with torch.no_grad():
        out = model(x_hourly, x_mask)

    expected = x_hourly[:, -1].unsqueeze(1) + out["residual"]
    assert torch.allclose(out["pred"], expected, atol=1e-6)
