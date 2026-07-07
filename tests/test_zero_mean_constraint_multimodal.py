import pytest

torch = pytest.importorskip("torch")

from src.models.htq_transformer import CausalHTQTransformer, HTQConfig


def test_causal_htq_transformer_multimodal_prediction_is_current_hourly_plus_residual():
    torch.manual_seed(0)
    model = CausalHTQTransformer(HTQConfig(use_meteo=True))
    model.eval()
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)
    x_meteo = torch.randn(2, 6, 5, 2)
    meteo_mask = torch.ones(2, 6, 5, 2, dtype=torch.bool)

    with torch.no_grad():
        out = model(x_hourly, x_mask, x_meteo=x_meteo, meteo_mask=meteo_mask)

    assert torch.allclose(out["pred"], x_hourly[:, -1].unsqueeze(1) + out["residual"], atol=1e-6)
    assert torch.isfinite(out["pred"]).all()
    assert torch.isfinite(out["residual"]).all()
