import pytest

torch = pytest.importorskip("torch")

from src.models.htq_transformer import CausalHTQTransformer, HTQConfig


def test_causal_htq_transformer_wind_only_fallback_without_meteo_batch():
    torch.manual_seed(0)
    model = CausalHTQTransformer(HTQConfig(use_meteo=True, use_static=True))
    model.eval()
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)

    with torch.no_grad():
        out = model(x_hourly, x_mask)

    assert out["pred"].shape == (2, 6, 6, 2)
    assert out["residual"].shape == (2, 6, 6, 2)
    assert out["encoder_memory"].shape == (2, 36, 64)
    assert out["fusion_info"] is None
