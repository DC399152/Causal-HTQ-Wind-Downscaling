import pytest

torch = pytest.importorskip("torch")

from src.models.htq_transformer import CausalHTQTransformer, HTQConfig


def test_causal_htq_transformer_wind_only_forward_when_multimodal_disabled():
    torch.manual_seed(0)
    model = CausalHTQTransformer(HTQConfig(use_meteo=False, use_static=False))
    model.eval()
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)

    with torch.no_grad():
        out = model(x_hourly, x_mask)

    assert out["pred"].shape == (2, 6, 6, 2)
    assert out["residual"].shape == (2, 6, 6, 2)
    assert out["encoder_memory"].shape == (2, 36, 64)
    assert out["fusion_info"] is None


def test_causal_htq_transformer_raises_when_meteo_enabled_but_missing():
    model = CausalHTQTransformer(HTQConfig(use_meteo=True))
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="x_meteo is missing"):
        model(x_hourly, x_mask)


def test_causal_htq_transformer_raises_when_meteo_mask_missing():
    model = CausalHTQTransformer(HTQConfig(use_meteo=True))
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)
    x_meteo = torch.randn(2, 6, 5, 2)

    with pytest.raises(ValueError, match="meteo_mask is missing"):
        model(x_hourly, x_mask, x_meteo=x_meteo)


def test_causal_htq_transformer_raises_when_static_enabled_but_missing():
    model = CausalHTQTransformer(HTQConfig(use_static=True))
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="x_static is missing"):
        model(x_hourly, x_mask)
