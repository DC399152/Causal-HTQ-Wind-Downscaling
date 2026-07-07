import pytest

torch = pytest.importorskip("torch")

from src.models.htq_transformer import CausalHTQTransformer, HTQConfig


def test_causal_htq_transformer_multimodal_forward_shapes_and_gate_info():
    torch.manual_seed(0)
    model = CausalHTQTransformer(HTQConfig(use_meteo=True))
    model.eval()
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)
    x_meteo = torch.randn(2, 6, 5, 2)
    meteo_mask = torch.ones(2, 6, 5, 2, dtype=torch.bool)

    with torch.no_grad():
        out = model(x_hourly, x_mask, x_meteo=x_meteo, meteo_mask=meteo_mask)

    assert out["pred"].shape == (2, 6, 6, 2)
    assert out["residual"].shape == (2, 6, 6, 2)
    assert out["encoder_memory"].shape == (2, 36, 64)
    assert out["fusion_info"] is not None
    assert not out["fusion_info"]["skipped"]
    assert 0.0 <= float(out["fusion_info"]["gate_mean"]) <= 1.0


def test_causal_htq_transformer_multimodal_forward_handles_masked_meteo_tokens():
    torch.manual_seed(0)
    model = CausalHTQTransformer(HTQConfig(use_meteo=True))
    model.eval()
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)
    x_meteo = torch.randn(2, 6, 5, 2)
    meteo_mask = torch.ones(2, 6, 5, 2, dtype=torch.bool)
    meteo_mask[0, :, :2, :] = False
    meteo_mask[1] = False

    with torch.no_grad():
        out = model(x_hourly, x_mask, x_meteo=x_meteo, meteo_mask=meteo_mask)

    assert out["pred"].shape == (2, 6, 6, 2)
    assert torch.isfinite(out["pred"]).all()
    assert torch.isfinite(out["residual"]).all()
