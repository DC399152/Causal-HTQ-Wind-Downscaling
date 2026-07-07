import torch

from src.models.htq_transformer import CausalHTQTransformer, HTQConfig


def test_causal_htq_transformer_meteo_static_forward_shapes():
    model = CausalHTQTransformer(
        HTQConfig(
            use_meteo=True,
            use_static=True,
            dropout=0.0,
            fusion_dropout=0.0,
            static_dropout=0.0,
        )
    )
    model.eval()
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)
    x_meteo = torch.randn(2, 6, 5, 2)
    meteo_mask = torch.ones(2, 6, 5, 2, dtype=torch.bool)
    x_static = torch.rand(2, 17)

    with torch.no_grad():
        out = model(
            x_hourly,
            x_mask,
            x_meteo=x_meteo,
            meteo_mask=meteo_mask,
            x_static=x_static,
        )

    assert out["pred"].shape == (2, 6, 6, 2)
    assert out["residual"].shape == (2, 6, 6, 2)
    assert out["encoder_memory"].shape == (2, 36, 64)
    assert out["fusion_info"] is not None
    assert bool(out["fusion_info"]["skipped"]) is False
    assert 0.0 <= float(out["fusion_info"]["gate_mean"]) <= 1.0
