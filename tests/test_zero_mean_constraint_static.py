import torch

from src.models.htq_transformer import CausalHTQTransformer, HTQConfig


def test_static_forward_prediction_is_current_hourly_plus_residual():
    model = CausalHTQTransformer(
        HTQConfig(
            use_static=True,
            dropout=0.0,
            static_dropout=0.0,
        )
    )
    model.eval()
    x_hourly = torch.randn(2, 6, 6, 2)
    x_mask = torch.ones(2, 6, 6, 2, dtype=torch.bool)
    x_static = torch.rand(2, 17)

    with torch.no_grad():
        out = model(x_hourly, x_mask, x_static=x_static)

    assert torch.allclose(out["pred"], x_hourly[:, -1].unsqueeze(1) + out["residual"], atol=1e-6)
    assert torch.isfinite(out["pred"]).all()
    assert torch.isfinite(out["residual"]).all()
