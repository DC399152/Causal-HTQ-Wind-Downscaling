import torch

from src.models.static_encoder import StaticEncoderConfig, StaticFeatureEncoder


def test_static_feature_encoder_outputs_one_token():
    encoder = StaticFeatureEncoder(
        StaticEncoderConfig(
            input_dim=17,
            d_model=64,
            hidden_dim=128,
            dropout=0.0,
            n_static_tokens=1,
        )
    )
    x_static = torch.rand(2, 17)

    tokens = encoder(x_static)

    assert tokens.shape == (2, 1, 64)
    assert torch.isfinite(tokens).all()
