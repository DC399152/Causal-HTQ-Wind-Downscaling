import pytest

torch = pytest.importorskip("torch")

from src.models.meteo_encoder import MeteoPressureLevelEncoder


def test_meteo_pressure_level_encoder_shapes_and_finite_outputs():
    torch.manual_seed(0)
    encoder = MeteoPressureLevelEncoder()
    x_meteo = torch.randn(2, 6, 5, 2)
    meteo_mask = torch.ones(2, 6, 5, 2, dtype=torch.bool)

    tokens = encoder(x_meteo, meteo_mask)

    assert tokens.shape == (2, 30, 64)
    assert torch.isfinite(tokens).all()
