import pytest

torch = pytest.importorskip("torch")

from src.models.tokenizer import HeightTimeTokenizer


def test_height_time_tokenizer_appends_mask_features():
    x = torch.randn(2, 6, 4, 2)
    mask = torch.ones(2, 6, 4, 2, dtype=torch.bool)
    mask[0, 0, 0, :] = False

    tokenizer = HeightTimeTokenizer(include_mask_features=True)
    out = tokenizer(x, mask)

    assert out.token_features.shape == (2, 6, 4, 4)
    assert out.token_valid.shape == (2, 6, 4)
    assert not out.token_valid[0, 0, 0]
    assert out.token_valid[0, 0, 1]
    assert torch.allclose(out.token_features[..., :2], x)
    assert torch.equal(out.token_features[..., 2:].bool(), mask)


def test_height_time_tokenizer_can_keep_values_only():
    x = torch.randn(2, 6, 4, 2)
    mask = torch.ones(2, 6, 4, 2, dtype=torch.bool)

    tokenizer = HeightTimeTokenizer(include_mask_features=False)
    out = tokenizer(x, mask)

    assert out.token_features.shape == (2, 6, 4, 2)
    assert out.token_valid.shape == (2, 6, 4)
