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


def test_height_time_tokenizer_delta_is_mask_aware():
    # One height and one channel makes the temporal transitions explicit:
    # t0 missing, t1 valid, t2 missing, t3 missing, t4 valid, t5 valid.
    x = torch.tensor([[[[10.0]], [[12.0]], [[20.0]], [[30.0]], [[40.0]], [[45.0]]]])
    mask = torch.tensor([[[[False]], [[True]], [[False]], [[False]], [[True]], [[True]]]])

    tokenizer = HeightTimeTokenizer(
        include_mask_features=True,
        include_delta_features=True,
        context_hours=6,
        height_levels=1,
        input_channels=1,
    )
    out = tokenizer(x, mask)

    delta = out.token_features[..., 1:2]
    expected = torch.tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[5.0]]]])
    assert torch.allclose(delta, expected)


def test_height_time_tokenizer_delta_keeps_legacy_behavior_without_mask():
    x = torch.tensor([[[[10.0]], [[12.0]], [[20.0]], [[30.0]], [[40.0]], [[45.0]]]])

    tokenizer = HeightTimeTokenizer(
        include_mask_features=False,
        include_delta_features=True,
        context_hours=6,
        height_levels=1,
        input_channels=1,
    )
    out = tokenizer(x)

    delta = out.token_features[..., 1:2]
    expected = torch.tensor([[[[0.0]], [[2.0]], [[8.0]], [[10.0]], [[10.0]], [[5.0]]]])
    assert torch.allclose(delta, expected)
