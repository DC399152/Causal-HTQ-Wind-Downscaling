from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from scripts.evaluate import model_config_from_checkpoint
from scripts.train import default_loss_config, save_checkpoint
from src.models.htq_cross_attention_reader import (
    CrossAttentionReader,
    CrossAttentionReaderConfig,
    HTQCrossAttentionReader,
)
from src.models.model_factory import build_model


def _reader(dropout: float = 0.0) -> CrossAttentionReader:
    return CrossAttentionReader(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=dropout,
    )


def _model_config() -> CrossAttentionReaderConfig:
    return CrossAttentionReaderConfig(
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        context_hours=3,
        target_steps=2,
        height_levels=2,
        query_builder_type="fixed",
        reader_nhead=4,
        reader_dim_feedforward=32,
        reader_dropout=0.0,
        output_head_type="multi_horizon_shared_trunk",
        output_head_hidden_dim=8,
        output_head_dropout=0.0,
    )


def _model_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "x_hourly": torch.randn(batch_size, 3, 2, 2),
        "x_mask": torch.ones(batch_size, 3, 2, 2, dtype=torch.bool),
        "current_hourly_reference": torch.randn(batch_size, 2, 2),
        "height_values": torch.tensor([[250.0, 275.0]]).repeat(batch_size, 1),
    }


def test_reader_shapes_attention_and_forbidden_modules() -> None:
    reader = _reader()
    queries = torch.randn(2, 12, 16)
    memory = torch.randn(2, 24, 16)
    mask = torch.zeros(2, 24, dtype=torch.bool)

    output, post_cross, attention = reader(
        queries,
        memory,
        mask,
        return_attention=True,
    )

    assert output.shape == (2, 12, 16)
    assert post_cross.shape == (2, 12, 16)
    assert attention is not None
    assert attention.shape == (2, 4, 12, 24)
    _, _, no_attention = reader(queries, memory, mask)
    assert no_attention is None
    assert sum(isinstance(module, nn.MultiheadAttention) for module in reader.modules()) == 1
    assert not any(
        isinstance(module, (nn.TransformerDecoder, nn.TransformerDecoderLayer))
        for module in reader.modules()
    )


def test_reader_query_independence() -> None:
    reader = _reader().eval()
    memory = torch.randn(1, 8, 16)
    queries_a = torch.randn(1, 5, 16)
    queries_b = queries_a.clone()
    queries_b[:, 2, 0] += 3.0

    output_a, _, _ = reader(queries_a, memory)
    output_b, _, _ = reader(queries_b, memory)

    unchanged = torch.tensor([0, 1, 3, 4])
    torch.testing.assert_close(output_a[:, unchanged], output_b[:, unchanged])
    assert not torch.allclose(output_a[:, 2], output_b[:, 2])


def test_reader_depends_on_valid_memory_and_ignores_masked_memory() -> None:
    reader = _reader().eval()
    queries = torch.randn(1, 4, 16)
    memory_a = torch.randn(1, 7, 16)
    mask = torch.zeros(1, 7, dtype=torch.bool)

    valid_changed = memory_a.clone()
    valid_changed[:, 1, 0] += 5.0
    output_a, _, _ = reader(queries, memory_a, mask)
    output_valid_changed, _, _ = reader(queries, valid_changed, mask)
    assert not torch.allclose(output_a, output_valid_changed)

    mask[:, -1] = True
    masked_changed = memory_a.clone()
    masked_changed[:, -1] += 1000.0
    output_masked_a, _, _ = reader(queries, memory_a, mask)
    output_masked_b, _, _ = reader(queries, masked_changed, mask)
    torch.testing.assert_close(output_masked_a, output_masked_b)


def test_full_reader_model_shapes_and_backward_gradients() -> None:
    model = HTQCrossAttentionReader(_model_config())
    inputs = _model_inputs()
    output = model(**inputs, return_features=True, return_attention=True)

    assert output["pred"].shape == (2, 2, 2, 2)
    assert output["residual"].shape == (2, 2, 2, 2)
    assert output["target_queries"].shape == (2, 4, 16)
    assert output["reader_post_cross"].shape == (2, 4, 16)
    assert output["target_features"].shape == (2, 2, 2, 16)
    assert output["reader_attention_weights"].shape == (2, 4, 4, 6)
    assert torch.isfinite(output["pred"]).all()

    output["pred"].square().mean().backward()
    module_groups = {
        "encoder": model.encoder,
        "query_builder": model.query_builder,
        "reader": model.reader,
        "residual_head": model.residual_head,
    }
    for name, module in module_groups.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients, f"{name} did not receive gradients"
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)

    assert not any(
        isinstance(module, (nn.TransformerDecoder, nn.TransformerDecoderLayer))
        for module in model.modules()
    )


def test_reader_checkpoint_factory_round_trip_strict_load(tmp_path) -> None:
    config = _model_config()
    model = build_model(config).eval()
    inputs = _model_inputs(batch_size=1)
    with torch.no_grad():
        expected = model(**inputs)["pred"]

    checkpoint_path = tmp_path / "reader.pt"
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        metrics={"val_MAE_ms": 1.0},
        model_config=config,
        loss_config=default_loss_config(),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["architecture"] == "htq_cross_attention_reader"
    assert checkpoint["model_config"]["reader_num_layers"] == 1
    assert checkpoint["model_config"]["reader_nhead"] == 4
    assert checkpoint["output_head_config"]["type"] == "multi_horizon_shared_trunk"
    assert checkpoint["model_config"] == asdict(config)
    rebuilt_config = model_config_from_checkpoint(checkpoint)
    rebuilt = build_model(rebuilt_config).eval()
    rebuilt.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with torch.no_grad():
        actual = rebuilt(**inputs)["pred"]

    torch.testing.assert_close(actual, expected)
