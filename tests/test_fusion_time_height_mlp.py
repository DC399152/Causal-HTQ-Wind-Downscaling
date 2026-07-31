from dataclasses import asdict, replace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from scripts.evaluate import model_config_from_checkpoint
from scripts.train import default_loss_config, save_checkpoint
from src.models.htq_fusion_time_height_mlp import (
    FusionTimeHeightMLPConfig,
    HTQFusionTimeHeightMLP,
)
from src.models.model_factory import build_model, model_config_from_dict
from src.models.time_height_mixer import (
    ChannelMixingMLP,
    HeightMixingMLP,
    TemporalMixingMLP,
    TemporalTargetProjection,
    TimeHeightMixerBlock,
)


def _mixer_dimensions():
    return {
        "context_hours": 4,
        "height_levels": 3,
        "d_model": 8,
    }


def _model_config(**overrides):
    values = {
        "d_model": 16,
        "mlp_d_model": 12,
        "context_hours": 4,
        "target_steps": 3,
        "height_levels": 3,
        "num_mixer_blocks": 2,
        "temporal_mixing_hidden_dim": 7,
        "height_mixing_hidden_dim": 5,
        "channel_expansion_ratio": 2,
        "mixer_dropout": 0.0,
        "target_projection_hidden_dim": 6,
        "target_projection_dropout": 0.0,
        "meteo_context_hours": 4,
        "fusion_nhead": 4,
        "fusion_dropout": 0.0,
        "static_dropout": 0.0,
        "output_head_hidden_dim": 10,
        "output_head_dropout": 0.0,
        "output_head_type": "multi_horizon_shared_trunk",
    }
    values.update(overrides)
    return FusionTimeHeightMLPConfig(**values)


def _model_inputs(batch_size=2, *, multimodal=False):
    inputs = {
        "x_hourly": torch.randn(batch_size, 4, 3, 2),
        "x_mask": torch.ones(batch_size, 4, 3, 2, dtype=torch.bool),
        "current_hourly_reference": torch.randn(batch_size, 3, 2),
        "height_values": torch.tensor(
            [[10.0, 80.0, 100.0]]
        ).repeat(batch_size, 1),
    }
    if multimodal:
        inputs.update(
            {
                "x_meteo": torch.randn(batch_size, 4, 5, 2),
                "meteo_mask": torch.ones(
                    batch_size, 4, 5, 2, dtype=torch.bool
                ),
                "x_static": torch.randn(batch_size, 17),
            }
        )
    return inputs


def test_axis_mixers_and_block_preserve_shape():
    dimensions = _mixer_dimensions()
    x = torch.randn(2, 4, 3, 8)
    modules = [
        TemporalMixingMLP(**dimensions, hidden_dim=7, dropout=0.0),
        HeightMixingMLP(**dimensions, hidden_dim=5, dropout=0.0),
        ChannelMixingMLP(**dimensions, expansion_ratio=2, dropout=0.0),
        TimeHeightMixerBlock(
            **dimensions,
            temporal_hidden_dim=7,
            height_hidden_dim=5,
            channel_expansion_ratio=2,
            dropout=0.0,
        ),
    ]
    for module in modules:
        output = module(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()


def test_temporal_mixing_does_not_mix_heights():
    module = TemporalMixingMLP(
        **_mixer_dimensions(), hidden_dim=7, dropout=0.0
    ).eval()
    original = torch.randn(1, 4, 3, 8)
    changed = original.clone()
    changed[:, :, 1] += 3.0
    with torch.no_grad():
        output_a = module(original)
        output_b = module(changed)
    torch.testing.assert_close(output_a[:, :, 0], output_b[:, :, 0])
    torch.testing.assert_close(output_a[:, :, 2], output_b[:, :, 2])


def test_height_mixing_does_not_mix_historical_times():
    module = HeightMixingMLP(
        **_mixer_dimensions(), hidden_dim=5, dropout=0.0
    ).eval()
    original = torch.randn(1, 4, 3, 8)
    changed = original.clone()
    changed[:, 2] += 3.0
    with torch.no_grad():
        output_a = module(original)
        output_b = module(changed)
    torch.testing.assert_close(output_a[:, 0], output_b[:, 0])
    torch.testing.assert_close(output_a[:, 1], output_b[:, 1])
    torch.testing.assert_close(output_a[:, 3], output_b[:, 3])


def test_channel_mixing_does_not_mix_time_height_positions():
    module = ChannelMixingMLP(
        **_mixer_dimensions(), expansion_ratio=2, dropout=0.0
    ).eval()
    original = torch.randn(1, 4, 3, 8)
    changed = original.clone()
    changed[:, 2, 1] += 3.0
    with torch.no_grad():
        output_a = module(original)
        output_b = module(changed)
    unchanged = torch.ones((4, 3), dtype=torch.bool)
    unchanged[2, 1] = False
    torch.testing.assert_close(
        output_a[:, unchanged],
        output_b[:, unchanged],
    )


def test_temporal_target_projection_shape_and_dependency():
    projection = TemporalTargetProjection(
        context_hours=4,
        target_steps=3,
        height_levels=3,
        d_model=8,
        hidden_dim=6,
        dropout=0.0,
    ).eval()
    original = torch.randn(1, 4, 3, 8)
    changed = original.clone()
    changed[:, 1, 2] += torch.linspace(0.1, 0.8, 8)
    with torch.no_grad():
        output_a = projection(original)
        output_b = projection(changed)
    assert output_a.shape == (1, 3, 3, 8)
    torch.testing.assert_close(output_a[:, :, :2], output_b[:, :, :2])
    assert not torch.equal(output_a[:, :, 2], output_b[:, :, 2])


def test_full_model_shapes_features_and_masked_input_stability():
    model = HTQFusionTimeHeightMLP(_model_config()).eval()
    inputs = _model_inputs(batch_size=2)
    inputs["x_mask"][:, 1, 2] = False
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["x_hourly"][:, 1, 2] = 1000.0
    with torch.no_grad():
        output = model(**inputs, return_features=True)
        changed_output = model(**changed)

    assert output["pred"].shape == (2, 3, 3, 2)
    assert output["residual"].shape == (2, 3, 3, 2)
    assert output["fused_features"].shape == (2, 4, 3, 16)
    assert output["context_features"].shape == (2, 4, 3, 12)
    assert output["target_projection_output"].shape == (2, 3, 3, 12)
    assert output["target_features"].shape == (2, 3, 3, 12)
    assert len(output["mixer_block_outputs"]) == 2
    assert torch.isfinite(output["pred"]).all()
    torch.testing.assert_close(output["pred"], changed_output["pred"])


def test_backbone_contains_only_mlp_operations():
    model = HTQFusionTimeHeightMLP(_model_config())
    backbone = nn.ModuleList(
        [
            model.input_projection,
            model.mixer_blocks,
            model.backbone_output_norm,
            model.target_projection,
            model.target_time_embedding,
        ]
    )
    banned = (
        nn.MultiheadAttention,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
        nn.RNNBase,
        nn.Conv1d,
        nn.Conv2d,
    )
    assert not any(isinstance(module, banned) for module in backbone.modules())


def test_multimodal_model_backward_has_finite_gradients():
    model = HTQFusionTimeHeightMLP(
        _model_config(use_meteo=True, use_static=True)
    )
    output = model(**_model_inputs(multimodal=True))
    output["pred"].square().mean().backward()
    groups = {
        "tokenizer": model.tokenizer,
        "meteo_encoder": model.meteo_encoder,
        "static_encoder": model.static_encoder,
        "fusion": model.fusion,
        "input_projection": model.input_projection,
        "mixer_blocks": model.mixer_blocks,
        "target_projection": model.target_projection,
        "target_time_embedding": model.target_time_embedding,
        "residual_head": model.residual_head,
    }
    for name, module in groups.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients, f"{name} did not receive gradients"
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_factory_checkpoint_round_trip_strict_load(tmp_path):
    config = _model_config()
    model = build_model(config).eval()
    inputs = _model_inputs(batch_size=1)
    with torch.no_grad():
        expected = model(**inputs)["pred"]

    path = tmp_path / "mlp.pt"
    optimizer = torch.optim.AdamW(model.parameters())
    dataset_metadata = {
        "dataset_path": "dataset.npz",
        "dataset_fingerprint": "sha256:test",
        "height_values": [[10.0, 80.0, 100.0]],
    }
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=2,
        metrics={"val_MAE_ms": 1.0},
        model_config=config,
        loss_config=default_loss_config(),
        norm_stats={"y_mean": [0.0, 0.0], "y_std": [1.0, 1.0]},
        dataset_metadata=dataset_metadata,
    )
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["architecture"] == "htq_fusion_time_height_mlp"
    assert checkpoint["model_config"] == asdict(config)
    assert checkpoint["dataset_fingerprint"] == "sha256:test"
    assert checkpoint["height_values"] == [[10.0, 80.0, 100.0]]
    restored_config = model_config_from_checkpoint(checkpoint)
    restored = build_model(restored_config).eval()
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with torch.no_grad():
        actual = restored(**inputs)["pred"]
    torch.testing.assert_close(actual, expected)


def test_nested_output_head_config_is_reconstructed_by_factory():
    config = asdict(_model_config())
    config["output_head"] = {
        "type": "multi_horizon_shared_trunk",
        "hidden_dim": 11,
        "dropout": 0.0,
    }
    resolved = model_config_from_dict(config)
    assert isinstance(resolved, FusionTimeHeightMLPConfig)
    assert resolved.output_head_hidden_dim == 11


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("context_hours", 0, "context_hours"),
        ("mlp_d_model", 0, "mlp_d_model"),
        ("num_mixer_blocks", 0, "num_mixer_blocks"),
        ("mixer_dropout", 1.0, "mixer_dropout"),
        ("target_projection_dropout", -0.1, "target_projection_dropout"),
    ],
)
def test_config_validation(field, value, message):
    with pytest.raises(ValueError, match=message):
        HTQFusionTimeHeightMLP(replace(_model_config(), **{field: value}))
