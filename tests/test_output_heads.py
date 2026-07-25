from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

from src.models.htq_encoder_only import EncoderOnlyConfig, HTQTargetTokenEncoderOnly
from src.models.htq_transformer import CausalHTQTransformer, HTQConfig
from src.models.model_factory import build_model, model_config_from_dict
from src.models.output_heads import (
    MultiHorizonIndependentMLPResidualHead,
    MultiHorizonSharedTrunkResidualHead,
    ResidualHeadConfig,
    build_residual_head,
    residual_head_parameter_counts,
)


@pytest.mark.parametrize(
    "head_type",
    [
        "shared_linear",
        "shared_mlp",
        "multi_horizon_shared_trunk",
        "multi_horizon_independent_mlp",
    ],
)
def test_output_head_shapes(head_type):
    config = ResidualHeadConfig(
        type=head_type,
        hidden_dim=12,
        dropout=0.0,
    )
    head = build_residual_head(
        d_model=16,
        target_steps=6,
        output_channels=2,
        config=config,
    )
    output = head(torch.randn(2, 6, 6, 16))
    assert output.shape == (2, 6, 6, 2)
    assert torch.isfinite(output).all()


def test_shared_trunk_horizon_heads_are_equal_but_independent():
    head = MultiHorizonSharedTrunkResidualHead(
        16,
        6,
        2,
        ResidualHeadConfig(
            type="multi_horizon_shared_trunk",
            hidden_dim=12,
            dropout=0.0,
            identical_horizon_init=True,
        ),
    )
    first, second = head.horizon_heads[:2]
    assert first is not second
    assert torch.equal(first.weight, second.weight)
    assert first.weight.data_ptr() != second.weight.data_ptr()


def test_independent_mlps_are_equal_but_independent():
    head = MultiHorizonIndependentMLPResidualHead(
        16,
        6,
        2,
        ResidualHeadConfig(
            type="multi_horizon_independent_mlp",
            hidden_dim=12,
            dropout=0.0,
            identical_horizon_init=True,
        ),
    )
    first, second = head.horizon_mlps[:2]
    assert first is not second
    assert torch.equal(first[1].weight, second[1].weight)
    assert first[1].weight.data_ptr() != second[1].weight.data_ptr()
    assert torch.equal(first[-1].weight, second[-1].weight)
    assert first[-1].weight.data_ptr() != second[-1].weight.data_ptr()


def _has_nonzero_gradient(module):
    return any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in module.parameters()
    )


def _has_no_gradient(module):
    return all(
        parameter.grad is None or bool(torch.all(parameter.grad == 0))
        for parameter in module.parameters()
    )


def test_shared_trunk_gradient_routes_only_to_selected_final_head():
    head = MultiHorizonSharedTrunkResidualHead(
        16,
        6,
        2,
        ResidualHeadConfig(
            type="multi_horizon_shared_trunk",
            hidden_dim=12,
            dropout=0.0,
        ),
    )
    head(torch.randn(2, 6, 6, 16))[:, 2].sum().backward()
    assert _has_nonzero_gradient(head.trunk)
    assert _has_nonzero_gradient(head.horizon_heads[2])
    assert all(
        _has_no_gradient(module)
        for index, module in enumerate(head.horizon_heads)
        if index != 2
    )


def test_independent_mlp_gradient_routes_only_to_selected_horizon():
    head = MultiHorizonIndependentMLPResidualHead(
        16,
        6,
        2,
        ResidualHeadConfig(
            type="multi_horizon_independent_mlp",
            hidden_dim=12,
            dropout=0.0,
        ),
    )
    head(torch.randn(2, 6, 6, 16))[:, 2].sum().backward()
    assert _has_nonzero_gradient(head.horizon_mlps[2])
    assert all(
        _has_no_gradient(module)
        for index, module in enumerate(head.horizon_mlps)
        if index != 2
    )


def _encoder_decoder_config(head_type):
    return HTQConfig(
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        context_hours=2,
        target_steps=6,
        height_levels=6,
        query_builder_type="fixed",
        output_head_type=head_type,
        output_head_hidden_dim=12,
        output_head_dropout=0.0,
    )


def _encoder_only_config(head_type):
    return EncoderOnlyConfig(
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        context_hours=2,
        target_steps=6,
        height_levels=6,
        residual_head_hidden_dim=12,
        residual_head_dropout=0.0,
        output_head_type=head_type,
        output_head_hidden_dim=12,
        output_head_dropout=0.0,
        fusion_nhead=4,
    )


def _inputs(context_hours=2, batch_size=2):
    return {
        "x_hourly": torch.randn(batch_size, context_hours, 6, 2),
        "x_mask": torch.ones(batch_size, context_hours, 6, 2, dtype=torch.bool),
        "current_hourly_reference": torch.randn(batch_size, 6, 2),
        "height_values": torch.tensor(
            [[175, 200, 225, 250, 275, 300]],
            dtype=torch.float32,
        ).repeat(batch_size, 1),
    }


@pytest.mark.parametrize(
    "head_type",
    [
        "shared_linear",
        "shared_mlp",
        "multi_horizon_shared_trunk",
        "multi_horizon_independent_mlp",
    ],
)
def test_encoder_decoder_supports_all_output_heads(head_type):
    model = CausalHTQTransformer(_encoder_decoder_config(head_type)).eval()
    output = model(**_inputs(), return_features=True)
    assert output["pred"].shape == (2, 6, 6, 2)
    assert output["residual"].shape == (2, 6, 6, 2)
    assert output["target_features"].shape == (2, 6, 6, 16)


@pytest.mark.parametrize(
    "head_type",
    [
        "shared_mlp",
        "multi_horizon_shared_trunk",
        "multi_horizon_independent_mlp",
    ],
)
def test_encoder_only_supports_configured_output_heads(head_type):
    model = HTQTargetTokenEncoderOnly(_encoder_only_config(head_type)).eval()
    output = model(**_inputs(), return_features=True)
    assert output["pred"].shape == (2, 6, 6, 2)
    assert output["residual"].shape == (2, 6, 6, 2)
    assert output["target_features"].shape == (2, 6, 6, 16)


@pytest.mark.parametrize(
    ("model", "inputs"),
    [
        (
            CausalHTQTransformer(
                _encoder_decoder_config("multi_horizon_shared_trunk")
            ),
            _inputs(),
        ),
        (
            HTQTargetTokenEncoderOnly(
                _encoder_only_config("multi_horizon_shared_trunk")
            ),
            _inputs(),
        ),
    ],
)
def test_multi_horizon_model_backward_has_finite_gradients(model, inputs):
    output = model(**inputs)
    output["pred"].square().mean().backward()
    body_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("residual_head.") and parameter.requires_grad
    ]
    head_gradients = [
        parameter.grad for parameter in model.residual_head.parameters()
    ]
    assert any(gradient is not None for gradient in body_gradients)
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in body_gradients
    )
    assert head_gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in head_gradients
    )


def test_legacy_encoder_decoder_state_dict_strict_load():
    legacy = CausalHTQTransformer(HTQConfig(query_builder_type="fixed"))
    assert "residual_head.weight" in legacy.state_dict()
    assert "residual_head.bias" in legacy.state_dict()
    config = model_config_from_dict(
        {
            "architecture": "htq_encoder_decoder",
            "query_builder_type": "fixed",
        }
    )
    restored = build_model(config)
    restored.load_state_dict(legacy.state_dict(), strict=True)


def test_legacy_encoder_only_state_dict_strict_load():
    legacy_config = EncoderOnlyConfig(
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=32,
        fusion_nhead=4,
        residual_head_hidden_dim=12,
    )
    legacy = HTQTargetTokenEncoderOnly(legacy_config)
    assert "residual_head.0.weight" in legacy.state_dict()
    assert "residual_head.4.weight" in legacy.state_dict()
    restored_config = model_config_from_dict(
        {
            "architecture": "htq_target_token_encoder_only",
            "d_model": 16,
            "nhead": 4,
            "num_encoder_layers": 1,
            "dim_feedforward": 32,
            "fusion_nhead": 4,
            "residual_head_hidden_dim": 12,
        }
    )
    restored = build_model(restored_config)
    restored.load_state_dict(legacy.state_dict(), strict=True)


def test_multi_horizon_save_load_output_is_identical(tmp_path):
    config = _encoder_decoder_config("multi_horizon_shared_trunk")
    model = build_model(config).eval()
    inputs = _inputs()
    with torch.no_grad():
        expected = model(**inputs)["pred"]
    path = tmp_path / "multi_horizon.pt"
    torch.save(
        {
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    checkpoint = torch.load(path, map_location="cpu")
    restored = build_model(model_config_from_dict(checkpoint["model_config"])).eval()
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with torch.no_grad():
        actual = restored(**inputs)["pred"]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_dim", 0, "hidden_dim"),
        ("dropout", 1.0, "dropout"),
        ("final_weight_std", -1.0, "final_weight_std"),
        ("share_across_heights", False, "share_across_heights"),
    ],
)
def test_output_head_config_validation(field, value, message):
    values = {"type": "multi_horizon_shared_trunk"}
    values[field] = value
    with pytest.raises(ValueError, match=message):
        build_residual_head(
            d_model=16,
            target_steps=6,
            output_channels=2,
            config=ResidualHeadConfig(**values),
        )


def test_output_head_parameter_counts_match_built_modules():
    dimensions = {
        "d_model": 16,
        "target_steps": 6,
        "output_channels": 2,
        "hidden_dim": 12,
    }
    expected = residual_head_parameter_counts(**dimensions)
    for head_type, parameter_count in expected.items():
        head = build_residual_head(
            d_model=dimensions["d_model"],
            target_steps=dimensions["target_steps"],
            output_channels=dimensions["output_channels"],
            config=ResidualHeadConfig(
                type=head_type,
                hidden_dim=dimensions["hidden_dim"],
            ),
        )
        assert sum(parameter.numel() for parameter in head.parameters()) == parameter_count
