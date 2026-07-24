from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.evaluate import model_config_from_checkpoint
from scripts.train import default_loss_config, model_forward, save_checkpoint
from src.data.dataset import WindDownscalingDataset
from src.models.htq_encoder_only import EncoderOnlyConfig, HTQTargetTokenEncoderOnly
from src.models.htq_transformer import CausalHTQTransformer, HTQConfig
from src.models.model_factory import build_model


def _config(**overrides):
    values = {
        "d_model": 32,
        "nhead": 4,
        "num_encoder_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "context_hours": 12,
        "target_steps": 6,
        "height_levels": 6,
        "use_meteo": True,
        "use_static": True,
        "meteo_context_hours": 12,
        "fusion_nhead": 4,
        "fusion_dropout": 0.0,
        "static_dropout": 0.0,
        "residual_head_dropout": 0.0,
    }
    values.update(overrides)
    return EncoderOnlyConfig(**values)


def _batch(batch_size=2):
    x_mask = torch.ones(batch_size, 12, 6, 2, dtype=torch.bool)
    x_mask[:, 2, 1, 0] = False  # One missing channel keeps the token valid.
    x_mask[:, 3, 2, :] = False  # Both missing channels mask the token.
    meteo_mask = torch.ones(batch_size, 12, 5, 2, dtype=torch.bool)
    meteo_mask[:, 1, 2, :] = False
    return {
        "x_hourly": torch.randn(batch_size, 12, 6, 2),
        "x_mask": x_mask,
        "x_meteo": torch.randn(batch_size, 12, 5, 2),
        "meteo_mask": meteo_mask,
        "x_static": torch.rand(batch_size, 17),
        "current_hourly_y_norm": torch.randn(batch_size, 6, 2),
        "height": torch.tensor([[250, 275, 300, 325, 350, 375]], dtype=torch.float32).repeat(
            batch_size, 1
        ),
    }


def test_model_factory_builds_old_and_encoder_only_models():
    old = build_model(HTQConfig())
    new = build_model(_config())
    assert isinstance(old, CausalHTQTransformer)
    assert isinstance(new, HTQTargetTokenEncoderOnly)


def test_encoder_only_forward_shapes_and_token_validity():
    model = HTQTargetTokenEncoderOnly(_config()).eval()
    batch = _batch()
    tokenized = model.tokenizer(batch["x_hourly"], batch["x_mask"])
    assert bool(tokenized.token_valid[:, 2, 1].all())
    assert not bool(tokenized.token_valid[:, 3, 2].any())

    with torch.no_grad():
        output = model_forward(model, batch)

    assert output["pred"].shape == (2, 6, 6, 2)
    assert output["residual"].shape == (2, 6, 6, 2)
    assert output["target_features"].shape == (2, 6, 6, 32)
    assert output["encoder_memory"].shape == (2, 72, 32)
    assert torch.isfinite(output["pred"]).all()
    assert torch.allclose(
        output["pred"],
        batch["current_hourly_y_norm"].unsqueeze(1) + output["residual"],
    )


def test_encoder_only_block_attention_mask_direction():
    mask = HTQTargetTokenEncoderOnly.build_attention_mask(
        72,
        36,
        use_block_attention_mask=True,
        allow_target_to_target_attention=True,
        device=torch.device("cpu"),
    )
    assert mask.shape == (108, 108)
    assert bool(mask[:72, 72:].all())
    assert not bool(mask[72:, :72].any())
    assert not bool(mask[72:, 72:].any())

    no_target_exchange = HTQTargetTokenEncoderOnly.build_attention_mask(
        72,
        36,
        use_block_attention_mask=True,
        allow_target_to_target_attention=False,
        device=torch.device("cpu"),
    )
    target_block = no_target_exchange[72:, 72:]
    assert not bool(target_block.diagonal().any())
    assert int(target_block.sum()) == 36 * 35


def test_encoder_only_backward_reaches_all_trainable_subsystems():
    model = HTQTargetTokenEncoderOnly(_config()).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    optimizer.zero_grad(set_to_none=True)
    output = model_forward(model, _batch())
    output["pred"].abs().mean().backward()

    modules = {
        "encoder": model.encoder,
        "target_tokens": model.target_token_builder,
        "height": model.physical_height_encoder,
        "residual_head": model.residual_head,
        "meteo": model.meteo_encoder,
        "static": model.static_encoder,
        "fusion": model.fusion,
    }
    for name, module in modules.items():
        assert module is not None
        gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert gradients, name
        assert all(gradient is not None for gradient in gradients), name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), name
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()


def test_encoder_only_residual_initialization_is_small_but_not_zero():
    torch.manual_seed(7)
    model = HTQTargetTokenEncoderOnly(_config()).eval()
    with torch.no_grad():
        residual = model_forward(model, _batch())["residual"]
    assert 0.0 < float(residual.abs().mean()) < 0.05


def test_encoder_only_checkpoint_roundtrip_and_legacy_default(tmp_path):
    config = _config()
    model = build_model(config).eval()
    batch = _batch()
    with torch.no_grad():
        expected = model_forward(model, batch)["pred"]

    path = tmp_path / "encoder_only.pt"
    torch.save(
        {
            "architecture": config.architecture,
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    checkpoint = torch.load(path, map_location="cpu")
    restored = build_model(model_config_from_checkpoint(checkpoint)).eval()
    restored.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        actual = model_forward(restored, batch)["pred"]
    assert torch.allclose(actual, expected)

    legacy_config = HTQConfig(query_builder_type="fixed")
    legacy_checkpoint = {"model_config": asdict(legacy_config)}
    assert isinstance(build_model(model_config_from_checkpoint(legacy_checkpoint)), CausalHTQTransformer)


def test_training_checkpoint_contains_architecture_scheduler_and_norm_stats(tmp_path):
    config = _config()
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "training.pt"
    norm_stats = {"x_mean": [0.0, 0.0], "x_std": [1.0, 1.0]}
    save_checkpoint(
        path,
        model,
        optimizer,
        3,
        {"val_MAE_ms": 1.0},
        config,
        default_loss_config(),
        scheduler=scheduler,
        norm_stats=norm_stats,
    )
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["architecture"] == "htq_target_token_encoder_only"
    assert checkpoint["scheduler_state_dict"] is not None
    assert checkpoint["norm_stats"] == norm_stats


def test_dataset_exposes_physical_height_values_in_training_batch(tmp_path):
    dataset_dir = tmp_path / "dataset"
    split_dir = dataset_dir / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("0\n", encoding="utf-8")
    heights = np.asarray([[250, 275, 300, 325, 350, 375]], dtype=np.float32)
    np.savez_compressed(
        dataset_dir / "dataset.npz",
        x_hourly=np.zeros((1, 12, 6, 2), dtype=np.float32),
        x_mask=np.ones((1, 12, 6, 2), dtype=bool),
        y_10min=np.zeros((1, 6, 6, 2), dtype=np.float32),
        y_mask=np.ones((1, 6, 6, 2), dtype=bool),
        current_hourly=np.zeros((1, 6, 2), dtype=np.float32),
        station_id=np.asarray(["station_a"]),
        target_time_start=np.asarray(["2026-01-01T00:00"]),
        target_times_10min=np.asarray(
            [[f"2026-01-01T00:{minute:02d}" for minute in range(0, 60, 10)]]
        ),
        height_values=heights,
        source_file=np.asarray(["dummy"]),
        split=np.asarray(["train"]),
    )
    item = WindDownscalingDataset(dataset_dir, split="train", return_metadata=False)[0]
    assert item["height_values"].dtype == torch.float32
    assert torch.equal(item["height_values"], item["height"])
