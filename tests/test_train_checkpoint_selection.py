import math

import pytest

torch = pytest.importorskip("torch")

from scripts.train import load_best_checkpoint_for_test, validate_monitor_value, write_metrics_json


def test_load_best_checkpoint_for_test_restores_best_epoch_state(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    best_path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "metrics": {"val_MAE_ms": 0.1},
            "model_config": {},
            "loss_config": {"type": "residual_physics"},
        },
        best_path,
    )

    with torch.no_grad():
        model.weight.fill_(4.0)
    checkpoint = load_best_checkpoint_for_test(best_path, model, torch.device("cpu"))

    assert checkpoint["epoch"] == 2
    assert model.weight.item() == pytest.approx(2.0)


def test_load_best_checkpoint_for_test_raises_when_missing(tmp_path):
    model = torch.nn.Linear(1, 1)
    with pytest.raises(FileNotFoundError, match="Best checkpoint was not created"):
        load_best_checkpoint_for_test(tmp_path / "best.pt", model, torch.device("cpu"))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_validate_monitor_value_rejects_non_finite(value):
    with pytest.raises(ValueError, match="Validation monitor MAE_ms is not finite"):
        validate_monitor_value("MAE_ms", value)


def test_validate_monitor_value_accepts_finite():
    validate_monitor_value("MAE_ms", 0.5)


def test_write_metrics_json_incrementally_replaces_history(tmp_path):
    import json

    path = tmp_path / "metrics.json"
    summary = {"status": "running", "history": [{"epoch": 1, "train_loss_norm_total": 1.0}]}
    write_metrics_json(path, summary)

    summary["history"].append({"epoch": 2, "train_loss_norm_total": 0.5})
    summary["status"] = "interrupted"
    write_metrics_json(path, summary)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "interrupted"
    assert [row["epoch"] for row in saved["history"]] == [1, 2]
    assert not path.with_suffix(".json.tmp").exists()
