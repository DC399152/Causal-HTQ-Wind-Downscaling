import numpy as np

from scripts.analyze_residual_predictability import (
    _knn_baselines,
    _knn_oracle_best_of_k_baselines,
    _metrics_from_residual_prediction,
    _topk_neighbor_indices,
)


def _dummy_residuals():
    # Shapes: train [3, T=2, H=1, C=2], val [1, T=2, H=1, C=2].
    train_features = np.asarray([[0.0], [0.1], [0.2]], dtype=np.float32)
    eval_features = np.asarray([[0.0]], dtype=np.float32)
    train_residual = np.asarray(
        [
            [[[[0.0, 0.0]], [[0.0, 0.0]]]],
            [[[[2.0, 2.0]], [[2.0, 2.0]]]],
            [[[[10.0, 10.0]], [[10.0, 10.0]]]],
        ],
        dtype=np.float32,
    ).reshape(3, 2, 1, 2)
    eval_residual = np.asarray([[[[2.0, 2.0]], [[2.0, 2.0]]]], dtype=np.float32).reshape(1, 2, 1, 2)
    train_mask = np.ones_like(train_residual, dtype=bool)
    eval_mask = np.ones_like(eval_residual, dtype=bool)
    residual_mag = np.abs(eval_residual[..., 0])
    return train_features, eval_features, train_residual, eval_residual, train_mask, eval_mask, residual_mag


def test_knn_oracle_outputs_valid_metrics_and_beats_mean_in_dummy_case():
    train_features, eval_features, train_residual, eval_residual, train_mask, eval_mask, residual_mag = _dummy_residuals()
    top_idx, top_dist = _topk_neighbor_indices(train_features, eval_features, max_k=3, chunk_size=2)

    mean_result = _knn_baselines(
        train_residual,
        train_mask,
        eval_residual,
        eval_mask,
        residual_mag,
        top_idx,
        ks=[3],
    )
    oracle = _knn_oracle_best_of_k_baselines(
        train_residual,
        eval_residual,
        eval_mask,
        residual_mag,
        top_idx,
        top_dist,
        ks=[3],
    )

    assert oracle["k3"]["metrics"]["MAE_ms"] < mean_result["k3"]["metrics"]["MAE_ms"]
    assert oracle["k3"]["metrics"]["MAE_ms"] == 0.0
    assert np.isfinite(oracle["k3"]["metrics"]["RMSE_ms"])
    assert oracle["k3"]["selected_neighbor_index"].shape == (1,)


def test_knn_oracle_prediction_shape_can_be_reconstructed_from_selected_indices():
    train_features, eval_features, train_residual, eval_residual, _, eval_mask, residual_mag = _dummy_residuals()
    top_idx, top_dist = _topk_neighbor_indices(train_features, eval_features, max_k=2, chunk_size=2)
    oracle = _knn_oracle_best_of_k_baselines(
        train_residual,
        eval_residual,
        eval_mask,
        residual_mag,
        top_idx,
        top_dist,
        ks=[2],
    )

    selected = oracle["k2"]["selected_neighbor_index"]
    pred = train_residual[selected]

    assert pred.shape == eval_residual.shape
    assert np.isfinite(pred).all()


def test_knn_oracle_mask_changes_candidate_selection():
    train_features = np.asarray([[0.0], [0.1]], dtype=np.float32)
    eval_features = np.asarray([[0.0]], dtype=np.float32)
    train_residual = np.asarray(
        [
            [[[[0.0, 0.0]], [[100.0, 100.0]]]],
            [[[[1.0, 1.0]], [[0.0, 0.0]]]],
        ],
        dtype=np.float32,
    ).reshape(2, 2, 1, 2)
    eval_residual = np.asarray([[[[1.0, 1.0]], [[100.0, 100.0]]]], dtype=np.float32).reshape(1, 2, 1, 2)
    eval_mask = np.asarray([[[[True, True]], [[False, False]]]], dtype=bool).reshape(1, 2, 1, 2)
    residual_mag = np.abs(eval_residual[..., 0])
    top_idx, top_dist = _topk_neighbor_indices(train_features, eval_features, max_k=2, chunk_size=2)

    oracle = _knn_oracle_best_of_k_baselines(
        train_residual,
        eval_residual,
        eval_mask,
        residual_mag,
        top_idx,
        top_dist,
        ks=[2],
    )

    assert int(oracle["k2"]["selected_neighbor_index"][0]) == 1
    assert oracle["k2"]["metrics"]["MAE_ms"] == 0.0


def test_topk_handles_k_larger_than_train_samples_safely():
    train_features, eval_features, train_residual, eval_residual, _, eval_mask, residual_mag = _dummy_residuals()
    top_idx, top_dist = _topk_neighbor_indices(train_features, eval_features, max_k=10, chunk_size=2)

    assert top_idx.shape == (1, 3)
    oracle = _knn_oracle_best_of_k_baselines(
        train_residual,
        eval_residual,
        eval_mask,
        residual_mag,
        top_idx,
        top_dist,
        ks=[3],
    )
    assert np.isfinite(oracle["k3"]["metrics"]["MAE_ms"])


def test_metrics_from_residual_prediction_no_nan_for_nonconstant_series():
    pred = np.asarray([[[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]]], dtype=np.float32)
    target = np.asarray([[[[0.0, 0.0]], [[2.0, 2.0]], [[4.0, 4.0]]]], dtype=np.float32)
    mask = np.ones_like(target, dtype=bool)

    metrics = _metrics_from_residual_prediction(pred, target, mask)

    assert np.isfinite(metrics["MAE_ms"])
    assert np.isfinite(metrics["RMSE_ms"])
    assert np.isfinite(metrics["residual_ACC"])
