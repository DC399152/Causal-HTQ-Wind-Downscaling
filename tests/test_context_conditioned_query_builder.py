import torch
import pytest

from src.models.query_builder import (
    ContextConditionedQueryBuilder,
    FixedTargetQueryBuilder,
    MultiScaleTrendEmbedding,
    TemporalContextPooling,
)


@pytest.mark.parametrize(
    ("mode_name", "use_temporal_context", "use_multiscale_trend"),
    [
        ("A_time_height_only", False, False),
        ("B_multiscale_trend", False, True),
        ("C_temporal_pooling", True, False),
        ("D_trend_plus_temporal_pooling", True, True),
    ],
)
def test_query_builder_ablation_modes_shape(mode_name, use_temporal_context, use_multiscale_trend):
    print(mode_name)
    builder = ContextConditionedQueryBuilder(
        d_model=64,
        target_steps=6,
        context_hours=6,
        height_levels=6,
        use_temporal_context=use_temporal_context,
        use_multiscale_trend=use_multiscale_trend,
        trend_scales=(1, 3, 5),
    )
    encoder_memory = torch.randn(2, 36, 64)

    queries = builder(encoder_memory)

    assert queries.shape == (2, 36, 64)
    assert torch.isfinite(queries).all()


def test_time_height_only_does_not_change_with_encoder_memory():
    builder = ContextConditionedQueryBuilder(
        d_model=64,
        target_steps=6,
        context_hours=6,
        height_levels=6,
        use_temporal_context=False,
        use_multiscale_trend=False,
    )
    builder.eval()
    encoder_memory_a = torch.randn(2, 36, 64)
    encoder_memory_b = encoder_memory_a + 0.5

    with torch.no_grad():
        queries_a = builder(encoder_memory_a)
        queries_b = builder(encoder_memory_b)

    assert torch.allclose(queries_a, queries_b)


def test_temporal_context_queries_change_with_encoder_memory():
    builder = ContextConditionedQueryBuilder(
        d_model=64,
        target_steps=6,
        context_hours=6,
        height_levels=6,
        use_temporal_context=True,
        use_multiscale_trend=False,
    )
    builder.eval()
    encoder_memory_a = torch.randn(2, 36, 64)
    encoder_memory_b = encoder_memory_a + 0.5

    with torch.no_grad():
        queries_a = builder(encoder_memory_a)
        queries_b = builder(encoder_memory_b)

    assert not torch.allclose(queries_a, queries_b)


def test_multiscale_trend_queries_change_with_encoder_memory():
    builder = ContextConditionedQueryBuilder(
        d_model=64,
        target_steps=6,
        context_hours=6,
        height_levels=6,
        use_temporal_context=False,
        use_multiscale_trend=True,
        trend_scales=(1, 3, 5),
    )
    builder.eval()
    encoder_memory_a = torch.randn(2, 36, 64)
    encoder_memory_b = encoder_memory_a.clone()
    encoder_memory_b[:, -6:] = encoder_memory_b[:, -6:] + 0.5

    with torch.no_grad():
        queries_a = builder(encoder_memory_a)
        queries_b = builder(encoder_memory_b)

    assert not torch.allclose(queries_a, queries_b)


def test_context_conditioned_queries_are_deterministic_in_eval_mode():
    builder = ContextConditionedQueryBuilder(
        d_model=64,
        target_steps=6,
        context_hours=6,
        height_levels=6,
        use_temporal_context=True,
        use_multiscale_trend=True,
    )
    builder.eval()
    encoder_memory = torch.randn(2, 36, 64)

    with torch.no_grad():
        queries_1 = builder(encoder_memory)
        queries_2 = builder(encoder_memory)

    assert torch.allclose(queries_1, queries_2)


def test_context_conditioned_queries_are_finite():
    builder = ContextConditionedQueryBuilder(
        d_model=64,
        target_steps=6,
        context_hours=6,
        height_levels=6,
        use_temporal_context=True,
        use_multiscale_trend=True,
    )
    encoder_memory = torch.randn(2, 36, 64)

    queries = builder(encoder_memory)

    assert torch.isfinite(queries).all()


def test_fixed_and_context_conditioned_queries_differ():
    torch.manual_seed(0)
    fixed_builder = FixedTargetQueryBuilder(d_model=64, target_steps=6, height_levels=6)
    context_builder = ContextConditionedQueryBuilder(
        d_model=64,
        target_steps=6,
        context_hours=6,
        height_levels=6,
        use_temporal_context=True,
    )
    encoder_memory = torch.randn(2, 36, 64)

    fixed_queries = fixed_builder(encoder_memory)
    context_queries = context_builder(encoder_memory)

    assert fixed_queries.shape == context_queries.shape
    assert not torch.allclose(fixed_queries, context_queries)


def test_temporal_context_pooling_shape():
    pooling = TemporalContextPooling(d_model=64)
    memory_4d = torch.randn(2, 6, 6, 64)

    context = pooling(memory_4d)

    assert context.shape == (2, 6, 64)
    assert torch.isfinite(context).all()


def test_multiscale_trend_handles_short_context_safely():
    trend = MultiScaleTrendEmbedding(d_model=64, trend_scales=(1, 3, 5))
    memory_4d = torch.randn(2, 3, 6, 64)

    context = trend(memory_4d)

    assert context.shape == (2, 6, 64)
    assert torch.isfinite(context).all()
