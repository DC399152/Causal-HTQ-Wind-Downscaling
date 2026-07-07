import torch

from src.models.query_builder import ContextConditionedQueryBuilder, FixedTargetQueryBuilder


def test_context_conditioned_query_builder_shape():
    builder = ContextConditionedQueryBuilder(d_model=64, target_steps=6, context_hours=6, height_levels=6)
    encoder_memory = torch.randn(2, 36, 64)

    queries = builder(encoder_memory)

    assert queries.shape == (2, 36, 64)


def test_context_conditioned_queries_change_with_encoder_memory():
    builder = ContextConditionedQueryBuilder(d_model=64, target_steps=6, context_hours=6, height_levels=6)
    builder.eval()
    encoder_memory_a = torch.randn(2, 36, 64)
    encoder_memory_b = encoder_memory_a + 0.5

    with torch.no_grad():
        queries_a = builder(encoder_memory_a)
        queries_b = builder(encoder_memory_b)

    assert not torch.allclose(queries_a, queries_b)


def test_context_conditioned_queries_are_deterministic_in_eval_mode():
    builder = ContextConditionedQueryBuilder(d_model=64, target_steps=6, context_hours=6, height_levels=6)
    builder.eval()
    encoder_memory = torch.randn(2, 36, 64)

    with torch.no_grad():
        queries_1 = builder(encoder_memory)
        queries_2 = builder(encoder_memory)

    assert torch.allclose(queries_1, queries_2)


def test_context_conditioned_queries_are_finite():
    builder = ContextConditionedQueryBuilder(d_model=64, target_steps=6, context_hours=6, height_levels=6)
    encoder_memory = torch.randn(2, 36, 64)

    queries = builder(encoder_memory)

    assert torch.isfinite(queries).all()


def test_fixed_and_context_conditioned_queries_differ():
    torch.manual_seed(0)
    fixed_builder = FixedTargetQueryBuilder(d_model=64, target_steps=6, height_levels=6)
    context_builder = ContextConditionedQueryBuilder(d_model=64, target_steps=6, context_hours=6, height_levels=6)
    encoder_memory = torch.randn(2, 36, 64)

    fixed_queries = fixed_builder(encoder_memory)
    context_queries = context_builder(encoder_memory)

    assert fixed_queries.shape == context_queries.shape
    assert not torch.allclose(fixed_queries, context_queries)
