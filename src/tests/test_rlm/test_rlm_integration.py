import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.config.rlm_config import RLMConfig, ToolType
from src.core.rlm.rlm_core import (
    RecursiveLanguageModel,
    RLMOrchestrator,
    RLMResult,
)


class TestRLMResult:
    def test_default_values(self):
        result = RLMResult(answer="test", success=True)
        assert result.answer == "test"
        assert result.success is True
        assert result.tool_trace == []
        assert result.total_tool_calls == 0

    def test_full_result(self):
        result = RLMResult(
            answer=jnp.zeros(10),
            success=True,
            tool_trace=[{"tool": "peek"}],
            total_tool_calls=5,
            max_recursion_depth_reached=2,
            execution_time=1.5,
            tokens_used=100,
        )
        assert result.total_tool_calls == 5
        assert result.max_recursion_depth_reached == 2


class TestRecursiveLanguageModelModule:
    @pytest.fixture
    def config(self):
        return RLMConfig(
            max_recursion_depth=3,
            tool_budget=10,
            context_peek_size=500,
        )

    def test_init_and_forward(self, config):
        def forward(query_emb, context_len, depth, calls):
            model = RecursiveLanguageModel(d_model=128, config=config)
            return model(query_emb, context_len, depth, calls)

        import haiku as hk
        init_fn, apply_fn = hk.transform(forward)

        rng = jax.random.PRNGKey(42)
        query_emb = jnp.zeros((1, 128))

        params = init_fn(rng, query_emb, 5000, 0, 0)
        assert params is not None

        tool_probs, term_prob, parameters, encoded_query = apply_fn(
            params, rng, query_emb, 5000, 0, 0
        )

        assert tool_probs.shape[-1] == len(ToolType)
        assert term_prob.shape[-1] == 1
        assert "peek_start" in parameters
        assert encoded_query.shape == (1, 128)

    def test_tool_probabilities_sum_to_one(self, config):
        def forward(query_emb, context_len, depth, calls):
            model = RecursiveLanguageModel(d_model=128, config=config)
            return model(query_emb, context_len, depth, calls)

        import haiku as hk
        init_fn, apply_fn = hk.transform(forward)

        rng = jax.random.PRNGKey(42)
        query_emb = jnp.zeros((1, 128))
        params = init_fn(rng, query_emb, 5000, 0, 0)

        tool_probs, _, _, _ = apply_fn(params, rng, query_emb, 5000, 0, 0)

        prob_sum = float(tool_probs.sum())
        assert abs(prob_sum - 1.0) < 0.01

    def test_answer_synthesis(self, config):
        def forward(query_emb, results):
            model = RecursiveLanguageModel(d_model=128, config=config)
            return model.synthesize_answer(query_emb, results)

        import haiku as hk
        init_fn, apply_fn = hk.transform(forward)

        rng = jax.random.PRNGKey(42)
        query_emb = jnp.zeros((1, 128))
        intermediate_results = [jnp.ones((1, 128)), jnp.ones((1, 128)) * 2]

        params = init_fn(rng, query_emb, intermediate_results)
        answer = apply_fn(params, rng, query_emb, intermediate_results)

        assert answer.shape == (1, 128)

    def test_answer_synthesis_empty_results(self, config):
        def forward(query_emb, results):
            model = RecursiveLanguageModel(d_model=128, config=config)
            return model.synthesize_answer(query_emb, results)

        import haiku as hk
        init_fn, apply_fn = hk.transform(forward)

        rng = jax.random.PRNGKey(42)
        query_emb = jnp.zeros((1, 128))

        params = init_fn(rng, query_emb, [])
        answer = apply_fn(params, rng, query_emb, [])

        assert answer.shape == (1, 128)


class TestRLMOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        config = RLMConfig(
            max_recursion_depth=3,
            tool_budget=10,
            direct_context_threshold=100,
            auto_partition_threshold=500,
            fallback_to_direct=True,
        )
        return RLMOrchestrator(d_model=128, config=config)

    def test_init(self, orchestrator):
        rng = jax.random.PRNGKey(42)
        params = orchestrator.init(rng)
        assert params is not None

    def test_solve_short_context_direct_pass(self, orchestrator):
        rng = jax.random.PRNGKey(42)
        orchestrator.init(rng)

        result = orchestrator.solve(
            query="What is this?",
            context="Short context.",
            rng=rng,
        )

        assert result.success is True
        assert result.answer == "Short context."
        assert result.tool_trace[0]["tool"] == "direct_pass"

    def test_solve_long_context(self, orchestrator):
        rng = jax.random.PRNGKey(42)
        orchestrator.init(rng)

        long_context = "A" * 5000
        result = orchestrator.solve(
            query="Analyze this document.",
            context=long_context,
            rng=rng,
        )

        assert result.success is True
        assert result.execution_time > 0

    def test_solve_with_embedding_fn(self):
        def mock_embedding(text):
            return np.random.randn(128).astype(np.float32)

        config = RLMConfig(
            direct_context_threshold=100,
            fallback_to_direct=True,
        )
        orchestrator = RLMOrchestrator(
            d_model=128,
            config=config,
            embedding_fn=mock_embedding,
        )

        rng = jax.random.PRNGKey(42)
        orchestrator.init(rng)

        result = orchestrator.solve(
            query="Test query",
            context="Short.",
            rng=rng,
        )

        assert result.success is True

    def test_solve_not_initialized(self, orchestrator):
        with pytest.raises(ValueError, match="not initialized"):
            orchestrator.solve("query", "context")

    def test_reset(self, orchestrator):
        rng = jax.random.PRNGKey(42)
        orchestrator.init(rng)

        orchestrator.context_store.store("test", "content")
        assert len(orchestrator.context_store) > 0

        orchestrator.reset()
        assert len(orchestrator.context_store) == 0

    def test_get_stats(self, orchestrator):
        rng = jax.random.PRNGKey(42)
        orchestrator.init(rng)

        stats = orchestrator.get_stats()
        assert "context_store" in stats
        assert "tool_stats" in stats
        assert "recursive_manager" in stats


class TestRLMConfig:
    def test_default_config(self):
        config = RLMConfig()
        assert config.enabled is True
        assert config.max_recursion_depth == 5
        assert config.tool_budget == 20

    def test_minimal_preset(self):
        config = RLMConfig.minimal()
        assert config.max_recursion_depth == 3
        assert config.tool_budget == 10

    def test_aggressive_preset(self):
        config = RLMConfig.aggressive()
        assert config.max_recursion_depth == 8
        assert config.tool_budget == 50

    def test_validation_recursion_depth(self):
        config = RLMConfig(max_recursion_depth=0)
        with pytest.raises(ValueError, match="max_recursion_depth"):
            config.validate()

    def test_validation_tool_budget(self):
        config = RLMConfig(tool_budget=0)
        with pytest.raises(ValueError, match="tool_budget"):
            config.validate()

    def test_validation_chunk_sizes(self):
        config = RLMConfig(min_chunk_size=1000, max_chunk_size=500)
        with pytest.raises(ValueError, match="min_chunk_size"):
            config.validate()

    def test_validation_temperature(self):
        config = RLMConfig(tool_temperature=5.0)
        with pytest.raises(ValueError, match="tool_temperature"):
            config.validate()
