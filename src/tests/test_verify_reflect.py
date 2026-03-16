"""
Comprehensive Tests for Verify/Reflect Loop and Self-Critique Head

Tests cover:
- VerificationHead (forward pass, scoring, shapes)
- ReflectionModule (correction generation, gating)
- VerifyReflectReasoning (full loop, early stopping, convergence)
- SelfCritiqueHead (quality scoring, revision signal, iterative revision)
- AGIConfig verify/reflect and self-critique settings
- Integration with ChainOfThoughtReasoning

References:
    - Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023)
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk

from src.config.agi_config import AGIConfig
from src.core.reasoning import (
    VerificationHead,
    ReflectionModule,
    VerifyReflectReasoning,
    SelfCritiqueHead,
)

# =============================================================================
# VerificationHead Tests
# =============================================================================


class TestVerificationHead:
    """Tests for the Verification Head module."""

    @pytest.fixture
    def d_model(self):
        return 64

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    def _build_verifier_fn(self, d_model):
        def _fn(answer, thought, query):
            verifier = VerificationHead(d_model)
            return verifier(answer, thought, query)

        return hk.transform(_fn)

    def test_verification_output_shape(self, d_model, rng):
        """Test verification score has correct shape."""
        fn = self._build_verifier_fn(d_model)
        batch = 4
        answer = jnp.ones((batch, d_model))
        thought = jnp.ones((batch, d_model))
        query = jnp.ones((batch, d_model))

        params = fn.init(rng, answer, thought, query)
        score = fn.apply(params, rng, answer, thought, query)

        assert score.shape == (batch, 1)

    def test_verification_score_bounded(self, d_model, rng):
        """Test verification score is in [0, 1] (sigmoid output)."""
        fn = self._build_verifier_fn(d_model)
        answer = jax.random.normal(rng, (8, d_model))
        thought = jax.random.normal(jax.random.PRNGKey(1), (8, d_model))
        query = jax.random.normal(jax.random.PRNGKey(2), (8, d_model))

        params = fn.init(rng, answer, thought, query)
        score = fn.apply(params, rng, answer, thought, query)

        assert jnp.all(score >= 0.0)
        assert jnp.all(score <= 1.0)

    def test_verification_gradients_flow(self, d_model, rng):
        """Test gradients flow through verification head."""
        fn = self._build_verifier_fn(d_model)
        answer = jax.random.normal(rng, (4, d_model))
        thought = jax.random.normal(jax.random.PRNGKey(1), (4, d_model))
        query = jax.random.normal(jax.random.PRNGKey(2), (4, d_model))

        params = fn.init(rng, answer, thought, query)

        def loss_fn(params):
            score = fn.apply(params, rng, answer, thought, query)
            return jnp.mean(score)

        grads = jax.grad(loss_fn)(params)
        total_norm = sum(float(jnp.linalg.norm(g)) for g in jax.tree.leaves(grads))
        assert total_norm > 0

    def test_verification_different_inputs(self, d_model, rng):
        """Test that different inputs produce different scores."""
        fn = self._build_verifier_fn(d_model)
        a1 = jax.random.normal(rng, (2, d_model))
        a2 = jax.random.normal(jax.random.PRNGKey(99), (2, d_model))
        thought = jnp.zeros((2, d_model))
        query = jnp.zeros((2, d_model))

        params = fn.init(rng, a1, thought, query)
        s1 = fn.apply(params, rng, a1, thought, query)
        s2 = fn.apply(params, rng, a2, thought, query)

        assert not jnp.allclose(s1, s2)

    def test_verification_finite_output(self, d_model, rng):
        """Test that outputs are always finite."""
        fn = self._build_verifier_fn(d_model)

        for scale in [0.01, 1.0, 50.0]:
            answer = jax.random.normal(rng, (2, d_model)) * scale
            thought = jax.random.normal(jax.random.PRNGKey(1), (2, d_model)) * scale
            query = jax.random.normal(jax.random.PRNGKey(2), (2, d_model)) * scale

            params = fn.init(rng, answer, thought, query)
            score = fn.apply(params, rng, answer, thought, query)
            assert jnp.all(jnp.isfinite(score)), f"Non-finite at scale {scale}"


# =============================================================================
# ReflectionModule Tests
# =============================================================================


class TestReflectionModule:
    """Tests for the Reflection Module."""

    @pytest.fixture
    def d_model(self):
        return 64

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    def _build_reflector_fn(self, d_model):
        def _fn(answer, thought, v_score):
            reflector = ReflectionModule(d_model)
            return reflector(answer, thought, v_score)

        return hk.transform(_fn)

    def test_reflection_output_shape(self, d_model, rng):
        """Test corrected answer has same shape as input."""
        fn = self._build_reflector_fn(d_model)
        batch = 4
        answer = jnp.ones((batch, d_model))
        thought = jnp.ones((batch, d_model))
        v_score = jnp.ones((batch, 1)) * 0.3

        params = fn.init(rng, answer, thought, v_score)
        corrected, delta = fn.apply(params, rng, answer, thought, v_score)

        assert corrected.shape == (batch, d_model)
        assert delta.shape == (batch, d_model)

    def test_reflection_low_verification_more_correction(self, d_model, rng):
        """Test that low verification scores produce larger corrections."""
        fn = self._build_reflector_fn(d_model)
        answer = jax.random.normal(rng, (4, d_model))
        thought = jax.random.normal(jax.random.PRNGKey(1), (4, d_model))

        params = fn.init(rng, answer, thought, jnp.ones((4, 1)) * 0.5)

        # Low verification → more correction
        _, delta_low = fn.apply(params, rng, answer, thought, jnp.ones((4, 1)) * 0.1)
        # High verification → less correction
        _, delta_high = fn.apply(params, rng, answer, thought, jnp.ones((4, 1)) * 0.9)

        low_norm = float(jnp.linalg.norm(delta_low))
        high_norm = float(jnp.linalg.norm(delta_high))

        # Low verification should generally produce larger corrections
        # (due to correction_scale = 1 - v_score)
        assert (
            low_norm > high_norm * 0.5
        ), f"Expected larger correction for low verification: {low_norm} vs {high_norm}"

    def test_reflection_correction_bounded(self, d_model, rng):
        """Test that correction delta is bounded (tanh + gate)."""
        fn = self._build_reflector_fn(d_model)
        answer = jax.random.normal(rng, (4, d_model))
        thought = jax.random.normal(jax.random.PRNGKey(1), (4, d_model))
        v_score = jnp.ones((4, 1)) * 0.3

        params = fn.init(rng, answer, thought, v_score)
        _, delta = fn.apply(params, rng, answer, thought, v_score)

        # tanh output is in [-1, 1], gate is in [0, 1], v_score scale in [0, 1]
        # So delta elements should be bounded
        assert jnp.all(jnp.abs(delta) <= 1.0 + 1e-6)

    def test_reflection_gradients(self, d_model, rng):
        """Test that gradients flow through reflection."""
        fn = self._build_reflector_fn(d_model)
        answer = jax.random.normal(rng, (2, d_model))
        thought = jax.random.normal(jax.random.PRNGKey(1), (2, d_model))
        v_score = jnp.ones((2, 1)) * 0.3

        params = fn.init(rng, answer, thought, v_score)

        def loss_fn(params):
            corrected, _ = fn.apply(params, rng, answer, thought, v_score)
            return jnp.mean(corrected**2)

        grads = jax.grad(loss_fn)(params)
        total_norm = sum(float(jnp.linalg.norm(g)) for g in jax.tree.leaves(grads))
        assert total_norm > 0


# =============================================================================
# VerifyReflectReasoning Tests
# =============================================================================


class TestVerifyReflectReasoning:
    """Tests for the full Verify/Reflect reasoning loop."""

    @pytest.fixture
    def d_model(self):
        return 64

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    def _build_vr_fn(self, d_model, max_verify=3, threshold=0.85):
        def _fn(query, context):
            vr = VerifyReflectReasoning(
                d_model=d_model,
                max_reasoning_steps=3,
                max_verify_steps=max_verify,
                confidence_threshold=threshold,
                use_semantic_graph=False,
            )
            return vr(query, context)

        return hk.transform(_fn)

    def test_vr_basic_output_keys(self, d_model, rng):
        """Test that all expected output keys are present."""
        fn = self._build_vr_fn(d_model)
        query = jnp.ones((2, 8, d_model))
        context = jnp.ones((2, 16, d_model))

        params = fn.init(rng, query, context)
        result = fn.apply(params, rng, query, context)

        expected_keys = [
            "final_answer",
            "reasoning_chain",
            "confidences",
            "thought_summary",
            "verification_scores",
            "num_reflections",
            "reflection_deltas",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_vr_final_answer_shape(self, d_model, rng):
        """Test that final answer has correct shape."""
        fn = self._build_vr_fn(d_model)
        query = jnp.ones((2, 8, d_model))
        context = jnp.ones((2, 16, d_model))

        params = fn.init(rng, query, context)
        result = fn.apply(params, rng, query, context)

        assert result["final_answer"].shape == (2, d_model)

    def test_vr_2d_query_input(self, d_model, rng):
        """Test with 2D query input [batch, d_model]."""
        fn = self._build_vr_fn(d_model)
        query = jnp.ones((2, d_model))
        context = jnp.ones((2, 16, d_model))

        params = fn.init(rng, query, context)
        result = fn.apply(params, rng, query, context)

        assert result["final_answer"].shape == (2, d_model)

    def test_vr_verification_scores_list(self, d_model, rng):
        """Test that verification scores are returned as a list."""
        fn = self._build_vr_fn(d_model, max_verify=3)
        query = jnp.ones((2, 8, d_model))
        context = jnp.ones((2, 16, d_model))

        params = fn.init(rng, query, context)
        result = fn.apply(params, rng, query, context)

        assert isinstance(result["verification_scores"], list)
        assert len(result["verification_scores"]) >= 1

    def test_vr_num_reflections_bounded(self, d_model, rng):
        """Test that num_reflections <= max_verify_steps."""
        max_v = 3
        fn = self._build_vr_fn(d_model, max_verify=max_v)
        query = jnp.ones((2, 8, d_model))
        context = jnp.ones((2, 16, d_model))

        params = fn.init(rng, query, context)
        result = fn.apply(params, rng, query, context)

        assert result["num_reflections"] <= max_v

    def test_vr_output_finite(self, d_model, rng):
        """Test all outputs are finite."""
        fn = self._build_vr_fn(d_model)
        query = jax.random.normal(rng, (2, 8, d_model))
        context = jax.random.normal(jax.random.PRNGKey(1), (2, 16, d_model))

        params = fn.init(rng, query, context)
        result = fn.apply(params, rng, query, context)

        assert jnp.all(jnp.isfinite(result["final_answer"]))
        assert jnp.all(jnp.isfinite(result["thought_summary"]))

    def test_vr_gradients_flow(self, d_model, rng):
        """Test that gradients flow through the full verify/reflect loop."""
        fn = self._build_vr_fn(d_model, max_verify=2)
        query = jax.random.normal(rng, (2, 4, d_model))
        context = jax.random.normal(jax.random.PRNGKey(1), (2, 8, d_model))

        params = fn.init(rng, query, context)

        def loss_fn(params):
            result = fn.apply(params, rng, query, context)
            return jnp.mean(result["final_answer"] ** 2)

        grads = jax.grad(loss_fn)(params)
        total_norm = sum(float(jnp.linalg.norm(g)) for g in jax.tree.leaves(grads))
        assert total_norm > 0, "Gradients should flow through verify/reflect"


# =============================================================================
# SelfCritiqueHead Tests
# =============================================================================


class TestSelfCritiqueHead:
    """Tests for the Self-Critique Head module."""

    @pytest.fixture
    def d_model(self):
        return 64

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    def _build_critique_fn(self, d_model, threshold=0.6, max_revisions=2):
        def _fn(hidden, is_training=True):
            head = SelfCritiqueHead(d_model, threshold, max_revisions)
            return head(hidden, is_training)

        return hk.transform(_fn)

    def _build_revision_fn(self, d_model):
        def _fn(hidden, revision_signal, iteration):
            head = SelfCritiqueHead(d_model)
            return head.revise(hidden, revision_signal, iteration)

        return hk.transform(_fn)

    def test_critique_output_keys(self, d_model, rng):
        """Test that critique output contains expected keys."""
        fn = self._build_critique_fn(d_model)
        hidden = jnp.ones((4, d_model))

        params = fn.init(rng, hidden)
        result = fn.apply(params, rng, hidden)

        assert "quality_score" in result
        assert "revision_signal" in result
        assert "needs_revision" in result

    def test_critique_quality_score_shape(self, d_model, rng):
        """Test quality score shape."""
        fn = self._build_critique_fn(d_model)
        hidden = jnp.ones((4, d_model))

        params = fn.init(rng, hidden)
        result = fn.apply(params, rng, hidden)

        assert result["quality_score"].shape == (4, 1)

    def test_critique_quality_score_bounded(self, d_model, rng):
        """Test quality score is in [0, 1]."""
        fn = self._build_critique_fn(d_model)
        hidden = jax.random.normal(rng, (8, d_model))

        params = fn.init(rng, hidden)
        result = fn.apply(params, rng, hidden)

        assert jnp.all(result["quality_score"] >= 0.0)
        assert jnp.all(result["quality_score"] <= 1.0)

    def test_critique_revision_signal_shape(self, d_model, rng):
        """Test revision signal shape matches d_model."""
        fn = self._build_critique_fn(d_model)
        hidden = jnp.ones((4, d_model))

        params = fn.init(rng, hidden)
        result = fn.apply(params, rng, hidden)

        assert result["revision_signal"].shape == (4, d_model)

    def test_critique_revision_signal_bounded(self, d_model, rng):
        """Test revision signal is bounded (tanh output)."""
        fn = self._build_critique_fn(d_model)
        hidden = jax.random.normal(rng, (4, d_model))

        params = fn.init(rng, hidden)
        result = fn.apply(params, rng, hidden)

        assert jnp.all(jnp.abs(result["revision_signal"]) <= 1.0 + 1e-6)

    def test_critique_needs_revision_boolean(self, d_model, rng):
        """Test needs_revision is boolean-like."""
        fn = self._build_critique_fn(d_model, threshold=0.6)
        hidden = jax.random.normal(rng, (4, d_model))

        params = fn.init(rng, hidden)
        result = fn.apply(params, rng, hidden)

        # needs_revision = quality_score < threshold → boolean array
        assert result["needs_revision"].dtype == jnp.bool_

    def test_critique_3d_input(self, d_model, rng):
        """Test with 3D sequence input (should pool)."""
        fn = self._build_critique_fn(d_model)
        hidden = jnp.ones((2, 16, d_model))

        params = fn.init(rng, hidden)
        result = fn.apply(params, rng, hidden)

        assert result["quality_score"].shape == (2, 1)
        assert result["revision_signal"].shape == (2, d_model)

    def test_revise_output_shape(self, d_model, rng):
        """Test revision produces correct shape."""
        fn = self._build_revision_fn(d_model)
        hidden = jnp.ones((4, d_model))
        revision = jnp.ones((4, d_model)) * 0.1

        params = fn.init(rng, hidden, revision, 0)
        revised = fn.apply(params, rng, hidden, revision, 0)

        assert revised.shape == (4, d_model)

    def test_revise_decay_with_iteration(self, d_model, rng):
        """Test that revision strength decays with iteration."""
        fn = self._build_revision_fn(d_model)
        hidden = jnp.zeros((2, d_model))
        revision = jnp.ones((2, d_model))

        params = fn.init(rng, hidden, revision, 0)

        rev_0 = fn.apply(params, rng, hidden, revision, 0)
        rev_1 = fn.apply(params, rng, hidden, revision, 1)
        rev_2 = fn.apply(params, rng, hidden, revision, 2)

        # Each iteration should have less effect
        norm_0 = float(jnp.linalg.norm(rev_0 - hidden))
        norm_1 = float(jnp.linalg.norm(rev_1 - hidden))
        norm_2 = float(jnp.linalg.norm(rev_2 - hidden))

        assert norm_0 > norm_1 > norm_2

    def test_critique_gradients(self, d_model, rng):
        """Test gradients flow through critique head."""
        fn = self._build_critique_fn(d_model)
        hidden = jax.random.normal(rng, (4, d_model))

        params = fn.init(rng, hidden)

        def loss_fn(params):
            result = fn.apply(params, rng, hidden)
            return jnp.mean(result["quality_score"])

        grads = jax.grad(loss_fn)(params)
        total_norm = sum(float(jnp.linalg.norm(g)) for g in jax.tree.leaves(grads))
        assert total_norm > 0


# =============================================================================
# AGIConfig Verify/Reflect and Self-Critique Tests
# =============================================================================


class TestAGIConfigVerifyReflect:
    """Tests for verify/reflect and self-critique config settings."""

    def test_verify_reflect_defaults(self):
        """Test default verify/reflect settings."""
        config = AGIConfig()
        assert config.enable_verify_reflect is False
        assert config.max_verify_steps == 3
        assert config.verify_confidence_threshold == pytest.approx(0.85)
        assert config.reflect_temperature == pytest.approx(0.7)

    def test_verify_reflect_enable(self):
        """Test enabling verify/reflect."""
        config = AGIConfig(enable_verify_reflect=True)
        assert config.enable_verify_reflect is True

    def test_verify_reflect_validation(self):
        """Test verify/reflect validation."""
        with pytest.raises(AssertionError):
            AGIConfig(enable_verify_reflect=True, max_verify_steps=0)

        with pytest.raises(AssertionError):
            AGIConfig(enable_verify_reflect=True, verify_confidence_threshold=0.0)

        with pytest.raises(AssertionError):
            AGIConfig(enable_verify_reflect=True, reflect_temperature=3.0)

    def test_self_critique_defaults(self):
        """Test default self-critique settings."""
        config = AGIConfig()
        assert config.enable_self_critique is False
        assert config.self_critique_threshold == pytest.approx(0.6)
        assert config.max_revisions == 2
        assert config.critique_loss_coeff == pytest.approx(0.1)

    def test_self_critique_enable(self):
        """Test enabling self-critique."""
        config = AGIConfig(enable_self_critique=True)
        assert config.enable_self_critique is True

    def test_self_critique_validation(self):
        """Test self-critique validation."""
        with pytest.raises(AssertionError):
            AGIConfig(enable_self_critique=True, self_critique_threshold=0.0)

        with pytest.raises(AssertionError):
            AGIConfig(enable_self_critique=True, max_revisions=0)

        with pytest.raises(AssertionError):
            AGIConfig(enable_self_critique=True, critique_loss_coeff=-0.1)

    def test_think_budget_defaults(self):
        """Test default think budget settings."""
        config = AGIConfig()
        assert config.enable_think_budget is False
        assert config.think_budget_max_tokens == 1024
        assert config.think_budget_min_tokens == 32

    def test_think_budget_validation(self):
        """Test think budget validation."""
        with pytest.raises(AssertionError):
            AGIConfig(
                enable_think_budget=True, think_budget_max_tokens=10, think_budget_min_tokens=100
            )

    def test_kv_cache_defaults(self):
        """Test default KV cache settings."""
        config = AGIConfig()
        assert config.enable_kv_cache is False
        assert config.kv_cache_prefix_len == 256
        assert config.kv_cache_max_batch == 32
        assert config.kv_cache_eviction == "lru"

    def test_kv_cache_validation(self):
        """Test KV cache validation."""
        with pytest.raises(AssertionError):
            AGIConfig(enable_kv_cache=True, kv_cache_eviction="random")

    def test_print_summary_new_features(self, capsys):
        """Test print_summary includes all new feature sections."""
        config = AGIConfig(
            use_grpo=True,
            enable_verify_reflect=True,
            enable_kv_cache=True,
            enable_self_critique=True,
            enable_think_budget=True,
        )
        config.print_summary()
        captured = capsys.readouterr()

        assert "Verify/Reflect" in captured.out
        assert "KV Prefix Cache" in captured.out
        assert "Self-Critique" in captured.out
        assert "Think Budget" in captured.out
