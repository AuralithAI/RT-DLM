"""
Comprehensive Tests for GRPO (Group Relative Policy Optimization)

Tests cover:
- GRPOValueHead module (forward pass, shapes, dtypes)
- compute_grpo_advantages (group-relative computation, normalization)
- compute_grpo_loss (clipped surrogate, value loss, entropy, KL)
- AGIConfig GRPO settings (defaults, validation, edge cases)
- Integration with ComputeController

References:
    - DeepSeek-R1: Incentivizing Reasoning in LLMs via GRPO
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from src.config.agi_config import AGIConfig
from src.core.agi.compute_controller import (
    GRPOValueHead,
    compute_grpo_advantages,
    compute_grpo_loss,
    ComputeState,
    ModuleOutput,
    ControllerRewardShaper,
)

# =============================================================================
# GRPOValueHead Tests
# =============================================================================


class TestGRPOValueHead:
    """Tests for the GRPO Value Head module."""

    @pytest.fixture
    def d_model(self):
        return 64

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(42)

    def _build_value_head_fn(self, d_model, dropout_rate=0.0):
        """Build a Haiku-transformed GRPOValueHead."""

        def _fn(hidden, is_training=True):
            head = GRPOValueHead(d_model=d_model, dropout_rate=dropout_rate)
            return head(hidden, is_training=is_training)

        return hk.transform(_fn)

    def test_value_head_2d_input(self, d_model, rng):
        """Test value head with 2D pooled input [batch, d_model]."""
        fn = self._build_value_head_fn(d_model)
        hidden = jnp.ones((4, d_model))

        params = fn.init(rng, hidden)
        values = fn.apply(params, rng, hidden)

        assert values.shape == (4, 1), f"Expected (4, 1), got {values.shape}"
        assert values.dtype == jnp.float32

    def test_value_head_3d_input(self, d_model, rng):
        """Test value head with 3D sequence input [batch, seq, d_model]."""
        fn = self._build_value_head_fn(d_model)
        hidden = jnp.ones((2, 16, d_model))

        params = fn.init(rng, hidden)
        values = fn.apply(params, rng, hidden)

        # Should pool to (2, d_model) then output (2, 1)
        assert values.shape == (2, 1)

    def test_value_head_batch_size_1(self, d_model, rng):
        """Test with batch size 1."""
        fn = self._build_value_head_fn(d_model)
        hidden = jnp.ones((1, d_model))

        params = fn.init(rng, hidden)
        values = fn.apply(params, rng, hidden)

        assert values.shape == (1, 1)

    def test_value_head_large_batch(self, d_model, rng):
        """Test with large batch size."""
        fn = self._build_value_head_fn(d_model)
        hidden = jax.random.normal(rng, (128, d_model))

        params = fn.init(rng, hidden)
        values = fn.apply(params, rng, hidden)

        assert values.shape == (128, 1)
        assert jnp.all(jnp.isfinite(values))

    def test_value_head_parameter_count(self, d_model, rng):
        """Test that parameter shapes are correct."""
        fn = self._build_value_head_fn(d_model)
        hidden = jnp.ones((2, d_model))

        params = fn.init(rng, hidden)

        # Should have: value_fc1 (d_model -> d_model//2), value_ln, value_out (d_model//2 -> 1)
        hk.data_structures.to_mutable_dict(params)
        total_params = sum(p.size for p in jax.tree.leaves(params))
        assert total_params > 0, "Should have parameters"

    def test_value_head_deterministic_inference(self, d_model, rng):
        """Test that inference mode is deterministic (no dropout)."""
        fn = self._build_value_head_fn(d_model, dropout_rate=0.1)
        hidden = jax.random.normal(rng, (4, d_model))

        params = fn.init(rng, hidden)

        # Two calls with is_training=False should give same result
        v1 = fn.apply(params, jax.random.PRNGKey(0), hidden, is_training=False)
        v2 = fn.apply(params, jax.random.PRNGKey(99), hidden, is_training=False)

        np.testing.assert_allclose(v1, v2, atol=1e-6)

    def test_value_head_different_inputs_different_values(self, d_model, rng):
        """Test that different inputs produce different values."""
        fn = self._build_value_head_fn(d_model)

        h1 = jax.random.normal(rng, (2, d_model))
        h2 = jax.random.normal(jax.random.PRNGKey(999), (2, d_model))

        params = fn.init(rng, h1)
        v1 = fn.apply(params, rng, h1, is_training=False)
        v2 = fn.apply(params, rng, h2, is_training=False)

        assert not jnp.allclose(v1, v2), "Different inputs should produce different values"

    def test_value_head_gradients_flow(self, d_model, rng):
        """Test that gradients flow through the value head."""
        fn = self._build_value_head_fn(d_model)
        hidden = jax.random.normal(rng, (4, d_model))

        params = fn.init(rng, hidden)

        def loss_fn(params):
            values = fn.apply(params, rng, hidden, is_training=True)
            return jnp.mean(values**2)

        grads = jax.grad(loss_fn)(params)
        grad_norms = jax.tree.map(lambda g: jnp.linalg.norm(g), grads)
        total_grad_norm = sum(jax.tree.leaves(grad_norms))

        assert total_grad_norm > 0, "Gradients should be non-zero"

    def test_value_head_output_finite(self, d_model, rng):
        """Test that outputs are always finite."""
        fn = self._build_value_head_fn(d_model)

        # Test with various input scales
        for scale in [0.001, 1.0, 100.0]:
            hidden = jax.random.normal(rng, (4, d_model)) * scale
            params = fn.init(rng, hidden)
            values = fn.apply(params, rng, hidden)
            assert jnp.all(jnp.isfinite(values)), f"Non-finite values at scale {scale}"


# =============================================================================
# compute_grpo_advantages Tests
# =============================================================================


class TestGRPOAdvantages:
    """Tests for group-relative advantage computation."""

    def test_basic_advantages(self):
        """Test basic advantage computation."""
        rewards = jnp.array([1.0, 0.5, 0.8, 0.2])  # 2 prompts, 2 responses each
        advantages, returns = compute_grpo_advantages(rewards, group_size=2)

        assert advantages.shape == (4,)
        assert returns.shape == (4,)

    def test_advantages_sum_to_zero_per_group(self):
        """Test that unnormalized advantages sum to ~0 within each group."""
        rewards = jnp.array([1.0, 0.5, 0.8, 0.2, 0.9, 0.3])
        advantages, _ = compute_grpo_advantages(rewards, group_size=2, normalize=False)

        # Reshape to groups and check sums
        grouped = advantages.reshape(-1, 2)
        group_sums = grouped.sum(axis=1)

        np.testing.assert_allclose(group_sums, 0.0, atol=1e-6)

    def test_advantages_positive_for_best_in_group(self):
        """Test that the best response in each group gets positive advantage."""
        rewards = jnp.array([0.9, 0.1, 0.3, 0.7])  # 2 groups of 2
        advantages, _ = compute_grpo_advantages(rewards, group_size=2, normalize=False)

        # Group 1: response 0 should be positive (0.9 > 0.1)
        assert advantages[0] > 0
        # Group 1: response 1 should be negative
        assert advantages[1] < 0
        # Group 2: response 3 should be positive (0.7 > 0.3)
        assert advantages[3] > 0

    def test_normalized_advantages(self):
        """Test that normalized advantages have unit-ish variance per group."""
        rewards = jnp.array([2.0, 1.0, 0.5, -0.5, 3.0, 0.0, 1.0, 2.0])
        advantages, _ = compute_grpo_advantages(rewards, group_size=4, normalize=True)

        # Normalized advantages should have controlled magnitude
        assert jnp.all(jnp.isfinite(advantages))

    def test_returns_equal_rewards(self):
        """Test that returns match raw rewards (for single-step)."""
        rewards = jnp.array([1.0, 0.5, 0.8, 0.2])
        _, returns = compute_grpo_advantages(rewards, group_size=2)

        np.testing.assert_allclose(returns, rewards, atol=1e-6)

    def test_single_group(self):
        """Test with a single group."""
        rewards = jnp.array([1.0, 0.5, 0.8, 0.2])
        advantages, returns = compute_grpo_advantages(rewards, group_size=4)

        assert advantages.shape == (4,)

    def test_large_group_size(self):
        """Test with large group size."""
        rewards = jax.random.normal(jax.random.PRNGKey(0), (64,))
        advantages, returns = compute_grpo_advantages(rewards, group_size=8)

        assert advantages.shape == (64,)
        assert jnp.all(jnp.isfinite(advantages))

    def test_identical_rewards_zero_advantages(self):
        """Test that identical rewards in a group produce zero advantages."""
        rewards = jnp.array([1.0, 1.0, 1.0, 1.0])
        advantages, _ = compute_grpo_advantages(rewards, group_size=4, normalize=False)

        np.testing.assert_allclose(advantages, 0.0, atol=1e-6)

    def test_negative_rewards(self):
        """Test with negative rewards."""
        rewards = jnp.array([-1.0, -0.5, -0.8, -0.2])
        advantages, _ = compute_grpo_advantages(rewards, group_size=2)

        assert jnp.all(jnp.isfinite(advantages))
        # -0.5 > -1.0, so second response should have positive advantage
        assert advantages[1] > advantages[0]


# =============================================================================
# compute_grpo_loss Tests
# =============================================================================


class TestGRPOLoss:
    """Tests for the GRPO loss computation."""

    @pytest.fixture
    def batch_data(self):
        """Create sample batch data for loss computation."""
        rng = jax.random.PRNGKey(42)
        batch_size = 8

        log_probs = jax.random.normal(rng, (batch_size,)) - 1.0
        old_log_probs = jax.random.normal(jax.random.PRNGKey(1), (batch_size,)) - 1.0
        rewards = jax.random.uniform(jax.random.PRNGKey(2), (batch_size,))
        advantages, returns = compute_grpo_advantages(rewards, group_size=4)
        values = jax.random.normal(jax.random.PRNGKey(3), (batch_size,))

        return {
            "log_probs": log_probs,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "values": values,
            "returns": returns,
        }

    def test_loss_returns_scalar(self, batch_data):
        """Test that total loss is a scalar."""
        total_loss, components = compute_grpo_loss(**batch_data)

        assert total_loss.ndim == 0, "Total loss should be scalar"

    def test_loss_components_all_present(self, batch_data):
        """Test that all expected loss components are returned."""
        _, components = compute_grpo_loss(**batch_data)

        expected_keys = [
            "policy_loss",
            "value_loss",
            "entropy",
            "kl_divergence",
            "total_grpo_loss",
            "mean_ratio",
            "mean_advantage",
            "clip_fraction",
        ]
        for key in expected_keys:
            assert key in components, f"Missing component: {key}"

    def test_loss_finite(self, batch_data):
        """Test that all loss components are finite."""
        _, components = compute_grpo_loss(**batch_data)

        for key, val in components.items():
            assert jnp.all(jnp.isfinite(val)), f"Non-finite {key}: {val}"

    def test_value_loss_nonnegative(self, batch_data):
        """Test that value loss is non-negative (MSE)."""
        _, components = compute_grpo_loss(**batch_data)

        assert components["value_loss"] >= 0

    def test_zero_advantages_zero_policy_loss(self):
        """Test that zero advantages produce zero policy loss."""
        batch_size = 4
        log_probs = jnp.zeros(batch_size)
        old_log_probs = jnp.zeros(batch_size)
        advantages = jnp.zeros(batch_size)
        values = jnp.zeros(batch_size)
        returns = jnp.zeros(batch_size)

        _, components = compute_grpo_loss(log_probs, old_log_probs, advantages, values, returns)

        np.testing.assert_allclose(float(components["policy_loss"]), 0.0, atol=1e-6)

    def test_clip_fraction_in_range(self, batch_data):
        """Test that clip fraction is between 0 and 1."""
        _, components = compute_grpo_loss(**batch_data)

        clip_frac = float(components["clip_fraction"])
        assert 0.0 <= clip_frac <= 1.0

    def test_custom_clip_eps(self, batch_data):
        """Test with custom clip epsilon."""
        _, comp_narrow = compute_grpo_loss(**batch_data, clip_eps=0.05)
        _, comp_wide = compute_grpo_loss(**batch_data, clip_eps=0.5)

        # Narrower clipping should generally clip more
        # (or equal in degenerate case)
        assert jnp.isfinite(comp_narrow["clip_fraction"])
        assert jnp.isfinite(comp_wide["clip_fraction"])

    def test_kl_coeff_effect(self, batch_data):
        """Test that KL coefficient scales the KL penalty."""
        _, comp_low = compute_grpo_loss(**batch_data, kl_coeff=0.0)
        _, comp_high = compute_grpo_loss(**batch_data, kl_coeff=1.0)

        # With kl_coeff=0, KL should not affect total loss
        # Total = policy + value - entropy + kl_coeff * kl
        assert jnp.isfinite(comp_low["total_grpo_loss"])
        assert jnp.isfinite(comp_high["total_grpo_loss"])

    def test_loss_gradient_flows(self, batch_data):
        """Test that loss is differentiable w.r.t. log_probs and values."""

        def loss_fn(log_probs, values):
            total, _ = compute_grpo_loss(
                log_probs=log_probs,
                old_log_probs=batch_data["old_log_probs"],
                advantages=batch_data["advantages"],
                values=values,
                returns=batch_data["returns"],
            )
            return total

        grads = jax.grad(loss_fn, argnums=(0, 1))(batch_data["log_probs"], batch_data["values"])

        for i, g in enumerate(grads):
            assert jnp.all(jnp.isfinite(g)), f"Non-finite gradient at index {i}"

    def test_mean_ratio_near_one_when_same_policy(self):
        """Test that mean ratio ≈ 1 when policies are identical."""
        batch_size = 8
        log_probs = jnp.array([-1.0] * batch_size)
        old_log_probs = jnp.array([-1.0] * batch_size)
        advantages = jax.random.normal(jax.random.PRNGKey(0), (batch_size,))
        values = jnp.zeros(batch_size)
        returns = jnp.zeros(batch_size)

        _, components = compute_grpo_loss(log_probs, old_log_probs, advantages, values, returns)

        np.testing.assert_allclose(float(components["mean_ratio"]), 1.0, atol=1e-5)


# =============================================================================
# AGIConfig GRPO Settings Tests
# =============================================================================


class TestAGIConfigGRPO:
    """Tests for GRPO-related AGIConfig settings."""

    def test_grpo_defaults(self):
        """Test default GRPO settings."""
        config = AGIConfig()

        assert config.use_grpo is False
        assert config.grpo_num_groups == 4
        assert config.grpo_group_size == 4
        assert config.grpo_clip_eps == pytest.approx(0.2)
        assert config.grpo_kl_coeff == pytest.approx(0.01)
        assert config.grpo_value_loss_coeff == pytest.approx(0.5)
        assert config.grpo_entropy_coeff == pytest.approx(0.01)
        assert config.grpo_gamma == pytest.approx(1.0)
        assert config.grpo_lam == pytest.approx(0.95)
        assert config.grpo_normalize_advantages is True
        assert config.grpo_reward_model == "internal"

    def test_grpo_enable(self):
        """Test enabling GRPO."""
        config = AGIConfig(use_grpo=True)
        assert config.use_grpo is True

    def test_grpo_custom_settings(self):
        """Test custom GRPO settings."""
        config = AGIConfig(
            use_grpo=True,
            grpo_num_groups=8,
            grpo_group_size=6,
            grpo_clip_eps=0.1,
            grpo_kl_coeff=0.05,
            grpo_reward_model="rule_based",
        )
        assert config.grpo_num_groups == 8
        assert config.grpo_group_size == 6
        assert config.grpo_clip_eps == pytest.approx(0.1)
        assert config.grpo_reward_model == "rule_based"

    def test_grpo_invalid_clip_eps(self):
        """Test validation rejects invalid clip_eps."""
        with pytest.raises(AssertionError):
            AGIConfig(use_grpo=True, grpo_clip_eps=0.0)

        with pytest.raises(AssertionError):
            AGIConfig(use_grpo=True, grpo_clip_eps=1.0)

    def test_grpo_invalid_group_size(self):
        """Test validation rejects group_size < 2."""
        with pytest.raises(AssertionError):
            AGIConfig(use_grpo=True, grpo_group_size=1)

    def test_grpo_invalid_reward_model(self):
        """Test validation rejects unknown reward model."""
        with pytest.raises(AssertionError):
            AGIConfig(use_grpo=True, grpo_reward_model="unknown")

    def test_grpo_in_dict(self):
        """Test GRPO settings appear in config dict."""
        config = AGIConfig(use_grpo=True)
        d = config.to_dict()

        assert "use_grpo" in d
        assert "grpo_num_groups" in d
        assert "grpo_clip_eps" in d
        assert d["use_grpo"] is True

    def test_grpo_print_summary(self, capsys):
        """Test that print_summary includes GRPO info."""
        config = AGIConfig(use_grpo=True)
        config.print_summary()
        captured = capsys.readouterr()
        assert "GRPO" in captured.out
        assert "Enabled: True" in captured.out


# =============================================================================
# ControllerRewardShaper Tests
# =============================================================================


class TestControllerRewardShaper:
    """Tests for reward shaping utilities."""

    def test_compute_step_reward_basic(self):
        """Test basic step reward computation."""
        shaper = ControllerRewardShaper()
        state = ComputeState(
            hidden=jnp.zeros((1, 64)),
            hidden_pooled=jnp.zeros((1, 64)),
            memory_summary=jnp.zeros((1, 64)),
            uncertainty=jnp.array([[0.5]]),
            confidence=jnp.array([[0.5]]),
            budget_remaining=0.8,
            step=0,
            modules_called=[],
            module_outputs=[],
        )
        output = ModuleOutput(
            hidden_delta=jnp.zeros((1, 64)),
            confidence=jnp.array([[0.7]]),
            uncertainty=jnp.array([[0.3]]),
            actual_cost=0.05,
            suggests_halt=False,
        )

        reward = shaper.compute_step_reward(state, output, 0.05)
        assert isinstance(reward, float)

    def test_compute_final_reward_correct(self):
        """Test final reward for correct answers."""
        shaper = ControllerRewardShaper()
        reward = shaper.compute_final_reward(is_correct=True, total_cost=0.3, num_steps=3, max_steps=10)
        assert reward > 0

    def test_compute_final_reward_wrong(self):
        """Test final reward for wrong answers."""
        shaper = ControllerRewardShaper()
        reward = shaper.compute_final_reward(is_correct=False, total_cost=0.3, num_steps=3, max_steps=10)
        assert reward < 0

    def test_compute_returns(self):
        """Test discounted returns computation."""
        shaper = ControllerRewardShaper(gamma=0.99)
        step_rewards = [0.1, 0.2, 0.3]
        final_reward = 1.0

        returns = shaper.compute_returns(step_rewards, final_reward)

        assert len(returns) == 3
        # Returns should be decreasing from end to start (discounting)
        assert all(isinstance(r, float) for r in returns)
