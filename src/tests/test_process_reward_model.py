"""Tests for the Process Reward Model."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.core.rlm.process_reward_model import (
    PRMConfig,
    ProcessRewardModel,
    combined_outcome_step_reward,
    outcome_loss,
    prm_total_loss,
    step_correctness_loss,
)


def test_step_correctness_loss_zero_on_perfect():
    """Loss should be near zero when logits perfectly match labels."""
    labels = jnp.array([[1.0, 0.0, 1.0]])
    mask = jnp.array([[1.0, 1.0, 1.0]])
    logits = jnp.where(labels > 0, 50.0, -50.0)
    loss = step_correctness_loss(logits, labels, mask)
    assert float(loss) < 1e-3


def test_outcome_loss_finite():
    """Outcome loss should produce a finite scalar."""
    logits = jnp.array([1.0, -1.0])
    labels = jnp.array([1.0, 0.0])
    loss = outcome_loss(logits, labels)
    assert jnp.isfinite(loss)


def test_combined_reward_blend():
    """Blended reward should equal alpha*outcome + (1-alpha)*mean_step."""
    outcome = jnp.array([1.0])
    steps = jnp.array([[0.0, 0.5, 1.0]])
    mask = jnp.ones_like(steps)
    out = combined_outcome_step_reward(outcome, steps, mask, alpha=0.5)
    assert float(out[0]) == pytest.approx(0.5 * 1.0 + 0.5 * 0.5, abs=1e-6)


def test_prm_forward_shapes():
    """PRM module should produce step and outcome logits with correct shapes."""

    def fwd(h, m):
        return ProcessRewardModel(PRMConfig(d_model=32, num_heads=4, num_layers=2))(h, m)

    transformed = hk.transform(fwd)
    rng = jax.random.PRNGKey(0)
    h = jnp.asarray(np.random.randn(2, 6, 32).astype(np.float32))
    m = jnp.ones((2, 6))
    params = transformed.init(rng, h, m)
    step_logits, outcome_logits = transformed.apply(params, rng, h, m)
    assert step_logits.shape == (2, 6)
    assert outcome_logits.shape == (2,)


def test_prm_total_loss_finite():
    """PRM combined loss must be finite for random inputs."""
    cfg = PRMConfig(step_weight=0.5, outcome_weight=0.5)
    step_logits = jnp.zeros((2, 4))
    out_logits = jnp.zeros((2,))
    step_lab = jnp.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    out_lab = jnp.array([1.0, 0.0])
    mask = jnp.ones((2, 4))
    loss = prm_total_loss(step_logits, out_logits, step_lab, out_lab, mask, cfg)
    assert jnp.isfinite(loss)
