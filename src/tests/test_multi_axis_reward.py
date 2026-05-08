"""Tests for the multi-axis Bradley-Terry reward model."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.core.ethics.multi_axis_reward import (
    DEFAULT_AXIS_WEIGHTS,
    MultiAxisRewardModel,
    RewardAxis,
    RewardModelConfig,
    aggregate_reward,
    bradley_terry_loss,
    expected_calibration_error,
    per_axis_bradley_terry_loss,
    reliability_diagram,
    temperature_search,
    total_reward_loss,
)


def test_default_weights_sum_to_one():
    """Axis weights must sum to 1 to be a valid probability mixture."""
    assert sum(DEFAULT_AXIS_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)


def test_reward_model_forward_shapes():
    """Reward model returns one scalar per axis per example."""

    def fwd(h):
        cfg = RewardModelConfig(d_model=32, num_heads=4, num_layers=2)
        return MultiAxisRewardModel(cfg)(h)

    transformed = hk.transform(fwd)
    rng = jax.random.PRNGKey(0)
    h = jnp.asarray(np.random.randn(3, 5, 32).astype(np.float32))
    params = transformed.init(rng, h)
    out = transformed.apply(params, rng, h)
    assert set(out.keys()) == {a.value for a in RewardAxis}
    for v in out.values():
        assert v.shape == (3,)


def test_aggregate_reward_weighted_sum():
    """Aggregate reward equals weighted sum of axis rewards."""
    rewards = {a.value: jnp.ones((2,)) for a in RewardAxis}
    agg = aggregate_reward(rewards)
    assert jnp.allclose(agg, jnp.ones((2,)) * sum(DEFAULT_AXIS_WEIGHTS.values()))


def test_bradley_terry_loss_zero_when_chosen_dominates():
    """BT loss should be near zero when chosen reward >> rejected."""
    chosen = jnp.array([10.0, 10.0])
    rejected = jnp.array([-10.0, -10.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert float(loss) < 1e-4


def test_per_axis_bradley_terry_loss_keys():
    """per_axis_bradley_terry_loss returns one entry per axis."""
    chosen = {a.value: jnp.ones((2,)) for a in RewardAxis}
    rejected = {a.value: jnp.zeros((2,)) for a in RewardAxis}
    losses = per_axis_bradley_terry_loss(chosen, rejected)
    assert set(losses.keys()) == set(chosen.keys())
    assert all(jnp.isfinite(v) for v in losses.values())


def test_total_reward_loss_finite():
    """total_reward_loss must produce a finite scalar."""
    per_axis = {a.value: jnp.array(0.5) for a in RewardAxis}
    total = total_reward_loss(per_axis)
    assert jnp.isfinite(total)


def test_ece_perfect_calibration():
    """ECE should be 0 when confidence equals accuracy."""
    conf = np.array([0.5, 0.5, 0.5, 0.5])
    correct = np.array([1.0, 0.0, 1.0, 0.0])
    ece = expected_calibration_error(conf, correct, n_bins=2)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_temperature_search_finds_finite_t():
    """Temperature search must return a value in the candidate set."""
    chosen = np.array([1.0, 2.0, 3.0])
    rejected = np.array([0.0, 0.5, 1.0])
    t, ece = temperature_search(chosen, rejected)
    assert t in {0.5, 0.7, 1.0, 1.2, 1.5}
    assert ece >= 0.0


def test_reliability_diagram_bin_count():
    """Reliability diagram returns one tuple per bin."""
    conf = np.linspace(0.0, 1.0, 100)
    correct = (conf > 0.5).astype(np.float32)
    diag = reliability_diagram(conf, correct, n_bins=10)
    assert len(diag) == 10
