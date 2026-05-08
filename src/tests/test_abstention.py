"""Tests for abstention training utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from src.core.ethics.abstention import (
    AbstentionConfig,
    abstention_loss,
    confidence_score,
    ensemble_uncertainty,
    expected_calibration_loss,
    predictive_entropy,
    should_abstain,
    synthesize_abstention_examples,
    uncertainty_routing_decision,
)


def test_predictive_entropy_uniform_is_max():
    """Uniform distribution has the highest entropy."""
    logits = jnp.zeros((1, 4))
    e = float(predictive_entropy(logits)[0])
    assert e == pytest.approx(float(jnp.log(4.0)), abs=1e-5)


def test_confidence_score_matches_softmax_max():
    """confidence_score should equal max softmax probability."""
    logits = jnp.array([[1.0, 5.0, 0.0]])
    s = float(confidence_score(logits)[0])
    expected = float(jnp.exp(5.0) / (jnp.exp(1.0) + jnp.exp(5.0) + jnp.exp(0.0)))
    assert s == pytest.approx(expected, abs=1e-6)


def test_should_abstain_low_confidence():
    """should_abstain returns True when max prob < threshold."""
    flat = jnp.zeros((1, 4))
    assert bool(should_abstain(flat, threshold=0.5)[0])


def test_abstention_loss_finite():
    """abstention_loss must return a finite total."""
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(2, 5, 8)).astype(np.float32))
    labels = jnp.zeros((2, 5), dtype=jnp.int32)
    abstain_mask = jnp.array([[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    out = abstention_loss(logits, labels, abstain_mask)
    assert jnp.isfinite(out["total"])
    assert {"ce", "abstain", "overconfidence", "total"}.issubset(out.keys())


def test_uncertainty_routing_decision():
    """Low-confidence routes to verify, high-confidence to commit."""
    assert uncertainty_routing_decision(0.4, 0.6) == "verify"
    assert uncertainty_routing_decision(0.9, 0.6) == "commit"


def test_calibration_loss_nonneg():
    """Calibration loss is non-negative."""
    conf = jnp.linspace(0.0, 1.0, 50)
    correct = (conf > 0.5).astype(jnp.float32)
    loss = expected_calibration_loss(conf, correct, n_bins=5)
    assert float(loss) >= 0.0


def test_synthesize_abstention_examples_balance():
    """Synthesized dataset must contain both factual and abstention examples."""
    out = synthesize_abstention_examples([("Q1", "A1"), ("Q2", "A2")], ["Unknown1", "Unknown2"])
    assert len(out) == 4
    abst = sum(1 for e in out if e["abstain"] == "1")
    assert abst == 2


def test_ensemble_uncertainty_zero_for_identical():
    """Identical ensemble members give zero variance."""
    logits = jnp.array([[[1.0, 2.0, 3.0]]])
    var = float(ensemble_uncertainty([logits, logits, logits])[0, 0])
    assert var == pytest.approx(0.0, abs=1e-6)


def test_abstention_config_default_weight():
    """Default abstain_loss_weight should be > 0."""
    cfg = AbstentionConfig()
    assert cfg.abstain_loss_weight > 0.0
