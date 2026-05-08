"""Tests for DPO / IPO losses."""

import jax.numpy as jnp
import numpy as np
import pytest

from src.core.ethics.dpo import DPOConfig, dpo_accuracy, dpo_loss, ipo_loss


def _fake_logits(batch: int, seq: int, vocab: int, seed: int) -> jnp.ndarray:
    """Build random logits of given shape."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(batch, seq, vocab)).astype(np.float32))


def test_dpo_loss_finite_with_reference():
    """DPO loss must be a finite scalar when reference logits are provided."""
    pol_c = _fake_logits(2, 4, 6, 0)
    pol_r = _fake_logits(2, 4, 6, 1)
    ref_c = _fake_logits(2, 4, 6, 2)
    ref_r = _fake_logits(2, 4, 6, 3)
    chosen_lab = jnp.zeros((2, 4), dtype=jnp.int32)
    rejected_lab = jnp.zeros((2, 4), dtype=jnp.int32)
    out = dpo_loss(pol_c, pol_r, chosen_lab, rejected_lab, ref_c, ref_r)
    assert jnp.isfinite(out["loss"])
    assert "margin" in out and "implicit_reward_chosen" in out


def test_dpo_reference_free_path():
    """Reference-free mode should still produce a finite loss."""
    pol_c = _fake_logits(2, 3, 5, 0)
    pol_r = _fake_logits(2, 3, 5, 1)
    chosen_lab = jnp.zeros((2, 3), dtype=jnp.int32)
    rejected_lab = jnp.zeros((2, 3), dtype=jnp.int32)
    out = dpo_loss(
        pol_c,
        pol_r,
        chosen_lab,
        rejected_lab,
        config=DPOConfig(reference_free=True),
    )
    assert jnp.isfinite(out["loss"])


def test_dpo_loss_decreases_with_chosen_advantage():
    """Larger chosen-vs-rejected gap should yield lower loss."""

    def build(advantage: float):
        vocab = 6
        pol_c = jnp.zeros((1, 2, vocab))
        pol_c = pol_c.at[..., 0].set(advantage)
        pol_r = jnp.zeros((1, 2, vocab))
        ref = jnp.zeros((1, 2, vocab))
        labels = jnp.zeros((1, 2), dtype=jnp.int32)
        return dpo_loss(pol_c, pol_r, labels, labels, ref, ref)["loss"]

    low = float(build(0.1))
    high = float(build(5.0))
    assert high < low


def test_ipo_loss_finite():
    """IPO loss must be finite."""
    pol_c = _fake_logits(2, 4, 6, 0)
    pol_r = _fake_logits(2, 4, 6, 1)
    ref_c = _fake_logits(2, 4, 6, 2)
    ref_r = _fake_logits(2, 4, 6, 3)
    labels = jnp.zeros((2, 4), dtype=jnp.int32)
    loss = ipo_loss(pol_c, pol_r, labels, labels, ref_c, ref_r, beta=0.1)
    assert jnp.isfinite(loss)


def test_dpo_accuracy_one_when_chosen_higher():
    """Accuracy is 1.0 if all chosen logp exceed rejected."""
    chosen = jnp.array([1.0, 2.0, 3.0])
    rejected = jnp.array([0.0, 0.5, 1.0])
    acc = dpo_accuracy(chosen, rejected)
    assert float(acc) == pytest.approx(1.0)
