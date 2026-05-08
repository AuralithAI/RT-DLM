"""Tests for constitutional self-critique."""

import jax.numpy as jnp
import numpy as np

from src.core.ethics.constitutional import (
    ConstitutionalRule,
    ConstitutionalRuleset,
    constitutional_self_critique_loss,
    critique_consistency_loss,
    default_ruleset,
    revision_imitation_loss,
    run_self_critique,
)


def test_default_ruleset_has_categories():
    """Default ruleset should expose multiple distinct categories."""
    rs = default_ruleset()
    cats = rs.all_categories()
    assert "cbrn" in cats and "privacy" in cats and len(cats) >= 10


def test_ruleset_for_category_filters():
    """`for_category` should return only matching rules."""
    rs = ConstitutionalRuleset()
    rs.add("R1", "a", "x")
    rs.add("R2", "b", "y")
    assert len(rs.for_category("a")) == 1
    assert rs.for_category("a")[0].rule_id == "R1"


def test_revision_imitation_loss_shape():
    """Imitation loss should produce a scalar."""
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(2, 4, 8)).astype(np.float32))
    labels = jnp.asarray(rng.integers(0, 8, size=(2, 4)))
    loss = revision_imitation_loss(logits, labels)
    assert loss.ndim == 0


def test_consistency_loss_negative_when_distributions_differ():
    """Pushing distributions apart should yield negative loss (we maximize KL)."""
    a = jnp.zeros((1, 1, 4))
    b = jnp.array([[[10.0, 0.0, 0.0, 0.0]]])
    mask = jnp.ones((1, 1))
    loss = critique_consistency_loss(a, b, mask)
    assert float(loss) < 0.0


def test_combined_loss_returns_total():
    """Combined loss dict must always include 'total' and 'imitation'."""
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(2, 3, 5)).astype(np.float32))
    labels = jnp.asarray(rng.integers(0, 5, size=(2, 3)))
    out = constitutional_self_critique_loss(logits, labels)
    assert "total" in out and "imitation" in out


def test_run_self_critique_invokes_revise_when_violated():
    """If any rule fires, revise_fn must be called."""
    rs = ConstitutionalRuleset()
    rs.add("R1", "x", "no")

    def critique(_, rule: ConstitutionalRule):
        return True, "violated"

    def revise(prompt, response, vios):
        return response + " [REVISED]"

    out = run_self_critique("p", "r", rs, critique, revise)
    assert "[REVISED]" in str(out["revised"])
    assert out["violations"] == ["R1"]


def test_run_self_critique_passthrough_when_clean():
    """If no rule fires, revised == original."""
    rs = ConstitutionalRuleset()
    rs.add("R1", "x", "no")

    def critique(_, rule):
        return False, ""

    def revise(prompt, response, vios):
        return "should-not-be-called"

    out = run_self_critique("p", "r", rs, critique, revise)
    assert out["revised"] == "r"
    assert out["violations"] == []
