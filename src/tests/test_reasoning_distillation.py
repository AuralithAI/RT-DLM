"""Tests for reasoning-trace knowledge distillation."""

import jax.numpy as jnp
import numpy as np
import pytest

from src.core.training.reasoning_distillation import (
    TraceDistillConfig,
    chain_to_adjacency,
    cosine_trace_loss,
    logit_distillation_loss,
    project_teacher_to_student_dim,
    reasoning_trace_distillation_loss,
    structural_alignment_loss,
)


def test_cosine_trace_loss_zero_for_identical():
    """Identical (normalized) traces yield zero cosine loss."""
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.normal(size=(2, 4, 8)).astype(np.float32))
    loss = cosine_trace_loss(a, a)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_cosine_trace_loss_handles_length_mismatch():
    """Length mismatch should be handled by truncation, not crash."""
    s = jnp.ones((1, 5, 4))
    t = jnp.ones((1, 3, 4))
    loss = cosine_trace_loss(s, t)
    assert jnp.isfinite(loss)


def test_structural_alignment_loss_zero_for_identical():
    """Identical adjacency matrices give zero structural loss."""
    a = chain_to_adjacency(5)
    loss = structural_alignment_loss(a, a)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_structural_alignment_loss_handles_dim_mismatch():
    """Loss should crop to the smaller dimension."""
    a = chain_to_adjacency(5)
    b = chain_to_adjacency(7)
    loss = structural_alignment_loss(a, b)
    assert jnp.isfinite(loss)


def test_chain_to_adjacency_structure():
    """chain_to_adjacency must have ones on the super-diagonal only."""
    a = chain_to_adjacency(4)
    expected = jnp.array(
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=jnp.float32
    )
    assert jnp.allclose(a, expected)


def test_logit_distillation_loss_zero_for_identical():
    """Identical teacher/student logits give zero KD loss."""
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(2, 4, 8)).astype(np.float32))
    loss = logit_distillation_loss(logits, logits, temperature=2.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-4)


def test_combined_loss_keys():
    """Combined loss must include 'total' and 'cosine'."""
    rng = np.random.default_rng(0)
    s = jnp.asarray(rng.normal(size=(2, 4, 8)).astype(np.float32))
    t = jnp.asarray(rng.normal(size=(2, 4, 8)).astype(np.float32))
    out = reasoning_trace_distillation_loss(s, t)
    assert "total" in out and "cosine" in out


def test_combined_loss_with_all_terms():
    """All-loss path should emit cosine, structural, logit, total."""
    rng = np.random.default_rng(0)
    s = jnp.asarray(rng.normal(size=(2, 3, 4)).astype(np.float32))
    t = jnp.asarray(rng.normal(size=(2, 3, 4)).astype(np.float32))
    sa = chain_to_adjacency(3)[None, ...].repeat(2, axis=0)
    ta = chain_to_adjacency(3)[None, ...].repeat(2, axis=0)
    sl = jnp.asarray(rng.normal(size=(2, 3, 6)).astype(np.float32))
    tl = jnp.asarray(rng.normal(size=(2, 3, 6)).astype(np.float32))
    out = reasoning_trace_distillation_loss(
        s, t, sa, ta, sl, tl, config=TraceDistillConfig(logit_weight=0.5)
    )
    assert {"cosine", "structural", "logit", "total"}.issubset(out.keys())


def test_project_teacher_dim_truncate_and_pad():
    """Projection should resize teacher embeddings to target dim."""
    t = jnp.ones((2, 3, 8))
    smaller = project_teacher_to_student_dim(t, 4)
    larger = project_teacher_to_student_dim(t, 12)
    same = project_teacher_to_student_dim(t, 8)
    assert smaller.shape == (2, 3, 4)
    assert larger.shape == (2, 3, 12)
    assert same.shape == (2, 3, 8)
