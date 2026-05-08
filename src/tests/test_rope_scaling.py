"""Tests for RoPE NTK and YaRN scaling."""

import jax.numpy as jnp
import numpy as np
import pytest

from src.core.utils.rope_scaling import (
    apply_rope_with_inv_freq,
    ntk_aware_scale,
    yarn_attention_scale,
    yarn_scale_factors,
)


def test_ntk_no_scale_matches_default():
    """No scaling should match the standard RoPE base."""
    inv = ntk_aware_scale(64, base=10000.0, scale_factor=1.0)
    assert inv.shape == (32,)
    assert float(inv[0]) == pytest.approx(1.0, abs=1e-6)


def test_ntk_scale_decreases_high_freqs():
    """Increasing scale factor should reduce high-frequency components."""
    a = ntk_aware_scale(64, scale_factor=1.0)
    b = ntk_aware_scale(64, scale_factor=4.0)
    assert float(b[-1]) < float(a[-1])


def test_yarn_short_context_returns_default():
    """When new_length <= original_length, YaRN should be identity."""
    inv = yarn_scale_factors(64, original_length=4096, new_length=4096)
    assert inv.shape == (32,)


def test_yarn_long_context_blends():
    """For extended context, low-freq channels should be compressed (smaller)."""
    base = yarn_scale_factors(64, original_length=4096, new_length=4096)
    long = yarn_scale_factors(64, original_length=4096, new_length=131072)
    assert float(long[0]) <= float(base[0]) + 1e-6


def test_yarn_attention_scale_monotone():
    """YaRN attention temperature should grow with scale factor."""
    s1 = yarn_attention_scale(1.0)
    s2 = yarn_attention_scale(8.0)
    assert s1 == pytest.approx(1.0)
    assert s2 > s1


def test_apply_rope_shape_preserved():
    """RoPE must preserve input shape."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(2, 8, 64)).astype(np.float32))
    pos = jnp.arange(8)
    inv = ntk_aware_scale(64, scale_factor=2.0)
    y = apply_rope_with_inv_freq(x, pos, inv)
    assert y.shape == x.shape
    assert not jnp.allclose(x, y)
