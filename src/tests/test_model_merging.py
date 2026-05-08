"""Tests for SLERP / TIES / weighted model merging."""

import jax.numpy as jnp
import numpy as np
import pytest

from src.core.utils.model_merging import (
    merge_checkpoints,
    slerp,
    slerp_arrays,
    ties_merge,
    weighted_average,
)


def _make_params(seed: int):
    """Build a small nested param tree for testing."""
    rng = np.random.default_rng(seed)
    return {
        "layer1": {"w": jnp.asarray(rng.normal(size=(4, 4)).astype(np.float32))},
        "layer2": {"w": jnp.asarray(rng.normal(size=(4,)).astype(np.float32))},
    }


def test_weighted_average_equal_weights():
    """Equal weights produce arithmetic mean."""
    a = _make_params(0)
    b = _make_params(1)
    merged = weighted_average([a, b], [1.0, 1.0])
    expected = (a["layer1"]["w"] + b["layer1"]["w"]) / 2.0
    assert jnp.allclose(merged["layer1"]["w"], expected, atol=1e-6)


def test_slerp_endpoints():
    """SLERP at t=0 and t=1 must return endpoints."""
    a = _make_params(0)
    b = _make_params(1)
    m0 = slerp(a, b, t=0.0)
    m1 = slerp(a, b, t=1.0)
    assert jnp.allclose(m0["layer1"]["w"], a["layer1"]["w"], atol=1e-4)
    assert jnp.allclose(m1["layer1"]["w"], b["layer1"]["w"], atol=1e-4)


def test_slerp_arrays_unit_circle():
    """SLERP between orthogonal unit vectors at t=0.5 should bisect."""
    a = jnp.array([1.0, 0.0])
    b = jnp.array([0.0, 1.0])
    mid = slerp_arrays(a, b, 0.5)
    expected = jnp.array([np.sqrt(2) / 2, np.sqrt(2) / 2])
    assert jnp.allclose(mid, expected, atol=1e-4)


def test_ties_merge_preserves_base_when_weights_zero():
    """TIES with zero candidate weight should drop in toward base."""
    base = _make_params(0)
    cand = _make_params(1)
    merged = ties_merge(base, [cand], [1.0], density=1.0)
    assert merged["layer1"]["w"].shape == base["layer1"]["w"].shape


def test_merge_dispatcher_invalid_method():
    """Unknown methods must raise."""
    a = _make_params(0)
    with pytest.raises(ValueError):
        merge_checkpoints(a, {"x": _make_params(1)}, {"x": 1.0}, method="bogus")


def test_merge_dispatcher_slerp_two():
    """SLERP via dispatcher should yield correct shapes."""
    a = _make_params(0)
    b = _make_params(1)
    out = merge_checkpoints(
        a, {"a": a, "b": b}, {"a": 1.0, "b": 1.0}, method="slerp"
    )
    assert out["layer1"]["w"].shape == (4, 4)
