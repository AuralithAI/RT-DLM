"""Tests for LoRA adapters."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from src.core.training.lora import (
    LoRAAdapter,
    LoRAConfig,
    LoRALinear,
    count_trainable_lora,
    is_lora_param,
    merge_lora_into_base,
    split_lora_params,
)


def test_is_lora_param():
    """Path matcher must recognize lora_A / lora_B."""
    assert is_lora_param("layer/lora_A")
    assert is_lora_param("layer/lora_B")
    assert not is_lora_param("layer/w")


def test_lora_linear_zero_init_matches_base():
    """At init, lora_B is zero so output should equal base linear."""

    def fwd(x):
        return LoRALinear(8, rank=4)(x, is_training=False)

    transformed = hk.transform(fwd)
    rng = jax.random.PRNGKey(0)
    x = jnp.asarray(np.random.randn(2, 6).astype(np.float32))
    params = transformed.init(rng, x)
    out = transformed.apply(params, rng, x)
    expected = jnp.dot(x, params["lo_ra_linear"]["w"]) + params["lo_ra_linear"]["b"]
    assert jnp.allclose(out, expected, atol=1e-5)


def test_lora_adapter_zero_init_returns_zeros():
    """Standalone LoRA adapter at init returns zero delta."""

    def fwd(x):
        return LoRAAdapter(6, 8, LoRAConfig(rank=4))(x, is_training=False)

    transformed = hk.transform(fwd)
    rng = jax.random.PRNGKey(0)
    x = jnp.asarray(np.random.randn(2, 6).astype(np.float32))
    params = transformed.init(rng, x)
    out = transformed.apply(params, rng, x)
    assert jnp.allclose(out, jnp.zeros_like(out))


def test_split_lora_params_roundtrip():
    """split_lora_params should partition keys without overlap."""
    params = {
        "layer1": {
            "w": jnp.ones((4, 4)),
            "lora_A": jnp.ones((4, 2)),
            "lora_B": jnp.zeros((2, 4)),
        }
    }
    base, lora = split_lora_params(params)
    assert "w" in base["layer1"]
    assert "lora_A" in lora["layer1"]
    assert "lora_B" in lora["layer1"]
    assert "w" not in lora["layer1"]
    assert "lora_A" not in base["layer1"]


def test_count_trainable_lora_only_lora_params():
    """count_trainable_lora must ignore base weights."""
    params = {
        "layer": {
            "w": jnp.ones((10, 10)),
            "lora_A": jnp.ones((10, 4)),
            "lora_B": jnp.ones((4, 10)),
        }
    }
    n = count_trainable_lora(params)
    assert n == 10 * 4 + 4 * 10


def test_merge_lora_into_base():
    """Merging lora_A @ lora_B into base produces a single 'w' tensor."""
    params = {
        "layer": {
            "w": jnp.zeros((4, 4)),
            "lora_A": jnp.ones((4, 2)),
            "lora_B": jnp.ones((2, 4)),
            "_scaling": 1.0,
        }
    }
    merged = merge_lora_into_base(params)
    assert "lora_A" not in merged["layer"]
    assert "lora_B" not in merged["layer"]
    expected = jnp.dot(jnp.ones((4, 2)), jnp.ones((2, 4)))
    assert jnp.allclose(merged["layer"]["w"], expected)
