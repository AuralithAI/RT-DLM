"""Tests for the dynamic fusion FFN and temporal proximity bias."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from src.modules.multimodal.fusion_module import (
    CrossModalAttention,
    MultiModalFusionLayer,
)


def test_cross_modal_attention_temporal_bias_shifts_output():
    """Adding temporal bias must change the attention output vs the no-bias case."""

    def fwd(q, k, v, qts, kts, use_bias):
        attn = CrossModalAttention(d_model=32, num_heads=4)
        if use_bias:
            return attn(q, k, v, query_timestamps=qts, key_timestamps=kts)[0]
        return attn(q, k, v)[0]

    transformed = hk.transform(fwd)
    rng = jax.random.PRNGKey(0)
    q = jnp.asarray(np.random.randn(2, 5, 32).astype(np.float32))
    k = jnp.asarray(np.random.randn(2, 7, 32).astype(np.float32))
    v = jnp.asarray(np.random.randn(2, 7, 32).astype(np.float32))
    qts = jnp.asarray(np.random.rand(2, 5).astype(np.float32))
    kts = jnp.asarray(np.random.rand(2, 7).astype(np.float32))
    params = transformed.init(rng, q, k, v, qts, kts, True)
    a = transformed.apply(params, rng, q, k, v, qts, kts, False)
    b = transformed.apply(params, rng, q, k, v, qts, kts, True)
    assert a.shape == (2, 5, 32)
    assert not jnp.allclose(a, b)


def test_fusion_layer_handles_three_modalities():
    """Fusion FFN must work with 3 modalities, validating the dynamic projection fix."""

    def fwd(inputs):
        layer = MultiModalFusionLayer(
            d_model=32, num_heads=4, modalities=["text", "audio", "video"]
        )
        return layer(inputs)[0]

    transformed = hk.transform(fwd)
    rng = jax.random.PRNGKey(0)
    inputs = {
        "text": jnp.asarray(np.random.randn(2, 4, 32).astype(np.float32)),
        "audio": jnp.asarray(np.random.randn(2, 4, 32).astype(np.float32)),
        "video": jnp.asarray(np.random.randn(2, 4, 32).astype(np.float32)),
    }
    params = transformed.init(rng, inputs)
    out = transformed.apply(params, rng, inputs)
    assert out.shape == (2, 4, 32)


def test_fusion_layer_single_modality_passthrough():
    """One-modality fusion must short-circuit and preserve shape."""

    def fwd(inputs):
        layer = MultiModalFusionLayer(d_model=16, num_heads=4, modalities=["text"])
        return layer(inputs)[0]

    transformed = hk.transform(fwd)
    rng = jax.random.PRNGKey(0)
    inputs = {"text": jnp.asarray(np.random.randn(2, 3, 16).astype(np.float32))}
    params = transformed.init(rng, inputs)
    out = transformed.apply(params, rng, inputs)
    assert out.shape == (2, 3, 16)
