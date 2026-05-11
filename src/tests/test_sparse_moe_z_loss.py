"""Unit tests for MoE z-loss and router-z-loss."""

import haiku as hk
import jax
import jax.numpy as jnp
import pytest

from src.core.model.sparse_moe import SparseMoE


def _make_forward(z_w: float, router_z_w: float):
    def _fn(x):
        moe = SparseMoE(
            d_model=16,
            num_experts=4,
            top_k=2,
            expert_capacity=8,
            z_loss_weight=z_w,
            router_z_loss_weight=router_z_w,
        )
        return moe(x)

    return hk.transform_with_state(_fn)


class TestRouterZLoss:
    def setup_method(self):
        self.rng = jax.random.PRNGKey(0)
        self.x = jax.random.normal(self.rng, (2, 8, 16))

    def test_metrics_contain_z_losses(self):
        fwd = _make_forward(1e-4, 1e-3)
        params, state = fwd.init(self.rng, self.x)
        (_, _, _, metrics), _ = fwd.apply(params, state, self.rng, self.x)
        assert "z_loss" in metrics
        assert "router_z_loss" in metrics
        assert jnp.all(jnp.isfinite(metrics["z_loss"]))
        assert jnp.all(jnp.isfinite(metrics["router_z_loss"]))

    def test_z_loss_non_negative(self):
        fwd = _make_forward(1e-4, 1e-3)
        params, state = fwd.init(self.rng, self.x)
        (_, _, _, metrics), _ = fwd.apply(params, state, self.rng, self.x)
        assert float(metrics["z_loss"]) >= 0
        assert float(metrics["router_z_loss"]) >= 0

    def test_zero_weights_match_legacy_loss(self):
        fwd_zero = _make_forward(0.0, 0.0)
        fwd_active = _make_forward(1.0, 1.0)
        params, state = fwd_zero.init(self.rng, self.x)
        (_, _, aux_zero, _), _ = fwd_zero.apply(params, state, self.rng, self.x)
        (_, _, aux_active, m_active), _ = fwd_active.apply(params, state, self.rng, self.x)
        expected = aux_zero + m_active["z_loss"] + m_active["router_z_loss"]
        assert jnp.allclose(aux_active, expected, atol=1e-5)

    def test_router_z_loss_uses_logsumexp(self):
        fwd = _make_forward(0.0, 1.0)
        params, state = fwd.init(self.rng, self.x)
        (_, _, _, metrics), _ = fwd.apply(params, state, self.rng, self.x)
        assert float(metrics["router_z_loss"]) > 0

    def test_return_tuple_shape_preserved(self):
        fwd = _make_forward(1e-4, 1e-3)
        params, state = fwd.init(self.rng, self.x)
        (out, top_k, aux, metrics), _ = fwd.apply(params, state, self.rng, self.x)
        assert out.shape == self.x.shape
        assert top_k.shape == (2, 8, 2)
        assert aux.shape == ()
        assert isinstance(metrics, dict)
