"""Unit tests for muP parameterization helpers."""

import jax
import jax.numpy as jnp
import pytest

from src.core.model.mup import (
    MuPConfig,
    build_lr_scale_tree,
    classify_param,
    lr_scale,
    output_logit_scale,
    width_multiplier,
)


class TestWidthMultiplier:
    def test_identity(self):
        assert width_multiplier(256, 256) == pytest.approx(1.0)

    def test_double_width(self):
        assert width_multiplier(512, 256) == pytest.approx(2.0)

    def test_invalid_base(self):
        with pytest.raises(ValueError):
            width_multiplier(256, 0)


class TestLrScale:
    def test_embedding_invariant(self):
        assert lr_scale("embedding", 1024, 256) == pytest.approx(1.0)

    def test_hidden_scales_inverse_width(self):
        assert lr_scale("hidden", 1024, 256) == pytest.approx(0.25)

    def test_readout_scales_inverse_width(self):
        assert lr_scale("readout", 2048, 256) == pytest.approx(0.125)

    def test_bias_invariant(self):
        assert lr_scale("bias", 4096, 256) == pytest.approx(1.0)

    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            lr_scale("attention", 1024, 256)


class TestOutputLogitScale:
    def test_inverse_width(self):
        assert output_logit_scale(1024, 256) == pytest.approx(0.25)


class TestClassifyParam:
    def test_bias_by_shape(self):
        assert classify_param("layer/b", (64,)) == "bias"

    def test_embedding_name(self):
        assert classify_param("tok_emb/w", (1000, 256)) == "embedding"

    def test_readout_name(self):
        assert classify_param("lm_head/w", (256, 1000)) == "readout"

    def test_hidden_default(self):
        assert classify_param("transformer/layer_0/mlp/w", (256, 1024)) == "hidden"


class TestMuPConfig:
    def test_defaults(self):
        cfg = MuPConfig()
        assert cfg.base_d_model == 256
        assert cfg.enabled is False

    def test_invalid_base(self):
        with pytest.raises(ValueError):
            MuPConfig(base_d_model=0)

    def test_invalid_lr(self):
        with pytest.raises(ValueError):
            MuPConfig(base_lr=0.0)


class TestBuildLrScaleTree:
    def test_tree_matches_structure(self):
        params = {
            "block_0": {"w": jnp.ones((256, 512)), "b": jnp.zeros((512,))},
            "lm_head": {"w": jnp.ones((512, 1000))},
        }
        tree = build_lr_scale_tree(params, d_model=1024, base_d_model=256)
        assert tree["block_0"]["w"].shape == params["block_0"]["w"].shape
        assert float(tree["block_0"]["w"][0, 0]) == pytest.approx(0.25)
        assert float(tree["block_0"]["b"][0]) == pytest.approx(1.0)
        assert float(tree["lm_head"]["w"][0, 0]) == pytest.approx(0.25)
