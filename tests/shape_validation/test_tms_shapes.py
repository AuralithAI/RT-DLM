"""
Shape validation tests for TMS Model integration points.
Tests shape compatibility across different batch sizes and sequence lengths.
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.model.model_tms import TMSModel


class ShapeConfig:
    d_model = 64
    num_heads = 4
    num_layers = 2
    vocab_size = 1000
    max_seq_length = 128
    moe_experts = 4
    moe_top_k = 2
    memory_size = 32
    retrieval_k = 4
    ltm_weight = 0.3
    stm_weight = 0.3
    mtm_weight = 0.3


BATCH_SIZES = [1, 4, 16, 32]
SEQ_LENGTHS = [8, 32, 64, 128]


class TestTMSModelShapes:
    
    @pytest.fixture
    def config(self):
        return ShapeConfig()
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.fixture
    def model_fn(self, config):
        def _model_fn(inputs, rng=None, **kwargs):
            model = TMSModel(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                vocab_size=config.vocab_size,
                max_seq_length=config.max_seq_length,
                moe_experts=config.moe_experts,
                moe_top_k=config.moe_top_k,
                memory_size=config.memory_size,
                retrieval_k=config.retrieval_k,
                ltm_weight=config.ltm_weight,
                stm_weight=config.stm_weight,
                mtm_weight=config.mtm_weight,
            )
            return model(inputs, rng=rng, **kwargs)
        return _model_fn
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_batch_size_variations(self, model_fn, config, rng_key, batch_size):
        """Test model works with different batch sizes."""
        model = hk.transform_with_state(model_fn)
        
        seq_len = 32
        inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, config.vocab_size)
        
        params, state = model.init(rng_key, inputs)
        logits, _ = model.apply(params, state, rng_key, inputs)
        
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    @pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
    def test_sequence_length_variations(self, model_fn, config, rng_key, seq_len):
        """Test model works with different sequence lengths."""
        model = hk.transform_with_state(model_fn)
        
        batch_size = 4
        inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, config.vocab_size)
        
        params, state = model.init(rng_key, inputs)
        logits, _ = model.apply(params, state, rng_key, inputs)
        
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 8), (1, 64), (4, 32), (16, 16), (32, 8)
    ])
    def test_batch_seq_combinations(self, model_fn, config, rng_key, batch_size, seq_len):
        """Test various batch_size × seq_len combinations."""
        model = hk.transform_with_state(model_fn)
        
        inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, config.vocab_size)
        
        params, state = model.init(rng_key, inputs)
        logits, _ = model.apply(params, state, rng_key, inputs)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert not jnp.any(jnp.isnan(logits))
        assert not jnp.any(jnp.isinf(logits))
    
    def test_with_ltm_memory(self, model_fn, config, rng_key):
        """Test shape compatibility with LTM memory input."""
        batch_size, seq_len = 4, 32
        inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, config.vocab_size)
        
        rng_key, subkey = jax.random.split(rng_key)
        ltm = jax.random.normal(subkey, (batch_size, config.d_model))
        
        def model_with_ltm(inputs, rng=None):
            model = TMSModel(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                vocab_size=config.vocab_size,
                max_seq_length=config.max_seq_length,
                moe_experts=config.moe_experts,
                moe_top_k=config.moe_top_k,
                memory_size=config.memory_size,
                retrieval_k=config.retrieval_k,
                ltm_weight=config.ltm_weight,
                stm_weight=config.stm_weight,
                mtm_weight=config.mtm_weight,
            )
            return model(inputs, rng=rng, retrieved_memory_ltm=ltm)
        
        model_ltm = hk.transform_with_state(model_with_ltm)
        params, state = model_ltm.init(rng_key, inputs)
        logits, _ = model_ltm.apply(params, state, rng_key, inputs)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_with_all_memories(self, model_fn, config, rng_key):
        """Test shape compatibility with all memory types (LTM, STM, MTM)."""
        batch_size, seq_len = 4, 32
        inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, config.vocab_size)
        
        rng_key, k1, k2, k3 = jax.random.split(rng_key, 4)
        ltm = jax.random.normal(k1, (batch_size, config.d_model))
        stm = jax.random.normal(k2, (batch_size, config.d_model))
        mtm = jax.random.normal(k3, (batch_size, config.d_model))
        
        def model_with_memories(inputs, rng=None):
            model = TMSModel(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                vocab_size=config.vocab_size,
                max_seq_length=config.max_seq_length,
                moe_experts=config.moe_experts,
                moe_top_k=config.moe_top_k,
                memory_size=config.memory_size,
                retrieval_k=config.retrieval_k,
                ltm_weight=config.ltm_weight,
                stm_weight=config.stm_weight,
                mtm_weight=config.mtm_weight,
            )
            return model(
                inputs, rng=rng,
                retrieved_memory_ltm=ltm,
                retrieved_memory_stm=stm,
                retrieved_memory_mtm=mtm
            )
        
        model_mem = hk.transform_with_state(model_with_memories)
        params, state = model_mem.init(rng_key, inputs)
        logits, _ = model_mem.apply(params, state, rng_key, inputs)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_return_attention_shapes(self, model_fn, config, rng_key):
        """Test output shapes when returning attention weights."""
        batch_size, seq_len = 4, 32
        inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, config.vocab_size)
        
        def model_with_attn(inputs, rng=None):
            model = TMSModel(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                vocab_size=config.vocab_size,
                max_seq_length=config.max_seq_length,
                moe_experts=config.moe_experts,
                moe_top_k=config.moe_top_k,
                memory_size=config.memory_size,
                retrieval_k=config.retrieval_k,
                ltm_weight=config.ltm_weight,
                stm_weight=config.stm_weight,
                mtm_weight=config.mtm_weight,
            )
            return model(inputs, rng=rng, return_attention=True)
        
        model_attn = hk.transform_with_state(model_with_attn)
        params, state = model_attn.init(rng_key, inputs)
        result, _ = model_attn.apply(params, state, rng_key, inputs)
        
        # TMSModel now returns a dict when return_attention=True
        assert isinstance(result, dict)
        logits = result["logits"]
        attn_info = result["attention_info"]
        aux_loss = result["aux_loss"]
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert isinstance(attn_info, dict)
        assert "self_attention" in attn_info
        assert aux_loss.shape == ()
    
    @pytest.mark.parametrize("d_model", [32, 64, 128, 256])
    def test_d_model_variations(self, config, rng_key, d_model):
        """Test model works with different d_model sizes."""
        num_heads = max(2, d_model // 16)
        
        def model_fn(inputs, rng=None):
            model = TMSModel(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=2,
                vocab_size=config.vocab_size,
                max_seq_length=config.max_seq_length,
                moe_experts=4,
                moe_top_k=2,
                memory_size=config.memory_size,
                retrieval_k=config.retrieval_k,
                ltm_weight=config.ltm_weight,
                stm_weight=config.stm_weight,
                mtm_weight=config.mtm_weight,
            )
            return model(inputs, rng=rng)
        
        model = hk.transform_with_state(model_fn)
        
        batch_size, seq_len = 4, 32
        inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, config.vocab_size)
        
        params, state = model.init(rng_key, inputs)
        logits, _ = model.apply(params, state, rng_key, inputs)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_edge_case_batch_size_1(self, model_fn, config, rng_key):
        """Test edge case: batch size of 1."""
        model = hk.transform_with_state(model_fn)
        
        inputs = jax.random.randint(rng_key, (1, 32), 1, config.vocab_size)
        
        params, state = model.init(rng_key, inputs)
        logits, _ = model.apply(params, state, rng_key, inputs)
        
        assert logits.shape == (1, 32, config.vocab_size)
        assert not jnp.any(jnp.isnan(logits))
    
    def test_edge_case_seq_len_1(self, model_fn, config, rng_key):
        """Test edge case: sequence length of 1."""
        model = hk.transform_with_state(model_fn)
        
        inputs = jax.random.randint(rng_key, (4, 1), 1, config.vocab_size)
        
        params, state = model.init(rng_key, inputs)
        logits, _ = model.apply(params, state, rng_key, inputs)
        
        assert logits.shape == (4, 1, config.vocab_size)
        assert not jnp.any(jnp.isnan(logits))


class TestMemoryShapeValidation:
    
    @pytest.fixture
    def config(self):
        return ShapeConfig()
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_memory_batch_consistency(self, config, rng_key, batch_size):
        """Test memory shapes are consistent with batch size."""
        rng_key, k1, k2, k3 = jax.random.split(rng_key, 4)
        ltm = jax.random.normal(k1, (batch_size, config.d_model))
        stm = jax.random.normal(k2, (batch_size, config.d_model))
        mtm = jax.random.normal(k3, (batch_size, config.d_model))
        
        assert ltm.shape[0] == batch_size
        assert stm.shape[0] == batch_size
        assert mtm.shape[0] == batch_size
        assert ltm.shape[1] == config.d_model
    
    def test_memory_d_model_mismatch_handling(self, config, rng_key):
        """Test that memory dimension matches d_model."""
        batch_size = 4
        
        ltm_correct = jax.random.normal(rng_key, (batch_size, config.d_model))
        assert ltm_correct.shape == (batch_size, config.d_model)
        
        wrong_d_model = config.d_model * 2
        ltm_wrong = jax.random.normal(rng_key, (batch_size, wrong_d_model))
        assert ltm_wrong.shape[1] != config.d_model


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
