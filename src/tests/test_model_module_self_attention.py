"""
Tests for Model Module Self Attention

Tests for unified self-attention module with multiple attention variants.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk


class TestSelfAttentionModel(unittest.TestCase):
    """Test SelfAttentionModel class."""
    
    def test_standard_attention_initialization(self):
        """Test standard attention initialization."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def init_fn():
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128,
                attention_type="standard"
            )
            x = jnp.zeros((1, 10), dtype=jnp.int32)
            return model(x)
        
        # Use transform_with_state because SelfAttentionModel uses hk.get_state
        init = hk.transform_with_state(init_fn)
        rng = jax.random.PRNGKey(42)
        params, state = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_gqa_attention(self):
        """Test grouped-query attention."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def init_fn():
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128,
                attention_type="gqa",
                num_kv_heads=2
            )
            x = jnp.zeros((1, 10), dtype=jnp.int32)
            return model(x)
        
        init = hk.transform_with_state(init_fn)
        rng = jax.random.PRNGKey(42)
        params, state = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_mqa_attention(self):
        """Test multi-query attention."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def init_fn():
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128,
                attention_type="mqa"
            )
            x = jnp.zeros((1, 10), dtype=jnp.int32)
            return model(x)
        
        init = hk.transform_with_state(init_fn)
        rng = jax.random.PRNGKey(42)
        params, state = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_sliding_window_attention(self):
        """Test sliding window attention."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def init_fn():
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128,
                attention_type="sliding",
                sliding_window_size=32
            )
            x = jnp.zeros((1, 10), dtype=jnp.int32)
            return model(x)
        
        init = hk.transform_with_state(init_fn)
        rng = jax.random.PRNGKey(42)
        params, state = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_linear_attention(self):
        """Test linear attention."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def init_fn():
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128,
                attention_type="linear"
            )
            x = jnp.zeros((1, 10), dtype=jnp.int32)
            return model(x)
        
        init = hk.transform_with_state(init_fn)
        rng = jax.random.PRNGKey(42)
        params, state = init.init(rng)
        
        self.assertIsNotNone(params)


class TestPositionEncoding(unittest.TestCase):
    """Test position encoding options."""
    
    def test_rope_encoding(self):
        """Test RoPE position encoding."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def init_fn():
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128,
                position_encoding="rope"
            )
            x = jnp.zeros((1, 10), dtype=jnp.int32)
            return model(x)
        
        init = hk.transform_with_state(init_fn)
        rng = jax.random.PRNGKey(42)
        params, state = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_learned_encoding(self):
        """Test learned position encoding."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def init_fn():
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128,
                position_encoding="learned"
            )
            x = jnp.zeros((1, 10), dtype=jnp.int32)
            return model(x)
        
        init = hk.transform_with_state(init_fn)
        rng = jax.random.PRNGKey(42)
        params, state = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_no_position_encoding(self):
        """Test no position encoding."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def init_fn():
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128,
                position_encoding="none"
            )
            x = jnp.zeros((1, 10), dtype=jnp.int32)
            return model(x)
        
        init = hk.transform_with_state(init_fn)
        rng = jax.random.PRNGKey(42)
        params, state = init.init(rng)
        
        self.assertIsNotNone(params)


class TestSpikingAttention(unittest.TestCase):
    """Test spiking attention feature."""
    
    def test_spiking_enabled(self):
        """Test spiking attention enabled."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def init_fn():
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128,
                use_spiking=True
            )
            x = jnp.zeros((1, 10), dtype=jnp.int32)
            return model(x)
        
        init = hk.transform_with_state(init_fn)
        rng = jax.random.PRNGKey(42)
        params, state = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_spiking_disabled(self):
        """Test spiking attention disabled."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def init_fn():
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128,
                use_spiking=False
            )
            x = jnp.zeros((1, 10), dtype=jnp.int32)
            return model(x)
        
        init = hk.transform_with_state(init_fn)
        rng = jax.random.PRNGKey(42)
        params, state = init.init(rng)
        
        self.assertIsNotNone(params)


class TestForwardPass(unittest.TestCase):
    """Test forward pass of attention models."""
    
    def test_output_shape(self):
        """Test output shape matches expected."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def forward_fn(x):
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128
            )
            return model(x)
        
        init = hk.transform_with_state(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.randint(rng, (2, 20), 0, 1000)
        params, state = init.init(rng, x)
        output, new_state = init.apply(params, state, rng, x)
        
        # Output should have vocab_size dimension for logits
        self.assertEqual(output.shape[0], 2)  # batch
        self.assertEqual(output.shape[1], 20)  # seq_len


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of attention."""
    
    def test_no_nan_output(self):
        """Test no NaN in output."""
        from src.core.model.model_module_self_attention import SelfAttentionModel
        
        def forward_fn(x):
            model = SelfAttentionModel(
                d_model=64,
                num_heads=4,
                vocab_size=1000,
                max_seq_length=128
            )
            return model(x)
        
        init = hk.transform_with_state(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.randint(rng, (2, 20), 0, 1000)
        params, state = init.init(rng, x)
        output, new_state = init.apply(params, state, rng, x)
        
        self.assertFalse(jnp.any(jnp.isnan(output)))


if __name__ == "__main__":
    unittest.main()
