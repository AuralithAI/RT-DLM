"""
Tests for Augmented Attention Module

Tests for retrieval-augmented attention mechanisms.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk


class TestCrossAttentionRetrieval(unittest.TestCase):
    """Test CrossAttentionRetrieval class."""
    
    def test_attention_initialization(self):
        """Test CrossAttentionRetrieval initialization."""
        from src.modules.retrieval.augmented_attention import CrossAttentionRetrieval
        
        def init_fn():
            attention = CrossAttentionRetrieval(
                d_model=64,
                num_heads=4,
                dropout_rate=0.1
            )
            query = jnp.zeros((1, 10, 64))
            retrieved = jnp.zeros((1, 5, 64))
            return attention(query, retrieved)
        
        init = hk.transform(init_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_attention_output_shape(self):
        """Test attention output shape matches query."""
        from src.modules.retrieval.augmented_attention import CrossAttentionRetrieval
        
        def forward_fn(query, retrieved):
            attention = CrossAttentionRetrieval(d_model=64, num_heads=4)
            return attention(query, retrieved)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 10, 64))
        retrieved = jax.random.normal(rng, (2, 5, 64))
        
        params = init.init(rng, query, retrieved)
        output = init.apply(params, rng, query, retrieved)
        
        # Output should match query shape
        self.assertEqual(output.shape, query.shape)
    
    def test_attention_with_mask(self):
        """Test attention with retrieved document mask."""
        from src.modules.retrieval.augmented_attention import CrossAttentionRetrieval
        
        def forward_fn(query, retrieved, mask):
            attention = CrossAttentionRetrieval(d_model=64, num_heads=4)
            return attention(query, retrieved, retrieved_mask=mask)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 10, 64))
        retrieved = jax.random.normal(rng, (2, 5, 64))
        mask = jnp.ones((2, 5), dtype=jnp.bool_)
        mask = mask.at[:, 4].set(False)  # Mask out last retrieved doc
        
        params = init.init(rng, query, retrieved, mask)
        output = init.apply(params, rng, query, retrieved, mask)
        
        self.assertEqual(output.shape, query.shape)
    
    def test_attention_training_mode(self):
        """Test attention in training mode with dropout."""
        from src.modules.retrieval.augmented_attention import CrossAttentionRetrieval
        
        def forward_fn(query, retrieved, is_training):
            attention = CrossAttentionRetrieval(
                d_model=64,
                num_heads=4,
                dropout_rate=0.1
            )
            return attention(query, retrieved, is_training=is_training)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 10, 64))
        retrieved = jax.random.normal(rng, (2, 5, 64))
        
        params = init.init(rng, query, retrieved, True)
        output = init.apply(params, rng, query, retrieved, True)
        
        self.assertEqual(output.shape, query.shape)


class TestRetrievalAugmentedAttention(unittest.TestCase):
    """Test RetrievalAugmentedAttention class."""
    
    def test_retrieval_augmented_attention_exists(self):
        """Test RetrievalAugmentedAttention exists."""
        try:
            from src.modules.retrieval.augmented_attention import (
                RetrievalAugmentedAttention
            )
            self.assertIsNotNone(RetrievalAugmentedAttention)
        except ImportError:
            self.skipTest("RetrievalAugmentedAttention not available")
    
    def test_retrieval_augmented_forward(self):
        """Test RetrievalAugmentedAttention forward pass."""
        try:
            from src.modules.retrieval.augmented_attention import (
                RetrievalAugmentedAttention
            )
            
            def forward_fn(x, retrieved):
                attention = RetrievalAugmentedAttention(
                    d_model=64,
                    num_heads=4
                )
                return attention(x, retrieved)
            
            init = hk.transform(forward_fn)
            rng = jax.random.PRNGKey(42)
            
            x = jax.random.normal(rng, (2, 10, 64))
            retrieved = jax.random.normal(rng, (2, 5, 64))
            
            params = init.init(rng, x, retrieved)
            output = init.apply(params, rng, x, retrieved)
            
            # Output can be a tuple (output, attn_weights) or just output
            if isinstance(output, tuple):
                self.assertEqual(output[0].shape, x.shape)
            else:
                self.assertEqual(output.shape, x.shape)
        except ImportError:
            self.skipTest("RetrievalAugmentedAttention not available")


class TestMultiHeadConfiguration(unittest.TestCase):
    """Test multi-head attention configuration."""
    
    def test_d_model_divisible_by_heads(self):
        """Test that d_model must be divisible by num_heads."""
        from src.modules.retrieval.augmented_attention import CrossAttentionRetrieval
        
        # Valid configuration
        def valid_init():
            return CrossAttentionRetrieval(d_model=64, num_heads=4)
        
        init = hk.transform(valid_init)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_head_dim_calculation(self):
        """Test head dimension calculation."""
        from src.modules.retrieval.augmented_attention import CrossAttentionRetrieval
        
        d_model = 64
        num_heads = 4
        
        def init_fn():
            attention = CrossAttentionRetrieval(d_model=d_model, num_heads=num_heads)
            return attention.head_dim
        
        init = hk.transform(init_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        head_dim = init.apply(params, rng)
        
        self.assertEqual(head_dim, d_model // num_heads)


class TestRetrievalIntegration(unittest.TestCase):
    """Test integration with retrieval systems."""
    
    def test_variable_num_retrieved(self):
        """Test attention with different numbers of retrieved documents."""
        from src.modules.retrieval.augmented_attention import CrossAttentionRetrieval
        
        for num_retrieved in [1, 5, 10]:
            def forward_fn(query, retrieved):
                attention = CrossAttentionRetrieval(d_model=64, num_heads=4)
                return attention(query, retrieved)
            
            init = hk.transform(forward_fn)
            rng = jax.random.PRNGKey(42)
            
            query = jax.random.normal(rng, (2, 10, 64))
            retrieved = jax.random.normal(rng, (2, num_retrieved, 64))
            
            params = init.init(rng, query, retrieved)
            output = init.apply(params, rng, query, retrieved)
            
            # Output shape should match query regardless of num_retrieved
            self.assertEqual(output.shape, (2, 10, 64))


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of retrieval attention."""
    
    def test_no_nan_output(self):
        """Test that attention doesn't produce NaN."""
        from src.modules.retrieval.augmented_attention import CrossAttentionRetrieval
        
        def forward_fn(query, retrieved):
            attention = CrossAttentionRetrieval(d_model=64, num_heads=4)
            return attention(query, retrieved)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 10, 64))
        retrieved = jax.random.normal(rng, (2, 5, 64))
        
        params = init.init(rng, query, retrieved)
        output = init.apply(params, rng, query, retrieved)
        
        self.assertFalse(jnp.any(jnp.isnan(output)))
    
    def test_empty_retrieval_handling(self):
        """Test handling of empty retrieval (all masked)."""
        from src.modules.retrieval.augmented_attention import CrossAttentionRetrieval
        
        def forward_fn(query, retrieved, mask):
            attention = CrossAttentionRetrieval(d_model=64, num_heads=4)
            return attention(query, retrieved, retrieved_mask=mask)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 10, 64))
        retrieved = jax.random.normal(rng, (2, 5, 64))
        # All retrieved documents masked
        mask = jnp.zeros((2, 5), dtype=jnp.bool_)
        
        params = init.init(rng, query, retrieved, mask)
        output = init.apply(params, rng, query, retrieved, mask)
        
        # Should still produce valid output
        self.assertEqual(output.shape, query.shape)


if __name__ == "__main__":
    unittest.main()
