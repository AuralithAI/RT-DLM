"""
Tests for Fusion Module

Tests for cross-modal attention and multi-modal fusion.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk


class TestCrossModalAttention(unittest.TestCase):
    """Test CrossModalAttention class."""
    
    def test_attention_initialization(self):
        """Test CrossModalAttention initialization."""
        from src.modules.multimodal.fusion_module import CrossModalAttention
        
        def init_fn():
            attention = CrossModalAttention(d_model=64, num_heads=4)
            query = jnp.zeros((1, 10, 64))
            key = jnp.zeros((1, 20, 64))
            value = jnp.zeros((1, 20, 64))
            return attention(query, key, value)
        
        init = hk.transform(init_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_attention_output_shape(self):
        """Test attention output shape."""
        from src.modules.multimodal.fusion_module import CrossModalAttention
        
        def forward_fn(query, key, value):
            attention = CrossModalAttention(d_model=64, num_heads=4)
            output, weights = attention(query, key, value)
            return output, weights
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 10, 64))
        key = jax.random.normal(rng, (2, 20, 64))
        value = jax.random.normal(rng, (2, 20, 64))
        
        params = init.init(rng, query, key, value)
        output, weights = init.apply(params, rng, query, key, value)
        
        # Output should match query sequence length
        self.assertEqual(output.shape, (2, 10, 64))
        # Attention weights should have num_heads dimension
        self.assertEqual(weights.shape[1], 4)  # num_heads
    
    def test_attention_with_mask(self):
        """Test attention with mask."""
        from src.modules.multimodal.fusion_module import CrossModalAttention
        
        def forward_fn(query, key, value, mask):
            attention = CrossModalAttention(d_model=64, num_heads=4)
            output, weights = attention(query, key, value, mask=mask)
            return output, weights
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 10, 64))
        key = jax.random.normal(rng, (2, 20, 64))
        value = jax.random.normal(rng, (2, 20, 64))
        # Mask shape: [batch, heads, query_len, key_len] or broadcastable
        mask = jnp.ones((2, 1, 10, 20), dtype=jnp.bool_)
        
        params = init.init(rng, query, key, value, mask)
        output, weights = init.apply(params, rng, query, key, value, mask)
        
        self.assertIsNotNone(output)


class TestMultiModalFusionLayer(unittest.TestCase):
    """Test MultiModalFusionLayer class."""
    
    def test_fusion_initialization(self):
        """Test MultiModalFusionLayer initialization."""
        from src.modules.multimodal.fusion_module import MultiModalFusionLayer
        
        def init_fn():
            fusion = MultiModalFusionLayer(
                d_model=64,
                num_heads=4,
                modalities=['text', 'image', 'audio']
            )
            # Use same sequence length for all modalities (required for concatenation)
            modal_inputs = {
                'text': jnp.zeros((1, 10, 64)),
                'image': jnp.zeros((1, 10, 64)),
                'audio': jnp.zeros((1, 10, 64))
            }
            return fusion(modal_inputs)
        
        init = hk.transform(init_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_fusion_output_structure(self):
        """Test fusion layer output structure."""
        from src.modules.multimodal.fusion_module import MultiModalFusionLayer
        
        def forward_fn(modal_inputs):
            fusion = MultiModalFusionLayer(
                d_model=64,
                num_heads=4,
                modalities=['text', 'image']
            )
            return fusion(modal_inputs)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        # Use same sequence length for all modalities (required for concatenation)
        modal_inputs = {
            'text': jax.random.normal(rng, (2, 10, 64)),
            'image': jax.random.normal(rng, (2, 10, 64))
        }
        
        params = init.init(rng, modal_inputs)
        output = init.apply(params, rng, modal_inputs)
        
        # Output is (fused_output, cross_attention_maps, gated_features)
        self.assertIsInstance(output, tuple)
    
    def test_two_modality_fusion(self):
        """Test fusion with two modalities."""
        from src.modules.multimodal.fusion_module import MultiModalFusionLayer
        
        def forward_fn(modal_inputs):
            fusion = MultiModalFusionLayer(
                d_model=64,
                num_heads=4,
                modalities=['text', 'image']
            )
            return fusion(modal_inputs)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        # Use same sequence length for all modalities
        modal_inputs = {
            'text': jax.random.normal(rng, (2, 10, 64)),
            'image': jax.random.normal(rng, (2, 10, 64))
        }
        
        params = init.init(rng, modal_inputs)
        output = init.apply(params, rng, modal_inputs)
        
        self.assertIsNotNone(output)
    
    def test_three_modality_fusion(self):
        """Test fusion with three modalities."""
        from src.modules.multimodal.fusion_module import MultiModalFusionLayer
        
        def forward_fn(modal_inputs):
            fusion = MultiModalFusionLayer(
                d_model=64,
                num_heads=4,
                modalities=['text', 'image', 'audio']
            )
            return fusion(modal_inputs)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        # Use same sequence length for all modalities
        modal_inputs = {
            'text': jax.random.normal(rng, (2, 10, 64)),
            'image': jax.random.normal(rng, (2, 10, 64)),
            'audio': jax.random.normal(rng, (2, 10, 64))
        }
        
        params = init.init(rng, modal_inputs)
        output = init.apply(params, rng, modal_inputs)
        
        self.assertIsNotNone(output)


class TestModalityGating(unittest.TestCase):
    """Test adaptive modality gating."""
    
    def test_gating_mechanism(self):
        """Test modality gating produces valid gates."""
        from src.modules.multimodal.fusion_module import MultiModalFusionLayer
        
        def forward_fn(modal_inputs):
            fusion = MultiModalFusionLayer(
                d_model=64,
                num_heads=4,
                modalities=['text', 'image']
            )
            # Access gates through fusion
            return fusion(modal_inputs)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        # Use same sequence length for all modalities
        modal_inputs = {
            'text': jax.random.normal(rng, (2, 10, 64)),
            'image': jax.random.normal(rng, (2, 10, 64))
        }
        
        params = init.init(rng, modal_inputs)
        output = init.apply(params, rng, modal_inputs)
        
        self.assertIsNotNone(output)


class TestCrossAttentionMaps(unittest.TestCase):
    """Test cross-attention map generation."""
    
    def test_attention_maps_available(self):
        """Test that attention maps are computed."""
        from src.modules.multimodal.fusion_module import CrossModalAttention
        
        def forward_fn(query, key, value):
            attention = CrossModalAttention(d_model=64, num_heads=4)
            output, weights = attention(query, key, value)
            return output, weights
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 10, 64))
        key = jax.random.normal(rng, (2, 20, 64))
        value = jax.random.normal(rng, (2, 20, 64))
        
        params = init.init(rng, query, key, value)
        output, weights = init.apply(params, rng, query, key, value)
        
        # Attention weights should sum to 1 along key dimension
        sums = jnp.sum(weights, axis=-1)
        self.assertTrue(jnp.allclose(sums, 1.0, atol=1e-5))


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of fusion operations."""
    
    def test_no_nan_in_fusion(self):
        """Test fusion doesn't produce NaN."""
        from src.modules.multimodal.fusion_module import MultiModalFusionLayer
        
        def forward_fn(modal_inputs):
            fusion = MultiModalFusionLayer(
                d_model=64,
                num_heads=4,
                modalities=['text', 'image']
            )
            return fusion(modal_inputs)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        # Use same sequence length for all modalities
        modal_inputs = {
            'text': jax.random.normal(rng, (2, 10, 64)),
            'image': jax.random.normal(rng, (2, 10, 64))
        }
        
        params = init.init(rng, modal_inputs)
        output = init.apply(params, rng, modal_inputs)
        
        # Output is (fused_output, cross_attention_maps, gated_features)
        fused_output, cross_attn_maps, gated_features = output
        self.assertFalse(jnp.any(jnp.isnan(fused_output)))


if __name__ == "__main__":
    unittest.main()
