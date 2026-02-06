"""
Unit Tests for Reusable Components

Tests the reusable attention wrapper and related components for:
1. ReusableAttention module instantiation and forward pass
2. SpikingMechanism threshold behavior
3. Component reusability across multiple instances
"""

import unittest
import sys
import os
import jax
import jax.numpy as jnp
import haiku as hk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.components.reusable_components import (
    AttentionConfig,
    ReusableAttention,
    ReusableTransformerBlock,
    SpikingMechanism,
    PruningManager,
    create_attention,
    apply_shared_spiking,
    compute_attention_sparsity,
)


class TestReusableAttention(unittest.TestCase):
    """Test ReusableAttention module instantiation and forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = jax.random.PRNGKey(42)
        self.batch_size = 2
        self.seq_length = 16
        self.d_model = 64
        self.num_heads = 4
        
    def test_attention_forward_pass(self):
        """Test that ReusableAttention produces correct output shape."""
        def forward_fn(x):
            config = AttentionConfig(d_model=self.d_model, num_heads=self.num_heads)
            attention = ReusableAttention(config=config, name="test_attention")
            output, attn_weights = attention(x, return_attention=True, spike_threshold=0.1)
            return output, attn_weights
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        # Create input
        inputs = jax.random.normal(
            self.rng, 
            (self.batch_size, self.seq_length, self.d_model)
        )
        
        # Initialize and apply
        params, state = init_fn(self.rng, inputs)
        (output, attn_weights), _ = apply_fn(params, state, self.rng, inputs)
        
        # Verify shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.d_model))
        self.assertEqual(attn_weights.shape[-1], self.d_model)
        
    def test_attention_with_config(self):
        """Test ReusableAttention with full AttentionConfig."""
        config = AttentionConfig(
            d_model=128,
            num_heads=8,
            spike_threshold=0.2,
            epsilon=1e-6,
            enable_spiking=True,
            enable_usage_tracking=True
        )
        
        def forward_fn(x):
            attention = ReusableAttention(config=config, name="configured_attention")
            return attention(x, return_attention=False)
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        inputs = jax.random.normal(
            self.rng,
            (self.batch_size, self.seq_length, 128)
        )
        
        params, state = init_fn(self.rng, inputs)
        (output, _), _ = apply_fn(params, state, self.rng, inputs)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, 128))
    
    def test_attention_without_spiking(self):
        """Test ReusableAttention with spiking disabled."""
        config = AttentionConfig(
            d_model=self.d_model,
            num_heads=self.num_heads,
            enable_spiking=False
        )
        
        def forward_fn(x):
            attention = ReusableAttention(config=config, name="no_spike_attention")
            return attention(x)
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        inputs = jax.random.normal(
            self.rng,
            (self.batch_size, self.seq_length, self.d_model)
        )
        
        params, state = init_fn(self.rng, inputs)
        (output, _), _ = apply_fn(params, state, self.rng, inputs)
        
        self.assertEqual(output.shape, inputs.shape)


class TestSpikingMechanism(unittest.TestCase):
    """Test SpikingMechanism threshold behavior."""
    
    def test_spiking_threshold_application(self):
        """Test that spiking correctly thresholds values."""
        mechanism = SpikingMechanism(spike_threshold=0.5, epsilon=1e-8)
        
        # Create scores with known values
        scores = jnp.array([[0.2, 0.6, 0.8], [0.3, 0.4, 0.9]])
        spiked = mechanism.apply(scores)
        
        # Values below threshold should be zero
        self.assertEqual(float(spiked[0, 0]), 0.0)  # 0.2 < 0.5
        self.assertEqual(float(spiked[1, 0]), 0.0)  # 0.3 < 0.5
        self.assertEqual(float(spiked[1, 1]), 0.0)  # 0.4 < 0.5
        
        # Values above threshold should be non-zero
        self.assertGreater(float(spiked[0, 1]), 0.0)  # 0.6 > 0.5
        self.assertGreater(float(spiked[0, 2]), 0.0)  # 0.8 > 0.5
        self.assertGreater(float(spiked[1, 2]), 0.0)  # 0.9 > 0.5
        
    def test_spiking_normalization(self):
        """Test that spiked scores are properly normalized."""
        mechanism = SpikingMechanism(spike_threshold=0.3, epsilon=1e-8)
        
        scores = jnp.array([[0.1, 0.4, 0.5]])
        spiked = mechanism.apply(scores)
        
        # Sum should be close to 1 after normalization
        row_sum = float(jnp.sum(spiked, axis=-1)[0])
        self.assertAlmostEqual(row_sum, 1.0, places=5)
        
    def test_invalid_threshold_passthrough(self):
        """Test that invalid threshold returns unchanged scores."""
        # Threshold out of range
        mechanism = SpikingMechanism(spike_threshold=1.5, epsilon=1e-8)
        scores = jnp.array([[0.2, 0.6, 0.8]])
        result = mechanism.apply(scores)
        
        # Should return unchanged
        self.assertTrue(jnp.allclose(result, scores))
        
    def test_sparsity_calculation(self):
        """Test sparsity ratio calculation."""
        mechanism = SpikingMechanism(spike_threshold=0.5, epsilon=1e-8)
        
        # 2 out of 4 values are above threshold
        scores = jnp.array([[0.2, 0.6], [0.8, 0.3]])
        sparsity = mechanism.get_sparsity(scores)
        
        # 50% are below threshold (0.2 and 0.3)
        self.assertAlmostEqual(sparsity, 0.5, places=1)


class TestComponentReusability(unittest.TestCase):
    """Test that components can be reused across multiple instances."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = jax.random.PRNGKey(0)
        self.d_model = 64
        self.num_heads = 4
        
    def test_multiple_attention_instances(self):
        """Test creating multiple independent attention instances."""
        def forward_fn(x):
            config = AttentionConfig(d_model=self.d_model, num_heads=self.num_heads)
            
            # Create two independent attention modules
            attn1 = ReusableAttention(config=config, name="attention_1")
            attn2 = ReusableAttention(config=config, name="attention_2")
            
            # Apply both
            out1, _ = attn1(x, spike_threshold=0.1)
            out2, _ = attn2(out1, spike_threshold=0.2)
            
            return out1, out2
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        inputs = jax.random.normal(self.rng, (2, 8, self.d_model))
        params, state = init_fn(self.rng, inputs)
        
        # Verify both modules created separate parameters
        param_keys = list(params.keys())
        attn1_keys = [k for k in param_keys if 'attention_1' in k]
        attn2_keys = [k for k in param_keys if 'attention_2' in k]
        
        self.assertGreater(len(attn1_keys), 0)
        self.assertGreater(len(attn2_keys), 0)
        
        # Apply and verify outputs
        (out1, out2), _ = apply_fn(params, state, self.rng, inputs)
        
        self.assertEqual(out1.shape, inputs.shape)
        self.assertEqual(out2.shape, inputs.shape)
        
    def test_factory_function_creates_unique_instances(self):
        """Test that factory functions create independent modules."""
        def forward_fn(x):
            attn1 = create_attention(d_model=self.d_model, num_heads=self.num_heads, name="factory_1")
            attn2 = create_attention(d_model=self.d_model, num_heads=self.num_heads, name="factory_2")
            
            out1, _ = attn1(x)
            out2, _ = attn2(x)
            
            return out1, out2
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        inputs = jax.random.normal(self.rng, (2, 8, self.d_model))
        params, state = init_fn(self.rng, inputs)
        
        (out1, out2), _ = apply_fn(params, state, self.rng, inputs)
        
        # Both outputs should be valid but different
        self.assertEqual(out1.shape, out2.shape)
        
    def test_transformer_block_stacking(self):
        """Test stacking multiple transformer blocks."""
        def forward_fn(x):
            config = AttentionConfig(d_model=self.d_model, num_heads=self.num_heads)
            
            # Stack multiple blocks
            block1 = ReusableTransformerBlock(config=config, name="block_1")
            block2 = ReusableTransformerBlock(config=config, name="block_2")
            
            out1, attn1 = block1(x, spike_threshold=0.1, return_attention=True)
            out2, attn2 = block2(out1, spike_threshold=0.1, return_attention=True)
            
            return out2, attn2
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        inputs = jax.random.normal(self.rng, (2, 8, self.d_model))
        params, state = init_fn(self.rng, inputs)
        
        # Verify blocks have separate parameters
        block1_params = [k for k in params.keys() if 'block_1' in k]
        block2_params = [k for k in params.keys() if 'block_2' in k]
        
        self.assertGreater(len(block1_params), 0)
        self.assertGreater(len(block2_params), 0)
        
        # Apply and verify output
        (output, attn_weights), _ = apply_fn(params, state, self.rng, inputs)
        
        self.assertEqual(output.shape, inputs.shape)


class TestPruningManager(unittest.TestCase):
    """Test PruningManager utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_heads = 4
        self.d_model = 64
        self.manager = PruningManager(
            num_heads=self.num_heads,
            d_model=self.d_model,
            head_threshold=0.01,
            ffn_threshold=0.01
        )
        
    def test_head_usage_computation_3d(self):
        """Test head usage computation from 3D attention weights."""
        # Shape: [batch, seq, d_model]
        attn_weights = jax.random.uniform(
            jax.random.PRNGKey(0),
            (2, 8, self.d_model)
        )
        
        head_usage = self.manager.compute_head_usage(attn_weights)
        
        self.assertEqual(head_usage.shape, (self.num_heads,))
        
    def test_head_usage_computation_4d(self):
        """Test head usage computation from 4D attention weights."""
        # Shape: [batch, heads, seq, seq]
        attn_weights = jax.random.uniform(
            jax.random.PRNGKey(0),
            (2, self.num_heads, 8, 8)
        )
        
        head_usage = self.manager.compute_head_usage(attn_weights)
        
        self.assertEqual(head_usage.shape, (self.num_heads,))
        
    def test_pruning_mask_generation(self):
        """Test pruning mask generation."""
        # High usage for some heads
        head_usage = jnp.array([0.1, 0.005, 0.2, 0.001])
        ffn_usage = jnp.ones(self.d_model) * 0.1
        
        active_heads, active_ffn = self.manager.get_pruning_mask(head_usage, ffn_usage)
        
        # Heads 0, 2 should be active (above 0.01)
        self.assertTrue(active_heads[0])
        self.assertFalse(active_heads[1])
        self.assertTrue(active_heads[2])
        self.assertFalse(active_heads[3])
        
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        active_heads = jnp.array([True, False, True, False])  # 2 of 4 active
        active_ffn = jnp.ones(self.d_model, dtype=bool)  # All active
        
        compression = self.manager.compute_compression_ratio(active_heads, active_ffn)
        
        self.assertAlmostEqual(compression['head_compression'], 0.5, places=2)
        self.assertAlmostEqual(compression['ffn_compression'], 0.0, places=2)
        self.assertEqual(compression['active_heads'], 2)


class TestUtilityFunctions(unittest.TestCase):
    """Test standalone utility functions."""
    
    def test_apply_shared_spiking(self):
        """Test standalone spiking function."""
        scores = jnp.array([[0.1, 0.5, 0.8]])
        
        spiked = apply_shared_spiking(scores, spike_threshold=0.4, epsilon=1e-8)
        
        # First value should be zero
        self.assertEqual(float(spiked[0, 0]), 0.0)
        # Other values should be non-zero
        self.assertGreater(float(spiked[0, 1]), 0.0)
        self.assertGreater(float(spiked[0, 2]), 0.0)
        
    def test_compute_attention_sparsity(self):
        """Test sparsity computation utility."""
        scores = jnp.array([[0.1, 0.6], [0.8, 0.2]])
        
        sparsity = compute_attention_sparsity(scores, spike_threshold=0.5)
        
        # 2 of 4 values below threshold
        self.assertAlmostEqual(sparsity, 0.5, places=1)


class TestAttentionConfig(unittest.TestCase):
    """Test AttentionConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AttentionConfig()
        
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.spike_threshold, 0.1)
        self.assertTrue(config.enable_spiking)
        
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = AttentionConfig(d_model=128, num_heads=4)
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict['d_model'], 128)
        self.assertEqual(config_dict['num_heads'], 4)
        
    def test_config_from_dict(self):
        """Test configuration deserialization."""
        config_dict = {'d_model': 512, 'num_heads': 16, 'spike_threshold': 0.2}
        config = AttentionConfig.from_dict(config_dict)
        
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.spike_threshold, 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

