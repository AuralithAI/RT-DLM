"""
Tests for Advanced Attention Mechanisms

Tests RoPE, GQA, Sliding Window, and Linear attention implementations.
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from core.model.advanced_attention import (
    AttentionConfig,
    precompute_rope_frequencies,
    apply_rope,
    RotaryEmbedding,
    GroupedQueryAttention,
    SlidingWindowAttention,
    LinearAttention,
    AdvancedSelfAttention,
    create_attention_config,
    compute_attention_flops,
    estimate_kv_cache_size,
)


class TestRoPE:
    """Test Rotary Position Embedding."""
    
    def test_precompute_frequencies_shape(self):
        """Test that precomputed frequencies have correct shape."""
        dim = 64
        max_seq_len = 512
        
        cos, sin = precompute_rope_frequencies(dim, max_seq_len)
        
        assert cos.shape == (max_seq_len, dim // 2)
        assert sin.shape == (max_seq_len, dim // 2)
        
    def test_rope_preserves_shape(self):
        """Test that RoPE preserves input shape."""
        batch_size = 2
        seq_len = 32
        dim = 64
        
        x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, seq_len, dim))
        cos, sin = precompute_rope_frequencies(dim, seq_len)
        
        rotated = apply_rope(x, cos, sin)
        
        assert rotated.shape == x.shape
        
    def test_rope_different_positions_different_output(self):
        """Test that different positions get different rotations."""
        dim = 64
        max_seq_len = 32
        
        x = jnp.ones((1, max_seq_len, dim))  # Same input at all positions
        cos, sin = precompute_rope_frequencies(dim, max_seq_len)
        
        rotated = apply_rope(x, cos, sin)
        
        # Different positions should have different outputs
        assert not jnp.allclose(rotated[0, 0], rotated[0, 1])
        
    def test_rope_scaling_factor(self):
        """Test that scaling factor modifies frequencies."""
        dim = 64
        max_seq_len = 512
        
        cos_base, sin_base = precompute_rope_frequencies(dim, max_seq_len)
        cos_scaled, sin_scaled = precompute_rope_frequencies(dim, max_seq_len, scaling_factor=2.0)
        
        # Scaled frequencies should be different
        assert not jnp.allclose(cos_base, cos_scaled)
        
    def test_rotary_embedding_module(self):
        """Test RotaryEmbedding Haiku module."""
        dim = 64
        max_seq_len = 128
        batch_size = 2
        seq_len = 32
        
        def forward(q, k):
            rope = RotaryEmbedding(dim, max_seq_len)
            return rope(q, k)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        q = jax.random.normal(rng, (batch_size, seq_len, dim))
        k = jax.random.normal(rng, (batch_size, seq_len, dim))
        
        params = forward_fn.init(rng, q, k)
        q_rot, k_rot = forward_fn.apply(params, rng, q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestGroupedQueryAttention:
    """Test Grouped-Query Attention."""
    
    def test_gqa_output_shape(self):
        """Test that GQA produces correct output shape."""
        num_heads = 8
        num_kv_heads = 2
        head_dim = 64
        batch_size = 2
        seq_len = 32
        d_model = num_heads * head_dim
        
        def forward(x):
            gqa = GroupedQueryAttention(num_heads, num_kv_heads, head_dim, use_rope=False)
            return gqa(x, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.normal(rng, (batch_size, seq_len, d_model))
        
        params = forward_fn.init(rng, x)
        output, attn_weights = forward_fn.apply(params, rng, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        
    def test_gqa_with_rope(self):
        """Test GQA with rotary embeddings."""
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32
        
        def forward(x):
            gqa = GroupedQueryAttention(
                num_heads, num_kv_heads, head_dim, 
                use_rope=True, max_seq_length=64
            )
            return gqa(x, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.normal(rng, (2, 16, num_heads * head_dim))
        
        params = forward_fn.init(rng, x)
        output, _ = forward_fn.apply(params, rng, x)
        
        assert output.shape == x.shape
        
    def test_mqa_single_kv_head(self):
        """Test Multi-Query Attention (MQA) with single KV head."""
        num_heads = 8
        num_kv_heads = 1  # MQA
        head_dim = 64
        
        def forward(x):
            mqa = GroupedQueryAttention(num_heads, num_kv_heads, head_dim, use_rope=False)
            return mqa(x, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.normal(rng, (2, 32, num_heads * head_dim))
        
        params = forward_fn.init(rng, x)
        output, attn_weights = forward_fn.apply(params, rng, x)
        
        assert output.shape == x.shape
        assert attn_weights.shape[1] == num_heads  # Still num_heads in output
        
    def test_gqa_with_mask(self):
        """Test GQA with attention mask."""
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32
        d_model = num_heads * head_dim
        batch_size = 2
        seq_len = 16
        
        def forward(x, mask):
            gqa = GroupedQueryAttention(num_heads, num_kv_heads, head_dim, use_rope=False)
            return gqa(x, mask=mask, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.normal(rng, (batch_size, seq_len, d_model))
        mask = jnp.tril(jnp.ones((batch_size, seq_len, seq_len)))  # Causal mask
        
        params = forward_fn.init(rng, x, mask)
        output, attn_weights = forward_fn.apply(params, rng, x, mask)
        
        assert output.shape == x.shape
        
    def test_gqa_num_heads_divisibility(self):
        """Test that GQA raises error when num_heads not divisible by num_kv_heads."""
        with pytest.raises(ValueError):
            def forward(x):
                gqa = GroupedQueryAttention(8, 3, 64)  # 8 not divisible by 3
                return gqa(x)
            
            forward_fn = hk.transform(forward)
            rng = jax.random.PRNGKey(42)
            x = jax.random.normal(rng, (2, 16, 512))
            forward_fn.init(rng, x)


class TestSlidingWindowAttention:
    """Test Sliding Window Attention."""
    
    def test_sliding_window_output_shape(self):
        """Test that sliding window attention produces correct shape."""
        num_heads = 4
        head_dim = 32
        window_size = 8
        batch_size = 2
        seq_len = 32
        
        def forward(x):
            swa = SlidingWindowAttention(num_heads, head_dim, window_size)
            return swa(x, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.normal(rng, (batch_size, seq_len, num_heads * head_dim))
        
        params = forward_fn.init(rng, x)
        output, attn_weights = forward_fn.apply(params, rng, x)
        
        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        
    def test_sliding_window_locality(self):
        """Test that attention is local within window."""
        num_heads = 2
        head_dim = 16
        window_size = 4
        seq_len = 16
        
        def forward(x):
            swa = SlidingWindowAttention(num_heads, head_dim, window_size)
            return swa(x, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.normal(rng, (1, seq_len, num_heads * head_dim))
        
        params = forward_fn.init(rng, x)
        _, attn_weights = forward_fn.apply(params, rng, x)
        
        # Check that attention outside window is zero (after softmax with -inf masking)
        # Position 10 should not attend to position 0 (outside window of 4)
        far_attention = attn_weights[0, 0, 10, 0]
        assert far_attention < 1e-6, "Attention outside window should be near zero"


class TestLinearAttention:
    """Test Linear Attention."""
    
    def test_linear_attention_output_shape(self):
        """Test that linear attention produces correct shape."""
        num_heads = 4
        head_dim = 32
        batch_size = 2
        seq_len = 64
        
        def forward(x):
            la = LinearAttention(num_heads, head_dim)
            return la(x, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        x = jax.random.normal(rng, (batch_size, seq_len, num_heads * head_dim))
        
        params = forward_fn.init(rng, x)
        output = forward_fn.apply(params, rng, x)
        
        assert output.shape == x.shape
        
    def test_linear_attention_feature_maps(self):
        """Test different feature maps."""
        num_heads = 2
        head_dim = 16
        
        for feature_map in ["elu", "relu"]:
            def forward(x):
                la = LinearAttention(num_heads, head_dim, feature_map=feature_map)
                return la(x, is_training=False)
            
            forward_fn = hk.transform(forward)
            rng = jax.random.PRNGKey(42)
            
            x = jax.random.normal(rng, (1, 16, num_heads * head_dim))
            
            params = forward_fn.init(rng, x)
            output = forward_fn.apply(params, rng, x)
            
            assert output.shape == x.shape


class TestAdvancedSelfAttention:
    """Test AdvancedSelfAttention module."""
    
    def test_standard_attention(self):
        """Test standard attention mode."""
        config = AttentionConfig(
            d_model=64, num_heads=4, attention_type="standard",
            position_encoding="rope", max_seq_length=64
        )
        vocab_size = 1000
        
        def forward(inputs):
            model = AdvancedSelfAttention(config, vocab_size)
            return model(inputs, return_attention=True, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        inputs = jax.random.randint(rng, (2, 32), 1, vocab_size)
        
        params = forward_fn.init(rng, inputs)
        logits, attn_weights = forward_fn.apply(params, rng, inputs)
        
        assert logits.shape == (2, 32, vocab_size)
        
    def test_gqa_attention(self):
        """Test GQA attention mode."""
        config = AttentionConfig(
            d_model=64, num_heads=8, num_kv_heads=2,
            attention_type="gqa", position_encoding="rope"
        )
        vocab_size = 1000
        
        def forward(inputs):
            model = AdvancedSelfAttention(config, vocab_size)
            return model(inputs, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        inputs = jax.random.randint(rng, (2, 16), 1, vocab_size)
        
        params = forward_fn.init(rng, inputs)
        logits = forward_fn.apply(params, rng, inputs)
        
        assert logits.shape == (2, 16, vocab_size)
        
    def test_sliding_window_attention(self):
        """Test sliding window attention mode."""
        config = AttentionConfig(
            d_model=64, num_heads=4, attention_type="sliding",
            sliding_window_size=8, position_encoding="learned"
        )
        vocab_size = 500
        
        def forward(inputs):
            model = AdvancedSelfAttention(config, vocab_size)
            return model(inputs, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        inputs = jax.random.randint(rng, (2, 32), 1, vocab_size)
        
        params = forward_fn.init(rng, inputs)
        logits = forward_fn.apply(params, rng, inputs)
        
        assert logits.shape == (2, 32, vocab_size)
        
    def test_linear_attention_mode(self):
        """Test linear attention mode."""
        config = AttentionConfig(
            d_model=64, num_heads=4, attention_type="linear",
            position_encoding="none"
        )
        vocab_size = 500
        
        def forward(inputs):
            model = AdvancedSelfAttention(config, vocab_size)
            return model(inputs, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        inputs = jax.random.randint(rng, (2, 64), 1, vocab_size)
        
        params = forward_fn.init(rng, inputs)
        logits = forward_fn.apply(params, rng, inputs)
        
        assert logits.shape == (2, 64, vocab_size)
        
    def test_spiking_integration(self):
        """Test spiking attention integration."""
        config = AttentionConfig(
            d_model=64, num_heads=4, enable_spiking=True, spike_threshold=0.1
        )
        vocab_size = 500
        
        def forward(inputs):
            model = AdvancedSelfAttention(config, vocab_size)
            return model(inputs, spike_threshold=0.1, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        inputs = jax.random.randint(rng, (2, 16), 1, vocab_size)
        
        params = forward_fn.init(rng, inputs)
        logits = forward_fn.apply(params, rng, inputs)
        
        assert logits.shape == (2, 16, vocab_size)


class TestConfigFactory:
    """Test configuration factory functions."""
    
    def test_create_standard_config(self):
        """Test creating standard attention config."""
        config = create_attention_config(d_model=512, num_heads=8)
        
        assert config.d_model == 512
        assert config.num_heads == 8
        assert config.attention_type == "standard"
        
    def test_create_gqa_config(self):
        """Test creating GQA config with auto num_kv_heads."""
        config = create_attention_config(
            d_model=512, num_heads=8, attention_type="gqa"
        )
        
        assert config.attention_type == "gqa"
        assert config.num_kv_heads == 2  # 8 // 4 = 2
        
    def test_create_mqa_config(self):
        """Test creating MQA config."""
        config = create_attention_config(
            d_model=512, num_heads=8, attention_type="mqa"
        )
        
        assert config.attention_type == "mqa"
        assert config.num_kv_heads == 1
        
    def test_create_rope_config(self):
        """Test creating config with RoPE."""
        config = create_attention_config(
            position_encoding="rope", rope_theta=10000.0
        )
        
        assert config.position_encoding == "rope"
        assert config.rope_theta == 10000.0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_compute_attention_flops_standard(self):
        """Test FLOP computation for standard attention."""
        flops = compute_attention_flops(
            batch_size=4, seq_len=1024, d_model=512, num_heads=8
        )
        
        # O(n² * d) - should be substantial
        assert flops > 0
        
    def test_compute_attention_flops_linear(self):
        """Test FLOP computation for linear attention."""
        flops_standard = compute_attention_flops(
            batch_size=4, seq_len=1024, d_model=512, num_heads=8,
            attention_type="standard"
        )
        flops_linear = compute_attention_flops(
            batch_size=4, seq_len=1024, d_model=512, num_heads=8,
            attention_type="linear"
        )
        
        # Linear should be much fewer FLOPs for long sequences
        assert flops_linear < flops_standard
        
    def test_estimate_kv_cache_size(self):
        """Test KV cache size estimation."""
        cache_info = estimate_kv_cache_size(
            batch_size=4, seq_len=2048, d_model=512,
            num_layers=24, num_kv_heads=8
        )
        
        assert "kv_cache_gb" in cache_info
        assert cache_info["kv_cache_gb"] > 0
        
    def test_kv_cache_reduction_with_gqa(self):
        """Test that GQA reduces KV cache size."""
        # Standard MHA with 8 KV heads
        cache_mha = estimate_kv_cache_size(
            batch_size=4, seq_len=2048, d_model=512,
            num_layers=24, num_kv_heads=8
        )
        
        # GQA with 2 KV heads
        cache_gqa = estimate_kv_cache_size(
            batch_size=4, seq_len=2048, d_model=512,
            num_layers=24, num_kv_heads=2
        )
        
        # GQA should have ~4x smaller cache
        assert cache_gqa["kv_cache_gb"] < cache_mha["kv_cache_gb"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
