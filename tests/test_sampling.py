"""
Tests for the TokenSampler and sampling strategies.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from core.sampling import (
    TokenSampler,
    SamplingConfig,
    SampleOutput,
    create_sampling_config_balanced,
    create_sampling_config_creative,
    create_sampling_config_precise,
    create_sampling_config_deterministic,
)


class TestTokenSampler:
    """Test suite for TokenSampler class."""
    
    @pytest.fixture
    def sampler(self):
        """Create a sampler instance."""
        return TokenSampler(vocab_size=1000)
    
    @pytest.fixture
    def rng_key(self):
        """Create a random key."""
        return jax.random.PRNGKey(42)
    
    @pytest.fixture
    def sample_logits(self):
        """Create sample logits for testing."""
        # Create logits with a clear peak at token 42
        logits = jnp.zeros((2, 1000))  # batch_size=2, vocab_size=1000
        logits = logits.at[:, 42].set(10.0)  # Strong preference for token 42
        logits = logits.at[:, 100].set(5.0)  # Second choice
        logits = logits.at[:, 200].set(3.0)  # Third choice
        return logits
    
    def test_apply_temperature_low(self, sampler, sample_logits):
        """Test that low temperature makes distribution sharper."""
        scaled = sampler.apply_temperature(sample_logits, temperature=0.1)
        # Low temperature should amplify differences
        assert jnp.max(scaled) > jnp.max(sample_logits)
    
    def test_apply_temperature_high(self, sampler, sample_logits):
        """Test that high temperature makes distribution flatter."""
        scaled = sampler.apply_temperature(sample_logits, temperature=2.0)
        # High temperature should reduce differences
        assert jnp.max(scaled) < jnp.max(sample_logits)
    
    def test_apply_temperature_one(self, sampler, sample_logits):
        """Test that temperature=1.0 leaves logits unchanged."""
        scaled = sampler.apply_temperature(sample_logits, temperature=1.0)
        assert jnp.allclose(scaled, sample_logits)
    
    def test_apply_top_k(self, sampler, sample_logits):
        """Test top-K filtering."""
        filtered = sampler.apply_top_k(sample_logits, top_k=2)
        
        # Only top 2 tokens should have valid logits
        probs = jax.nn.softmax(filtered, axis=-1)
        # Token 42 and 100 should have almost all probability
        top_prob = probs[:, 42] + probs[:, 100]
        assert jnp.all(top_prob > 0.99)
    
    def test_apply_top_k_disabled(self, sampler, sample_logits):
        """Test that top_k=0 disables filtering."""
        filtered = sampler.apply_top_k(sample_logits, top_k=0)
        assert jnp.allclose(filtered, sample_logits)
    
    def test_apply_top_p(self, sampler, sample_logits):
        """Test nucleus (top-P) filtering."""
        filtered = sampler.apply_top_p(sample_logits, top_p=0.5)
        
        # With top_p=0.5, only the most probable tokens should remain
        probs = jax.nn.softmax(filtered, axis=-1)
        # Check that low-probability tokens are filtered out
        assert probs[:, 500].sum() < 1e-5  # Random token should be filtered
    
    def test_apply_top_p_disabled(self, sampler, sample_logits):
        """Test that top_p=1.0 disables filtering."""
        filtered = sampler.apply_top_p(sample_logits, top_p=1.0)
        assert jnp.allclose(filtered, sample_logits)
    
    def test_sample_returns_correct_shape(self, sampler, sample_logits, rng_key):
        """Test that sample returns correct output shapes."""
        config = SamplingConfig(temperature=1.0, top_k=50, top_p=0.9)
        output = sampler.sample(sample_logits, config, rng_key)
        
        batch_size = sample_logits.shape[0]
        
        assert isinstance(output, SampleOutput)
        assert output.token_id.shape == (batch_size, 1)
        assert output.token_prob.shape == (batch_size, 1)
        assert output.token_log_prob.shape == (batch_size, 1)
        assert output.entropy.shape == (batch_size,)
    
    def test_sample_with_3d_logits(self, sampler, rng_key):
        """Test sampling from 3D logits (batch, seq_len, vocab)."""
        logits_3d = jnp.zeros((2, 10, 1000))  # batch=2, seq_len=10, vocab=1000
        logits_3d = logits_3d.at[:, -1, 42].set(10.0)  # Peak at last position
        
        config = SamplingConfig()
        output = sampler.sample(logits_3d, config, rng_key)
        
        # Should take last position and return shape (batch, 1)
        assert output.token_id.shape == (2, 1)
    
    def test_greedy_sample(self, sampler, sample_logits):
        """Test greedy decoding."""
        output = sampler.greedy_sample(sample_logits)
        
        # Should always select token 42 (highest logit)
        assert jnp.all(output.token_id == 42)
    
    def test_sample_deterministic_config(self, sampler, sample_logits, rng_key):
        """Test deterministic sampling config produces consistent results."""
        config = create_sampling_config_deterministic()
        
        output1 = sampler.sample(sample_logits, config, rng_key)
        output2 = sampler.sample(sample_logits, config, rng_key)
        
        # With near-zero temperature and top_k=1, should be deterministic
        assert jnp.all(output1.token_id == output2.token_id)
    
    def test_entropy_calculation(self, sampler, sample_logits, rng_key):
        """Test entropy is computed correctly."""
        config = SamplingConfig()
        output = sampler.sample(sample_logits, config, rng_key)
        
        # Entropy should be non-negative
        assert jnp.all(output.entropy >= 0)
    
    def test_top_k_tokens_returned(self, sampler, sample_logits, rng_key):
        """Test that top-K alternatives are returned."""
        config = SamplingConfig(top_log_probs=5)
        output = sampler.sample(sample_logits, config, rng_key)
        
        batch_size = sample_logits.shape[0]
        assert output.top_k_token_ids.shape == (batch_size, 5)
        assert output.top_k_probs.shape == (batch_size, 5)
        
        # Token 42 should be in top-K for all batches
        assert jnp.any(output.top_k_token_ids == 42)


class TestSamplingConfigs:
    """Test preset sampling configurations."""
    
    def test_creative_config(self):
        """Test creative config has high temperature."""
        config = create_sampling_config_creative()
        assert config.temperature > 1.0
        assert config.top_k > 50
    
    def test_precise_config(self):
        """Test precise config has low temperature."""
        config = create_sampling_config_precise()
        assert config.temperature < 0.5
        assert config.top_k < 50
    
    def test_balanced_config(self):
        """Test balanced config has moderate settings."""
        config = create_sampling_config_balanced()
        assert 0.5 <= config.temperature <= 1.0
        assert 0.85 <= config.top_p <= 0.95
    
    def test_deterministic_config(self):
        """Test deterministic config is near-greedy."""
        config = create_sampling_config_deterministic()
        assert config.temperature < 0.1
        assert config.top_k == 1


class TestMinPFiltering:
    """Test min-P filtering strategy."""
    
    @pytest.fixture
    def sampler(self):
        return TokenSampler(vocab_size=100)
    
    def test_min_p_filters_low_prob(self, sampler):
        """Test that min_p filters very low probability tokens."""
        # Create logits with one dominant token
        logits = jnp.zeros((1, 100))
        logits = logits.at[0, 0].set(10.0)  # Dominant token
        logits = logits.at[0, 1].set(1.0)   # Should be filtered with min_p=0.5
        
        filtered = sampler.apply_min_p(logits, min_p=0.5)
        
        probs = jax.nn.softmax(filtered, axis=-1)
        # Token 1 should have very low probability after filtering
        assert probs[0, 1] < 0.01
    
    def test_min_p_disabled(self, sampler):
        """Test that min_p=0 disables filtering."""
        logits = jnp.ones((1, 100))
        filtered = sampler.apply_min_p(logits, min_p=0.0)
        assert jnp.allclose(filtered, logits)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
