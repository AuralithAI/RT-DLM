"""
Tests for Sampling Module

Tests for token sampling strategies including temperature, top-k, top-p,
repetition penalty, and various sampling configurations.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np


class TestSamplingConfig(unittest.TestCase):
    """Test SamplingConfig dataclass."""
    
    def test_default_config(self):
        """Test default sampling configuration."""
        from src.core.sampling import SamplingConfig
        
        config = SamplingConfig()
        
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_k, 50)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.min_p, 0.0)
        self.assertEqual(config.repetition_penalty, 1.0)
        self.assertEqual(config.max_tokens, 512)
        self.assertTrue(config.log_probs)
    
    def test_custom_config(self):
        """Test custom sampling configuration."""
        from src.core.sampling import SamplingConfig
        
        config = SamplingConfig(
            temperature=0.7,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.2,
            max_tokens=256,
            stop_tokens=[0, 1, 2]
        )
        
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_k, 40)
        self.assertEqual(config.top_p, 0.95)
        self.assertEqual(config.repetition_penalty, 1.2)
        self.assertEqual(config.max_tokens, 256)
        self.assertEqual(config.stop_tokens, [0, 1, 2])


class TestSampleOutput(unittest.TestCase):
    """Test SampleOutput named tuple."""
    
    def test_sample_output_fields(self):
        """Test SampleOutput has correct fields."""
        from src.core.sampling import SampleOutput
        
        output = SampleOutput(
            token_id=jnp.array([1]),
            token_prob=jnp.array([0.5]),
            token_log_prob=jnp.array([-0.693]),
            top_k_token_ids=jnp.array([[1, 2, 3]]),
            top_k_probs=jnp.array([[0.5, 0.3, 0.2]]),
            top_k_log_probs=jnp.array([[-0.693, -1.2, -1.6]]),
            entropy=jnp.array([1.5])
        )
        
        self.assertEqual(output.token_id.shape, (1,))
        self.assertEqual(output.token_prob.shape, (1,))
        self.assertAlmostEqual(float(output.token_prob[0]), 0.5, places=2)


class TestTokenSampler(unittest.TestCase):
    """Test TokenSampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 100
        self.batch_size = 2
        self.rng = jax.random.PRNGKey(42)
    
    def test_sampler_initialization(self):
        """Test TokenSampler initialization."""
        from src.core.sampling import TokenSampler
        
        sampler = TokenSampler(vocab_size=self.vocab_size)
        self.assertEqual(sampler.vocab_size, self.vocab_size)
        
        sampler_no_vocab = TokenSampler()
        self.assertIsNone(sampler_no_vocab.vocab_size)
    
    def test_apply_temperature(self):
        """Test temperature scaling."""
        from src.core.sampling import TokenSampler
        
        sampler = TokenSampler()
        logits = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        
        # High temperature -> more uniform
        scaled_high = sampler.apply_temperature(logits, temperature=2.0)
        self.assertEqual(scaled_high.shape, logits.shape)
        self.assertTrue(jnp.allclose(scaled_high, logits / 2.0))
        
        # Low temperature -> more peaked
        scaled_low = sampler.apply_temperature(logits, temperature=0.5)
        self.assertTrue(jnp.allclose(scaled_low, logits / 0.5))
    
    def test_apply_temperature_invalid(self):
        """Test temperature with invalid values."""
        from src.core.sampling import TokenSampler
        
        sampler = TokenSampler()
        logits = jnp.array([[1.0, 2.0, 3.0]])
        
        with self.assertRaises(ValueError):
            sampler.apply_temperature(logits, temperature=0.0)
        
        with self.assertRaises(ValueError):
            sampler.apply_temperature(logits, temperature=-1.0)
    
    def test_apply_top_k(self):
        """Test top-K filtering."""
        from src.core.sampling import TokenSampler
        
        sampler = TokenSampler()
        logits = jnp.array([[1.0, 5.0, 2.0, 4.0, 3.0]])
        
        # Keep top 3
        filtered = sampler.apply_top_k(logits, top_k=3)
        
        # Check that non-top-k values are set to very negative
        self.assertTrue(filtered[0, 0] < -1e9)  # 1.0 should be filtered
        self.assertTrue(filtered[0, 2] < -1e9)  # 2.0 should be filtered
        
        # Top values should remain
        self.assertAlmostEqual(float(filtered[0, 1]), 5.0, places=5)
        self.assertAlmostEqual(float(filtered[0, 3]), 4.0, places=5)
        self.assertAlmostEqual(float(filtered[0, 4]), 3.0, places=5)
    
    def test_apply_top_k_zero(self):
        """Test top-K with k=0 (no filtering)."""
        from src.core.sampling import TokenSampler
        
        sampler = TokenSampler()
        logits = jnp.array([[1.0, 2.0, 3.0]])
        
        filtered = sampler.apply_top_k(logits, top_k=0)
        self.assertTrue(jnp.allclose(filtered, logits))
    
    def test_apply_top_p(self):
        """Test nucleus (top-P) filtering."""
        from src.core.sampling import TokenSampler
        
        sampler = TokenSampler()
        # Create logits that produce clear probability distribution
        logits = jnp.array([[4.0, 2.0, 1.0, 0.5, 0.1]])
        
        # Apply top-p filtering
        filtered = sampler.apply_top_p(logits, top_p=0.9)
        
        # Check shape is preserved
        self.assertEqual(filtered.shape, logits.shape)
        
        # The highest probability token should always be kept
        probs_before = jax.nn.softmax(logits)
        max_idx = jnp.argmax(probs_before)
        self.assertFalse(filtered[0, max_idx] < -1e9)
    
    def test_apply_top_p_no_filtering(self):
        """Test top-P with p=1.0 (no filtering)."""
        from src.core.sampling import TokenSampler
        
        sampler = TokenSampler()
        logits = jnp.array([[1.0, 2.0, 3.0]])
        
        filtered = sampler.apply_top_p(logits, top_p=1.0)
        self.assertTrue(jnp.allclose(filtered, logits))
    
    def test_apply_min_p(self):
        """Test min-P filtering."""
        from src.core.sampling import TokenSampler
        
        sampler = TokenSampler()
        logits = jnp.array([[5.0, 2.0, 0.5, 0.1]])
        
        filtered = sampler.apply_min_p(logits, min_p=0.1)
        
        # Shape should be preserved
        self.assertEqual(filtered.shape, logits.shape)
    
    def test_apply_min_p_zero(self):
        """Test min-P with p=0 (no filtering)."""
        from src.core.sampling import TokenSampler
        
        sampler = TokenSampler()
        logits = jnp.array([[1.0, 2.0, 3.0]])
        
        filtered = sampler.apply_min_p(logits, min_p=0.0)
        self.assertTrue(jnp.allclose(filtered, logits))
    
    def test_apply_repetition_penalty(self):
        """Test repetition penalty."""
        from src.core.sampling import TokenSampler
        
        sampler = TokenSampler()
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        generated_tokens = jnp.array([[1, 3]])  # Tokens 1 and 3 were generated
        
        penalized = sampler.apply_repetition_penalty(logits, generated_tokens, penalty=1.5)
        
        # Shape should be preserved
        self.assertEqual(penalized.shape, logits.shape)
    
    def test_apply_repetition_penalty_no_penalty(self):
        """Test repetition penalty with penalty=1.0 (no change)."""
        from src.core.sampling import TokenSampler
        
        sampler = TokenSampler()
        logits = jnp.array([[1.0, 2.0, 3.0]])
        generated_tokens = jnp.array([[1]])
        
        penalized = sampler.apply_repetition_penalty(logits, generated_tokens, penalty=1.0)
        self.assertTrue(jnp.allclose(penalized, logits))


class TestSamplingIntegration(unittest.TestCase):
    """Integration tests for sampling pipeline."""
    
    def test_full_sampling_pipeline(self):
        """Test complete sampling with all strategies."""
        from src.core.sampling import TokenSampler, SamplingConfig
        
        sampler = TokenSampler(vocab_size=100)
        config = SamplingConfig(
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        rng = jax.random.PRNGKey(42)
        logits = jax.random.normal(rng, (2, 100))  # batch_size=2, vocab_size=100
        
        # Apply all transformations
        processed = sampler.apply_temperature(logits, config.temperature)
        processed = sampler.apply_top_k(processed, config.top_k)
        processed = sampler.apply_top_p(processed, config.top_p)
        
        # Convert to probabilities
        probs = jax.nn.softmax(processed, axis=-1)
        
        # Verify probabilities sum to 1
        self.assertTrue(jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5))


class TestSpeculativeDecoding(unittest.TestCase):
    """Test speculative decoding utilities."""
    
    def test_speculative_decoding_exists(self):
        """Test that speculative decoding utilities exist."""
        from src.core.sampling import SpeculativeDecoder
        
        # SpeculativeDecoder exists, test basic instantiation
        def mock_forward(params, tokens):
            return jnp.ones((1, tokens.shape[1], 100)) 
        
        decoder = SpeculativeDecoder(
            target_forward_fn=mock_forward,
            draft_forward_fn=mock_forward,
            num_speculative_tokens=4,
            temperature=1.0
        )
        
        self.assertEqual(decoder.num_speculative_tokens, 4)
        self.assertEqual(decoder.temperature, 1.0)


if __name__ == "__main__":
    unittest.main()
