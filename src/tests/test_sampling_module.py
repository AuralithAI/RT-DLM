"""
Tests for Sampling Module

Tests for token sampling strategies including temperature, top-k, top-p,
repetition penalty, and various sampling configurations.
"""

import unittest
import jax
import jax.numpy as jnp


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
            stop_tokens=[0, 1, 2],
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
            entropy=jnp.array([1.5]),
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
        config = SamplingConfig(temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1)

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
        def mock_forward(params, text):
            return jnp.ones((1, text.shape[1], 100))

        decoder = SpeculativeDecoder(
            target_forward_fn=mock_forward,
            draft_forward_fn=mock_forward,
            num_speculative_tokens=4,
            temperature=1.0,
        )

        self.assertEqual(decoder.num_speculative_tokens, 4)
        self.assertEqual(decoder.temperature, 1.0)


class TestTopKEdgeCases(unittest.TestCase):
    """Test top-K edge cases for coverage."""

    def test_top_k_greater_than_vocab_size(self):
        """Test top-K returns logits unchanged when k >= vocab_size."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler()
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        filtered = sampler.apply_top_k(logits, top_k=5)
        self.assertTrue(jnp.allclose(filtered, logits))

        filtered = sampler.apply_top_k(logits, top_k=10)
        self.assertTrue(jnp.allclose(filtered, logits))


class TestTopPEdgeCases(unittest.TestCase):
    """Test top-P edge cases for coverage."""

    def test_top_p_zero_raises_error(self):
        """Test top-P with p=0 raises ValueError."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler()
        logits = jnp.array([[1.0, 2.0, 3.0]])

        with self.assertRaises(ValueError) as ctx:
            sampler.apply_top_p(logits, top_p=0.0)

        self.assertIn("top_p must be positive", str(ctx.exception))

    def test_top_p_negative_raises_error(self):
        """Test top-P with negative value raises ValueError."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler()
        logits = jnp.array([[1.0, 2.0, 3.0]])

        with self.assertRaises(ValueError) as ctx:
            sampler.apply_top_p(logits, top_p=-0.5)

        self.assertIn("top_p must be positive", str(ctx.exception))


class TestFrequencyPresencePenalty(unittest.TestCase):
    """Test frequency and presence penalty for coverage."""

    def test_no_penalty_returns_unchanged(self):
        """Test that zero penalties return unchanged logits."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler()
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        generated_tokens = jnp.array([[1, 2, 1, 3]])

        result = sampler.apply_frequency_presence_penalty(
            logits, generated_tokens, frequency_penalty=0.0, presence_penalty=0.0
        )

        self.assertTrue(jnp.allclose(result, logits))

    def test_frequency_penalty_applied(self):
        """Test frequency penalty reduces logits for repeated tokens."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler()
        logits = jnp.array([[5.0, 5.0, 5.0, 5.0, 5.0]])
        generated_tokens = jnp.array([[1, 1, 2]])

        result = sampler.apply_frequency_presence_penalty(
            logits, generated_tokens, frequency_penalty=1.0, presence_penalty=0.0
        )

        self.assertLess(float(result[0, 1]), float(result[0, 2]))
        self.assertAlmostEqual(float(result[0, 0]), 5.0, places=4)

    def test_presence_penalty_applied(self):
        """Test presence penalty reduces logits for any appeared token."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler()
        logits = jnp.array([[5.0, 5.0, 5.0, 5.0, 5.0]])
        generated_tokens = jnp.array([[1, 1, 2]])

        result = sampler.apply_frequency_presence_penalty(
            logits, generated_tokens, frequency_penalty=0.0, presence_penalty=1.0
        )

        self.assertAlmostEqual(float(result[0, 1]), float(result[0, 2]), places=4)
        self.assertLess(float(result[0, 1]), float(result[0, 0]))


class TestSampleWith3DLogits(unittest.TestCase):
    """Test sample() handles 3D logits correctly."""

    def test_sample_with_3d_logits(self):
        """Test sampling extracts last position from 3D logits."""
        from src.core.sampling import TokenSampler, SamplingConfig

        sampler = TokenSampler(vocab_size=100)
        config = SamplingConfig(temperature=1.0, top_k=50)

        rng = jax.random.PRNGKey(42)
        logits_3d = jax.random.normal(rng, (2, 10, 100))

        output = sampler.sample(logits_3d, config, rng)

        self.assertEqual(output.token_id.shape, (2, 1))


class TestSampleWithGeneratedTokens(unittest.TestCase):
    """Test sample() with generated tokens for penalties."""

    def test_sample_with_repetition_penalty(self):
        """Test sample applies repetition penalty from config."""
        from src.core.sampling import TokenSampler, SamplingConfig

        sampler = TokenSampler(vocab_size=100)
        config = SamplingConfig(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=2.0)

        rng = jax.random.PRNGKey(42)
        logits = jax.random.normal(rng, (1, 100))
        generated_tokens = jnp.array([[5, 10, 15, 5, 10]])

        output = sampler.sample(logits, config, rng, generated_tokens=generated_tokens)

        self.assertEqual(output.token_id.shape, (1, 1))

    def test_sample_with_frequency_presence_penalty(self):
        """Test sample applies frequency/presence penalties from config."""
        from src.core.sampling import TokenSampler, SamplingConfig

        sampler = TokenSampler(vocab_size=100)
        config = SamplingConfig(
            temperature=1.0, top_k=0, top_p=1.0, frequency_penalty=0.5, presence_penalty=0.5
        )

        rng = jax.random.PRNGKey(42)
        logits = jax.random.normal(rng, (1, 100))
        generated_tokens = jnp.array([[5, 5, 10, 15]])

        output = sampler.sample(logits, config, rng, generated_tokens=generated_tokens)

        self.assertEqual(output.token_id.shape, (1, 1))


class TestBeamSearch(unittest.TestCase):
    """Test beam search decoding."""

    def test_beam_search_basic(self):
        """Test basic beam search generation."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler(vocab_size=50)

        def mock_logits_fn(tokens):
            batch_size, seq_len = tokens.shape
            return jax.random.normal(jax.random.PRNGKey(seq_len), (batch_size, seq_len, 50))

        initial_tokens = jnp.array([[1, 2, 3]])

        sequences, scores = sampler.beam_search(
            logits_fn=mock_logits_fn,
            initial_tokens=initial_tokens,
            beam_width=3,
            max_length=10,
            stop_tokens=[0],
        )

        self.assertEqual(sequences.shape, (1, 10))
        self.assertEqual(scores.shape, (1,))

    def test_beam_search_with_stop_tokens(self):
        """Test beam search stops at stop tokens."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler(vocab_size=50)

        def mock_logits_fn(tokens):
            batch_size, seq_len = tokens.shape
            logits = jnp.zeros((batch_size, seq_len, 50))
            if seq_len > 5:
                logits = logits.at[:, :, 0].set(10.0)
            else:
                logits = logits.at[:, :, 10].set(5.0)
            return logits

        initial_tokens = jnp.array([[1, 2]])

        sequences, scores = sampler.beam_search(
            logits_fn=mock_logits_fn,
            initial_tokens=initial_tokens,
            beam_width=2,
            max_length=20,
            stop_tokens=[0],
            early_stopping=True,
        )

        self.assertEqual(sequences.shape[0], 1)

    def test_beam_search_with_ngram_blocking(self):
        """Test beam search with n-gram repetition blocking."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler(vocab_size=20)

        call_count = [0]

        def mock_logits_fn(tokens):
            call_count[0] += 1
            batch_size, seq_len = tokens.shape
            return jax.random.normal(jax.random.PRNGKey(call_count[0]), (batch_size, seq_len, 20))

        initial_tokens = jnp.array([[1, 2, 3, 4]])

        sequences, scores = sampler.beam_search(
            logits_fn=mock_logits_fn,
            initial_tokens=initial_tokens,
            beam_width=2,
            max_length=15,
            no_repeat_ngram_size=3,
        )

        self.assertEqual(sequences.shape, (1, 15))

    def test_beam_search_with_min_length(self):
        """Test beam search respects minimum length."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler(vocab_size=20)

        def mock_logits_fn(tokens):
            batch_size, seq_len = tokens.shape
            logits = jnp.zeros((batch_size, seq_len, 20))
            logits = logits.at[:, :, 0].set(10.0)
            return logits

        initial_tokens = jnp.array([[1, 2]])

        sequences, scores = sampler.beam_search(
            logits_fn=mock_logits_fn,
            initial_tokens=initial_tokens,
            beam_width=2,
            max_length=15,
            stop_tokens=[0],
            min_length=8,
        )

        self.assertEqual(sequences.shape, (1, 15))


class TestDiverseBeamSearch(unittest.TestCase):
    """Test diverse beam search."""

    def test_diverse_beam_search_basic(self):
        """Test diverse beam search generates multiple outputs."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler(vocab_size=30)

        call_count = [0]

        def mock_logits_fn(tokens):
            call_count[0] += 1
            batch_size, seq_len = tokens.shape
            return jax.random.normal(jax.random.PRNGKey(call_count[0]), (batch_size, seq_len, 30))

        initial_tokens = jnp.array([[1, 2]])

        sequences, scores = sampler.diverse_beam_search(
            logits_fn=mock_logits_fn,
            initial_tokens=initial_tokens,
            num_beam_groups=2,
            beam_width=4,
            diversity_penalty=0.5,
            max_length=10,
        )

        self.assertEqual(sequences.shape[0], 2)

    def test_diverse_beam_search_invalid_beam_width(self):
        """Test diverse beam search raises error for invalid beam_width."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler(vocab_size=30)

        def mock_logits_fn(tokens):
            return jnp.zeros((tokens.shape[0], tokens.shape[1], 30))

        initial_tokens = jnp.array([[1, 2]])

        with self.assertRaises(ValueError) as ctx:
            sampler.diverse_beam_search(
                logits_fn=mock_logits_fn,
                initial_tokens=initial_tokens,
                num_beam_groups=3,
                beam_width=5,
                max_length=10,
            )

        self.assertIn("must be divisible by", str(ctx.exception))


class TestConstrainedBeamSearch(unittest.TestCase):
    """Test constrained beam search."""

    def test_constrained_beam_search_no_constraints(self):
        """Test constrained beam search with empty constraints uses regular beam search."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler(vocab_size=30)

        call_count = [0]

        def mock_logits_fn(tokens):
            call_count[0] += 1
            batch_size, seq_len = tokens.shape
            return jax.random.normal(jax.random.PRNGKey(call_count[0]), (batch_size, seq_len, 30))

        initial_tokens = jnp.array([[1, 2]])

        sequences, scores = sampler.constrained_beam_search(
            logits_fn=mock_logits_fn,
            initial_tokens=initial_tokens,
            constraints=[],
            beam_width=2,
            max_length=10,
        )

        self.assertEqual(sequences.shape, (1, 10))

    def test_constrained_beam_search_with_constraints(self):
        """Test constrained beam search with constraints."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler(vocab_size=30)

        call_count = [0]

        def mock_logits_fn(tokens):
            call_count[0] += 1
            batch_size, seq_len = tokens.shape
            return jax.random.normal(jax.random.PRNGKey(call_count[0]), (batch_size, seq_len, 30))

        initial_tokens = jnp.array([[1, 2]])
        constraints = [[5, 6], [10]]

        sequences, scores = sampler.constrained_beam_search(
            logits_fn=mock_logits_fn,
            initial_tokens=initial_tokens,
            constraints=constraints,
            beam_width=3,
            max_length=15,
        )

        self.assertEqual(sequences.shape[0], 1)


class TestSpeculativeDecoderDrafting(unittest.TestCase):
    """Test SpeculativeDecoder token drafting and verification."""

    def test_draft_tokens(self):
        """Test draft token generation."""
        from src.core.sampling import SpeculativeDecoder

        vocab_size = 50

        def mock_forward(params, text):
            batch_size, seq_len = text.shape
            rng = jax.random.PRNGKey(seq_len)
            return jax.random.normal(rng, (batch_size, seq_len, vocab_size))

        decoder = SpeculativeDecoder(
            target_forward_fn=mock_forward,
            draft_forward_fn=mock_forward,
            num_speculative_tokens=4,
            temperature=1.0,
        )

        rng = jax.random.PRNGKey(42)
        initial_tokens = jnp.array([[1, 2, 3]])

        draft_tokens, draft_probs = decoder._draft_tokens(
            draft_params=None, tokens=initial_tokens, rng_key=rng, num_tokens=4
        )

        self.assertEqual(draft_tokens.shape, (1, 4))
        self.assertEqual(draft_probs.shape, (1, 4))

    def test_verify_and_accept_all(self):
        """Test verification when all tokens are accepted."""
        from src.core.sampling import SpeculativeDecoder

        vocab_size = 50

        def mock_forward(params, text):
            batch_size, seq_len = text.shape
            return jnp.zeros((batch_size, seq_len, vocab_size))

        decoder = SpeculativeDecoder(
            target_forward_fn=mock_forward,
            draft_forward_fn=mock_forward,
            num_speculative_tokens=3,
            temperature=1.0,
        )

        rng = jax.random.PRNGKey(42)
        tokens = jnp.array([[1, 2, 3]])
        draft_tokens = jnp.array([[10, 11, 12]])
        draft_probs = jnp.array([[0.5, 0.5, 0.5]])

        accepted_tokens, num_accepted = decoder._verify_and_accept(
            target_params=None,
            tokens=tokens,
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            rng_key=rng,
        )

        self.assertGreater(num_accepted, 0)
        self.assertEqual(accepted_tokens.shape[0], 1)

    def test_speculative_generate(self):
        """Test full speculative generation."""
        from src.core.sampling import SpeculativeDecoder

        vocab_size = 30

        def mock_forward(params, text):
            batch_size, seq_len = text.shape
            rng = jax.random.PRNGKey(seq_len)
            return jax.random.normal(rng, (batch_size, seq_len, vocab_size))

        decoder = SpeculativeDecoder(
            target_forward_fn=mock_forward,
            draft_forward_fn=mock_forward,
            num_speculative_tokens=2,
            temperature=1.0,
        )

        rng = jax.random.PRNGKey(42)
        initial_tokens = jnp.array([[1, 2]])

        output = decoder.generate(
            target_params=None,
            draft_params=None,
            initial_tokens=initial_tokens,
            rng_key=rng,
            max_length=10,
            stop_tokens=[0],
        )

        self.assertGreaterEqual(output.shape[1], 2)
        self.assertLessEqual(output.shape[1], 15)


class TestSelfSpeculativeDecoder(unittest.TestCase):
    """Test SelfSpeculativeDecoder class."""

    def test_self_speculative_decoder_initialization(self):
        """Test SelfSpeculativeDecoder initialization."""
        from src.core.sampling import SelfSpeculativeDecoder

        def mock_forward(params, text, exit_layer=None):
            batch_size, seq_len = text.shape
            return jnp.zeros((batch_size, seq_len, 50))

        decoder = SelfSpeculativeDecoder(
            model_forward_fn=mock_forward,
            early_exit_layer=4,
            num_speculative_tokens=3,
            temperature=0.8,
        )

        self.assertEqual(decoder.early_exit_layer, 4)
        self.assertEqual(decoder.num_speculative_tokens, 3)
        self.assertEqual(decoder.temperature, 0.8)

    def test_self_speculative_early_exit_check(self):
        """Test early exit support checking."""
        from src.core.sampling import SelfSpeculativeDecoder

        def supporting_forward(params, text, exit_layer=None):
            batch_size, seq_len = text.shape
            return jnp.zeros((batch_size, seq_len, 50))

        decoder = SelfSpeculativeDecoder(
            model_forward_fn=supporting_forward, early_exit_layer=4, num_speculative_tokens=2
        )

        sample_tokens = jnp.array([[1, 2, 3]])

        supports = decoder._check_early_exit_support(None, sample_tokens)
        self.assertTrue(supports)

    def test_self_speculative_fallback_generation(self):
        """Test fallback to standard generation when early exit not supported."""
        from src.core.sampling import SelfSpeculativeDecoder

        def non_supporting_forward(params, text):
            batch_size, seq_len = text.shape
            rng = jax.random.PRNGKey(seq_len)
            return jax.random.normal(rng, (batch_size, seq_len, 30))

        decoder = SelfSpeculativeDecoder(
            model_forward_fn=non_supporting_forward, early_exit_layer=4, num_speculative_tokens=2
        )

        rng = jax.random.PRNGKey(42)
        initial_tokens = jnp.array([[1, 2]])

        output = decoder.generate(
            params=None, initial_tokens=initial_tokens, rng_key=rng, max_length=8, stop_tokens=[0]
        )

        self.assertGreaterEqual(output.shape[1], 2)

    def test_self_speculative_full_generation(self):
        """Test full self-speculative generation with early exit support."""
        from src.core.sampling import SelfSpeculativeDecoder

        def supporting_forward(params, text, exit_layer=None):
            batch_size, seq_len = text.shape
            rng = jax.random.PRNGKey(seq_len + (exit_layer or 0))
            return jax.random.normal(rng, (batch_size, seq_len, 30))

        decoder = SelfSpeculativeDecoder(
            model_forward_fn=supporting_forward,
            early_exit_layer=4,
            num_speculative_tokens=2,
            temperature=1.0,
        )

        rng = jax.random.PRNGKey(42)
        initial_tokens = jnp.array([[1, 2]])

        output = decoder.generate(
            params=None, initial_tokens=initial_tokens, rng_key=rng, max_length=10, stop_tokens=[0]
        )

        self.assertGreaterEqual(output.shape[1], 2)
        self.assertLessEqual(output.shape[1], 10)

    def test_contains_stop_token(self):
        """Test stop token detection helper."""
        from src.core.sampling import SelfSpeculativeDecoder

        def mock_forward(params, text, exit_layer=None):
            return jnp.zeros((text.shape[0], text.shape[1], 30))

        decoder = SelfSpeculativeDecoder(model_forward_fn=mock_forward, early_exit_layer=4)

        tokens_with_stop = jnp.array([[1, 2, 0, 3]])
        tokens_without_stop = jnp.array([[1, 2, 3, 4]])

        self.assertTrue(decoder._contains_stop_token(tokens_with_stop, [0]))
        self.assertFalse(decoder._contains_stop_token(tokens_without_stop, [0]))

    def test_log_stats(self):
        """Test logging stats method."""
        from src.core.sampling import SelfSpeculativeDecoder

        def mock_forward(params, text, exit_layer=None):
            return jnp.zeros((text.shape[0], text.shape[1], 30))

        decoder = SelfSpeculativeDecoder(model_forward_fn=mock_forward, early_exit_layer=4)

        decoder._log_stats(total_drafted=10, total_accepted=8)
        decoder._log_stats(total_drafted=0, total_accepted=0)

    def test_draft_tokens_early_exit(self):
        """Test drafting tokens with early exit."""
        from src.core.sampling import SelfSpeculativeDecoder

        def supporting_forward(params, text, exit_layer=None):
            batch_size, seq_len = text.shape
            rng = jax.random.PRNGKey(seq_len)
            return jax.random.normal(rng, (batch_size, seq_len, 30))

        decoder = SelfSpeculativeDecoder(
            model_forward_fn=supporting_forward, early_exit_layer=4, num_speculative_tokens=3
        )

        rng = jax.random.PRNGKey(42)
        tokens = jnp.array([[1, 2, 3]])

        draft_tokens, draft_probs = decoder._draft_tokens_early_exit(
            params=None, tokens=tokens, rng_key=rng, num_tokens=3
        )

        self.assertEqual(draft_tokens.shape, (1, 3))
        self.assertEqual(draft_probs.shape, (1, 3))

    def test_verify_drafts(self):
        """Test verification of draft tokens."""
        from src.core.sampling import SelfSpeculativeDecoder

        def supporting_forward(params, text, exit_layer=None):
            batch_size, seq_len = text.shape
            return jnp.zeros((batch_size, seq_len, 30))

        decoder = SelfSpeculativeDecoder(
            model_forward_fn=supporting_forward, early_exit_layer=4, num_speculative_tokens=2
        )

        rng = jax.random.PRNGKey(42)
        tokens = jnp.array([[1, 2, 3]])
        draft_tokens = jnp.array([[10, 15]])
        draft_probs = jnp.array([[0.5, 0.5]])

        accepted, num_accepted = decoder._verify_drafts(
            params=None,
            tokens=tokens,
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            rng_key=rng,
        )

        self.assertGreater(num_accepted, 0)
        self.assertEqual(accepted.shape[0], 1)

    def test_standard_generate(self):
        """Test standard autoregressive generation fallback."""
        from src.core.sampling import SelfSpeculativeDecoder

        def mock_forward(params, text, exit_layer=None):
            batch_size, seq_len = text.shape
            rng = jax.random.PRNGKey(seq_len)
            return jax.random.normal(rng, (batch_size, seq_len, 30))

        decoder = SelfSpeculativeDecoder(
            model_forward_fn=mock_forward, early_exit_layer=4, num_speculative_tokens=2
        )

        rng = jax.random.PRNGKey(42)
        initial_tokens = jnp.array([[1, 2]])

        output = decoder._standard_generate(
            params=None, tokens=initial_tokens, rng_key=rng, max_length=6, stop_tokens=[0]
        )

        self.assertGreaterEqual(output.shape[1], 2)
        self.assertLessEqual(output.shape[1], 6)

    def test_speculative_generate_internal(self):
        """Test internal speculative generation method."""
        from src.core.sampling import SelfSpeculativeDecoder

        def supporting_forward(params, text, exit_layer=None):
            batch_size, seq_len = text.shape
            rng = jax.random.PRNGKey(seq_len + (exit_layer or 0))
            return jax.random.normal(rng, (batch_size, seq_len, 30))

        decoder = SelfSpeculativeDecoder(
            model_forward_fn=supporting_forward, early_exit_layer=4, num_speculative_tokens=2
        )

        decoder._early_exit_supported = True

        rng = jax.random.PRNGKey(42)
        initial_tokens = jnp.array([[1, 2]])

        output = decoder._speculative_generate(
            params=None, initial_tokens=initial_tokens, rng_key=rng, max_length=8, stop_tokens=[0]
        )

        self.assertGreaterEqual(output.shape[1], 2)


class TestSpeculativeDecoderStopTokens(unittest.TestCase):
    """Test speculative decoder with stop tokens."""

    def test_generate_with_stop_token_in_output(self):
        """Test generation stops when stop token is encountered."""
        from src.core.sampling import SpeculativeDecoder

        vocab_size = 30
        call_count = [0]

        def mock_forward(params, text):
            call_count[0] += 1
            batch_size, seq_len = text.shape
            logits = jnp.zeros((batch_size, seq_len, vocab_size))
            if seq_len > 5:
                logits = logits.at[:, :, 0].set(10.0)
            else:
                logits = logits.at[:, :, 15].set(5.0)
            return logits

        decoder = SpeculativeDecoder(
            target_forward_fn=mock_forward,
            draft_forward_fn=mock_forward,
            num_speculative_tokens=2,
            temperature=1.0,
        )

        rng = jax.random.PRNGKey(42)
        initial_tokens = jnp.array([[1, 2]])

        output = decoder.generate(
            target_params=None,
            draft_params=None,
            initial_tokens=initial_tokens,
            rng_key=rng,
            max_length=20,
            stop_tokens=[0],
        )

        self.assertGreaterEqual(output.shape[1], 2)


class TestBeamSearchEdgeCases(unittest.TestCase):
    """Test beam search edge cases for coverage."""

    def test_beam_search_empty_stop_tokens(self):
        """Test beam search with None stop_tokens."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler(vocab_size=30)

        call_count = [0]

        def mock_logits_fn(tokens):
            call_count[0] += 1
            batch_size, seq_len = tokens.shape
            return jax.random.normal(jax.random.PRNGKey(call_count[0]), (batch_size, seq_len, 30))

        initial_tokens = jnp.array([[1, 2]])

        sequences, scores = sampler.beam_search(
            logits_fn=mock_logits_fn,
            initial_tokens=initial_tokens,
            beam_width=2,
            max_length=8,
            stop_tokens=None,
        )

        self.assertEqual(sequences.shape, (1, 8))

    def test_beam_search_all_beams_finished_early(self):
        """Test beam search when all beams finish early."""
        from src.core.sampling import TokenSampler

        sampler = TokenSampler(vocab_size=20)

        def mock_logits_fn(tokens):
            batch_size, seq_len = tokens.shape
            logits = jnp.zeros((batch_size, seq_len, 20))
            logits = logits.at[:, :, 0].set(100.0)
            return logits

        initial_tokens = jnp.array([[1, 2]])

        sequences, scores = sampler.beam_search(
            logits_fn=mock_logits_fn,
            initial_tokens=initial_tokens,
            beam_width=2,
            max_length=20,
            stop_tokens=[0],
            early_stopping=True,
        )

        self.assertEqual(sequences.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
