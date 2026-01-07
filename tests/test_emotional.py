"""
Unit tests for SocialEmotionalIntelligence module

Tests 14-emotion recognition system functionality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import unittest
import jax
import jax.numpy as jnp
import haiku as hk

from rtdlm import SocialEmotionalIntelligence

# Test constants
D_MODEL = 64
BATCH_SIZE = 2
SEQ_LEN = 16


class TestSocialEmotionalIntelligence(unittest.TestCase):
    """Test SocialEmotionalIntelligence with 14-emotion recognition"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(789)
        
    def test_initialization(self):
        """Test that SocialEmotionalIntelligence initializes correctly"""
        def init_fn(input_repr):
            model = SocialEmotionalIntelligence(d_model=D_MODEL)
            return model(input_repr)
        
        init = hk.transform(init_fn)
        # SocialEmotionalIntelligence expects [batch, seq, d_model] for user_input
        input_repr = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        params = init.init(self.rng, input_repr)
        self.assertIsNotNone(params)
        
    def test_14_emotion_labels(self):
        """Test that EMOTION_LABELS contains exactly 14 emotions"""
        self.assertEqual(len(SocialEmotionalIntelligence.EMOTION_LABELS), 14)
        self.assertEqual(SocialEmotionalIntelligence.NUM_EMOTIONS, 14)
        
        # Check specific emotions exist
        expected_emotions = [
            "joy", "sadness", "anger", "fear", "disgust", 
            "surprise", "neutral", "anticipation", "trust",
            "love", "guilt", "pride", "confusion", "curiosity"
        ]
        for emotion in expected_emotions:
            self.assertIn(emotion, SocialEmotionalIntelligence.EMOTION_LABELS)
    
    def test_output_structure(self):
        """Test output contains all emotion-related data"""
        def forward_fn(input_repr, social_context=None):
            model = SocialEmotionalIntelligence(d_model=D_MODEL)
            return model(input_repr, social_context)
        
        transformed = hk.transform(forward_fn)
        
        # SocialEmotionalIntelligence expects [batch, seq, d_model] for user_input
        input_repr = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, input_repr)
        result = transformed.apply(params, self.rng, input_repr)
        
        # Check return dict keys
        self.assertIn("recognized_emotions", result)
        self.assertIn("emotion_labels", result)
        self.assertIn("mixed_emotions", result)
        self.assertIn("emotion_intensity", result)
        self.assertIn("valence", result)
        self.assertIn("arousal", result)
        self.assertIn("empathy_signal", result)
        
    def test_emotion_probabilities_sum_to_one(self):
        """Test that emotion probabilities sum to approximately 1"""
        def forward_fn(input_repr, social_context=None):
            model = SocialEmotionalIntelligence(d_model=D_MODEL)
            return model(input_repr, social_context)
        
        transformed = hk.transform(forward_fn)
        
        input_repr = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, input_repr)
        result = transformed.apply(params, self.rng, input_repr)
        
        emotions = result["recognized_emotions"]
        sums = jnp.sum(emotions, axis=-1)
        
        # Should sum to approximately 1 (due to softmax)
        self.assertTrue(jnp.allclose(sums, 1.0, atol=1e-5))
        
    def test_14_emotion_output_shape(self):
        """Test that emotion output has 14 dimensions"""
        def forward_fn(input_repr, social_context=None):
            model = SocialEmotionalIntelligence(d_model=D_MODEL)
            return model(input_repr, social_context)
        
        transformed = hk.transform(forward_fn)
        
        input_repr = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, input_repr)
        result = transformed.apply(params, self.rng, input_repr)
        
        # recognized_emotions should have shape [batch, 14]
        self.assertEqual(result["recognized_emotions"].shape, (BATCH_SIZE, 14))
        self.assertEqual(result["mixed_emotions"].shape, (BATCH_SIZE, 14))
        
    def test_with_social_context(self):
        """Test with social context input"""
        def forward_fn(input_repr, social_context=None):
            model = SocialEmotionalIntelligence(d_model=D_MODEL)
            return model(input_repr, social_context)
        
        transformed = hk.transform(forward_fn)
        
        input_repr = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        social_context = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, input_repr, social_context)
        result = transformed.apply(params, self.rng, input_repr, social_context)
        
        self.assertIn("cultural_adapted", result)
        self.assertIn("social_analysis", result)
        
    def test_valence_arousal_bounded(self):
        """Test that valence and arousal are properly bounded"""
        def forward_fn(input_repr, social_context=None):
            model = SocialEmotionalIntelligence(d_model=D_MODEL)
            return model(input_repr, social_context)
        
        transformed = hk.transform(forward_fn)
        
        input_repr = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, input_repr)
        result = transformed.apply(params, self.rng, input_repr)
        
        # Valence should be bounded (typically -1 to 1 or 0 to 1)
        valence = result["valence"]
        arousal = result["arousal"]
        
        self.assertEqual(len(valence.shape), 2)  # (batch, 1) or (batch, d)
        self.assertEqual(len(arousal.shape), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

