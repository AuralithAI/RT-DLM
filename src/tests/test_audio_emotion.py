"""
Unit tests for AudioEmotionModule integration

Tests 14-emotion audio integration with SocialEmotionalIntelligence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import unittest
import jax
import jax.numpy as jnp
import haiku as hk

from src.rtdlm import SocialEmotionalIntelligence
from src.modules.multimodal.hybrid_audio_module import AudioEmotionModule

# Test constants
D_MODEL = 64
BATCH_SIZE = 2
SEQ_LEN = 16


class TestAudioEmotionIntegration(unittest.TestCase):
    """Tests for AudioEmotionModule 14-emotion integration with SocialEmotionalIntelligence"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(3000)
        
    def test_audio_emotion_14_classes(self):
        """Test that AudioEmotionModule produces 14 emotion classes"""
        def forward_fn(audio_features):
            model = AudioEmotionModule(d_model=D_MODEL)
            return model(audio_features)
        
        transformed = hk.transform(forward_fn)
        
        audio_features = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, audio_features)
        result = transformed.apply(params, self.rng, audio_features)
        
        # Should have 14 emotion probabilities
        self.assertEqual(result["emotion_probabilities"].shape, (BATCH_SIZE, 14))
        self.assertEqual(result["num_emotions"], 14)
        self.assertEqual(len(result["emotion_labels"]), 14)
        
    def test_audio_emotion_labels_match_social(self):
        """Test that AudioEmotionModule labels match SocialEmotionalIntelligence"""
        # Both should have the same 14 emotion labels
        audio_labels = AudioEmotionModule.EMOTION_LABELS
        social_labels = SocialEmotionalIntelligence.EMOTION_LABELS
        
        self.assertEqual(len(audio_labels), len(social_labels))
        self.assertEqual(set(audio_labels), set(social_labels))
        
    def test_audio_emotion_probability_sum(self):
        """Test that audio emotion probabilities sum to 1"""
        def forward_fn(audio_features):
            model = AudioEmotionModule(d_model=D_MODEL)
            return model(audio_features)
        
        transformed = hk.transform(forward_fn)
        
        audio_features = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, audio_features)
        result = transformed.apply(params, self.rng, audio_features)
        
        # Probabilities should sum to 1 (due to softmax)
        probs_sum = jnp.sum(result["emotion_probabilities"], axis=-1)
        self.assertTrue(jnp.allclose(probs_sum, 1.0, atol=1e-5))
        
    def test_audio_emotion_dominant_detection(self):
        """Test that dominant emotion index is correctly computed"""
        def forward_fn(audio_features):
            model = AudioEmotionModule(d_model=D_MODEL)
            return model(audio_features)
        
        transformed = hk.transform(forward_fn)
        
        audio_features = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, audio_features)
        result = transformed.apply(params, self.rng, audio_features)
        
        # Dominant emotion should be valid index
        self.assertEqual(result["dominant_emotion_idx"].shape, (BATCH_SIZE,))
        self.assertTrue(jnp.all(result["dominant_emotion_idx"] >= 0))
        self.assertTrue(jnp.all(result["dominant_emotion_idx"] < 14))
        
    def test_audio_valence_arousal(self):
        """Test that valence and arousal are computed"""
        def forward_fn(audio_features):
            model = AudioEmotionModule(d_model=D_MODEL)
            return model(audio_features)
        
        transformed = hk.transform(forward_fn)
        
        audio_features = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, audio_features)
        result = transformed.apply(params, self.rng, audio_features)
        
        # Check valence and arousal exist
        self.assertIn("valence", result)
        self.assertIn("arousal", result)
        
        # Valence should be in [-1, 1] (tanh output)
        self.assertTrue(jnp.all(result["valence"] >= -1))
        self.assertTrue(jnp.all(result["valence"] <= 1))
        
        # Arousal should be in [0, 1] (sigmoid output)
        self.assertTrue(jnp.all(result["arousal"] >= 0))
        self.assertTrue(jnp.all(result["arousal"] <= 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)

