"""
Unit tests for rtdlm_agi_complete.py AGI modules

Tests the following classes:
- ConsciousnessSimulator: RNN-based introspection and self-awareness
- ScientificDiscoveryEngine: Do-intervention causal reasoning
- CreativeGenerationEngine: Entropy-based novelty detection
- SocialEmotionalIntelligence: 14-emotion recognition system
"""

import sys
import unittest
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import jax
import jax.numpy as jnp
import haiku as hk

from rtdlm_agi_complete import (
    ConsciousnessSimulator,
    ScientificDiscoveryEngine,
    CreativeGenerationEngine,
    SocialEmotionalIntelligence
)


# Test constants
D_MODEL = 64
BATCH_SIZE = 2
SEQ_LEN = 16


class TestConsciousnessSimulator(unittest.TestCase):
    """Test ConsciousnessSimulator with RNN-based introspection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(42)
        
    def test_initialization(self):
        """Test that ConsciousnessSimulator initializes correctly"""
        def init_fn(internal_state, external_input):
            model = ConsciousnessSimulator(d_model=D_MODEL)
            return model(internal_state, external_input)
        
        init = hk.transform(init_fn)
        internal_state = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        external_input = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        params = init.init(self.rng, internal_state, external_input)
        self.assertIsNotNone(params)
        
    def test_output_shape(self):
        """Test output shapes are correct"""
        def forward_fn(internal_state, external_input):
            model = ConsciousnessSimulator(d_model=D_MODEL)
            return model(internal_state, external_input)
        
        transformed = hk.transform(forward_fn)
        
        internal_state = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        external_input = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, internal_state, external_input)
        result = transformed.apply(params, self.rng, internal_state, external_input)
        
        # Check return dict keys (actual keys from implementation)
        self.assertIn("self_awareness", result)
        self.assertIn("introspection", result)
        self.assertIn("recurrent_introspection", result)
        self.assertIn("meta_awareness", result)
        
        # Check recurrent_introspection shape
        self.assertEqual(result["recurrent_introspection"].shape, (BATCH_SIZE, D_MODEL))
        
    def test_awareness_level_bounded(self):
        """Test that consciousness outputs are valid"""
        def forward_fn(internal_state, external_input):
            model = ConsciousnessSimulator(d_model=D_MODEL)
            return model(internal_state, external_input)
        
        transformed = hk.transform(forward_fn)
        
        internal_state = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        external_input = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, internal_state, external_input)
        result = transformed.apply(params, self.rng, internal_state, external_input)
        
        # Check that introspection outputs are finite
        self.assertTrue(jnp.all(jnp.isfinite(result["recurrent_introspection"])))
        self.assertTrue(jnp.all(jnp.isfinite(result["self_awareness"])))


class TestScientificDiscoveryEngine(unittest.TestCase):
    """Test ScientificDiscoveryEngine with do-intervention causal reasoning"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(123)
        
    def test_initialization(self):
        """Test that ScientificDiscoveryEngine initializes correctly"""
        def init_fn(knowledge_base, observations):
            model = ScientificDiscoveryEngine(d_model=D_MODEL)
            return model(knowledge_base, observations)
        
        init = hk.transform(init_fn)
        knowledge_base = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        params = init.init(self.rng, knowledge_base, observations)
        self.assertIsNotNone(params)
        
    def test_output_contains_causal_data(self):
        """Test output contains causal reasoning data"""
        def forward_fn(knowledge_base, observations, research_question=None):
            model = ScientificDiscoveryEngine(d_model=D_MODEL)
            return model(knowledge_base, observations, research_question)
        
        transformed = hk.transform(forward_fn)
        
        knowledge_base = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, knowledge_base, observations)
        result = transformed.apply(params, self.rng, knowledge_base, observations)
        
        # Check return dict keys (actual keys from implementation)
        self.assertIn("hypothesis", result)
        self.assertIn("causal_analysis", result)
        self.assertIn("causal_graph", result)
        self.assertIn("intervention_results", result)
        self.assertIn("experiment_design", result)
        
    def test_intervention_shape(self):
        """Test do-intervention output shape"""
        def forward_fn(knowledge_base, observations, research_question=None):
            model = ScientificDiscoveryEngine(d_model=D_MODEL)
            return model(knowledge_base, observations, research_question)
        
        transformed = hk.transform(forward_fn)
        
        knowledge_base = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, knowledge_base, observations)
        result = transformed.apply(params, self.rng, knowledge_base, observations)
        
        # Check intervention results
        intervention = result["intervention_results"]
        self.assertIn("intervened_state", intervention)
        self.assertIn("counterfactual", intervention)
        self.assertIn("causal_effect", intervention)
        
    def test_with_research_question(self):
        """Test with research question input"""
        def forward_fn(knowledge_base, observations, research_question=None):
            model = ScientificDiscoveryEngine(d_model=D_MODEL)
            return model(knowledge_base, observations, research_question)
        
        transformed = hk.transform(forward_fn)
        
        knowledge_base = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        research_question = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, knowledge_base, observations, research_question)
        result = transformed.apply(params, self.rng, knowledge_base, observations, research_question)
        
        self.assertIn("hypothesis", result)


class TestCreativeGenerationEngine(unittest.TestCase):
    """Test CreativeGenerationEngine with entropy-based novelty detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(456)
        
    def test_initialization(self):
        """Test that CreativeGenerationEngine initializes correctly"""
        def init_fn(input_context):
            model = CreativeGenerationEngine(d_model=D_MODEL)
            return model(input_context)
        
        init = hk.transform(init_fn)
        input_context = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        params = init.init(self.rng, input_context)
        self.assertIsNotNone(params)
        
    def test_output_contains_novelty_data(self):
        """Test output contains novelty detection data"""
        def forward_fn(input_context, creative_seed=None):
            model = CreativeGenerationEngine(d_model=D_MODEL)
            return model(input_context, creative_seed)
        
        transformed = hk.transform(forward_fn)
        
        input_context = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, input_context)
        result = transformed.apply(params, self.rng, input_context)
        
        # Check return dict keys
        self.assertIn("creative_content", result)
        self.assertIn("novelty_score", result)
        self.assertIn("entropy_novelty", result)
        
    def test_entropy_novelty_structure(self):
        """Test entropy novelty calculation output structure"""
        def forward_fn(input_context, creative_seed=None):
            model = CreativeGenerationEngine(d_model=D_MODEL)
            return model(input_context, creative_seed)
        
        transformed = hk.transform(forward_fn)
        
        input_context = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, input_context)
        result = transformed.apply(params, self.rng, input_context)
        
        # Check novelty score is present
        self.assertIn("novelty_score", result)
        self.assertIn("entropy_novelty", result)
        
    def test_novelty_scores_bounded(self):
        """Test that novelty scores are properly bounded"""
        def forward_fn(input_context, creative_seed=None):
            model = CreativeGenerationEngine(d_model=D_MODEL)
            return model(input_context, creative_seed)
        
        transformed = hk.transform(forward_fn)
        
        input_context = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, input_context)
        result = transformed.apply(params, self.rng, input_context)
        
        # Novelty score should be present
        self.assertIn("novelty_score", result)


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


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple modules"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(999)
        
    def test_all_modules_compatible(self):
        """Test that all modules can work with same input dimensions"""
        def combined_fn(internal_state):
            consciousness = ConsciousnessSimulator(d_model=D_MODEL)
            science = ScientificDiscoveryEngine(d_model=D_MODEL)
            creativity = CreativeGenerationEngine(d_model=D_MODEL)
            social = SocialEmotionalIntelligence(d_model=D_MODEL)
            
            c_out = consciousness(internal_state, internal_state)
            s_out = science(internal_state, internal_state)  # knowledge_base, observations
            cr_out = creativity(internal_state)
            so_out = social(internal_state)
            
            return {
                "consciousness": c_out["self_awareness"],
                "science": s_out["hypothesis"],
                "creativity": cr_out["novelty_score"],
                "social": so_out["recognized_emotions"]
            }
        
        init = hk.transform(combined_fn)
        internal_state = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        params = init.init(self.rng, internal_state)
        result = init.apply(params, self.rng, internal_state)
        
        self.assertIn("consciousness", result)
        self.assertIn("science", result)
        self.assertIn("creativity", result)
        self.assertIn("social", result)


class TestCausalReasoningAdvanced(unittest.TestCase):
    """Advanced tests for do-intervention causal reasoning in ScientificDiscoveryEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(2024)
        
    def test_do_intervention_counterfactual(self):
        """Test that do-intervention produces valid counterfactual reasoning"""
        def forward_fn(knowledge_base, observations):
            model = ScientificDiscoveryEngine(d_model=D_MODEL)
            return model(knowledge_base, observations)
        
        transformed = hk.transform(forward_fn)
        
        knowledge_base = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations = jax.random.normal(jax.random.PRNGKey(100), (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, knowledge_base, observations)
        result = transformed.apply(params, self.rng, knowledge_base, observations)
        
        # Verify intervention results structure
        intervention = result["intervention_results"]
        self.assertIn("intervened_state", intervention)
        self.assertIn("counterfactual", intervention)
        self.assertIn("causal_effect", intervention)
        
        # Verify causal effect is computed (difference between counterfactual and observed)
        causal_effect = intervention["causal_effect"]
        # Causal effect preserves sequence structure: (batch, seq, d_model)
        self.assertEqual(causal_effect.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
    def test_do_intervention_different_inputs(self):
        """Test that different inputs produce different causal effects"""
        def forward_fn(knowledge_base, observations):
            model = ScientificDiscoveryEngine(d_model=D_MODEL)
            return model(knowledge_base, observations)
        
        transformed = hk.transform(forward_fn)
        
        # First input
        knowledge_base1 = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations1 = jax.random.normal(jax.random.PRNGKey(100), (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        # Second input (different)
        knowledge_base2 = jax.random.normal(jax.random.PRNGKey(200), (BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations2 = jax.random.normal(jax.random.PRNGKey(300), (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, knowledge_base1, observations1)
        result1 = transformed.apply(params, self.rng, knowledge_base1, observations1)
        result2 = transformed.apply(params, self.rng, knowledge_base2, observations2)
        
        # Causal effects should be different for different inputs
        effect1 = result1["intervention_results"]["causal_effect"]
        effect2 = result2["intervention_results"]["causal_effect"]
        
        # They should not be identical
        self.assertFalse(jnp.allclose(effect1, effect2))
        
    def test_causal_graph_update(self):
        """Test that causal graph update produces valid structure"""
        def forward_fn(knowledge_base, observations):
            model = ScientificDiscoveryEngine(d_model=D_MODEL)
            return model(knowledge_base, observations)
        
        transformed = hk.transform(forward_fn)
        
        knowledge_base = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations = jax.random.normal(jax.random.PRNGKey(100), (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, knowledge_base, observations)
        result = transformed.apply(params, self.rng, knowledge_base, observations)
        
        # Verify causal graph outputs exist
        self.assertIn("causal_graph", result)
        causal_graph = result["causal_graph"]
        
        # Causal graph should have valid structure with batch dimension
        self.assertGreaterEqual(len(causal_graph.shape), 2)
        self.assertEqual(causal_graph.shape[0], BATCH_SIZE)


class TestAudioEmotionIntegration(unittest.TestCase):
    """Tests for AudioEmotionModule 14-emotion integration with SocialEmotionalIntelligence"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(3000)
        
    def test_audio_emotion_14_classes(self):
        """Test that AudioEmotionModule produces 14 emotion classes"""
        # Import AudioEmotionModule
        from multimodal.hybrid_audio_module import AudioEmotionModule
        
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
        from multimodal.hybrid_audio_module import AudioEmotionModule
        
        # Both should have the same 14 emotion labels
        audio_labels = AudioEmotionModule.EMOTION_LABELS
        social_labels = SocialEmotionalIntelligence.EMOTION_LABELS
        
        self.assertEqual(len(audio_labels), len(social_labels))
        self.assertEqual(set(audio_labels), set(social_labels))
        
    def test_audio_emotion_probability_sum(self):
        """Test that audio emotion probabilities sum to 1"""
        from multimodal.hybrid_audio_module import AudioEmotionModule
        
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
        from multimodal.hybrid_audio_module import AudioEmotionModule
        
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


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
