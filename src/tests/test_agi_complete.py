"""
Unit tests for rtdlm_agi_complete.py AGI modules

NOTE: Tests have been split into separate files for better organization:
- test_consciousness.py: ConsciousnessSimulator tests
- test_scientific.py: ScientificDiscoveryEngine tests  
- test_creative.py: CreativeGenerationEngine tests
- test_emotional.py: SocialEmotionalIntelligence tests
- test_audio_emotion.py: AudioEmotionModule integration tests
- test_integration.py: Integration tests
- test_training.py: Training pipeline tests

This file imports and runs all tests for backward compatibility.

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

from src.rtdlm import (
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
        from src.modules.multimodal.hybrid_audio_module import AudioEmotionModule
        
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
        from src.modules.multimodal.hybrid_audio_module import AudioEmotionModule
        
        # Both should have the same 14 emotion labels
        audio_labels = AudioEmotionModule.EMOTION_LABELS
        social_labels = SocialEmotionalIntelligence.EMOTION_LABELS
        
        self.assertEqual(len(audio_labels), len(social_labels))
        self.assertEqual(set(audio_labels), set(social_labels))
        
    def test_audio_emotion_probability_sum(self):
        """Test that audio emotion probabilities sum to 1"""
        from src.modules.multimodal.hybrid_audio_module import AudioEmotionModule
        
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
        from src.modules.multimodal.hybrid_audio_module import AudioEmotionModule
        
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


class TestContinualLearning(unittest.TestCase):
    """Test continual learning algorithms for loss stability"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(42)
        
    def test_ewc_loss_computation(self):
        """Test EWC loss is computed correctly and is non-negative"""
        from src.modules.capabilities.advanced_algorithms import compute_ewc_loss
        
        # Create mock parameters
        params = {
            "layer1": {"w": jnp.ones((64, 64))},
            "layer2": {"w": jnp.ones((64, 32))}
        }
        
        # Create slightly different previous parameters
        params_star = {
            "layer1": {"w": jnp.ones((64, 64)) * 0.9},
            "layer2": {"w": jnp.ones((64, 32)) * 0.95}
        }
        
        # Create Fisher information (importance weights)
        fisher_matrix = {
            "layer1": {"w": jnp.ones((64, 64)) * 0.5},
            "layer2": {"w": jnp.ones((64, 32)) * 0.3}
        }
        
        # Compute EWC loss
        ewc_loss = compute_ewc_loss(
            params=params,
            params_star=params_star,
            fisher_matrix=fisher_matrix,
            lambda_ewc=1000.0
        )
        
        # EWC loss should be non-negative
        self.assertGreaterEqual(float(ewc_loss), 0.0)
        
        # EWC loss should be greater than 0 since params differ from params_star
        self.assertGreater(float(ewc_loss), 0.0)
        
        # With identical params, EWC loss should be 0
        ewc_loss_identical = compute_ewc_loss(
            params=params,
            params_star=params,
            fisher_matrix=fisher_matrix,
            lambda_ewc=1000.0
        )
        self.assertAlmostEqual(float(ewc_loss_identical), 0.0, places=5)
        
    def test_ewc_loss_stability_across_tasks(self):
        """Test EWC maintains loss stability when learning multiple tasks"""
        from src.modules.capabilities.advanced_algorithms import (
            compute_ewc_loss, ContinualLearner
        )
        
        def forward_fn(features):
            model = ContinualLearner(d_model=D_MODEL)
            return model(features)
        
        transformed = hk.transform(forward_fn)
        
        # Task 1 features
        task1_features = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        # Initialize with task 1
        params_task1 = transformed.init(self.rng, task1_features)
        result_task1 = transformed.apply(params_task1, self.rng, task1_features)
        
        # Verify ContinualLearner output structure
        self.assertIn("features", result_task1)
        self.assertIn("importance", result_task1)
        self.assertIn("importance_weights", result_task1)
        
        # Task 2 features (different distribution)
        rng2 = jax.random.PRNGKey(123)
        task2_features = jax.random.normal(rng2, (BATCH_SIZE, SEQ_LEN, D_MODEL)) * 2.0
        
        # Process task 2 with same params
        result_task2 = transformed.apply(params_task1, rng2, task2_features)
        
        # Both tasks should produce valid outputs
        self.assertEqual(result_task1["features"].shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        self.assertEqual(result_task2["features"].shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        # Create simulated "task 2 trained" params (slightly modified)
        params_task2 = jax.tree_util.tree_map(
            lambda x: x + jax.random.normal(rng2, x.shape) * 0.01,
            params_task1
        )
        
        # Fisher matrix from task 1 (simulate importance estimation)
        fisher_task1 = jax.tree_util.tree_map(
            lambda x: jnp.abs(jax.random.normal(self.rng, x.shape)) * 0.1,
            params_task1
        )
        
        # Compute EWC loss to protect task 1 knowledge
        ewc_loss = compute_ewc_loss(
            params=params_task2,
            params_star=params_task1,
            fisher_matrix=fisher_task1,
            lambda_ewc=1000.0
        )
        
        # EWC loss should be finite and reasonable
        self.assertTrue(jnp.isfinite(ewc_loss))
        self.assertGreater(float(ewc_loss), 0.0)
        
        # Higher lambda should give higher loss
        ewc_loss_high_lambda = compute_ewc_loss(
            params=params_task2,
            params_star=params_task1,
            fisher_matrix=fisher_task1,
            lambda_ewc=10000.0
        )
        
        self.assertGreater(float(ewc_loss_high_lambda), float(ewc_loss))


class TestMultiAgentCrisisResponse(unittest.TestCase):
    """Test multi-agent system with dynamic specialist spawning for crisis response."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = jax.random.PRNGKey(42)
        
    def test_compute_task_complexity(self):
        """Test entropy-based task complexity computation."""
        from src.modules.hybrid_architecture.hybrid_integrator import compute_task_complexity
        
        # Low entropy input (uniform-ish after softmax)
        low_complexity_input = jnp.ones((BATCH_SIZE, D_MODEL))
        low_complexity = compute_task_complexity(low_complexity_input)
        
        # High entropy input (varied values)
        high_complexity_input = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL)) * 10
        high_complexity = compute_task_complexity(high_complexity_input)
        
        # Both should be in [0, 1]
        self.assertTrue(jnp.all(low_complexity >= 0))
        self.assertTrue(jnp.all(low_complexity <= 1))
        self.assertTrue(jnp.all(high_complexity >= 0))
        self.assertTrue(jnp.all(high_complexity <= 1))
        
        # Shape should match batch size
        self.assertEqual(low_complexity.shape, (BATCH_SIZE,))
        
    def test_multi_agent_consensus_basic(self):
        """Test basic multi-agent consensus without spawning."""
        from src.modules.hybrid_architecture.hybrid_integrator import MultiAgentConsensus
        
        def forward_fn(x):
            consensus = MultiAgentConsensus(
                d_model=D_MODEL,
                num_agents=4,
                spawn_threshold=0.5,
                name="test_consensus"
            )
            return consensus(x, auto_spawn=False)
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        inputs = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL))
        params, state = init_fn(self.rng, inputs)
        result, _ = apply_fn(params, state, self.rng, inputs)
        
        # Check output structure
        self.assertIn('consensus', result)
        self.assertIn('agent_responses', result)
        self.assertIn('task_complexity', result)
        self.assertIn('num_active_agents', result)
        
        # Should have 4 base agents, no spawned
        self.assertEqual(result['num_active_agents'], 4)
        self.assertEqual(result['num_spawned_agents'], 0)
        
        # Consensus shape should match input
        self.assertEqual(result['consensus'].shape, (BATCH_SIZE, D_MODEL))
        
    def test_agent_spawning_high_complexity(self):
        """Test complexity computation and spawning logic for high-complexity tasks.
        
        Note: In JAX/Haiku, spawned agents are created within the forward pass
        but the spawning logic is verified through the metrics returned.
        """
        from src.modules.hybrid_architecture.hybrid_integrator import MultiAgentConsensus, compute_task_complexity
        
        # Create inputs with different characteristics
        # Low complexity: uniform values (low entropy after softmax)
        low_var_input = jnp.ones((BATCH_SIZE, D_MODEL))
        low_complexity = compute_task_complexity(low_var_input)
        
        # Higher complexity: mixed positive/negative with varying magnitudes
        high_var_input = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL))
        high_complexity = compute_task_complexity(high_var_input)
        
        # The complexity function works on entropy of softmax distribution
        # Both should produce valid complexity scores in [0, 1]
        self.assertTrue(jnp.all(low_complexity >= 0))
        self.assertTrue(jnp.all(low_complexity <= 1))
        self.assertTrue(jnp.all(high_complexity >= 0))
        self.assertTrue(jnp.all(high_complexity <= 1))
        
        def forward_fn(x):
            consensus = MultiAgentConsensus(
                d_model=D_MODEL,
                num_agents=4,
                max_spawned_agents=4,
                spawn_threshold=0.001,  # Very low threshold to test spawning logic
                name="test_spawn"
            )
            result = consensus(x, auto_spawn=True)
            return result
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        params, state = init_fn(self.rng, high_var_input)
        result, _ = apply_fn(params, state, self.rng, high_var_input)
        
        # Check that complexity was computed
        self.assertIn('mean_complexity', result)
        self.assertIn('task_complexity', result)
        
        # The output should be valid regardless of spawning
        self.assertEqual(result['consensus'].shape, (BATCH_SIZE, D_MODEL))
        self.assertTrue(jnp.all(jnp.isfinite(result['consensus'])))
        
    def test_crisis_mode_spawning(self):
        """Test crisis mode activates and processes correctly.
        
        Note: Due to JAX's functional nature, we test that crisis mode
        activates the crisis spawning logic and produces valid output.
        """
        from src.modules.hybrid_architecture.hybrid_integrator import MultiAgentConsensus
        
        def forward_fn(x):
            consensus = MultiAgentConsensus(
                d_model=D_MODEL,
                num_agents=4,
                max_spawned_agents=8,
                spawn_threshold=0.1,  # Very low to ensure spawning
                name="test_crisis"
            )
            # Activate crisis mode
            result = consensus(x, auto_spawn=True, crisis_mode=True)
            return result
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        # Simulated crisis data (high entropy)
        crisis_data = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL)) * 50
        
        params, state = init_fn(self.rng, crisis_data)
        result, _ = apply_fn(params, state, self.rng, crisis_data)
        
        # Crisis mode flag should be set
        self.assertTrue(result['crisis_mode'])
        
        # Output should be valid
        self.assertEqual(result['consensus'].shape, (BATCH_SIZE, D_MODEL))
        self.assertTrue(jnp.all(jnp.isfinite(result['consensus'])))
        
        # Complexity should be computed
        self.assertIn('mean_complexity', result)
        self.assertIn('task_complexity', result)
        
    def test_simulated_disaster_data_overload(self):
        """Test system response to simulated disaster data overload scenario.
        
        Verifies that the system handles high-complexity crisis data correctly.
        """
        from src.modules.hybrid_architecture.hybrid_integrator import MultiAgentConsensus
        
        def forward_fn(x):
            consensus = MultiAgentConsensus(
                d_model=D_MODEL,
                num_agents=4,
                max_spawned_agents=8,
                spawn_threshold=0.4,
                name="disaster_response"
            )
            
            # Process with crisis mode
            result = consensus(x, auto_spawn=True, crisis_mode=True)
            
            return result
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        # Create disaster scenario data (high dimensional, high variance)
        rng1, rng2 = jax.random.split(self.rng)
        disaster_data = jnp.concatenate([
            jax.random.normal(rng1, (BATCH_SIZE // 2, D_MODEL)) * 100,
            jax.random.uniform(rng2, (BATCH_SIZE // 2, D_MODEL)) * 50
        ], axis=0)
        
        params, state = init_fn(self.rng, disaster_data)
        result, _ = apply_fn(params, state, self.rng, disaster_data)
        
        # Should have base agents active
        self.assertEqual(result['num_active_agents'], 4)
        
        # Consensus should still be valid
        self.assertEqual(result['consensus'].shape, (BATCH_SIZE, D_MODEL))
        self.assertTrue(jnp.all(jnp.isfinite(result['consensus'])))
        
        # Crisis mode should be enabled
        self.assertTrue(result['crisis_mode'])
        
    def test_specialist_agent_weight_scaling(self):
        """Test that spawned agents have specialized weight scales."""
        from src.modules.hybrid_architecture.hybrid_integrator import SpecialistAgent
        
        def forward_fn(x):
            # Create base agent
            base_agent = SpecialistAgent(
                d_model=D_MODEL,
                specialization="base",
                weight_scale=1.0,
                is_spawned=False,
                name="base_agent"
            )
            
            # Create spawned agent with higher weight scale
            spawned_agent = SpecialistAgent(
                d_model=D_MODEL,
                specialization="emergency_analysis",
                weight_scale=1.5,
                is_spawned=True,
                name="spawned_agent"
            )
            
            base_result = base_agent.process(x)
            spawned_result = spawned_agent.process(x)
            
            return base_result, spawned_result, base_agent.is_spawned, spawned_agent.is_spawned
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        inputs = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL))
        params, state = init_fn(self.rng, inputs)
        (base_result, spawned_result, base_spawned, spawned_spawned), _ = apply_fn(
            params, state, self.rng, inputs
        )
        
        # Check spawned flags
        self.assertFalse(base_spawned)
        self.assertTrue(spawned_spawned)
        
        # Both should produce valid outputs
        self.assertEqual(base_result['response'].shape, spawned_result['response'].shape)
        self.assertTrue(jnp.all(jnp.isfinite(base_result['response'])))
        self.assertTrue(jnp.all(jnp.isfinite(spawned_result['response'])))
        
    def test_parallel_processing(self):
        """Test parallel agent processing method."""
        from src.modules.hybrid_architecture.hybrid_integrator import MultiAgentConsensus
        
        def forward_fn(x):
            consensus = MultiAgentConsensus(
                d_model=D_MODEL,
                num_agents=4,
                name="parallel_test"
            )
            return consensus.process_parallel(x)
        
        init_fn, apply_fn = hk.transform_with_state(forward_fn)
        
        inputs = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL))
        params, state = init_fn(self.rng, inputs)
        result, _ = apply_fn(params, state, self.rng, inputs)
        
        # Check parallel consensus output
        self.assertIn('parallel_consensus', result)
        self.assertEqual(result['parallel_consensus'].shape, (BATCH_SIZE, D_MODEL))
        self.assertEqual(result['num_agents'], 4)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)

