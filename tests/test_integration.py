"""
Integration tests combining multiple AGI modules

Tests that all modules work together with compatible dimensions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import unittest
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
        
    def test_consciousness_to_creativity_pipeline(self):
        """Test consciousness output can feed into creativity"""
        def pipeline_fn(internal_state, external_input):
            consciousness = ConsciousnessSimulator(d_model=D_MODEL)
            creativity = CreativeGenerationEngine(d_model=D_MODEL)
            
            c_out = consciousness(internal_state, external_input)
            # self_awareness is (batch, d_model), need to expand for creativity input
            awareness_expanded = jnp.expand_dims(c_out["self_awareness"], axis=1)
            awareness_expanded = jnp.broadcast_to(
                awareness_expanded, 
                (BATCH_SIZE, SEQ_LEN, D_MODEL)
            )
            cr_out = creativity(awareness_expanded)
            
            return {
                "awareness": c_out["self_awareness"],
                "novelty": cr_out["novelty_score"]
            }
        
        transformed = hk.transform(pipeline_fn)
        
        internal_state = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        external_input = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, internal_state, external_input)
        result = transformed.apply(params, self.rng, internal_state, external_input)
        
        self.assertIn("awareness", result)
        self.assertIn("novelty", result)
        
    def test_science_to_social_pipeline(self):
        """Test scientific output can inform social reasoning"""
        def pipeline_fn(knowledge_base, observations):
            science = ScientificDiscoveryEngine(d_model=D_MODEL)
            social = SocialEmotionalIntelligence(d_model=D_MODEL)
            
            s_out = science(knowledge_base, observations)
            # Use hypothesis as context for social reasoning
            so_out = social(s_out["hypothesis"], s_out["hypothesis"])
            
            return {
                "hypothesis": s_out["hypothesis"],
                "emotions": so_out["recognized_emotions"]
            }
        
        transformed = hk.transform(pipeline_fn)
        
        knowledge_base = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, knowledge_base, observations)
        result = transformed.apply(params, self.rng, knowledge_base, observations)
        
        self.assertIn("hypothesis", result)
        self.assertIn("emotions", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
