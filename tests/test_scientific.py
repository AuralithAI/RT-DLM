"""
Unit tests for ScientificDiscoveryEngine module

Tests do-intervention causal reasoning functionality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import unittest
import jax
import jax.numpy as jnp
import haiku as hk

from rtdlm_agi_complete import ScientificDiscoveryEngine

# Test constants
D_MODEL = 64
BATCH_SIZE = 2
SEQ_LEN = 16


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
        
    def test_intervention_shape(self):
        """Test that intervention results have correct shape"""
        def forward_fn(knowledge_base, observations):
            model = ScientificDiscoveryEngine(d_model=D_MODEL)
            return model(knowledge_base, observations)
        
        transformed = hk.transform(forward_fn)
        
        knowledge_base = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, knowledge_base, observations)
        result = transformed.apply(params, self.rng, knowledge_base, observations)
        
        self.assertIn("intervention_results", result)
        intervention = result["intervention_results"]
        self.assertIn("intervened_state", intervention)
        
    def test_output_contains_causal_data(self):
        """Test output contains all causal reasoning data"""
        def forward_fn(knowledge_base, observations):
            model = ScientificDiscoveryEngine(d_model=D_MODEL)
            return model(knowledge_base, observations)
        
        transformed = hk.transform(forward_fn)
        
        knowledge_base = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
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


class TestCausalReasoningAdvanced(unittest.TestCase):
    """Advanced tests for do-intervention causal reasoning"""
    
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
