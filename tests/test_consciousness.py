"""
Unit tests for ConsciousnessSimulator module

Tests RNN-based introspection and self-awareness functionality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import unittest
import jax
import jax.numpy as jnp
import haiku as hk

from rtdlm_agi_complete import ConsciousnessSimulator

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
        """Test output shape matches expected dimensions"""
        def forward_fn(internal_state, external_input):
            model = ConsciousnessSimulator(d_model=D_MODEL)
            return model(internal_state, external_input)
        
        transformed = hk.transform(forward_fn)
        
        internal_state = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        external_input = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, internal_state, external_input)
        result = transformed.apply(params, self.rng, internal_state, external_input)
        
        # Check return dict keys
        self.assertIn("self_awareness", result)
        self.assertIn("introspection", result)
        self.assertIn("recurrent_introspection", result)
        
    def test_awareness_level_bounded(self):
        """Test that self_awareness has correct dimensions"""
        def forward_fn(internal_state, external_input):
            model = ConsciousnessSimulator(d_model=D_MODEL)
            return model(internal_state, external_input)
        
        transformed = hk.transform(forward_fn)
        
        internal_state = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        external_input = jax.random.normal(jax.random.PRNGKey(100), (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, internal_state, external_input)
        result = transformed.apply(params, self.rng, internal_state, external_input)
        
        awareness = result["self_awareness"]
        self.assertEqual(len(awareness.shape), 2)  # (batch, d_model)
        self.assertEqual(awareness.shape[0], BATCH_SIZE)
        
    def test_recurrent_introspection_output(self):
        """Test that recurrent introspection produces valid output"""
        def forward_fn(internal_state, external_input):
            model = ConsciousnessSimulator(d_model=D_MODEL)
            return model(internal_state, external_input)
        
        transformed = hk.transform(forward_fn)
        
        internal_state = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        external_input = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, internal_state, external_input)
        result = transformed.apply(params, self.rng, internal_state, external_input)
        
        # Check recurrent_introspection exists and has valid shape
        self.assertIn("recurrent_introspection", result)
        introspection = result["recurrent_introspection"]
        self.assertEqual(len(introspection.shape), 2)  # (batch, d_model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
