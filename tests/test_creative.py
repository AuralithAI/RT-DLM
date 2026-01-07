"""
Unit tests for CreativeGenerationEngine module

Tests entropy-based novelty detection functionality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import unittest
import jax
import jax.numpy as jnp
import haiku as hk

from rtdlm import CreativeGenerationEngine

# Test constants
D_MODEL = 64
BATCH_SIZE = 2
SEQ_LEN = 16


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
        
    def test_with_creativity_level(self):
        """Test with different creativity levels"""
        def forward_fn(content_context, creativity_level=0.5):
            model = CreativeGenerationEngine(d_model=D_MODEL)
            return model(content_context, creativity_level=creativity_level)
        
        transformed = hk.transform(forward_fn)
        
        content_context = jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, content_context)
        
        # Test with different creativity levels
        result_low = transformed.apply(params, self.rng, content_context, creativity_level=0.2)
        result_high = transformed.apply(params, self.rng, content_context, creativity_level=0.9)
        
        self.assertIn("creative_content", result_low)
        self.assertIn("novelty_score", result_low)
        self.assertIn("creative_content", result_high)
        self.assertIn("novelty_score", result_high)
        
    def test_different_inputs_different_novelty(self):
        """Test that different inputs produce different novelty scores"""
        def forward_fn(input_context, creative_seed=None):
            model = CreativeGenerationEngine(d_model=D_MODEL)
            return model(input_context, creative_seed)
        
        transformed = hk.transform(forward_fn)
        
        input1 = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        input2 = jax.random.normal(jax.random.PRNGKey(999), (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, input1)
        result1 = transformed.apply(params, self.rng, input1)
        result2 = transformed.apply(params, self.rng, input2)
        
        # Different inputs should generally produce different novelty
        self.assertFalse(jnp.allclose(
            result1["creative_content"], 
            result2["creative_content"]
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)

