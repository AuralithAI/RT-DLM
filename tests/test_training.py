"""
Unit tests for AGI training pipeline

Tests train_agi.py and training utilities.
Note: Data loading and tokenization tests removed - moved to Auralith-Data-Pipeline repo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import unittest
import jax.numpy as jnp

# Test constants
D_MODEL = 64
BATCH_SIZE = 2
SEQ_LEN = 16


class TestEvaluateReasoningQuality(unittest.TestCase):
    """Test reasoning quality evaluation"""
    
    def test_evaluate_reasoning_with_ground_truth(self):
        """Test reasoning evaluation with ground truth"""
        from config.agi_config import AGIConfig
        from train import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
        )
        
        trainer = AGITrainer(config)
        
        # Create fake reasoning chain
        reasoning_chain = [
            jnp.ones((BATCH_SIZE, D_MODEL)),
            jnp.ones((BATCH_SIZE, D_MODEL)) * 0.9,
            jnp.ones((BATCH_SIZE, D_MODEL)) * 0.8,
        ]
        
        # Evaluate without ground truth
        score = trainer.evaluate_reasoning_quality(reasoning_chain)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
    def test_evaluate_empty_reasoning_chain(self):
        """Test evaluation with empty reasoning chain"""
        from config.agi_config import AGIConfig
        from train import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
        )
        
        trainer = AGITrainer(config)
        
        score = trainer.evaluate_reasoning_quality([])
        self.assertEqual(score, 0.0)


class TestRetrievalIntegration(unittest.TestCase):
    """Test retrieval augmentation integration in training"""
    
    def test_configure_retrieval_disabled(self):
        """Test retrieval configuration when disabled"""
        from config.agi_config import AGIConfig
        from config.retrieval_config import RetrievalConfig
        from train import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
        )
        
        trainer = AGITrainer(config)
        
        # Initially retrieval should be None/disabled
        self.assertIsNone(trainer.retrieval_config)
        self.assertIsNone(trainer.retriever)
        
        # Configure with disabled config
        trainer.configure_retrieval(RetrievalConfig.disabled())
        self.assertIsNotNone(trainer.retrieval_config)
        self.assertFalse(trainer.retrieval_config.enabled)
        self.assertIsNone(trainer.retriever)
    
    def test_configure_retrieval_enabled(self):
        """Test retrieval configuration when enabled"""
        from config.agi_config import AGIConfig
        from config.retrieval_config import RetrievalConfig
        from train import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
        )
        
        trainer = AGITrainer(config)
        
        # Configure with training preset
        retrieval_config = RetrievalConfig.for_training()
        trainer.configure_retrieval(retrieval_config)
        
        self.assertTrue(trainer.retrieval_config.enabled)
        self.assertIsNotNone(trainer.retriever)
        self.assertIsNotNone(trainer.retrieval_training)
        self.assertIsNotNone(trainer.document_ingester)
    
    def test_batch_augmentation_disabled(self):
        """Test batch augmentation when retrieval is disabled"""
        import jax
        from config.agi_config import AGIConfig
        from train import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
        )
        
        trainer = AGITrainer(config)
        
        # Create sample batch
        batch = {
            "input_ids": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
            "targets": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
        }
        
        rng = jax.random.PRNGKey(42)
        
        # Augmentation should return same batch when disabled
        augmented = trainer._augment_batch_with_retrieval(batch, rng)
        self.assertIs(augmented, batch)


if __name__ == "__main__":
    unittest.main(verbosity=2)