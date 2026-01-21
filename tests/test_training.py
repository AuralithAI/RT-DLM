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

    def test_batch_augmentation_enabled_with_documents(self):
        """Test batch augmentation when retrieval is enabled with documents"""
        import jax
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
        
        # Configure retrieval with high augmentation probability to ensure it triggers
        # No mock needed - we now use hash-based embeddings that match d_model
        retrieval_config = RetrievalConfig.for_training()
        retrieval_config.augmentation_probability = 1.0  # Always augment for test
        trainer.configure_retrieval(retrieval_config)
        
        # Ingest some documents
        documents = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn.",
            "Deep learning uses neural networks with many layers for complex pattern recognition.",
            "Transformers revolutionized natural language processing with attention mechanisms.",
            "Reinforcement learning trains agents through reward signals and environment interaction.",
        ]
        trainer.ingest_documents(documents)
        
        # Create sample batch
        batch = {
            "input_ids": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
            "targets": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
        }
        
        rng = jax.random.PRNGKey(42)
        
        # Augmentation should return modified batch with retrieval context
        augmented = trainer._augment_batch_with_retrieval(batch, rng)
        
        # Verify augmented batch is different from original
        self.assertIsNot(augmented, batch)
        
        # Verify original keys are preserved
        self.assertIn("input_ids", augmented)
        self.assertIn("targets", augmented)
        
        # Verify retrieval keys are added (if augmentation was applied)
        # Note: These may or may not be present depending on retrieval results
        if "retrieved_embeddings" in augmented:
            self.assertIn("retrieval_mask", augmented)


class TestMemoryProfilerIntegration(unittest.TestCase):
    """Test memory profiler integration in training"""
    
    def test_memory_profiler_initialization(self):
        """Test memory profiler is initialized in trainer"""
        from config.agi_config import AGIConfig
        from train import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
            enable_memory_profiling=True,
        )
        
        trainer = AGITrainer(config)
        
        # Memory profiler should be initialized
        self.assertIsNotNone(trainer.memory_profiler)
        self.assertTrue(trainer.memory_profiler.enabled)
    
    def test_memory_profiler_disabled_by_default(self):
        """Test memory profiler is disabled by default"""
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
        
        # Memory profiler should exist but be disabled
        self.assertIsNotNone(trainer.memory_profiler)
        self.assertFalse(trainer.memory_profiler.enabled)


class TestGradientAccumulationIntegration(unittest.TestCase):
    """Test gradient accumulation integration in training"""
    
    def test_gradient_accumulation_default(self):
        """Test gradient accumulation defaults to 1 (disabled)"""
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
        
        # Default should be 1 (no accumulation)
        self.assertEqual(trainer.gradient_accumulation_steps, 1)
    
    def test_gradient_accumulation_configured(self):
        """Test gradient accumulation when configured"""
        from config.agi_config import AGIConfig
        from train import AGITrainer
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            batch_size=BATCH_SIZE,
            gradient_accumulation_steps=4,
        )
        
        trainer = AGITrainer(config)
        
        # Should match config
        self.assertEqual(trainer.gradient_accumulation_steps, 4)
    
    def test_compute_grads_method(self):
        """Test _compute_grads method exists and has correct signature"""
        import inspect
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
        
        # Verify _compute_grads method exists
        self.assertTrue(hasattr(trainer, '_compute_grads'))
        self.assertTrue(callable(trainer._compute_grads))
        
        # Check method signature has expected parameters
        sig = inspect.signature(trainer._compute_grads)
        params = list(sig.parameters.keys())
        self.assertIn('params', params)
        self.assertIn('batch', params)
        self.assertIn('rng', params)


if __name__ == "__main__":
    unittest.main(verbosity=2)