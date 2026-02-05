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
        from src.config.agi_config import AGIConfig
        from src.train import AGITrainer
        
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
        from src.config.agi_config import AGIConfig
        from src.train import AGITrainer
        
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
        from src.config.agi_config import AGIConfig
        from src.config.retrieval_config import RetrievalConfig
        from src.train import AGITrainer
        
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
        from src.config.agi_config import AGIConfig
        from src.config.retrieval_config import RetrievalConfig
        from src.train import AGITrainer
        
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
        from src.config.agi_config import AGIConfig
        from src.train import AGITrainer
        
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
        from src.config.agi_config import AGIConfig
        from src.config.retrieval_config import RetrievalConfig
        from src.train import AGITrainer
        
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
        from src.config.agi_config import AGIConfig
        from src.train import AGITrainer
        
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
        from src.config.agi_config import AGIConfig
        from src.train import AGITrainer
        
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
        from src.config.agi_config import AGIConfig
        from src.train import AGITrainer
        
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
        from src.config.agi_config import AGIConfig
        from src.train import AGITrainer
        
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
    
    def test_gradient_accumulator_standalone(self):
        """Test BatchGradientAccumulator works correctly in isolation"""
        import jax
        from src.core.gradient_accumulation import BatchGradientAccumulator
        
        # Create a simple model and loss for testing
        def simple_model_apply(params, rng, **batch):
            """Simple model that returns logits based on params and input."""
            x = batch["input_ids"].astype(jnp.float32)
            # Simple linear layer: y = x @ W + b
            logits = jnp.dot(x, params["W"]) + params["b"]
            return {"logits": logits}
        
        def simple_loss_fn(outputs, batch):
            """MSE loss between logits and targets."""
            logits = outputs["logits"]
            targets = batch["targets"].astype(jnp.float32)
            return jnp.mean((logits - targets) ** 2)
        
        # Initialize simple params
        rng = jax.random.PRNGKey(42)
        params = {
            "W": jax.random.normal(rng, (SEQ_LEN, SEQ_LEN)),
            "b": jnp.zeros((SEQ_LEN,)),
        }
        
        # Create accumulator
        accumulator = BatchGradientAccumulator(
            accumulation_steps=2,
            loss_fn=simple_loss_fn,
            model_apply_fn=simple_model_apply,
        )
        
        # Create micro-batches
        batch1 = {
            "input_ids": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
            "targets": jnp.zeros((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
        }
        batch2 = {
            "input_ids": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32) * 2,
            "targets": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
        }
        
        # Accumulate first batch - should not be complete
        done = accumulator.accumulate(params, batch1, rng)
        self.assertFalse(done)
        self.assertEqual(accumulator.current_step, 1)
        
        # Accumulate second batch - should be complete
        done = accumulator.accumulate(params, batch2, jax.random.fold_in(rng, 1))
        self.assertTrue(done)
        self.assertEqual(accumulator.current_step, 2)
        
        # Get accumulated gradients
        grads = accumulator.get_accumulated_grads()
        self.assertIsNotNone(grads)
        self.assertIn("W", grads)
        self.assertIn("b", grads)
        
        # Verify gradients have correct shapes
        self.assertEqual(grads["W"].shape, params["W"].shape)
        self.assertEqual(grads["b"].shape, params["b"].shape)
        
        # Verify average loss is computed
        avg_loss = accumulator.get_accumulated_loss()
        self.assertFalse(jnp.isnan(avg_loss))
        self.assertGreater(float(avg_loss), 0.0)  # Should be positive for non-zero inputs
        
        # Reset and verify state is cleared
        accumulator.reset()
        self.assertEqual(accumulator.current_step, 0)
        self.assertFalse(accumulator.is_complete)
    
    def test_gradient_accumulator_nan_handling(self):
        """Test that BatchGradientAccumulator properly handles NaN gradients"""
        import jax
        from src.core.gradient_accumulation import BatchGradientAccumulator
        
        call_count = [0]  # Mutable to track calls
        
        def model_with_nan(params, rng, **batch):
            """Model that produces NaN on first call."""
            call_count[0] += 1
            x = batch["input_ids"].astype(jnp.float32)
            logits = jnp.dot(x, params["W"]) + params["b"]
            # Produce NaN on first call
            if call_count[0] == 1:
                logits = logits * jnp.nan
            return {"logits": logits}
        
        def simple_loss_fn(outputs, batch):
            logits = outputs["logits"]
            targets = batch["targets"].astype(jnp.float32)
            return jnp.mean((logits - targets) ** 2)
        
        # Initialize params
        rng = jax.random.PRNGKey(42)
        params = {
            "W": jax.random.normal(rng, (SEQ_LEN, SEQ_LEN)),
            "b": jnp.zeros((SEQ_LEN,)),
        }
        
        # Create accumulator
        accumulator = BatchGradientAccumulator(
            accumulation_steps=2,
            loss_fn=simple_loss_fn,
            model_apply_fn=model_with_nan,
        )
        
        batch = {
            "input_ids": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
            "targets": jnp.zeros((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
        }
        
        # Accumulate both batches (first produces NaN, second is valid)
        accumulator.accumulate(params, batch, rng)
        accumulator.accumulate(params, batch, jax.random.fold_in(rng, 1))
        
        # Get gradients - should not contain NaN
        grads = accumulator.get_accumulated_grads()
        
        # Check no NaN in gradients (NaN should have been zeroed)
        for key, grad in grads.items():
            self.assertFalse(jnp.any(jnp.isnan(grad)), f"NaN found in gradient for {key}")

    def test_model_apply_wrapper_pattern(self):
        """Test that models with different signatures can be wrapped for BatchGradientAccumulator"""
        import jax
        from src.core.gradient_accumulation import BatchGradientAccumulator
        
        # Simulate a model that expects inputs={"text": ...} like RTDLMModel
        def model_with_inputs_dict(params, rng, inputs, return_reasoning=False):
            """Model that expects inputs dict, not **batch."""
            x = inputs["text"].astype(jnp.float32)
            logits = jnp.dot(x, params["W"]) + params["b"]
            result = {"logits": logits}
            if return_reasoning:
                result["reasoning"] = "test"
            return result
        
        # Wrapper that adapts batch format to model signature
        def model_apply_wrapper(params, rng, **batch):
            inputs = {"text": batch["input_ids"]}
            return model_with_inputs_dict(params, rng, inputs=inputs, return_reasoning=True)
        
        def simple_loss_fn(outputs, batch):
            logits = outputs["logits"]
            targets = batch["targets"].astype(jnp.float32)
            return jnp.mean((logits - targets) ** 2)
        
        # Initialize params
        rng = jax.random.PRNGKey(42)
        params = {
            "W": jax.random.normal(rng, (SEQ_LEN, SEQ_LEN)),
            "b": jnp.zeros((SEQ_LEN,)),
        }
        
        # Create accumulator with wrapper
        accumulator = BatchGradientAccumulator(
            accumulation_steps=2,
            loss_fn=simple_loss_fn,
            model_apply_fn=model_apply_wrapper,  # Use wrapper, not raw model
        )
        
        batch = {
            "input_ids": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
            "targets": jnp.zeros((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
        }
        
        # Should work without error
        accumulator.accumulate(params, batch, rng)
        done = accumulator.accumulate(params, batch, jax.random.fold_in(rng, 1))
        
        self.assertTrue(done)
        grads = accumulator.get_accumulated_grads()
        self.assertIn("W", grads)
        self.assertIn("b", grads)

    def test_structured_batch_auto_detection(self):
        """Test that BatchGradientAccumulator auto-detects structured batches with 'inputs' key"""
        import jax
        from src.core.gradient_accumulation import BatchGradientAccumulator
        
        # Model that expects inputs={"text": ...} and multimodal_inputs (like RTDLMModel)
        def model_with_structured_args(params, rng, inputs, multimodal_inputs=None, return_reasoning=False):
            """Model that expects structured inputs, not **batch."""
            x = inputs["text"].astype(jnp.float32)
            logits = jnp.dot(x, params["W"]) + params["b"]
            result = {"logits": logits}
            if return_reasoning:
                result["reasoning"] = "test"
            return result
        
        def simple_loss_fn(outputs, batch):
            logits = outputs["logits"]
            targets = batch["targets"].astype(jnp.float32)
            return jnp.mean((logits - targets) ** 2)
        
        # Initialize params
        rng = jax.random.PRNGKey(42)
        params = {
            "W": jax.random.normal(rng, (SEQ_LEN, SEQ_LEN)),
            "b": jnp.zeros((SEQ_LEN,)),
        }
        
        # Create accumulator with raw model (no wrapper) - relies on _apply_model detection
        accumulator = BatchGradientAccumulator(
            accumulation_steps=2,
            loss_fn=simple_loss_fn,
            model_apply_fn=model_with_structured_args,
        )
        
        # Structured batch with 'inputs' key - triggers auto-detection
        structured_batch = {
            "inputs": {"text": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)},
            "multimodal_inputs": None,
            "targets": jnp.zeros((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
        }
        
        # Should work without wrapper because _apply_model detects 'inputs' key
        accumulator.accumulate(params, structured_batch, rng)
        done = accumulator.accumulate(params, structured_batch, jax.random.fold_in(rng, 1))
        
        self.assertTrue(done)
        grads = accumulator.get_accumulated_grads()
        self.assertIn("W", grads)
        self.assertIn("b", grads)
        
        # Verify gradients are valid
        for key, grad in grads.items():
            self.assertFalse(jnp.any(jnp.isnan(grad)), f"NaN found in gradient for {key}")


if __name__ == "__main__":
    unittest.main(verbosity=2)