"""
Tests for rtdlm

Covers:
- compute_multimodal_alignment_loss
- compute_controller_loss
- create_rtdlm_agi
- compute_agi_loss with various configurations
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import jax.numpy as jnp
import jax
import numpy as np


# Test constants
D_MODEL = 64
BATCH_SIZE = 2
SEQ_LEN = 16
VOCAB_SIZE = 1000


class TestComputeMultimodalAlignmentLoss(unittest.TestCase):
    """Test multimodal alignment loss computation."""
    
    def test_alignment_loss_with_embeddings(self):
        """Test alignment loss when text and image embeddings present."""
        from src.rtdlm import compute_multimodal_alignment_loss
        
        batch_size = 4
        aux_outputs = {
            "text_embeddings": jnp.ones((batch_size, D_MODEL)),
            "image_embeddings": jnp.ones((batch_size, D_MODEL)) * 0.5,
        }
        
        loss = compute_multimodal_alignment_loss(aux_outputs)
        
        # Loss should be finite
        self.assertTrue(jnp.isfinite(loss).item())
        self.assertGreaterEqual(float(loss), 0.0)
    
    def test_alignment_loss_perfect_match(self):
        """Test alignment loss when embeddings are identical."""
        from src.rtdlm import compute_multimodal_alignment_loss
        
        batch_size = 4
        # Identical normalized embeddings should give low/zero loss
        embeddings = jnp.eye(batch_size)  # Orthogonal embeddings
        
        aux_outputs = {
            "text_embeddings": embeddings,
            "image_embeddings": embeddings,  # Same as text
        }
        
        loss = compute_multimodal_alignment_loss(aux_outputs)
        
        # With identical embeddings, loss should be very low
        self.assertTrue(jnp.isfinite(loss).item())
    
    def test_alignment_loss_no_embeddings(self):
        """Test alignment loss returns 0 when no embeddings."""
        from src.rtdlm import compute_multimodal_alignment_loss
        
        aux_outputs = {}
        
        loss = compute_multimodal_alignment_loss(aux_outputs)
        
        self.assertEqual(float(loss), 0.0)
    
    def test_alignment_loss_only_text(self):
        """Test alignment loss returns 0 when only text embeddings."""
        from src.rtdlm import compute_multimodal_alignment_loss
        
        aux_outputs = {
            "text_embeddings": jnp.ones((4, D_MODEL)),
        }
        
        loss = compute_multimodal_alignment_loss(aux_outputs)
        
        self.assertEqual(float(loss), 0.0)
    
    def test_alignment_loss_audio_embeddings(self):
        """Test alignment loss with audio embeddings."""
        from src.rtdlm import compute_multimodal_alignment_loss
        
        batch_size = 4
        aux_outputs = {
            "text_embeddings": jnp.ones((batch_size, D_MODEL)),
            "audio_embeddings": jnp.ones((batch_size, D_MODEL)) * 0.8,
        }
        
        loss = compute_multimodal_alignment_loss(aux_outputs)
        
        self.assertTrue(jnp.isfinite(loss).item())
        self.assertGreaterEqual(float(loss), 0.0)


class TestControllerLossComputer(unittest.TestCase):
    """Test ControllerLossComputer class."""
    
    def test_controller_loss_basic(self):
        """Test controller loss with basic execution trace."""
        from src.core.agi.compute_controller import ControllerLossComputer
        
        loss_computer = ControllerLossComputer()
        
        task_loss = jnp.array(1.0)
        execution_trace = {
            "total_cost": 0.3,
            "modules_executed": [],
            "halt_probs": [],
        }
        predicted_confidence = jnp.array([0.8])
        actual_accuracy = jnp.array([1.0])
        
        total_loss, components = loss_computer.compute_total_loss(
            task_loss=task_loss,
            execution_trace=execution_trace,
            predicted_confidence=predicted_confidence,
            actual_accuracy=actual_accuracy,
        )
        
        self.assertTrue(jnp.isfinite(total_loss).item())
        self.assertIn("task_loss", components)
    
    def test_controller_loss_with_modules(self):
        """Test controller loss with modules executed."""
        from src.core.agi.compute_controller import ControllerLossComputer
        
        loss_computer = ControllerLossComputer()
        
        task_loss = jnp.array(1.0)
        execution_trace = {
            "total_cost": 0.5,
            "modules_executed": ["MEMORY_RETRIEVAL", "GRAPH_REASONING"],
            "halt_probs": [jnp.array([0.3]), jnp.array([0.7])],
        }
        predicted_confidence = jnp.array([0.9])
        actual_accuracy = jnp.array([1.0])
        
        total_loss, components = loss_computer.compute_total_loss(
            task_loss=task_loss,
            execution_trace=execution_trace,
            predicted_confidence=predicted_confidence,
            actual_accuracy=actual_accuracy,
        )
        
        self.assertTrue(jnp.isfinite(total_loss).item())
        self.assertIn("efficiency_loss", components)


class TestCreateRtdlmAgi(unittest.TestCase):
    """Test create_rtdlm_agi factory function."""
    
    def test_create_with_default_config(self):
        """Test creating model with default AGI config."""
        from src.rtdlm import create_rtdlm_agi
        from src.config.agi_config import AGIConfig
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=VOCAB_SIZE,
        )
        
        model = create_rtdlm_agi(config)
        
        self.assertIsNotNone(model)
        # model should be a Haiku transformed function
        self.assertTrue(hasattr(model, 'init'))
        self.assertTrue(hasattr(model, 'apply'))
    
    def test_create_with_use_state_true(self):
        """Test creating model with use_state=True (default)."""
        from src.rtdlm import create_rtdlm_agi
        from src.config.agi_config import AGIConfig
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=VOCAB_SIZE,
        )
        
        model = create_rtdlm_agi(config)
        
        self.assertIsNotNone(model)
    
    def test_create_with_use_state_false(self):
        """Test creating model with use_state=False."""
        from src.rtdlm import create_rtdlm_agi
        from src.config.agi_config import AGIConfig
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=VOCAB_SIZE,
        )
        
        model = create_rtdlm_agi(config, use_state=False)
        
        self.assertIsNotNone(model)


class TestComputeAgiLoss(unittest.TestCase):
    """Test compute_agi_loss function."""
    
    def test_basic_loss_computation(self):
        """Test basic loss computation with logits and targets."""
        from src.rtdlm import compute_agi_loss
        from src.config.agi_config import AGIConfig
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=VOCAB_SIZE,
        )
        
        # Create fake logits and targets
        logits = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        targets = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
        
        loss = compute_agi_loss(logits, targets, config=config)
        
        self.assertTrue(jnp.isfinite(loss).item())
        self.assertGreater(float(loss), 0.0)
    
    def test_loss_with_aux_outputs(self):
        """Test loss computation with auxiliary outputs."""
        from src.rtdlm import compute_agi_loss
        from src.config.agi_config import AGIConfig
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=VOCAB_SIZE,
        )
        
        logits = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        targets = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
        
        aux_outputs = {
            "text_embeddings": jnp.ones((BATCH_SIZE, D_MODEL)),
            "image_embeddings": jnp.ones((BATCH_SIZE, D_MODEL)),
            "controller_outputs": {
                "halt_prob": jnp.array([0.5]),
            },
        }
        
        loss = compute_agi_loss(logits, targets, aux_outputs=aux_outputs, config=config)
        
        self.assertTrue(jnp.isfinite(loss).item())
        self.assertGreater(float(loss), 0.0)
    
    def test_loss_with_reasoning_outputs(self):
        """Test loss computation with reasoning outputs."""
        from src.rtdlm import compute_agi_loss
        from src.config.agi_config import AGIConfig
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=VOCAB_SIZE,
        )
        
        logits = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        targets = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
        
        aux_outputs = {
            "reasoning_chain": [jnp.ones((BATCH_SIZE, D_MODEL)) for _ in range(3)],
        }
        
        loss = compute_agi_loss(logits, targets, aux_outputs=aux_outputs, config=config)
        
        self.assertTrue(jnp.isfinite(loss).item())


class TestCreateAgiOptimizer(unittest.TestCase):
    """Test create_agi_optimizer function."""
    
    def test_create_optimizer_default(self):
        """Test creating optimizer with default settings."""
        from src.rtdlm import create_agi_optimizer
        from src.config.agi_config import AGIConfig
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=VOCAB_SIZE,
            learning_rate=1e-4,
        )
        
        optimizer = create_agi_optimizer(config)
        
        self.assertIsNotNone(optimizer)
        # Optimizer should be an optax GradientTransformation
        self.assertTrue(hasattr(optimizer, 'init'))
        self.assertTrue(hasattr(optimizer, 'update'))


class TestAGISystemForward(unittest.TestCase):
    """Test AGI system forward pass."""
    
    def test_forward_text_only(self):
        """Test forward pass with text input only."""
        from src.rtdlm import create_rtdlm_agi
        from src.config.agi_config import AGIConfig
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=VOCAB_SIZE,
            max_seq_length=SEQ_LEN,
            multimodal_enabled=False,
        )
        
        model = create_rtdlm_agi(config)
        
        # Initialize model
        rng = jax.random.PRNGKey(42)
        input_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
        
        params, state = model.init(
            rng,
            inputs={"text": input_ids},
            multimodal_inputs=None,
            is_training=False,
        )
        
        # Forward pass
        output, _ = model.apply(
            params, state, rng,
            inputs={"text": input_ids},
            multimodal_inputs=None,
            is_training=False,
        )
        
        self.assertIn("logits", output)
        self.assertEqual(output["logits"].shape, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
    
    def test_forward_with_return_reasoning(self):
        """Test forward pass with return_reasoning=True."""
        from src.rtdlm import create_rtdlm_agi
        from src.config.agi_config import AGIConfig
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=VOCAB_SIZE,
            max_seq_length=SEQ_LEN,
            multimodal_enabled=False,
        )
        
        model = create_rtdlm_agi(config)
        
        rng = jax.random.PRNGKey(42)
        input_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
        
        params, state = model.init(
            rng,
            inputs={"text": input_ids},
            multimodal_inputs=None,
            is_training=False,
            return_reasoning=True,
        )
        
        output, _ = model.apply(
            params, state, rng,
            inputs={"text": input_ids},
            multimodal_inputs=None,
            is_training=False,
            return_reasoning=True,
        )
        
        self.assertIn("logits", output)


class TestAGISystemTrainingMode(unittest.TestCase):
    """Test AGI system in training mode."""
    
    def test_training_mode_deterministic(self):
        """Test that is_training=False gives deterministic outputs."""
        from src.rtdlm import create_rtdlm_agi
        from src.config.agi_config import AGIConfig
        
        config = AGIConfig(
            d_model=D_MODEL,
            num_heads=4,
            num_layers=2,
            vocab_size=VOCAB_SIZE,
            max_seq_length=SEQ_LEN,
            multimodal_enabled=False,
        )
        
        model = create_rtdlm_agi(config)
        
        rng = jax.random.PRNGKey(42)
        input_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
        
        params, state = model.init(
            rng,
            inputs={"text": input_ids},
            multimodal_inputs=None,
            is_training=False,
        )
        
        # Two forward passes with is_training=False should be identical
        output1, _ = model.apply(
            params, state, rng,
            inputs={"text": input_ids},
            multimodal_inputs=None,
            is_training=False,
        )
        
        output2, _ = model.apply(
            params, state, rng,
            inputs={"text": input_ids},
            multimodal_inputs=None,
            is_training=False,
        )
        
        # Outputs should be identical
        np.testing.assert_array_almost_equal(
            np.array(output1["logits"]),
            np.array(output2["logits"]),
            decimal=5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
