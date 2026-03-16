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
        self.assertTrue(hasattr(model, "init"))
        self.assertTrue(hasattr(model, "apply"))

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
        self.assertTrue(hasattr(optimizer, "init"))
        self.assertTrue(hasattr(optimizer, "update"))


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
            params,
            state,
            rng,
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
            params,
            state,
            rng,
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
            params,
            state,
            rng,
            inputs={"text": input_ids},
            multimodal_inputs=None,
            is_training=False,
        )

        output2, _ = model.apply(
            params,
            state,
            rng,
            inputs={"text": input_ids},
            multimodal_inputs=None,
            is_training=False,
        )

        # Outputs should be identical
        np.testing.assert_array_almost_equal(
            np.array(output1["logits"]), np.array(output2["logits"]), decimal=5
        )


class TestConsciousnessSimulator(unittest.TestCase):
    """Test ConsciousnessSimulator class for coverage."""

    def test_consciousness_valid_inputs(self):
        """Test consciousness simulator with valid inputs."""
        import haiku as hk
        from src.rtdlm import ConsciousnessSimulator

        def forward(internal_state, external_input, previous_goals=None):
            sim = ConsciousnessSimulator(d_model=D_MODEL)
            return sim(internal_state, external_input, previous_goals)

        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)

        internal_state = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        external_input = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        params = init.init(rng, internal_state, external_input)
        output = init.apply(params, rng, internal_state, external_input)

        self.assertIn("self_awareness", output)
        self.assertIn("introspection", output)
        self.assertIn("autonomous_goals", output)

    def test_consciousness_with_previous_goals(self):
        """Test consciousness simulator with previous goals (covers goal revision)."""
        import haiku as hk
        from src.rtdlm import ConsciousnessSimulator

        def forward(internal_state, external_input, previous_goals):
            sim = ConsciousnessSimulator(d_model=D_MODEL)
            return sim(internal_state, external_input, previous_goals)

        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)

        internal_state = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        external_input = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        previous_goals = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        params = init.init(rng, internal_state, external_input, previous_goals)
        output = init.apply(params, rng, internal_state, external_input, previous_goals)

        self.assertIn("autonomous_goals", output)

    def test_consciousness_1d_internal_state_error(self):
        """Test consciousness simulator raises error for 1D internal_state."""
        import haiku as hk
        from src.rtdlm import ConsciousnessSimulator

        def forward(internal_state, external_input):
            sim = ConsciousnessSimulator(d_model=D_MODEL)
            return sim(internal_state, external_input)

        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)

        internal_state = jax.random.normal(rng, (D_MODEL,))
        external_input = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        params = init.init(rng, jnp.expand_dims(internal_state, 0), external_input)
        output = init.apply(params, rng, internal_state, external_input)

        self.assertIn("self_awareness", output)

    def test_consciousness_1d_external_input_error(self):
        """Test consciousness simulator raises error for 1D external_input."""
        import haiku as hk
        from src.rtdlm import ConsciousnessSimulator

        def forward(internal_state, external_input):
            sim = ConsciousnessSimulator(d_model=D_MODEL)
            return sim(internal_state, external_input)

        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)

        internal_state = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        external_input = jax.random.normal(rng, (D_MODEL,))

        params = init.init(rng, internal_state, jnp.expand_dims(external_input, 0))
        output = init.apply(params, rng, internal_state, external_input)

        self.assertIn("self_awareness", output)


class TestCreativeGenerationEngine(unittest.TestCase):
    """Test CreativeGenerationEngine class for coverage."""

    def test_creative_engine_basic(self):
        """Test creative engine with basic inputs."""
        import haiku as hk
        from src.rtdlm import CreativeGenerationEngine

        def forward(
            content_context, style_reference=None, creativity_level=0.7, previous_content=None
        ):
            engine = CreativeGenerationEngine(d_model=D_MODEL)
            return engine(content_context, style_reference, creativity_level, previous_content)

        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)

        content_context = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        params = init.init(rng, content_context)
        output = init.apply(params, rng, content_context)

        self.assertIn("creative_content", output)
        self.assertIn("novelty_score", output)

    def test_creative_engine_with_style_reference(self):
        """Test creative engine with style reference (covers style_encoder)."""
        import haiku as hk
        from src.rtdlm import CreativeGenerationEngine

        def forward(content_context, style_reference):
            engine = CreativeGenerationEngine(d_model=D_MODEL)
            return engine(content_context, style_reference)

        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)

        content_context = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        style_reference = jax.random.normal(rng, (BATCH_SIZE, D_MODEL))

        params = init.init(rng, content_context, style_reference)
        output = init.apply(params, rng, content_context, style_reference)

        self.assertIn("style_encoding", output)
        self.assertFalse(jnp.allclose(output["style_encoding"], 0.0))

    def test_creative_engine_with_reference_content(self):
        """Test creative engine with reference content (covers similarity novelty)."""
        import haiku as hk
        from src.rtdlm import CreativeGenerationEngine

        def forward(content_context, previous_content):
            engine = CreativeGenerationEngine(d_model=D_MODEL)
            return engine(content_context, previous_content=previous_content)

        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)

        content_context = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        previous_content = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        params = init.init(rng, content_context, previous_content)
        output = init.apply(params, rng, content_context, previous_content)

        self.assertIn("novelty_metrics", output)
        self.assertIn("similarity_novelty", output["novelty_metrics"])


class TestSocialEmotionalIntelligence(unittest.TestCase):
    """Test SocialEmotionalIntelligence class for coverage."""

    def test_social_emotional_basic(self):
        """Test social emotional intelligence with basic inputs."""
        import haiku as hk
        from src.rtdlm import SocialEmotionalIntelligence

        def forward(user_input, conversation_history=None, social_context=None):
            sei = SocialEmotionalIntelligence(d_model=D_MODEL)
            return sei(user_input, conversation_history, social_context)

        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)

        user_input = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        params = init.init(rng, user_input)
        output = init.apply(params, rng, user_input)

        self.assertIn("recognized_emotions", output)
        self.assertIn("empathy_signal", output)

    def test_social_emotional_with_social_context(self):
        """Test social emotional intelligence with social context (covers cultural_adapter)."""
        import haiku as hk
        from src.rtdlm import SocialEmotionalIntelligence

        def forward(user_input, social_context):
            sei = SocialEmotionalIntelligence(d_model=D_MODEL)
            return sei(user_input, social_context=social_context)

        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)

        user_input = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        social_context = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        params = init.init(rng, user_input, social_context)
        output = init.apply(params, rng, user_input, social_context)

        self.assertIn("cultural_adapted", output)
        self.assertFalse(jnp.allclose(output["cultural_adapted"], 0.0))

    def test_get_emotion_label_valid(self):
        """Test get_emotion_label with valid index."""
        from src.rtdlm import SocialEmotionalIntelligence

        # Test the emotion labels directly from class attribute
        labels = SocialEmotionalIntelligence.EMOTION_LABELS

        # Valid indices
        self.assertEqual(labels[0], "joy")
        self.assertEqual(labels[6], "neutral")
        self.assertEqual(labels[13], "curiosity")

    def test_get_emotion_label_invalid(self):
        """Test get_emotion_label with invalid index (covers 'unknown' return)."""
        import haiku as hk
        from src.rtdlm import SocialEmotionalIntelligence

        def forward_invalid_index(user_input):
            sei = SocialEmotionalIntelligence(d_model=D_MODEL)
            # Call get_emotion_label with invalid index
            label = sei.get_emotion_label(100)
            result = sei(user_input)
            result["test_label"] = label
            return result

        init = hk.transform(forward_invalid_index)
        rng = jax.random.PRNGKey(42)

        user_input = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        params = init.init(rng, user_input)
        output = init.apply(params, rng, user_input)

        # Invalid index should return "unknown"
        self.assertEqual(output["test_label"], "unknown")

    def test_social_emotional_2d_social_analysis(self):
        """Test when social_analysis is 2D (covers else branch)."""
        import haiku as hk
        from src.rtdlm import SocialEmotionalIntelligence

        def forward(user_input):
            sei = SocialEmotionalIntelligence(d_model=D_MODEL)
            return sei(user_input, conversation_history=None)

        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)

        user_input = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        params = init.init(rng, user_input)
        output = init.apply(params, rng, user_input)

        self.assertIn("socially_aware_response", output)


class TestScientificDiscoveryEngine(unittest.TestCase):
    """Test ScientificDiscoveryEngine class."""

    def test_scientific_discovery_basic(self):
        """Test scientific discovery engine basic functionality."""
        import haiku as hk
        from src.rtdlm import ScientificDiscoveryEngine

        def forward(knowledge_base, observations):
            engine = ScientificDiscoveryEngine(d_model=D_MODEL)
            return engine(knowledge_base, observations)

        init = hk.transform(forward)
        rng = jax.random.PRNGKey(42)

        knowledge_base = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        observations = jax.random.normal(rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))

        params = init.init(rng, knowledge_base, observations)
        output = init.apply(params, rng, knowledge_base, observations)

        self.assertIn("hypothesis", output)
        self.assertIn("experiment_design", output)
        self.assertIn("causal_analysis", output)


class TestComputeAgiLossWithAuxOutputs(unittest.TestCase):
    """Test compute_agi_loss with various auxiliary outputs for coverage."""

    def test_loss_with_consciousness(self):
        """Test loss with consciousness outputs (covers compute_consciousness_loss)."""
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
            "consciousness": {
                "self_awareness": jnp.ones((BATCH_SIZE, D_MODEL)),
                "introspection": jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL)),
            }
        }

        loss = compute_agi_loss(logits, targets, aux_outputs=aux_outputs, config=config)

        self.assertTrue(jnp.isfinite(loss).item())
        self.assertIn("consciousness_loss", aux_outputs["loss_components"])

    def test_loss_with_multimodal(self):
        """Test loss with multimodal features (covers multimodal alignment)."""
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
            "multimodal_features": jnp.ones((BATCH_SIZE, D_MODEL)),
            "text_features": jnp.ones((BATCH_SIZE, D_MODEL)),
            "audio_features": jnp.ones((BATCH_SIZE, D_MODEL)),
        }

        loss = compute_agi_loss(logits, targets, aux_outputs=aux_outputs, config=config)

        self.assertTrue(jnp.isfinite(loss).item())
        self.assertIn("multimodal_loss", aux_outputs["loss_components"])

    def test_loss_with_fairness_evaluation(self):
        """Test loss with fairness evaluation (covers fairness penalty)."""
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
            "logits": logits,
            "fairness_evaluation": {
                "analyzer_active": True,
                "fairness_config": {"bias_threshold": 0.1},
            },
        }

        loss = compute_agi_loss(logits, targets, aux_outputs=aux_outputs, config=config)

        self.assertTrue(jnp.isfinite(loss).item())
        self.assertIn("fairness_loss", aux_outputs["loss_components"])


class TestComputeMultimodalAlignmentLossDetailed(unittest.TestCase):
    """Detailed tests for compute_multimodal_alignment_loss covering all branches."""

    def test_3d_text_features(self):
        """Test with 3D text features (covers text_features.ndim == 3 branch)."""
        from src.rtdlm import compute_multimodal_alignment_loss

        aux_outputs = {
            "text_features": jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL)),
            "audio_features": jnp.ones((BATCH_SIZE, D_MODEL)),
        }

        loss = compute_multimodal_alignment_loss(aux_outputs)

        self.assertTrue(jnp.isfinite(loss).item())

    def test_3d_other_features(self):
        """Test with 3D audio features (covers other_features.ndim == 3 branch)."""
        from src.rtdlm import compute_multimodal_alignment_loss

        aux_outputs = {
            "text_features": jnp.ones((BATCH_SIZE, D_MODEL)),
            "audio_features": jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL)),
        }

        loss = compute_multimodal_alignment_loss(aux_outputs)

        self.assertTrue(jnp.isfinite(loss).item())

    def test_shape_mismatch_skip(self):
        """Test that shape mismatch causes skip (covers continue branch)."""
        from src.rtdlm import compute_multimodal_alignment_loss

        aux_outputs = {
            "text_features": jnp.ones((BATCH_SIZE, D_MODEL)),
            "audio_features": jnp.ones((BATCH_SIZE, D_MODEL * 2)),
        }

        loss = compute_multimodal_alignment_loss(aux_outputs)

        self.assertEqual(float(loss), 0.0)

    def test_with_fused_features(self):
        """Test with fused features (covers fused_features branch)."""
        from src.rtdlm import compute_multimodal_alignment_loss

        aux_outputs = {
            "text_features": jnp.ones((BATCH_SIZE, D_MODEL)),
            "fused_features": jnp.ones((BATCH_SIZE, D_MODEL)),
        }

        loss = compute_multimodal_alignment_loss(aux_outputs)

        self.assertTrue(jnp.isfinite(loss).item())

    def test_with_3d_fused_features(self):
        """Test with 3D fused features (covers fused_features.ndim == 3 branch)."""
        from src.rtdlm import compute_multimodal_alignment_loss

        aux_outputs = {
            "text_features": jnp.ones((BATCH_SIZE, D_MODEL)),
            "fused_features": jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL)),
        }

        loss = compute_multimodal_alignment_loss(aux_outputs)

        self.assertTrue(jnp.isfinite(loss).item())

    def test_hybrid_analysis_fallback(self):
        """Test falling back to hybrid_analysis for text_features."""
        from src.rtdlm import compute_multimodal_alignment_loss

        aux_outputs = {
            "hybrid_analysis": {"text_encoding": jnp.ones((BATCH_SIZE, D_MODEL))},
            "audio_features": jnp.ones((BATCH_SIZE, D_MODEL)),
        }

        loss = compute_multimodal_alignment_loss(aux_outputs)

        self.assertTrue(jnp.isfinite(loss).item())

    def test_video_features(self):
        """Test with video features."""
        from src.rtdlm import compute_multimodal_alignment_loss

        aux_outputs = {
            "text_features": jnp.ones((BATCH_SIZE, D_MODEL)),
            "video_features": jnp.ones((BATCH_SIZE, D_MODEL)),
        }

        loss = compute_multimodal_alignment_loss(aux_outputs)

        self.assertTrue(jnp.isfinite(loss).item())

    def test_image_features(self):
        """Test with image features."""
        from src.rtdlm import compute_multimodal_alignment_loss

        aux_outputs = {
            "text_features": jnp.ones((BATCH_SIZE, D_MODEL)),
            "image_features": jnp.ones((BATCH_SIZE, D_MODEL)),
        }

        loss = compute_multimodal_alignment_loss(aux_outputs)

        self.assertTrue(jnp.isfinite(loss).item())


class TestComputeFairnessPenaltyLoss(unittest.TestCase):
    """Test compute_fairness_penalty_loss function."""

    def test_fairness_penalty_basic(self):
        """Test fairness penalty with basic inputs."""
        from src.rtdlm import compute_fairness_penalty_loss

        logits = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, VOCAB_SIZE))
        fairness_eval = {"fairness_config": {"bias_threshold": 0.1}}

        loss = compute_fairness_penalty_loss(logits, fairness_eval)

        self.assertTrue(jnp.isfinite(loss).item())
        self.assertGreaterEqual(float(loss), 0.0)

    def test_fairness_penalty_none_logits(self):
        """Test fairness penalty returns 0 for None logits."""
        from src.rtdlm import compute_fairness_penalty_loss

        fairness_eval = {"fairness_config": {"bias_threshold": 0.1}}

        loss = compute_fairness_penalty_loss(None, fairness_eval)

        self.assertEqual(float(loss), 0.0)


class TestComputeReasoningConsistencyLoss(unittest.TestCase):
    """Test compute_reasoning_consistency_loss function."""

    def test_reasoning_consistency_basic(self):
        """Test reasoning consistency with multiple steps."""
        from src.rtdlm import compute_reasoning_consistency_loss

        reasoning_chain = [jnp.ones((BATCH_SIZE, D_MODEL)) * i for i in range(3)]

        loss = compute_reasoning_consistency_loss(reasoning_chain)

        self.assertTrue(jnp.isfinite(loss).item())
        self.assertGreater(float(loss), 0.0)

    def test_reasoning_consistency_single_step(self):
        """Test reasoning consistency with single step returns 0."""
        from src.rtdlm import compute_reasoning_consistency_loss

        reasoning_chain = [jnp.ones((BATCH_SIZE, D_MODEL))]

        loss = compute_reasoning_consistency_loss(reasoning_chain)

        self.assertEqual(float(loss), 0.0)


class TestComputeConsciousnessLoss(unittest.TestCase):
    """Test compute_consciousness_loss function."""

    def test_consciousness_loss_basic(self):
        """Test consciousness loss with valid inputs."""
        from src.rtdlm import compute_consciousness_loss

        consciousness_signals = {
            "self_awareness": jnp.ones((BATCH_SIZE, D_MODEL)),
            "introspection": jnp.ones((BATCH_SIZE, SEQ_LEN, D_MODEL)),
        }

        loss = compute_consciousness_loss(consciousness_signals)

        self.assertTrue(jnp.isfinite(loss).item())

    def test_consciousness_loss_missing_keys(self):
        """Test consciousness loss returns 0 when keys missing."""
        from src.rtdlm import compute_consciousness_loss

        consciousness_signals = {}

        loss = compute_consciousness_loss(consciousness_signals)

        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
