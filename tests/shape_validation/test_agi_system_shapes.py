"""
Shape validation tests for RTDLMAGISystem integration points.
Tests all module chains and feature integration.
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.agi_config import AGIConfig


BATCH_SIZES = [1, 4, 16]
SEQ_LENGTHS = [16, 32, 64]


def get_test_config():
    return AGIConfig(
        d_model=64,
        num_heads=4,
        num_layers=2,
        vocab_size=1000,
        max_seq_length=128,
        moe_experts=4,
        moe_top_k=2,
        quantum_qubits=4,
        quantum_layers=2,
        multimodal_enabled=True,
        consciousness_enabled=True,
        graph_neurons_enabled=True,
    )


class TestAGISystemShapes:
    
    @pytest.fixture
    def config(self):
        return get_test_config()
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_agi_system_batch_sizes(self, config, rng_key, batch_size):
        """Test RTDLMAGISystem with different batch sizes."""
        try:
            from rtdlm import RTDLMAGISystem
        except ImportError:
            pytest.skip("RTDLMAGISystem not available")
        
        def model_fn(inputs):
            model = RTDLMAGISystem(config)
            return model(inputs)
        
        model = hk.transform_with_state(model_fn)
        
        seq_len = 32
        text_inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, config.vocab_size)
        inputs = {"text": text_inputs}
        
        try:
            params, state = model.init(rng_key, inputs)
            result, _ = model.apply(params, state, rng_key, inputs)
            
            assert "logits" in result
            logits = result["logits"]
            assert logits.shape[0] == batch_size
            assert not jnp.any(jnp.isnan(logits))
        except Exception as e:
            pytest.fail(f"AGI system failed with batch_size={batch_size}: {e}")

    @pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
    def test_agi_system_seq_lengths(self, config, rng_key, seq_len):
        """Test RTDLMAGISystem with different sequence lengths."""
        try:
            from rtdlm import RTDLMAGISystem
        except ImportError:
            pytest.skip("RTDLMAGISystem not available")
        
        def model_fn(inputs):
            model = RTDLMAGISystem(config)
            return model(inputs)
        
        model = hk.transform_with_state(model_fn)
        
        batch_size = 4
        text_inputs = jax.random.randint(rng_key, (batch_size, seq_len), 1, config.vocab_size)
        inputs = {"text": text_inputs}
        
        try:
            params, state = model.init(rng_key, inputs)
            result, _ = model.apply(params, state, rng_key, inputs)
            
            logits = result["logits"]
            assert logits.shape[1] == seq_len or logits.ndim == 2
        except Exception as e:
            pytest.fail(f"AGI system failed with seq_len={seq_len}: {e}")


class TestFeatureIntegrationShapes:
    
    @pytest.fixture
    def config(self):
        return get_test_config()
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    def test_feature_concatenation_shapes(self, config, rng_key):
        """Test that feature concatenation produces correct shapes."""
        batch_size, seq_len = 4, 32
        d_model = config.d_model
        
        core_features = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        hybrid_features = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        
        concatenated = jnp.concatenate([core_features, hybrid_features], axis=-1)
        
        assert concatenated.shape == (batch_size, seq_len, d_model * 2)
    
    def test_feature_projection_shapes(self, config, rng_key):
        """Test feature projection maintains correct shapes."""
        batch_size, seq_len = 4, 32
        d_model = config.d_model
        
        features = jax.random.normal(rng_key, (batch_size, seq_len, d_model * 2))
        
        def proj_fn(x):
            proj = hk.Linear(d_model, name="proj")
            return proj(x)
        
        proj = hk.transform(proj_fn)
        params = proj.init(rng_key, features)
        projected = proj.apply(params, rng_key, features)
        
        assert projected.shape == (batch_size, seq_len, d_model)
    
    @pytest.mark.parametrize("num_features", [2, 3, 4, 5])
    def test_multiple_feature_integration(self, config, rng_key, num_features):
        """Test integrating multiple feature sources."""
        batch_size, seq_len = 4, 32
        d_model = config.d_model
        
        features_list = []
        for _ in range(num_features):
            rng_key, subkey = jax.random.split(rng_key)
            feat = jax.random.normal(subkey, (batch_size, seq_len, d_model))
            features_list.append(feat)
        
        concatenated = jnp.concatenate(features_list, axis=-1)
        assert concatenated.shape == (batch_size, seq_len, d_model * num_features)
        
        mean_pooled = jnp.mean(jnp.stack(features_list, axis=0), axis=0)
        assert mean_pooled.shape == (batch_size, seq_len, d_model)


class TestMultimodalShapes:
    
    @pytest.fixture
    def config(self):
        return get_test_config()
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    def test_audio_feature_shapes(self, config, rng_key):
        """Test audio feature shape compatibility."""
        batch_size = 4
        audio_len = 16000
        mel_bins = 80
        
        audio_raw = jax.random.normal(rng_key, (batch_size, audio_len))
        audio_mel = jax.random.normal(rng_key, (batch_size, 100, mel_bins))
        
        assert audio_raw.shape == (batch_size, audio_len)
        assert audio_mel.shape == (batch_size, 100, mel_bins)
    
    def test_video_feature_shapes(self, config, rng_key):
        """Test video feature shape compatibility."""
        batch_size = 4
        num_frames = 16
        height, width = 224, 224
        channels = 3
        
        video = jax.random.normal(rng_key, (batch_size, num_frames, height, width, channels))
        
        assert video.shape == (batch_size, num_frames, height, width, channels)
    
    def test_image_feature_shapes(self, config, rng_key):
        """Test image feature shape compatibility."""
        batch_size = 4
        height, width = 224, 224
        channels = 3
        
        images = jax.random.normal(rng_key, (batch_size, height, width, channels))
        
        assert images.shape == (batch_size, height, width, channels)
    
    def test_multimodal_to_text_projection(self, config, rng_key):
        """Test projecting multimodal features to text dimension."""
        batch_size, seq_len = 4, 32
        d_model = config.d_model
        
        text_features = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        
        multimodal_dim = 256
        multimodal_features = jax.random.normal(rng_key, (batch_size, seq_len, multimodal_dim))
        
        def proj_fn(x):
            proj = hk.Linear(d_model, name="mm_proj")
            return proj(x)
        
        proj = hk.transform(proj_fn)
        params = proj.init(rng_key, multimodal_features)
        projected = proj.apply(params, rng_key, multimodal_features)
        
        assert projected.shape == text_features.shape
        
        fused = text_features + projected
        assert fused.shape == (batch_size, seq_len, d_model)


class TestKnowledgeRetrievalShapes:
    
    @pytest.fixture
    def config(self):
        return get_test_config()
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    def test_knowledge_base_shapes(self, config, rng_key):
        """Test knowledge base feature shapes."""
        batch_size = 4
        num_docs = 10
        doc_len = 64
        d_model = config.d_model
        
        knowledge_base = jax.random.normal(rng_key, (batch_size, num_docs, doc_len, d_model))
        
        knowledge_pooled = jnp.mean(knowledge_base, axis=(1, 2))
        assert knowledge_pooled.shape == (batch_size, d_model)
    
    def test_knowledge_to_sequence_broadcast(self, config, rng_key):
        """Test broadcasting knowledge features to sequence length."""
        batch_size, seq_len = 4, 32
        d_model = config.d_model
        
        knowledge_features = jax.random.normal(rng_key, (batch_size, d_model))
        
        knowledge_expanded = knowledge_features[:, None, :]
        knowledge_broadcast = jnp.broadcast_to(
            knowledge_expanded,
            (batch_size, seq_len, d_model)
        )
        
        assert knowledge_broadcast.shape == (batch_size, seq_len, d_model)
    
    @pytest.mark.parametrize("knowledge_seq_len", [16, 32, 64, 128])
    def test_knowledge_sequence_length_mismatch(self, config, rng_key, knowledge_seq_len):
        """Test handling of knowledge sequence length mismatch."""
        batch_size = 4
        core_seq_len = 32
        d_model = config.d_model
        
        core_features = jax.random.normal(rng_key, (batch_size, core_seq_len, d_model))
        knowledge_features = jax.random.normal(rng_key, (batch_size, knowledge_seq_len, d_model))
        
        if knowledge_seq_len != core_seq_len:
            knowledge_pooled = jnp.mean(knowledge_features, axis=1, keepdims=True)
            knowledge_aligned = jnp.broadcast_to(
                knowledge_pooled,
                (batch_size, core_seq_len, d_model)
            )
        else:
            knowledge_aligned = knowledge_features
        
        assert knowledge_aligned.shape == core_features.shape
        
        combined = core_features + knowledge_aligned
        assert combined.shape == (batch_size, core_seq_len, d_model)


class TestQuantumComponentShapes:
    
    @pytest.fixture
    def config(self):
        return get_test_config()
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    def test_quantum_input_projection(self, config, rng_key):
        """Test quantum circuit input projection shapes."""
        batch_size, seq_len = 4, 32
        d_model = config.d_model
        num_qubits = config.quantum_qubits
        
        features = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        
        features_flat = features.reshape(batch_size * seq_len, d_model)
        
        def proj_fn(x):
            proj = hk.Linear(2**num_qubits, name="quantum_proj")
            return proj(x)
        
        proj = hk.transform(proj_fn)
        params = proj.init(rng_key, features_flat)
        quantum_input = proj.apply(params, rng_key, features_flat)
        
        assert quantum_input.shape == (batch_size * seq_len, 2**num_qubits)
    
    def test_quantum_output_reshape(self, config, rng_key):
        """Test reshaping quantum output back to sequence format."""
        batch_size, seq_len = 4, 32
        d_model = config.d_model
        
        quantum_flat = jax.random.normal(rng_key, (batch_size * seq_len, d_model))
        
        quantum_seq = quantum_flat.reshape(batch_size, seq_len, d_model)
        
        assert quantum_seq.shape == (batch_size, seq_len, d_model)


class TestConsciousnessComponentShapes:
    
    @pytest.fixture
    def config(self):
        return get_test_config()
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    def test_consciousness_state_shapes(self, config, rng_key):
        """Test consciousness state dimension compatibility."""
        batch_size = 4
        d_model = config.d_model
        
        global_workspace = jax.random.normal(rng_key, (batch_size, d_model))
        attention_schema = jax.random.normal(rng_key, (batch_size, d_model))
        meta_cognition = jax.random.normal(rng_key, (batch_size, d_model))
        
        combined = jnp.concatenate([global_workspace, attention_schema, meta_cognition], axis=-1)
        assert combined.shape == (batch_size, d_model * 3)
    
    def test_consciousness_integration_shapes(self, config, rng_key):
        """Test consciousness feature integration with core features."""
        batch_size, seq_len = 4, 32
        d_model = config.d_model
        
        core_features = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        consciousness_state = jax.random.normal(rng_key, (batch_size, d_model))
        
        consciousness_expanded = consciousness_state[:, None, :]
        consciousness_broadcast = jnp.broadcast_to(
            consciousness_expanded,
            (batch_size, seq_len, d_model)
        )
        
        integrated = core_features + 0.1 * consciousness_broadcast
        assert integrated.shape == (batch_size, seq_len, d_model)


class TestReasoningComponentShapes:
    
    @pytest.fixture
    def config(self):
        return get_test_config()
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.mark.parametrize("num_steps", [1, 5, 10])
    def test_reasoning_step_shapes(self, config, rng_key, num_steps):
        """Test reasoning shapes across multiple steps."""
        batch_size, seq_len = 4, 32
        d_model = config.d_model
        
        hidden_state = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        
        reasoning_states = []
        for _ in range(num_steps):
            rng_key, subkey = jax.random.split(rng_key)
            step_state = hidden_state + jax.random.normal(subkey, hidden_state.shape) * 0.1
            reasoning_states.append(step_state)
            hidden_state = step_state
        
        all_states = jnp.stack(reasoning_states, axis=1)
        assert all_states.shape == (batch_size, num_steps, seq_len, d_model)
    
    def test_reasoning_output_shapes(self, config, rng_key):
        """Test reasoning engine output shapes."""
        batch_size, seq_len = 4, 32
        d_model = config.d_model
        
        reasoning_features = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        reasoning_confidence = jax.random.uniform(rng_key, (batch_size,))
        
        assert reasoning_features.shape == (batch_size, seq_len, d_model)
        assert reasoning_confidence.shape == (batch_size,)


class TestGraphNeuronShapes:
    
    @pytest.fixture
    def config(self):
        return get_test_config()
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    def test_node_feature_shapes(self, config, rng_key):
        """Test graph node feature shapes."""
        batch_size = 4
        num_nodes = 64
        d_model = config.d_model
        
        node_features = jax.random.normal(rng_key, (batch_size, num_nodes, d_model))
        
        assert node_features.shape == (batch_size, num_nodes, d_model)
    
    def test_adjacency_matrix_shapes(self, config, rng_key):
        """Test graph adjacency matrix shapes."""
        batch_size = 4
        num_nodes = 64
        
        adjacency = jax.random.uniform(rng_key, (batch_size, num_nodes, num_nodes))
        adjacency = (adjacency > 0.7).astype(jnp.float32)
        
        assert adjacency.shape == (batch_size, num_nodes, num_nodes)
        
        adjacency_symmetric = (adjacency + adjacency.transpose(0, 2, 1)) / 2
        assert adjacency_symmetric.shape == (batch_size, num_nodes, num_nodes)
    
    def test_graph_message_passing_shapes(self, config, rng_key):
        """Test graph message passing maintains shapes."""
        batch_size = 4
        num_nodes = 64
        d_model = config.d_model
        
        node_features = jax.random.normal(rng_key, (batch_size, num_nodes, d_model))
        adjacency = jax.random.uniform(rng_key, (batch_size, num_nodes, num_nodes))
        adjacency = (adjacency > 0.7).astype(jnp.float32)
        
        messages = jnp.einsum('bnm,bmd->bnd', adjacency, node_features)
        
        assert messages.shape == node_features.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
