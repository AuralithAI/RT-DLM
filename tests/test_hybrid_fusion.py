"""
Tests for Hybrid Architecture Fusion

Tests the hybrid integrator's ability to combine traditional ML,
deep learning, symbolic reasoning, and probabilistic approaches.
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import haiku as hk


class TestHybridFusion:
    """Test suite for hybrid architecture fusion accuracy"""
    
    @pytest.fixture
    def rng_key(self):
        """Create a random key for tests"""
        return jax.random.PRNGKey(42)
    
    @pytest.fixture
    def d_model(self):
        """Model dimension"""
        return 64
    
    def test_ensemble_fusion_initialization(self, rng_key, d_model):
        """Test EnsembleFusion module initializes correctly"""
        from modules.hybrid_architecture.hybrid_integrator import EnsembleFusion
        
        def forward(model_outputs, weights):
            fusion = EnsembleFusion(d_model)
            return fusion(model_outputs, weights)
        
        model = hk.transform(forward)
        
        # Create sample inputs: 4 model outputs
        batch_size = 2
        num_models = 4
        model_outputs = [
            jnp.ones((batch_size, d_model)) * (i + 1) 
            for i in range(num_models)
        ]
        weights = jnp.ones((batch_size, num_models)) / num_models
        
        # Initialize
        params = model.init(rng_key, model_outputs, weights)
        assert params is not None
        
        # Forward pass
        output = model.apply(params, rng_key, model_outputs, weights)
        assert output.shape == (batch_size, d_model)
    
    def test_cross_paradigm_interaction(self, rng_key, d_model):
        """Test that cross-paradigm interaction (outer product) is computed"""
        from modules.hybrid_architecture.hybrid_integrator import EnsembleFusion
        
        def forward(model_outputs, weights):
            fusion = EnsembleFusion(d_model)
            return fusion(model_outputs, weights)
        
        model = hk.transform(forward)
        
        batch_size = 2
        num_models = 4
        
        # Create distinct outputs for each paradigm
        statistical = jnp.array([[1.0] * d_model, [2.0] * d_model])
        deep = jnp.array([[0.5] * d_model, [0.25] * d_model])
        symbolic = jnp.ones((batch_size, d_model)) * 0.3
        probabilistic = jnp.ones((batch_size, d_model)) * 0.4
        
        model_outputs = [statistical, deep, symbolic, probabilistic]
        weights = jnp.ones((batch_size, num_models)) / num_models
        
        params = model.init(rng_key, model_outputs, weights)
        output = model.apply(params, rng_key, model_outputs, weights)
        
        # Output should have non-trivial values due to interaction
        assert not jnp.allclose(output, jnp.zeros_like(output))
        assert output.shape == (batch_size, d_model)
    
    def test_fusion_with_sequence_input(self, rng_key, d_model):
        """Test fusion with 3D sequence inputs"""
        from modules.hybrid_architecture.hybrid_integrator import EnsembleFusion
        
        def forward(model_outputs, weights):
            fusion = EnsembleFusion(d_model)
            return fusion(model_outputs, weights)
        
        model = hk.transform(forward)
        
        batch_size = 2
        seq_len = 8
        num_models = 4
        
        # 3D inputs: [batch, seq, d_model]
        model_outputs = [
            jnp.ones((batch_size, seq_len, d_model)) * (i + 1) 
            for i in range(num_models)
        ]
        weights = jnp.ones((batch_size, num_models)) / num_models
        
        params = model.init(rng_key, model_outputs, weights)
        output = model.apply(params, rng_key, model_outputs, weights)
        
        # Output shape depends on implementation (could be pooled or kept as sequence)
        assert output is not None
        assert len(output.shape) >= 2
    
    def test_rbf_kernel_svm(self, rng_key, d_model):
        """Test RBF kernel in SVM-like classifier"""
        from modules.hybrid_architecture.hybrid_integrator import SVMLikeClassifier
        
        def forward(x):
            svm = SVMLikeClassifier(d_model, num_support_vectors=16, gamma=0.1)
            return svm(x)
        
        model = hk.transform(forward)
        
        batch_size = 4
        inputs = jax.random.normal(rng_key, (batch_size, d_model))
        
        params = model.init(rng_key, inputs)
        output = model.apply(params, rng_key, inputs)
        
        assert output.shape == (batch_size, d_model)
        # RBF kernel should produce smooth outputs (no extreme values)
        assert jnp.all(jnp.isfinite(output))
    
    def test_cnn_branch_vectorized(self, rng_key, d_model):
        """Test vectorized CNN branch processing"""
        from modules.hybrid_architecture.hybrid_integrator import CNNBranch
        
        def forward(x):
            cnn = CNNBranch(d_model)
            return cnn(x)
        
        model = hk.transform(forward)
        
        batch_size = 4
        seq_len = 16
        inputs = jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        
        params = model.init(rng_key, inputs)
        output = model.apply(params, rng_key, inputs)
        
        assert output.shape == (batch_size, d_model)
        assert jnp.all(jnp.isfinite(output))
    
    def test_full_hybrid_integrator(self, rng_key, d_model):
        """Test complete hybrid architecture integrator"""
        from modules.hybrid_architecture.hybrid_integrator import HybridArchitectureIntegrator
        
        def forward(inputs, task_type=None):
            integrator = HybridArchitectureIntegrator(d_model)
            return integrator(inputs, task_type)
        
        model = hk.transform(forward)
        
        batch_size = 2
        seq_len = 8
        inputs = {
            'text': jax.random.normal(rng_key, (batch_size, seq_len, d_model))
        }
        
        params = model.init(rng_key, inputs)
        output = model.apply(params, rng_key, inputs)
        
        # Should return dictionary with all outputs
        assert isinstance(output, dict)
        assert 'ensemble_output' in output
        assert 'traditional_ml' in output
        assert 'deep_learning' in output
        assert 'symbolic_reasoning' in output
        assert 'probabilistic' in output
        assert 'approach_weights' in output
        assert 'confidence' in output
    
    def test_fusion_deterministic(self, rng_key, d_model):
        """Test that fusion is deterministic with same inputs"""
        from modules.hybrid_architecture.hybrid_integrator import EnsembleFusion
        
        def forward(model_outputs, weights):
            fusion = EnsembleFusion(d_model)
            return fusion(model_outputs, weights)
        
        model = hk.transform(forward)
        
        batch_size = 2
        num_models = 4
        model_outputs = [jnp.ones((batch_size, d_model)) for _ in range(num_models)]
        weights = jnp.ones((batch_size, num_models)) / num_models
        
        params = model.init(rng_key, model_outputs, weights)
        
        # Run twice with same inputs
        output1 = model.apply(params, rng_key, model_outputs, weights)
        output2 = model.apply(params, rng_key, model_outputs, weights)
        
        assert jnp.allclose(output1, output2)


def test_ensemble_fusion_basic():
    """Basic test for ensemble fusion without fixtures"""
    from modules.hybrid_architecture.hybrid_integrator import EnsembleFusion
    
    def forward(model_outputs, weights):
        fusion = EnsembleFusion(d_model=32)
        return fusion(model_outputs, weights)
    
    model = hk.transform(forward)
    rng = jax.random.PRNGKey(0)
    
    batch_size = 2
    d_model = 32
    model_outputs = [jnp.ones((batch_size, d_model)) for _ in range(4)]
    weights = jnp.ones((batch_size, 4)) / 4
    
    params = model.init(rng, model_outputs, weights)
    output = model.apply(params, rng, model_outputs, weights)
    
    assert output.shape == (batch_size, d_model)
    print("[PASS] test_ensemble_fusion_basic")


def test_multi_agent_consensus():
    """Test multi-agent system with consensus loop"""
    from modules.hybrid_architecture.hybrid_integrator import MultiAgentConsensus
    
    def forward(inputs):
        mac = MultiAgentConsensus(d_model=32, num_agents=4)
        # Disable auto-spawning to get exactly 4 agents
        return mac(inputs, auto_spawn=False)
    
    model = hk.transform(forward)
    rng = jax.random.PRNGKey(0)
    
    batch_size = 2
    d_model = 32
    inputs = jax.random.normal(rng, (batch_size, d_model))
    
    params = model.init(rng, inputs)
    output = model.apply(params, rng, inputs)
    
    # Check output structure
    assert isinstance(output, dict)
    assert 'consensus' in output
    assert 'agent_responses' in output
    assert 'agent_confidences' in output
    assert 'confidence_weights' in output
    
    # Check shapes
    assert output['consensus'].shape == (batch_size, d_model)
    assert len(output['agent_responses']) == 4
    assert len(output['agent_confidences']) == 4
    
    print("[PASS] test_multi_agent_consensus")


def test_specialist_agent():
    """Test individual specialist agent"""
    from modules.hybrid_architecture.hybrid_integrator import SpecialistAgent
    
    def forward(inputs):
        agent = SpecialistAgent(d_model=32, specialization="reasoning")
        return agent.process(inputs)
    
    model = hk.transform(forward)
    rng = jax.random.PRNGKey(0)
    
    batch_size = 2
    d_model = 32
    inputs = jax.random.normal(rng, (batch_size, d_model))
    
    params = model.init(rng, inputs)
    output = model.apply(params, rng, inputs)
    
    assert 'response' in output
    assert 'confidence' in output
    assert output['response'].shape == (batch_size, d_model)
    
    print("[PASS] test_specialist_agent")


if __name__ == "__main__":
    # Run basic tests
    test_ensemble_fusion_basic()
    test_multi_agent_consensus()
    test_specialist_agent()
    
    # Run pytest tests
    pytest.main([__file__, "-v"])

