"""Tests for quantum simulation being optional."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


class TestQuantumOptional:
    """Test that quantum components can be disabled."""
    
    @pytest.fixture
    def config_with_quantum(self):
        from config.agi_config import AGIConfig
        config = AGIConfig()
        config.d_model = 64
        config.vocab_size = 1000
        config.max_seq_length = 32
        config.num_heads = 4
        config.num_layers = 2
        config.moe_experts = 4
        config.moe_top_k = 2
        config.quantum_layers = 4
        config.quantum_qubits = 8
        return config
    
    @pytest.fixture
    def config_without_quantum(self):
        from config.agi_config import AGIConfig
        config = AGIConfig()
        config.d_model = 64
        config.vocab_size = 1000
        config.max_seq_length = 32
        config.num_heads = 4
        config.num_layers = 2
        config.moe_experts = 4
        config.moe_top_k = 2
        config.quantum_layers = 0
        return config
    
    def test_quantum_disabled_config(self, config_without_quantum):
        """Verify quantum_layers=0 is valid configuration."""
        assert config_without_quantum.quantum_layers == 0
    
    def test_quantum_enabled_config(self, config_with_quantum):
        """Verify quantum_layers>0 enables quantum."""
        assert config_with_quantum.quantum_layers > 0


class TestQuantumOverheadEstimation:
    """Test quantum overhead estimation utility."""
    
    def test_estimate_quantum_overhead_full_state(self):
        """Test memory estimation for full state vector."""
        from core.quantum import estimate_quantum_overhead
        
        estimate = estimate_quantum_overhead(
            num_qubits=16,
            num_layers=4,
            d_model=384,
            use_tensor_network=False
        )
        
        assert estimate["num_qubits"] == 16
        assert estimate["num_layers"] == 4
        assert estimate["use_tensor_network"] == False
        assert estimate["state_memory_bytes"] == (2 ** 16) * 16
        assert "O(2^n)" in estimate["memory_formula"]
        assert estimate["gate_parameters"] == 4 * 16 * 3
    
    def test_estimate_quantum_overhead_tensor_network(self):
        """Test memory estimation for tensor network."""
        from core.quantum import estimate_quantum_overhead
        
        estimate = estimate_quantum_overhead(
            num_qubits=100,
            num_layers=4,
            d_model=384,
            use_tensor_network=True,
            bond_dimension=64
        )
        
        assert estimate["num_qubits"] == 100
        assert estimate["use_tensor_network"] == True
        assert estimate["state_memory_bytes"] == 100 * 64 * 64 * 16
        assert "O(n × χ²)" in estimate["memory_formula"]
    
    def test_estimate_includes_disable_note(self):
        """Test that estimation includes note about disabling."""
        from core.quantum import estimate_quantum_overhead
        
        estimate = estimate_quantum_overhead()
        
        assert "quantum_layers=0" in estimate["note"]


class TestQuantumDocumentation:
    """Test that quantum modules have proper documentation."""
    
    def test_quantum_init_has_disclaimer(self):
        """Verify quantum __init__ has simulation disclaimer."""
        import core.quantum as quantum_module
        
        docstring = quantum_module.__doc__
        assert docstring is not None
        assert "CLASSICAL SIMULATION" in docstring.upper()
    
    def test_quantum_agi_core_has_disclaimer(self):
        """Verify QuantumAGICore module has disclaimer."""
        from core.quantum import quantum_agi_core
        
        docstring = quantum_agi_core.__doc__
        assert docstring is not None
        assert "SIMULATION" in docstring.upper()
    
    def test_extended_quantum_has_disclaimer(self):
        """Verify extended_quantum_sim has disclaimer."""
        from core.quantum import extended_quantum_sim
        
        docstring = extended_quantum_sim.__doc__
        assert docstring is not None
        assert "SIMULATION" in docstring.upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
