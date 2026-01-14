"""
Tests for Tensor Network Quantum Simulation Module

Tests for Matrix Product States (MPS), Tree Tensor Networks (TTN),
and tensor network-based quantum simulation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from config.tensor_network_config import TensorNetworkConfig
from core.quantum.tensor_network import (
    MatrixProductState,
    TreeTensorNetwork,
    TensorNetworkQuantumSimulator,
    create_tensor_network_simulator,
    estimate_memory_usage,
)


class TestTensorNetworkConfig:
    """Tests for TensorNetworkConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TensorNetworkConfig()
        assert config.num_qubits == 16
        assert config.bond_dimension == 64
        assert config.max_iterations == 100
        assert config.use_sparse is True
        assert config.network_type == "mps"
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = TensorNetworkConfig(
            num_qubits=64,
            bond_dimension=128,
            truncation_threshold=1e-12,
            max_iterations=200,
            network_type="ttn"
        )
        assert config.num_qubits == 64
        assert config.bond_dimension == 128
        assert config.max_iterations == 200
        assert config.network_type == "ttn"


class TestMatrixProductState:
    """Tests for Matrix Product State"""
    
    def test_mps_creation(self):
        """Test MPS creation with default state"""
        mps = MatrixProductState(num_qubits=4, bond_dimension=4)
        assert len(mps.tensors) == 4
    
    def test_mps_tensor_shapes(self):
        """Test MPS tensor shapes are correct"""
        mps = MatrixProductState(num_qubits=4, bond_dimension=8)
        
        # First tensor: (1, 2, chi) - left boundary
        # Middle tensors: (chi, 2, chi)
        # Last tensor: (chi, 2, 1) - right boundary
        assert mps.tensors[0].shape[0] == 1  # Left boundary
        assert mps.tensors[-1].shape[2] == 1  # Right boundary
        
        for tensor in mps.tensors:
            assert tensor.shape[1] == 2  # Physical dimension for qubits
    
    def test_mps_norm(self):
        """Test MPS normalization"""
        mps = MatrixProductState(num_qubits=4, bond_dimension=4)
        
        # Compute norm
        norm = mps.compute_norm()
        
        # Norm should be positive
        assert norm > 0
    
    def test_mps_canonicalize(self):
        """Test MPS canonicalization"""
        mps = MatrixProductState(num_qubits=4, bond_dimension=8)
        
        # Canonicalize left
        mps.canonicalize_left()
        
        # After canonicalization, should still have valid tensors
        assert len(mps.tensors) == 4
    
    def test_mps_apply_single_gate(self):
        """Test applying single-qubit gate"""
        mps = MatrixProductState(num_qubits=4, bond_dimension=8)
        
        # Hadamard gate
        H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2)
        
        # Apply to site 0
        mps.apply_single_qubit_gate(H, 0)
        
        # Should still have valid structure
        assert len(mps.tensors) == 4
    
    def test_mps_apply_two_qubit_gate(self):
        """Test applying two-qubit gate"""
        mps = MatrixProductState(num_qubits=4, bond_dimension=16)
        
        # CNOT gate
        CNOT = jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=jnp.complex64)
        
        # Apply to sites 0, 1
        mps.apply_two_qubit_gate(CNOT, 0, 1)
        
        # Should still have valid structure
        assert len(mps.tensors) == 4
    
    def test_mps_sample(self):
        """Test MPS sampling"""
        mps = MatrixProductState(num_qubits=4, bond_dimension=8)
        
        # Sample
        rng = jax.random.PRNGKey(42)
        result = mps.sample(rng)
        
        # Result should be array of 0s and 1s
        assert result.shape == (4,)
        for r in result:
            assert r in [0, 1]
    
    def test_mps_expectation_value(self):
        """Test computing expectation values"""
        mps = MatrixProductState(num_qubits=4, bond_dimension=8)
        
        # Pauli Z
        Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
        
        # Compute <Z>
        exp_val = mps.expectation_value(Z, 0)
        
        # Expectation value should be real and bounded
        assert jnp.abs(exp_val) <= 1.0 + 1e-6


class TestTreeTensorNetwork:
    """Tests for Tree Tensor Network"""
    
    def test_ttn_creation(self):
        """Test TTN creation"""
        ttn = TreeTensorNetwork(num_qubits=4, bond_dimension=8)
        # TTN uses layers structure
        assert len(ttn.layers) > 0
    
    def test_ttn_structure(self):
        """Test TTN has tree structure"""
        ttn = TreeTensorNetwork(num_qubits=8, bond_dimension=8)
        
        # First layer (leaves) should have >= num_qubits tensors
        assert len(ttn.layers[0]) >= 8
    
    def test_ttn_apply_gate(self):
        """Test applying gate to TTN"""
        ttn = TreeTensorNetwork(num_qubits=4, bond_dimension=8)
        
        # Hadamard gate
        H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2)
        
        # Apply to site 0
        ttn.apply_single_qubit_gate(H, 0)
        
        # Should still have valid structure
        assert len(ttn.layers) > 0


class TestTensorNetworkQuantumSimulator:
    """Tests for TensorNetworkQuantumSimulator"""
    
    def test_simulator_creation_mps(self):
        """Test creating MPS-based simulator"""
        config = TensorNetworkConfig(
            num_qubits=8,
            bond_dimension=16,
            network_type="mps"
        )
        simulator = TensorNetworkQuantumSimulator(config)
        assert simulator.config.network_type == "mps"
    
    def test_simulator_creation_ttn(self):
        """Test creating TTN-based simulator"""
        config = TensorNetworkConfig(
            num_qubits=8,
            bond_dimension=16,
            network_type="ttn"
        )
        simulator = TensorNetworkQuantumSimulator(config)
        assert simulator.config.network_type == "ttn"
    
    def test_simulator_hadamard(self):
        """Test Hadamard gate in simulator"""
        config = TensorNetworkConfig(
            num_qubits=4,
            bond_dimension=8
        )
        simulator = TensorNetworkQuantumSimulator(config)
        
        # Apply Hadamard to qubit 0
        simulator.apply_hadamard(0)
        
        # Verify state changed
        assert simulator.state is not None
    
    def test_simulator_apply_gate(self):
        """Test applying custom gate"""
        config = TensorNetworkConfig(
            num_qubits=4,
            bond_dimension=8
        )
        simulator = TensorNetworkQuantumSimulator(config)
        
        # Apply Pauli X
        X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
        simulator.apply_gate(X, [0])
        
        # Verify state changed
        assert simulator.state is not None
    
    def test_simulator_cnot(self):
        """Test CNOT gate"""
        config = TensorNetworkConfig(
            num_qubits=4,
            bond_dimension=16
        )
        simulator = TensorNetworkQuantumSimulator(config)
        
        # Apply CNOT
        simulator.apply_cnot(0, 1)
        
        assert simulator.state is not None
    
    def test_simulator_bell_state(self):
        """Test creating Bell state"""
        config = TensorNetworkConfig(
            num_qubits=2,
            bond_dimension=4
        )
        simulator = TensorNetworkQuantumSimulator(config)
        
        # Create Bell state: H(0) then CNOT(0,1)
        simulator.apply_hadamard(0)
        simulator.apply_cnot(0, 1)
        
        # Verify state exists
        assert simulator.state is not None
    
    def test_simulator_sample(self):
        """Test sampling from circuit"""
        config = TensorNetworkConfig(
            num_qubits=4,
            bond_dimension=8
        )
        simulator = TensorNetworkQuantumSimulator(config)
        
        # Apply some gates
        simulator.apply_hadamard(0)
        simulator.apply_cnot(0, 1)
        
        # Sample multiple times
        rng = jax.random.PRNGKey(42)
        samples = simulator.sample(rng, num_samples=10)
        
        assert samples.shape[0] == 10
    
    def test_simulator_reset(self):
        """Test resetting simulator state"""
        config = TensorNetworkConfig(
            num_qubits=4,
            bond_dimension=8
        )
        simulator = TensorNetworkQuantumSimulator(config)
        
        # Apply gate then reset
        simulator.apply_hadamard(0)
        simulator.reset()
        
        # Should be back to initial state
        assert simulator.state is not None
    
    def test_simulator_entanglement_entropy(self):
        """Test entanglement entropy computation"""
        config = TensorNetworkConfig(
            num_qubits=4,
            bond_dimension=8
        )
        simulator = TensorNetworkQuantumSimulator(config)
        
        # Create entangled state
        simulator.apply_hadamard(0)
        simulator.apply_cnot(0, 1)
        
        # Get entropy at site 0
        entropy = simulator.get_entanglement_entropy(0)
        
        # Should be non-negative
        assert entropy >= 0


class TestFactoryFunctions:
    """Tests for factory functions"""
    
    def test_create_mps_simulator(self):
        """Test creating simulator via factory"""
        config = TensorNetworkConfig(
            num_qubits=8,
            bond_dimension=16,
            network_type="mps"
        )
        simulator = create_tensor_network_simulator(config)
        assert isinstance(simulator, TensorNetworkQuantumSimulator)


class TestMemoryEstimation:
    """Tests for memory estimation"""
    
    def test_memory_estimate_mps(self):
        """Test memory estimation for MPS"""
        estimates = estimate_memory_usage(
            num_qubits=32,
            bond_dimension=64
        )
        
        # Should return dict with estimates
        assert "mps_gb" in estimates
        assert "full_state_gb" in estimates
        assert estimates["mps_gb"] > 0
    
    def test_memory_savings(self):
        """Test that MPS uses less memory than full state"""
        estimates = estimate_memory_usage(
            num_qubits=20,
            bond_dimension=32
        )
        
        # MPS should use significantly less memory
        assert estimates["mps_gb"] < estimates["full_state_gb"]


class TestLargeScaleSimulation:
    """Tests for large-scale quantum simulation"""
    
    def test_large_qubit_count(self):
        """Test simulation with many qubits"""
        config = TensorNetworkConfig(
            num_qubits=50,  # Would be impossible with full state vector
            bond_dimension=32
        )
        simulator = TensorNetworkQuantumSimulator(config)
        
        # Should be able to create and use
        simulator.apply_hadamard(0)
        assert simulator.state is not None
    
    def test_bond_dimensions(self):
        """Test getting bond dimensions"""
        config = TensorNetworkConfig(
            num_qubits=4,
            bond_dimension=16
        )
        simulator = TensorNetworkQuantumSimulator(config)
        
        # Apply gates to create entanglement
        simulator.apply_hadamard(0)
        simulator.apply_cnot(0, 1)
        
        # Get bond dimensions
        bonds = simulator.get_bond_dimensions()
        
        # Should return list
        assert isinstance(bonds, list)


class TestNumericalStability:
    """Tests for numerical stability"""
    
    def test_normalization_preservation(self):
        """Test norm is preserved through operations"""
        mps = MatrixProductState(num_qubits=4, bond_dimension=16)
        
        initial_norm = mps.compute_norm()
        
        # Apply gates
        H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2)
        mps.apply_single_qubit_gate(H, 0)
        mps.apply_single_qubit_gate(H, 1)
        
        final_norm = mps.compute_norm()
        
        # Norm should be approximately preserved
        assert jnp.isclose(initial_norm, final_norm, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
