"""
Tests for Extended Quantum Simulation

Tests for:
- Sparse state vector representation
- Chunked quantum simulation
- Extended quantum simulator (64+ qubits)
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))


class TestExtendedQuantumConfig:
    """Tests for extended quantum configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        from src.core.quantum.extended_quantum_sim import ExtendedQuantumConfig
        
        config = ExtendedQuantumConfig()
        assert config.max_qubits == 64
        assert config.chunk_size == 16
        assert config.use_sparse
        
    def test_custom_config(self):
        """Test custom configuration"""
        from src.core.quantum.extended_quantum_sim import ExtendedQuantumConfig
        
        config = ExtendedQuantumConfig(
            max_qubits=128,
            chunk_size=8,
            use_sparse=False
        )
        assert config.max_qubits == 128
        assert config.chunk_size == 8
        assert not config.use_sparse


class TestSparseStateVector:
    """Tests for sparse quantum state representation"""
    
    def test_initialization(self):
        """Test sparse state initialization to |0>"""
        from src.core.quantum.extended_quantum_sim import SparseStateVector
        
        state = SparseStateVector(num_qubits=10)
        assert state.num_qubits == 10
        assert state.state_dim == 1024
        assert state.num_nonzero == 1
        assert 0 in state.amplitudes
        
    def test_sparsity(self):
        """Test sparsity calculation"""
        from src.core.quantum.extended_quantum_sim import SparseStateVector
        
        state = SparseStateVector(num_qubits=10)
        assert state.sparsity > 0.99  # Only 1 of 1024 amplitudes non-zero
        
    def test_to_dense(self):
        """Test conversion to dense representation"""
        from src.core.quantum.extended_quantum_sim import SparseStateVector
        
        state = SparseStateVector(num_qubits=4)
        dense = state.to_dense()
        
        assert dense.shape == (16,)
        assert jnp.abs(dense[0]) == 1.0
        assert jnp.allclose(jnp.sum(jnp.abs(dense[1:])), 0.0)
        
    def test_from_dense(self):
        """Test creation from dense state"""
        from src.core.quantum.extended_quantum_sim import SparseStateVector
        
        dense = jnp.zeros(16, dtype=jnp.complex64)
        dense = dense.at[0].set(1 / jnp.sqrt(2))
        dense = dense.at[15].set(1 / jnp.sqrt(2))
        
        sparse = SparseStateVector.from_dense(dense)
        
        assert sparse.num_nonzero == 2
        assert 0 in sparse.amplitudes
        assert 15 in sparse.amplitudes
        
    def test_normalize(self):
        """Test state normalization"""
        from src.core.quantum.extended_quantum_sim import SparseStateVector
        
        state = SparseStateVector(num_qubits=4)
        state.amplitudes = {0: 2.0, 1: 2.0}
        state.normalize()
        
        probs = state.measure_probabilities()
        total_prob = sum(probs.values())
        assert jnp.isclose(total_prob, 1.0, atol=1e-6)
        
    def test_measure_probabilities(self):
        """Test measurement probability calculation"""
        from src.core.quantum.extended_quantum_sim import SparseStateVector
        
        state = SparseStateVector(num_qubits=4)
        probs = state.measure_probabilities()
        
        assert len(probs) == 1
        assert probs[0] == 1.0


class TestChunkedQuantumSimulator:
    """Tests for chunked quantum simulation"""
    
    def test_initialize_state(self):
        """Test chunked state initialization"""
        from src.core.quantum.extended_quantum_sim import (
            ChunkedQuantumSimulator, ExtendedQuantumConfig
        )
        
        config = ExtendedQuantumConfig(chunk_size=8)
        sim = ChunkedQuantumSimulator(config)
        
        # 20 qubits = 3 chunks of 8, 8, 4
        chunks = sim.initialize_state(20)
        
        assert len(chunks) == 3
        assert chunks[0].shape == (256,)  # 2^8
        assert chunks[1].shape == (256,)  # 2^8
        assert chunks[2].shape == (16,)   # 2^4
        
    def test_apply_single_qubit_gate(self):
        """Test single qubit gate application"""
        from src.core.quantum.extended_quantum_sim import (
            ChunkedQuantumSimulator, ExtendedQuantumConfig
        )
        
        config = ExtendedQuantumConfig(chunk_size=4)
        sim = ChunkedQuantumSimulator(config)
        
        chunks = sim.initialize_state(8)
        
        # Apply Hadamard to qubit 0
        H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2)
        new_chunks = sim.apply_single_qubit_gate(chunks, H, 0, 8)
        
        # After Hadamard, qubit 0 should be in superposition
        assert not jnp.allclose(new_chunks[0], chunks[0])
        
    def test_get_chunk_for_qubit(self):
        """Test qubit to chunk mapping"""
        from src.core.quantum.extended_quantum_sim import (
            ChunkedQuantumSimulator, ExtendedQuantumConfig
        )
        
        config = ExtendedQuantumConfig(chunk_size=8)
        sim = ChunkedQuantumSimulator(config)
        
        # Qubit 0-7 in chunk 0, 8-15 in chunk 1
        chunk_idx, local = sim._get_chunk_for_qubit(5)
        assert chunk_idx == 0
        assert local == 5
        
        chunk_idx, local = sim._get_chunk_for_qubit(10)
        assert chunk_idx == 1
        assert local == 2


class TestExtendedQuantumSimulator:
    """Tests for extended quantum simulator"""
    
    def test_create_small_state_dense(self):
        """Test small state uses dense representation"""
        from src.core.quantum.extended_quantum_sim import ExtendedQuantumSimulator
        
        sim = ExtendedQuantumSimulator()
        state = sim.create_state(8)
        
        assert state["type"] == "dense"
        assert state["num_qubits"] == 8
        
    def test_create_medium_state_sparse(self):
        """Test medium state uses sparse representation"""
        from src.core.quantum.extended_quantum_sim import ExtendedQuantumSimulator
        
        sim = ExtendedQuantumSimulator()
        state = sim.create_state(24)
        
        assert state["type"] == "sparse"
        assert state["num_qubits"] == 24
        
    def test_create_large_state_chunked(self):
        """Test large state uses chunked representation"""
        from src.core.quantum.extended_quantum_sim import ExtendedQuantumSimulator
        
        sim = ExtendedQuantumSimulator()
        state = sim.create_state(48)
        
        assert state["type"] == "chunked"
        assert state["num_qubits"] == 48
        
    def test_apply_hadamard_dense(self):
        """Test Hadamard gate on dense state"""
        from src.core.quantum.extended_quantum_sim import ExtendedQuantumSimulator
        
        sim = ExtendedQuantumSimulator()
        state = sim.create_state(4)
        
        new_state = sim.apply_hadamard(state, 0)
        
        probs = sim.get_probabilities(new_state)
        # After H on |0>, should have equal probability for |0> and |1>
        assert jnp.isclose(probs[0], 0.5, atol=1e-5)
        assert jnp.isclose(probs[1], 0.5, atol=1e-5)
        
    def test_apply_pauli_x(self):
        """Test Pauli-X gate (NOT gate)"""
        from src.core.quantum.extended_quantum_sim import ExtendedQuantumSimulator
        
        sim = ExtendedQuantumSimulator()
        state = sim.create_state(4)
        
        new_state = sim.apply_pauli_x(state, 0)
        
        probs = sim.get_probabilities(new_state)
        # After X on |0000>, should get |0001>
        assert jnp.isclose(probs[1], 1.0, atol=1e-5)
        
    def test_apply_rotation_y(self):
        """Test Y rotation gate"""
        from src.core.quantum.extended_quantum_sim import ExtendedQuantumSimulator
        
        sim = ExtendedQuantumSimulator()
        state = sim.create_state(4)
        
        # Rotate by pi should flip the qubit
        new_state = sim.apply_rotation_y(state, 0, jnp.pi)
        
        probs = sim.get_probabilities(new_state)
        assert jnp.isclose(probs[1], 1.0, atol=1e-5)
        
    def test_variational_layer(self):
        """Test variational quantum layer"""
        from src.core.quantum.extended_quantum_sim import ExtendedQuantumSimulator
        
        sim = ExtendedQuantumSimulator()
        state = sim.create_state(4)
        
        # Random parameters
        params = jnp.array([0.1, 0.2, 0.3] * 4)  # 3 params per qubit
        
        new_state = sim.variational_layer(state, params)
        
        assert new_state["type"] == "dense"
        assert new_state["num_qubits"] == 4


class TestFactoryFunction:
    """Tests for factory function"""
    
    def test_create_from_config(self):
        """Test creating simulator from AGI config"""
        from src.core.quantum.extended_quantum_sim import create_extended_quantum_simulator
        from src.config.agi_config import AGIConfig
        
        config = AGIConfig(
            quantum_max_qubits=64,
            quantum_sparse_mode=True
        )
        
        sim = create_extended_quantum_simulator(config)
        
        assert sim.config.max_qubits == 64
        assert sim.config.use_sparse


class TestQuantumScaling:
    """Tests for quantum simulation scaling"""
    
    def test_40_qubit_state(self):
        """Test creating 40 qubit state (chunked)"""
        from src.core.quantum.extended_quantum_sim import ExtendedQuantumSimulator
        
        sim = ExtendedQuantumSimulator()
        state = sim.create_state(40)
        
        assert state["type"] == "chunked"
        assert len(state["chunks"]) == 3  # ceil(40/16)
        
    def test_64_qubit_state(self):
        """Test creating 64 qubit state (chunked)"""
        from src.core.quantum.extended_quantum_sim import ExtendedQuantumSimulator
        
        sim = ExtendedQuantumSimulator()
        state = sim.create_state(64)
        
        assert state["type"] == "chunked"
        assert state["num_qubits"] == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
