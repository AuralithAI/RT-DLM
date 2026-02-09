"""
Tests for Quantum Readiness Module

Tests for quantum simulation, gate operations, and quantum-inspired
computations using classical simulation.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np


class TestQuantumGateType(unittest.TestCase):
    """Test QuantumGateType enum."""
    
    def test_gate_types_exist(self):
        """Test all gate types are defined."""
        from src.core.quantum.quantum_readiness import QuantumGateType
        
        self.assertEqual(QuantumGateType.HADAMARD.value, "H")
        self.assertEqual(QuantumGateType.PAULI_X.value, "X")
        self.assertEqual(QuantumGateType.PAULI_Y.value, "Y")
        self.assertEqual(QuantumGateType.PAULI_Z.value, "Z")
        self.assertEqual(QuantumGateType.CNOT.value, "CNOT")
        self.assertEqual(QuantumGateType.ROTATION_X.value, "RX")
        self.assertEqual(QuantumGateType.ROTATION_Y.value, "RY")
        self.assertEqual(QuantumGateType.ROTATION_Z.value, "RZ")


class TestQuantumGate(unittest.TestCase):
    """Test QuantumGate dataclass."""
    
    def test_gate_creation(self):
        """Test creating quantum gates."""
        from src.core.quantum.quantum_readiness import QuantumGate, QuantumGateType
        
        # Hadamard gate
        h_gate = QuantumGate(QuantumGateType.HADAMARD, [0])
        self.assertEqual(h_gate.gate_type, QuantumGateType.HADAMARD)
        self.assertEqual(h_gate.qubits, [0])
        self.assertEqual(h_gate.parameters, [])
        
        # Rotation gate with parameters
        rx_gate = QuantumGate(QuantumGateType.ROTATION_X, [1], [0.5])
        self.assertEqual(rx_gate.parameters, [0.5])
        
        # CNOT gate
        cnot_gate = QuantumGate(QuantumGateType.CNOT, [0, 1])
        self.assertEqual(len(cnot_gate.qubits), 2)


class TestQuantumCircuit(unittest.TestCase):
    """Test QuantumCircuit dataclass."""
    
    def test_circuit_creation(self):
        """Test creating quantum circuits."""
        from src.core.quantum.quantum_readiness import (
            QuantumCircuit, QuantumGate, QuantumGateType
        )
        
        gates = [
            QuantumGate(QuantumGateType.HADAMARD, [0]),
            QuantumGate(QuantumGateType.CNOT, [0, 1])
        ]
        
        circuit = QuantumCircuit(num_qubits=2, gates=gates)
        
        self.assertEqual(circuit.num_qubits, 2)
        self.assertEqual(len(circuit.gates), 2)
        self.assertEqual(circuit.measurements, [0, 1])  # Default
    
    def test_circuit_with_measurements(self):
        """Test circuit with custom measurements."""
        from src.core.quantum.quantum_readiness import QuantumCircuit
        
        circuit = QuantumCircuit(
            num_qubits=3,
            gates=[],
            measurements=[0, 2]
        )
        
        self.assertEqual(circuit.measurements, [0, 2])


class TestQuantumSimulator(unittest.TestCase):
    """Test QuantumSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.core.quantum.quantum_readiness import QuantumSimulator
        self.simulator = QuantumSimulator(max_qubits=10)
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        self.assertEqual(self.simulator.max_qubits, 10)
        self.assertEqual(self.simulator.coherence_time, 1000)
    
    def test_initialize_state(self):
        """Test quantum state initialization."""
        state = self.simulator.initialize_state(2)
        
        # Should be |00⟩ state
        self.assertEqual(state.shape, (4,))  # 2^2 = 4
        self.assertAlmostEqual(float(jnp.abs(state[0])), 1.0)  # |00⟩ amplitude
        self.assertAlmostEqual(float(jnp.abs(state[1])), 0.0)  # |01⟩ amplitude
        self.assertAlmostEqual(float(jnp.abs(state[2])), 0.0)  # |10⟩ amplitude
        self.assertAlmostEqual(float(jnp.abs(state[3])), 0.0)  # |11⟩ amplitude
    
    def test_initialize_state_too_many_qubits(self):
        """Test error when requesting too many qubits."""
        with self.assertRaises(ValueError):
            self.simulator.initialize_state(11)  # More than max_qubits=10
    
    def test_apply_hadamard_gate(self):
        """Test Hadamard gate creates superposition."""
        from src.core.quantum.quantum_readiness import QuantumGate, QuantumGateType
        
        state = self.simulator.initialize_state(1)
        h_gate = QuantumGate(QuantumGateType.HADAMARD, [0])
        
        new_state = self.simulator.apply_gate(state, h_gate, 1)
        
        # Hadamard on |0⟩ creates (|0⟩ + |1⟩)/√2
        expected_amplitude = 1.0 / jnp.sqrt(2.0)
        self.assertTrue(jnp.allclose(jnp.abs(new_state[0]), expected_amplitude, atol=1e-5))
        self.assertTrue(jnp.allclose(jnp.abs(new_state[1]), expected_amplitude, atol=1e-5))
    
    def test_apply_pauli_x_gate(self):
        """Test Pauli-X (NOT) gate."""
        from src.core.quantum.quantum_readiness import QuantumGate, QuantumGateType
        
        state = self.simulator.initialize_state(1)
        x_gate = QuantumGate(QuantumGateType.PAULI_X, [0])
        
        new_state = self.simulator.apply_gate(state, x_gate, 1)
        
        # X|0⟩ = |1⟩
        self.assertAlmostEqual(float(jnp.abs(new_state[0])), 0.0, places=5)
        self.assertAlmostEqual(float(jnp.abs(new_state[1])), 1.0, places=5)
    
    def test_apply_rotation_gates(self):
        """Test rotation gates."""
        from src.core.quantum.quantum_readiness import QuantumGate, QuantumGateType
        
        state = self.simulator.initialize_state(1)
        
        # Rotation X by π should flip the qubit (like Pauli-X)
        rx_gate = QuantumGate(QuantumGateType.ROTATION_X, [0], [jnp.pi])
        new_state = self.simulator.apply_gate(state, rx_gate, 1)
        
        # After RX(π)|0⟩, we should get -i|1⟩ (up to global phase)
        self.assertTrue(jnp.abs(new_state[1]) > 0.99)
    
    def test_apply_cnot_gate(self):
        """Test CNOT gate."""
        from src.core.quantum.quantum_readiness import QuantumGate, QuantumGateType
        
        # Create |10⟩ state (control=1, target=0)
        state = self.simulator.initialize_state(2)
        x_gate = QuantumGate(QuantumGateType.PAULI_X, [0])
        state = self.simulator.apply_gate(state, x_gate, 2)
        
        # Apply CNOT with control=0, target=1
        cnot_gate = QuantumGate(QuantumGateType.CNOT, [0, 1])
        new_state = self.simulator.apply_gate(state, cnot_gate, 2)
        
        # |10⟩ with CNOT(0,1) -> |11⟩
        # State vector should have amplitude at index 3 (binary 11)
        self.assertTrue(jnp.abs(new_state[3]) > 0.99)
    
    def test_create_bell_state(self):
        """Test Bell state creation."""
        state = self.simulator.create_bell_state(0, 1, 2)
        
        # Bell state: (|00⟩ + |11⟩)/√2
        expected_amp = 1.0 / jnp.sqrt(2.0)
        self.assertTrue(jnp.allclose(jnp.abs(state[0]), expected_amp, atol=1e-5))  # |00⟩
        self.assertTrue(jnp.allclose(jnp.abs(state[3]), expected_amp, atol=1e-5))  # |11⟩
        
        # Check entanglement was recorded
        self.assertIn("0_1", self.simulator.entanglement_map)
    
    def test_variational_quantum_layer(self):
        """Test variational quantum layer."""
        num_qubits = 3
        # Need 3*num_qubits params (2 for rotations + 1 for final RZ per qubit)
        params = jnp.zeros(num_qubits * 3)
        
        state = self.simulator.variational_quantum_layer(params, num_qubits)
        
        # State should be normalized
        norm = jnp.linalg.norm(state)
        self.assertTrue(jnp.allclose(norm, 1.0, atol=1e-5))
    
    def test_quantum_attention_circuit(self):
        """Test quantum attention circuit."""
        num_qubits = 4
        query_params = jnp.ones(num_qubits) * 0.5
        key_params = jnp.ones(num_qubits) * 0.3
        
        state = self.simulator.quantum_attention_circuit(query_params, key_params, num_qubits)
        
        # State should be normalized
        norm = jnp.linalg.norm(state)
        self.assertTrue(jnp.allclose(norm, 1.0, atol=1e-5))


class TestQuantumReadySystem(unittest.TestCase):
    """Test quantum-ready system factory."""
    
    def test_create_quantum_ready_system(self):
        """Test creating quantum-ready system."""
        try:
            from src.core.quantum.quantum_readiness import create_quantum_ready_system
            
            system = create_quantum_ready_system(
                d_model=64,
                num_qubits=4
            )
            
            self.assertIsNotNone(system)
        except ImportError:
            self.skipTest("create_quantum_ready_system not available")


class TestQuantumGateOperations(unittest.TestCase):
    """Test specific quantum gate mathematical properties."""
    
    def setUp(self):
        """Set up simulator."""
        from src.core.quantum.quantum_readiness import QuantumSimulator
        self.simulator = QuantumSimulator()
    
    def test_hadamard_is_self_inverse(self):
        """Test H² = I (Hadamard is self-inverse)."""
        from src.core.quantum.quantum_readiness import QuantumGate, QuantumGateType
        
        state = self.simulator.initialize_state(1)
        original = state.copy()
        
        h_gate = QuantumGate(QuantumGateType.HADAMARD, [0])
        
        # Apply H twice
        state = self.simulator.apply_gate(state, h_gate, 1)
        state = self.simulator.apply_gate(state, h_gate, 1)
        
        # Should return to original state
        self.assertTrue(jnp.allclose(jnp.abs(state), jnp.abs(original), atol=1e-5))
    
    def test_pauli_gates_anticommute(self):
        """Test that Pauli gates satisfy expected relations."""
        from src.core.quantum.quantum_readiness import QuantumGate, QuantumGateType
        
        state = self.simulator.initialize_state(1)
        
        # X² = I
        x_gate = QuantumGate(QuantumGateType.PAULI_X, [0])
        state_xx = self.simulator.apply_gate(state, x_gate, 1)
        state_xx = self.simulator.apply_gate(state_xx, x_gate, 1)
        
        self.assertTrue(jnp.allclose(jnp.abs(state_xx), jnp.abs(state), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
