"""
Tests for Quantum AGI Core Module

Tests for quantum-inspired attention and optimization patterns
using classical simulation of quantum concepts.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk


class TestQuantumInspiredAttention(unittest.TestCase):
    """Test QuantumInspiredAttention class."""
    
    def test_attention_initialization(self):
        """Test QuantumInspiredAttention initialization."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def init_fn():
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            x = jnp.ones((1, 10, 64))
            return attention(x)
        
        init = hk.transform(init_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_attention_forward_pass(self):
        """Test attention forward pass."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def forward_fn(x):
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            return attention(x)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 10, 64))
        
        params = init.init(rng, x)
        output = init.apply(params, rng, x)
        
        # Output should have same shape as input
        self.assertEqual(output.shape, x.shape)
    
    def test_hadamard_gate_initialization(self):
        """Test Hadamard gate initialization."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def init_fn():
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            # Access the hadamard gate parameter
            return attention.hadamard_gate
        
        init = hk.transform(init_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        hadamard = init.apply(params, rng)
        
        # Hadamard gate should be square matrix
        self.assertEqual(hadamard.shape[0], hadamard.shape[1])
    
    def test_rotation_gates(self):
        """Test rotation gates are initialized."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def init_fn():
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            return attention.rotation_gates
        
        init = hk.transform(init_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        rotations = init.apply(params, rng)
        
        # Should have shape [num_heads, num_qubits, 3]
        self.assertEqual(rotations.shape[0], 4)  # num_heads
        self.assertEqual(rotations.shape[-1], 3)  # X, Y, Z rotations


class TestQuantumGates(unittest.TestCase):
    """Test quantum gate simulations."""
    
    def test_apply_quantum_gates(self):
        """Test quantum gate application."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def test_fn():
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            # Create quantum state
            quantum_state = jnp.ones(8) / jnp.sqrt(8)
            result = attention.apply_quantum_gates(quantum_state, head_idx=0)
            return result
        
        init = hk.transform(test_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        result = init.apply(params, rng)
        
        # Result should be real-valued
        self.assertTrue(jnp.all(jnp.isreal(result)))


class TestQuantumEntanglement(unittest.TestCase):
    """Test quantum entanglement simulation."""
    
    def test_entanglement_operation(self):
        """Test quantum entanglement between states."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def test_fn():
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            q_states = jnp.ones((2, 8)) / jnp.sqrt(8)
            k_states = jnp.ones((2, 8)) / jnp.sqrt(8)
            entangled = attention.quantum_entanglement(q_states, k_states)
            return entangled
        
        init = hk.transform(test_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        result = init.apply(params, rng)
        
        # Entangled state should be normalized
        norms = jnp.linalg.norm(result, axis=-1)
        self.assertTrue(jnp.allclose(norms, 1.0, atol=1e-5))


class TestQuantumStatePreparation(unittest.TestCase):
    """Test quantum state preparation."""
    
    def test_state_prep_network(self):
        """Test state preparation network."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def test_fn(x):
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            # State prep should produce num_qubits * 2 outputs
            return attention.state_prep(x)
        
        init = hk.transform(test_fn)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 64))
        
        params = init.init(rng, x)
        result = init.apply(params, rng, x)
        
        # Output should be num_qubits * 2 (real and imaginary)
        self.assertEqual(result.shape[-1], 16)  # 8 * 2


class TestQueryKeyValueProjections(unittest.TestCase):
    """Test QKV projections."""
    
    def test_query_projection(self):
        """Test query projection."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def test_fn(x):
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            return attention.query_proj(x)
        
        init = hk.transform(test_fn)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 10, 64))
        
        params = init.init(rng, x)
        result = init.apply(params, rng, x)
        
        self.assertEqual(result.shape, (2, 10, 64))
    
    def test_key_projection(self):
        """Test key projection."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def test_fn(x):
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            return attention.key_proj(x)
        
        init = hk.transform(test_fn)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 10, 64))
        
        params = init.init(rng, x)
        result = init.apply(params, rng, x)
        
        self.assertEqual(result.shape, (2, 10, 64))
    
    def test_value_projection(self):
        """Test value projection."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def test_fn(x):
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            return attention.value_proj(x)
        
        init = hk.transform(test_fn)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 10, 64))
        
        params = init.init(rng, x)
        result = init.apply(params, rng, x)
        
        self.assertEqual(result.shape, (2, 10, 64))


class TestOutputProjection(unittest.TestCase):
    """Test output projection."""
    
    def test_output_projection(self):
        """Test output projection maintains d_model."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def test_fn(x):
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            return attention.output_proj(x)
        
        init = hk.transform(test_fn)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 10, 64))
        
        params = init.init(rng, x)
        result = init.apply(params, rng, x)
        
        self.assertEqual(result.shape[-1], 64)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of quantum operations."""
    
    def test_no_nan_in_output(self):
        """Test that outputs don't contain NaN."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def forward_fn(x):
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            return attention(x)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 10, 64))
        
        params = init.init(rng, x)
        output = init.apply(params, rng, x)
        
        self.assertFalse(jnp.any(jnp.isnan(output)))
    
    def test_no_inf_in_output(self):
        """Test that outputs don't contain Inf."""
        from src.core.quantum.quantum_agi_core import QuantumInspiredAttention
        
        def forward_fn(x):
            attention = QuantumInspiredAttention(
                d_model=64,
                num_heads=4,
                num_qubits=8
            )
            return attention(x)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 10, 64))
        
        params = init.init(rng, x)
        output = init.apply(params, rng, x)
        
        self.assertFalse(jnp.any(jnp.isinf(output)))


if __name__ == "__main__":
    unittest.main()
