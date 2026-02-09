"""
Tests for Module Adapters

Tests for adapters that wrap RT-DLM modules for the ComputeController.
"""

import unittest
import jax
import jax.numpy as jnp
import haiku as hk


class TestModuleType(unittest.TestCase):
    """Test ModuleType enumeration."""
    
    def test_module_type_enum_values(self):
        """Test ModuleType enum has expected values."""
        from src.core.agi.compute_controller import ModuleType
        
        # Check that required module types exist
        self.assertIsNotNone(ModuleType.MEMORY_RETRIEVAL)
        self.assertIsNotNone(ModuleType.GRAPH_REASONING)
        self.assertIsNotNone(ModuleType.SYMBOLIC_REASONING)
        self.assertIsNotNone(ModuleType.PROBABILISTIC)
        self.assertIsNotNone(ModuleType.QUANTUM_SIMULATION)
        self.assertIsNotNone(ModuleType.MOE_ROUTING)
        self.assertIsNotNone(ModuleType.ATTENTION_REFINEMENT)
        self.assertIsNotNone(ModuleType.OUTPUT_GENERATION)
    
    def test_module_type_comparison(self):
        """Test ModuleType enum comparison."""
        from src.core.agi.compute_controller import ModuleType
        
        module_type = ModuleType.MEMORY_RETRIEVAL
        self.assertEqual(module_type, ModuleType.MEMORY_RETRIEVAL)
        self.assertNotEqual(module_type, ModuleType.GRAPH_REASONING)


class TestModuleContract(unittest.TestCase):
    """Test ModuleContract dataclass."""
    
    def test_module_contract_creation(self):
        """Test creating ModuleContract."""
        from src.core.agi.compute_controller import ModuleContract, ModuleType
        
        contract = ModuleContract(
            module_type=ModuleType.MEMORY_RETRIEVAL,
            name="memory_adapter",
            base_cost=0.1
        )
        
        self.assertEqual(contract.name, "memory_adapter")
        self.assertEqual(contract.base_cost, 0.1)
        self.assertEqual(contract.module_type, ModuleType.MEMORY_RETRIEVAL)
        self.assertTrue(contract.skippable)
    
    def test_module_contract_with_dependencies(self):
        """Test ModuleContract with dependencies."""
        from src.core.agi.compute_controller import ModuleContract, ModuleType
        
        contract = ModuleContract(
            module_type=ModuleType.OUTPUT_GENERATION,
            name="output_adapter",
            base_cost=0.2,
            dependencies=[ModuleType.ATTENTION_REFINEMENT],
            max_calls=1
        )
        
        self.assertEqual(contract.max_calls, 1)
        self.assertIn(ModuleType.ATTENTION_REFINEMENT, contract.dependencies)


class TestModuleOutput(unittest.TestCase):
    """Test ModuleOutput dataclass."""
    
    def test_module_output_creation(self):
        """Test creating ModuleOutput."""
        from src.core.agi.compute_controller import ModuleOutput
        
        output = ModuleOutput(
            hidden_delta=jnp.zeros((1, 64)),
            confidence=jnp.array([[0.9]]),
            uncertainty=jnp.array([[0.1]]),
            actual_cost=0.15
        )
        
        self.assertEqual(output.hidden_delta.shape, (1, 64))
        self.assertEqual(output.actual_cost, 0.15)
    
    def test_module_output_with_evidence(self):
        """Test ModuleOutput with evidence."""
        from src.core.agi.compute_controller import ModuleOutput
        
        evidence_dict = {"source": "memory", "relevance": 0.9}
        output = ModuleOutput(
            hidden_delta=jnp.zeros((1, 64)),
            confidence=jnp.array([[0.85]]),
            uncertainty=jnp.array([[0.15]]),
            actual_cost=0.2,
            evidence=evidence_dict,
            suggests_halt=False,
            recommended_next=None
        )
        
        self.assertIsNotNone(output.evidence)
        if output.evidence is not None:
            self.assertEqual(output.evidence["source"], "memory")
        self.assertFalse(output.suggests_halt)


class TestComputeState(unittest.TestCase):
    """Test ComputeState NamedTuple."""
    
    def test_compute_state_creation(self):
        """Test creating ComputeState."""
        from src.core.agi.compute_controller import ComputeState
        
        state = ComputeState(
            hidden=jnp.zeros((1, 10, 64)),
            hidden_pooled=jnp.zeros((1, 64)),
            memory_summary=jnp.zeros((1, 64)),
            uncertainty=jnp.array([[0.5]]),
            confidence=jnp.array([[0.5]]),
            budget_remaining=1.0,
            step=0,
            modules_called=[],
            module_outputs=[]
        )
        
        self.assertEqual(state.hidden.shape, (1, 10, 64))
        self.assertEqual(state.hidden_pooled.shape, (1, 64))
        self.assertEqual(state.step, 0)
        self.assertEqual(state.budget_remaining, 1.0)


class TestModuleAdapter(unittest.TestCase):
    """Test ModuleAdapter base class."""
    
    def test_module_adapter_initialization(self):
        """Test ModuleAdapter initialization."""
        from src.core.agi.module_adapters import ModuleAdapter
        from src.core.agi.compute_controller import ModuleType
        
        def init_adapter():
            adapter = ModuleAdapter(
                d_model=64,
                module_type=ModuleType.MEMORY_RETRIEVAL,
                name="test_adapter"
            )
            return adapter
        
        # Initialize in Haiku context
        init_fn = hk.transform(lambda: init_adapter())
        rng = jax.random.PRNGKey(42)
        params = init_fn.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_adapter_confidence_estimator(self):
        """Test confidence estimator network."""
        from src.core.agi.module_adapters import ModuleAdapter
        from src.core.agi.compute_controller import ModuleType
        
        def test_fn():
            adapter = ModuleAdapter(
                d_model=64,
                module_type=ModuleType.ATTENTION_REFINEMENT
            )
            output = jnp.ones((1, 10, 64))
            confidence = adapter._compute_confidence(output)
            return confidence
        
        init_fn = hk.transform(test_fn)
        rng = jax.random.PRNGKey(42)
        params = init_fn.init(rng)
        result = init_fn.apply(params, rng)
        
        # Confidence should be between 0 and 1 (sigmoid output)
        self.assertTrue(jnp.all(result >= 0))
        self.assertTrue(jnp.all(result <= 1))
    
    def test_adapter_uncertainty_estimator(self):
        """Test uncertainty estimator network."""
        from src.core.agi.module_adapters import ModuleAdapter
        from src.core.agi.compute_controller import ModuleType
        
        def test_fn():
            adapter = ModuleAdapter(
                d_model=64,
                module_type=ModuleType.MEMORY_RETRIEVAL
            )
            output = jnp.ones((1, 10, 64))
            input_state = jnp.zeros((1, 10, 64))
            uncertainty = adapter._compute_uncertainty(output, input_state)
            return uncertainty
        
        init_fn = hk.transform(test_fn)
        rng = jax.random.PRNGKey(42)
        params = init_fn.init(rng)
        result = init_fn.apply(params, rng)
        
        # Uncertainty should be between 0 and 1
        self.assertTrue(jnp.all(result >= 0))
        self.assertTrue(jnp.all(result <= 1))
    
    def test_delta_projector_shape(self):
        """Test delta projector maintains shape."""
        from src.core.agi.module_adapters import ModuleAdapter
        from src.core.agi.compute_controller import ModuleType
        
        def test_fn():
            adapter = ModuleAdapter(
                d_model=64,
                module_type=ModuleType.ATTENTION_REFINEMENT
            )
            input_data = jnp.ones((1, 10, 64))
            delta = adapter.delta_projector(input_data)
            return delta
        
        init_fn = hk.transform(test_fn)
        rng = jax.random.PRNGKey(42)
        params = init_fn.init(rng)
        result = init_fn.apply(params, rng)
        
        # Delta should have same d_model dimension
        self.assertEqual(result.shape[-1], 64)


class TestHaltSuggestion(unittest.TestCase):
    """Test halt suggestion network."""
    
    def test_halt_suggester_output(self):
        """Test halt suggester produces valid output."""
        from src.core.agi.module_adapters import ModuleAdapter
        from src.core.agi.compute_controller import ModuleType
        
        def test_fn():
            adapter = ModuleAdapter(
                d_model=64,
                module_type=ModuleType.ATTENTION_REFINEMENT
            )
            input_data = jnp.ones((1, 64))
            halt = adapter.halt_suggester(input_data)
            return halt
        
        init_fn = hk.transform(test_fn)
        rng = jax.random.PRNGKey(42)
        params = init_fn.init(rng)
        result = init_fn.apply(params, rng)
        
        # Halt suggestion should be between 0 and 1
        self.assertTrue(jnp.all(result >= 0))
        self.assertTrue(jnp.all(result <= 1))


class TestAdapterIntegration(unittest.TestCase):
    """Test adapter integration with controller."""
    
    def test_adapter_output_format(self):
        """Test adapter produces correct output format."""
        from src.core.agi.compute_controller import ModuleOutput
        
        # Create a ModuleOutput with correct fields
        output = ModuleOutput(
            hidden_delta=jnp.zeros((1, 64)),
            confidence=jnp.array([[0.8]]),
            uncertainty=jnp.array([[0.2]]),
            actual_cost=0.1
        )
        
        # Verify structure
        self.assertTrue(hasattr(output, 'hidden_delta'))
        self.assertTrue(hasattr(output, 'confidence'))
        self.assertTrue(hasattr(output, 'uncertainty'))
        self.assertTrue(hasattr(output, 'actual_cost'))


if __name__ == "__main__":
    unittest.main()
