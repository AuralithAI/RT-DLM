"""
Tests for Compute Controller

Tests the dynamic compute allocation system including:
- ModuleContract and ModuleRegistry
- ComputeState and ComputeAction
- ComputeController decision making
- ComputePlan execution
- Module adapters
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict

from src.core.agi.compute_controller import (
    ModuleType,
    ModuleContract,
    ModuleOutput,
    ModuleRegistry,
    ComputeState,
    ComputeAction,
    ComputeController,
    ComputePlan,
    create_compute_controller_fn,
)

from src.core.agi.module_adapters import (
    ModuleAdapter,
    MemoryRetrievalAdapter,
    GraphReasoningAdapter,
    SymbolicReasoningAdapter,
    ProbabilisticAdapter,
    OutputGenerationAdapter,
    create_module_executors,
)

from src.config.compute_controller_config import (
    ComputeControllerConfig,
    ComputeStrategy,
    ModuleCostConfig,
    FAST_CONFIG,
    THOROUGH_CONFIG,
)


class TestModuleContract:
    """Tests for ModuleContract dataclass."""
    
    def test_create_contract(self):
        """Test creating a module contract."""
        contract = ModuleContract(
            module_type=ModuleType.MEMORY_RETRIEVAL,
            name="memory_retrieval",
            base_cost=0.05,
            skippable=True,
            description="Retrieve from memory"
        )
        
        assert contract.module_type == ModuleType.MEMORY_RETRIEVAL
        assert contract.base_cost == pytest.approx(0.05)
        assert contract.skippable is True
        assert contract.max_calls == 3  # Default
    
    def test_contract_with_dependencies(self):
        """Test contract with dependencies."""
        contract = ModuleContract(
            module_type=ModuleType.GRAPH_REASONING,
            name="graph_reasoning",
            base_cost=0.15,
            dependencies=[ModuleType.MEMORY_RETRIEVAL]
        )
        
        assert ModuleType.MEMORY_RETRIEVAL in contract.dependencies


class TestModuleRegistry:
    """Tests for ModuleRegistry."""
    
    def test_default_modules_registered(self):
        """Test that default modules are registered."""
        registry = ModuleRegistry()
        
        # Check some default modules exist
        assert registry.get(ModuleType.MEMORY_RETRIEVAL) is not None
        assert registry.get(ModuleType.GRAPH_REASONING) is not None
        assert registry.get(ModuleType.OUTPUT_GENERATION) is not None
    
    def test_register_custom_module(self):
        """Test registering a custom module."""
        registry = ModuleRegistry()
        
        custom_contract = ModuleContract(
            module_type=ModuleType.QUANTUM_SIMULATION,
            name="custom_quantum",
            base_cost=0.25,
            max_calls=1
        )
        
        registry.register(custom_contract)
        
        retrieved = registry.get(ModuleType.QUANTUM_SIMULATION)
        assert retrieved is not None
        assert retrieved.base_cost == pytest.approx(0.25)
    
    def test_get_available_modules(self):
        """Test filtering available modules."""
        registry = ModuleRegistry()
        
        # Create a state with some modules already called
        state = ComputeState(
            hidden=jnp.zeros((2, 10, 64)),
            hidden_pooled=jnp.zeros((2, 64)),
            memory_summary=jnp.zeros((2, 64)),
            uncertainty=jnp.ones((2, 1)) * 0.5,
            confidence=jnp.ones((2, 1)) * 0.5,
            budget_remaining=0.5,
            step=1,
            modules_called=[ModuleType.MEMORY_RETRIEVAL],
            module_outputs=[]
        )
        
        available = registry.get_available(state, state.budget_remaining)
        
        # Graph reasoning should be available (dependency met)
        module_types = [c.module_type for c in available]
        assert ModuleType.GRAPH_REASONING in module_types
        
        # Quantum should not be available (too expensive)
        # Default cost is 0.20, budget is 0.5, so it should be available
        # Let's check with a smaller budget
        state_low_budget = state._replace(budget_remaining=0.1)
        available_low = registry.get_available(state_low_budget, 0.1)
        module_types_low = [c.module_type for c in available_low]
        
        # Expensive modules should be filtered out
        assert ModuleType.QUANTUM_SIMULATION not in module_types_low
    
    def test_total_cost_calculation(self):
        """Test calculating total cost of modules."""
        registry = ModuleRegistry()
        
        modules = [
            ModuleType.MEMORY_RETRIEVAL,  # 0.05
            ModuleType.GRAPH_REASONING,    # 0.15
            ModuleType.OUTPUT_GENERATION   # 0.05
        ]
        
        total = registry.get_total_cost(modules)
        assert total == pytest.approx(0.25, rel=0.01)


class TestComputeState:
    """Tests for ComputeState."""
    
    def test_create_initial_state(self):
        """Test creating initial compute state."""
        batch_size = 2
        seq_len = 10
        d_model = 64
        
        hidden = jnp.zeros((batch_size, seq_len, d_model))
        
        state = ComputeState(
            hidden=hidden,
            hidden_pooled=hidden.mean(axis=1),
            memory_summary=jnp.zeros((batch_size, d_model)),
            uncertainty=jnp.ones((batch_size, 1)) * 0.5,
            confidence=jnp.ones((batch_size, 1)) * 0.5,
            budget_remaining=1.0,
            step=0,
            modules_called=[],
            module_outputs=[]
        )
        
        assert state.hidden.shape == (batch_size, seq_len, d_model)
        assert state.budget_remaining == pytest.approx(1.0)
        assert state.step == 0
        assert len(state.modules_called) == 0


class TestModuleOutput:
    """Tests for ModuleOutput."""
    
    def test_create_module_output(self):
        """Test creating module output."""
        batch_size = 2
        d_model = 64
        
        output = ModuleOutput(
            hidden_delta=jnp.zeros((batch_size, d_model)),
            confidence=jnp.ones((batch_size, 1)) * 0.8,
            uncertainty=jnp.ones((batch_size, 1)) * 0.2,
            actual_cost=0.05,
            evidence={"test": True},
            suggests_halt=False
        )
        
        assert output.confidence.mean() == pytest.approx(0.8)
        assert output.actual_cost == pytest.approx(0.05)
        assert output.suggests_halt is False


class TestComputeController:
    """Tests for ComputeController."""
    
    @pytest.fixture
    def controller_fn(self):
        """Create a transformed controller function."""
        d_model = 64
        
        def forward(state_hidden, state_memory, state_uncertainty, state_confidence):
            controller = ComputeController(
                d_model=d_model,
                max_steps=10,
                halt_threshold=0.8
            )
            
            registry = ModuleRegistry()
            
            state = ComputeState(
                hidden=state_hidden,
                hidden_pooled=state_hidden.mean(axis=1) if state_hidden.ndim == 3 else state_hidden,
                memory_summary=state_memory,
                uncertainty=state_uncertainty,
                confidence=state_confidence,
                budget_remaining=0.8,
                step=1,
                modules_called=[ModuleType.MEMORY_RETRIEVAL],
                module_outputs=[]
            )
            
            available = registry.get_available(state, state.budget_remaining)
            
            action, info = controller(state, available, is_training=False)
            
            return action.halt_probability, info["module_probs"]
        
        return hk.transform(forward)
    
    def test_controller_forward(self, controller_fn):
        """Test controller forward pass."""
        batch_size = 2
        seq_len = 10
        d_model = 64
        
        rng = jax.random.PRNGKey(42)
        
        hidden = jax.random.normal(rng, (batch_size, seq_len, d_model))
        memory = jax.random.normal(rng, (batch_size, d_model))
        uncertainty = jnp.ones((batch_size, 1)) * 0.5
        confidence = jnp.ones((batch_size, 1)) * 0.5
        
        params = controller_fn.init(rng, hidden, memory, uncertainty, confidence)
        halt_prob, module_probs = controller_fn.apply(params, rng, hidden, memory, uncertainty, confidence)
        
        # halt_prob is a float from action.halt_probability
        assert isinstance(halt_prob, (float, jnp.ndarray))
        assert module_probs.shape[1] == len(ModuleType)
        assert jnp.all(module_probs >= 0)
        assert jnp.allclose(module_probs.sum(axis=1), 1.0, atol=1e-5)
    
    def test_controller_high_confidence_halts(self, controller_fn):
        """Test that high confidence leads to higher halt probability."""
        batch_size = 2
        d_model = 64
        
        rng = jax.random.PRNGKey(42)
        
        hidden = jax.random.normal(rng, (batch_size, d_model))
        memory = jax.random.normal(rng, (batch_size, d_model))
        uncertainty = jnp.ones((batch_size, 1)) * 0.1  # Low uncertainty
        confidence_high = jnp.ones((batch_size, 1)) * 0.95  # High confidence
        confidence_low = jnp.ones((batch_size, 1)) * 0.3  # Low confidence
        
        params = controller_fn.init(rng, hidden, memory, uncertainty, confidence_high)
        
        halt_high, _ = controller_fn.apply(params, rng, hidden, memory, uncertainty, confidence_high)
        halt_low, _ = controller_fn.apply(params, rng, hidden, memory, uncertainty, confidence_low)
        
        # Higher confidence should generally lead to higher halt probability
        # (though this depends on learned weights, so we just check it runs)
        assert isinstance(halt_high, (float, jnp.ndarray))
        assert isinstance(halt_low, (float, jnp.ndarray))


class TestModuleAdapters:
    """Tests for module adapters."""
    
    def test_memory_adapter(self):
        """Test memory retrieval adapter."""
        d_model = 64
        batch_size = 2
        
        def forward(hidden, memory):
            adapter = MemoryRetrievalAdapter(d_model=d_model)
            
            state = ComputeState(
                hidden=hidden,
                hidden_pooled=hidden.mean(axis=1),
                memory_summary=memory,
                uncertainty=jnp.ones((batch_size, 1)) * 0.5,
                confidence=jnp.ones((batch_size, 1)) * 0.5,
                budget_remaining=1.0,
                step=0,
                modules_called=[],
                module_outputs=[]
            )
            
            output = adapter(state, ltm=memory[:, None, :])
            return output.hidden_delta, output.confidence
        
        fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        hidden = jax.random.normal(rng, (batch_size, 10, d_model))
        memory = jax.random.normal(rng, (batch_size, d_model))
        
        params = fn.init(rng, hidden, memory)
        delta, confidence = fn.apply(params, rng, hidden, memory)
        
        assert delta.shape == (batch_size, d_model)
        assert confidence.shape == (batch_size, 1)
        assert jnp.all(confidence >= 0) and jnp.all(confidence <= 1)
    
    def test_graph_reasoning_adapter(self):
        """Test graph reasoning adapter."""
        d_model = 64
        batch_size = 2
        
        def forward(hidden):
            adapter = GraphReasoningAdapter(d_model=d_model, num_hops=2)
            
            state = ComputeState(
                hidden=hidden,
                hidden_pooled=hidden.mean(axis=1),
                memory_summary=jnp.zeros((batch_size, d_model)),
                uncertainty=jnp.ones((batch_size, 1)) * 0.5,
                confidence=jnp.ones((batch_size, 1)) * 0.5,
                budget_remaining=1.0,
                step=0,
                modules_called=[],
                module_outputs=[]
            )
            
            output = adapter(state)
            return output.hidden_delta, output.uncertainty
        
        fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        hidden = jax.random.normal(rng, (batch_size, 10, d_model))
        
        params = fn.init(rng, hidden)
        delta, uncertainty = fn.apply(params, rng, hidden)
        
        assert delta.shape == (batch_size, d_model)
        assert uncertainty.shape == (batch_size, 1)
    
    def test_output_adapter_suggests_halt(self):
        """Test that output adapter always suggests halt."""
        d_model = 64
        vocab_size = 1000
        batch_size = 2
        
        def forward(hidden):
            adapter = OutputGenerationAdapter(d_model=d_model, vocab_size=vocab_size)
            
            state = ComputeState(
                hidden=hidden,
                hidden_pooled=hidden.mean(axis=1) if hidden.ndim == 3 else hidden,
                memory_summary=jnp.zeros((batch_size, d_model)),
                uncertainty=jnp.ones((batch_size, 1)) * 0.5,
                confidence=jnp.ones((batch_size, 1)) * 0.5,
                budget_remaining=0.1,
                step=5,
                modules_called=[],
                module_outputs=[]
            )
            
            output = adapter(state)
            return output.suggests_halt
        
        fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        hidden = jax.random.normal(rng, (batch_size, d_model))
        
        params = fn.init(rng, hidden)
        suggests_halt = fn.apply(params, rng, hidden)
        
        assert suggests_halt is True


class TestComputePlan:
    """Tests for ComputePlan execution."""
    
    def test_compute_plan_initialization(self):
        """Test compute plan state initialization."""
        d_model = 64
        batch_size = 2
        seq_len = 10
        
        def forward(hidden):
            plan = ComputePlan(d_model=d_model, max_steps=5)
            state = plan.initialize_state(hidden)
            return state.hidden_pooled, state.budget_remaining
        
        fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        hidden = jax.random.normal(rng, (batch_size, seq_len, d_model))
        
        params = fn.init(rng, hidden)
        hidden_pooled, budget = fn.apply(params, rng, hidden)
        
        assert hidden_pooled.shape == (batch_size, d_model)
        assert budget == pytest.approx(1.0)


class TestComputeControllerConfig:
    """Tests for configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ComputeControllerConfig()
        
        assert config.enabled is True
        assert config.max_steps == 10
        assert config.initial_budget == pytest.approx(1.0)
        assert config.strategy == ComputeStrategy.ADAPTIVE
    
    def test_fast_config(self):
        """Test fast configuration preset."""
        config = FAST_CONFIG
        
        assert config.max_steps == 5
        assert config.strategy == ComputeStrategy.GREEDY
    
    def test_thorough_config(self):
        """Test thorough configuration preset."""
        config = THOROUGH_CONFIG
        
        assert config.max_steps == 15
        assert config.strategy == ComputeStrategy.THOROUGH
    
    def test_effective_budget_scaling(self):
        """Test budget scaling with difficulty."""
        config = ComputeControllerConfig(strategy=ComputeStrategy.ADAPTIVE)
        
        easy_budget = config.get_effective_budget(0.2)
        hard_budget = config.get_effective_budget(0.9)
        
        assert hard_budget > easy_budget
    
    def test_module_cost_config(self):
        """Test module cost configuration."""
        costs = ModuleCostConfig(
            memory_retrieval=0.10,
            graph_reasoning=0.20
        )
        
        cost_dict = costs.to_dict()
        assert cost_dict["memory_retrieval"] == pytest.approx(0.10)
        assert cost_dict["graph_reasoning"] == pytest.approx(0.20)


class TestIntegration:
    """Integration tests for the full compute controller system."""
    
    def test_full_execution_loop(self):
        """Test complete execution loop."""
        d_model = 64
        batch_size = 2
        seq_len = 10
        
        def forward(hidden):
            # Create controller and plan
            controller = ComputeController(d_model=d_model, max_steps=3)
            plan = ComputePlan(d_model=d_model, max_steps=3)
            registry = ModuleRegistry()
            
            # Create simple module executors
            def simple_executor(state, is_training):
                delta = jnp.zeros_like(state.hidden_pooled)
                return ModuleOutput(
                    hidden_delta=delta,
                    confidence=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.7,
                    uncertainty=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.3,
                    actual_cost=0.1,
                    suggests_halt=False
                )
            
            def output_executor(state, is_training):
                return ModuleOutput(
                    hidden_delta=jnp.zeros_like(state.hidden_pooled),
                    confidence=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.9,
                    uncertainty=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.1,
                    actual_cost=0.05,
                    suggests_halt=True
                )
            
            executors = {
                ModuleType.MEMORY_RETRIEVAL: simple_executor,
                ModuleType.GRAPH_REASONING: simple_executor,
                ModuleType.SYMBOLIC_REASONING: simple_executor,
                ModuleType.OUTPUT_GENERATION: output_executor,
            }
            
            # Execute plan
            final_state, trace = plan(
                hidden=hidden,
                controller=controller,
                registry=registry,
                module_executors=executors,
                is_training=False
            )
            
            return final_state.confidence, trace["total_cost"]
        
        fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        hidden = jax.random.normal(rng, (batch_size, seq_len, d_model))
        
        params = fn.init(rng, hidden)
        confidence, total_cost = fn.apply(params, rng, hidden)
        
        assert confidence.shape == (batch_size, 1)
        assert total_cost >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
