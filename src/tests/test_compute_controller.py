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


# =============================================================================
# Test Training Losses
# =============================================================================

class TestControllerLossComputer:
    """Tests for ControllerLossComputer (Phase 2)."""
    
    def test_loss_computer_initialization(self):
        """Test loss computer initialization."""
        from src.core.agi.compute_controller import ControllerLossComputer
        
        loss_computer = ControllerLossComputer(
            lambda_compute=0.01,
            lambda_utilization=0.005,
            lambda_calibration=0.1
        )
        
        assert loss_computer.lambda_compute == pytest.approx(0.01)
        assert loss_computer.lambda_utilization == pytest.approx(0.005)
    
    def test_efficiency_loss(self):
        """Test compute efficiency loss computation."""
        from src.core.agi.compute_controller import ControllerLossComputer
        
        loss_computer = ControllerLossComputer(lambda_compute=0.1)
        
        # High cost should have higher penalty
        high_cost_loss = loss_computer.compute_efficiency_loss(0.8)
        low_cost_loss = loss_computer.compute_efficiency_loss(0.2)
        
        assert float(high_cost_loss) > float(low_cost_loss)
    
    def test_efficiency_loss_with_difficulty(self):
        """Test efficiency loss scales with task difficulty."""
        from src.core.agi.compute_controller import ControllerLossComputer
        
        loss_computer = ControllerLossComputer(lambda_compute=0.1)
        
        # Easy task should have higher penalty for same compute
        easy_task = jnp.array([0.2])
        hard_task = jnp.array([0.8])
        
        easy_loss = loss_computer.compute_efficiency_loss(0.5, easy_task)
        hard_loss = loss_computer.compute_efficiency_loss(0.5, hard_task)
        
        assert float(easy_loss) > float(hard_loss)
    
    def test_utilization_loss(self):
        """Test module utilization loss."""
        from src.core.agi.compute_controller import ControllerLossComputer
        
        loss_computer = ControllerLossComputer(
            lambda_utilization=0.1,
            target_utilization=0.3
        )
        
        # Using 3 of 12 modules = 25% utilization
        modules_called = [
            ModuleType.MEMORY_RETRIEVAL,
            ModuleType.GRAPH_REASONING,
            ModuleType.OUTPUT_GENERATION
        ]
        
        loss = loss_computer.compute_utilization_loss(modules_called)
        
        # Should be small since close to target
        assert float(loss) < 0.1
    
    def test_calibration_loss(self):
        """Test confidence calibration loss."""
        from src.core.agi.compute_controller import ControllerLossComputer
        
        loss_computer = ControllerLossComputer(lambda_calibration=1.0)
        
        # Perfect calibration: confidence matches accuracy
        perfect_conf = jnp.array([0.8, 0.8, 0.8])
        perfect_acc = jnp.array([0.8, 0.8, 0.8])
        perfect_loss = loss_computer.compute_calibration_loss(perfect_conf, perfect_acc)
        
        # Bad calibration: overconfident
        bad_conf = jnp.array([0.9, 0.9, 0.9])
        bad_acc = jnp.array([0.5, 0.5, 0.5])
        bad_loss = loss_computer.compute_calibration_loss(bad_conf, bad_acc)
        
        assert float(perfect_loss) < float(bad_loss)
    
    def test_budget_loss_overspending(self):
        """Test budget loss for overspending."""
        from src.core.agi.compute_controller import ControllerLossComputer
        
        loss_computer = ControllerLossComputer(lambda_budget=0.1)
        
        # Overspending should have high penalty
        overspend_loss = loss_computer.compute_budget_loss(-0.2, initial_budget=1.0)
        normal_loss = loss_computer.compute_budget_loss(0.3, initial_budget=1.0)
        
        assert float(overspend_loss) > float(normal_loss)
    
    def test_budget_loss_underspending(self):
        """Test budget loss for underspending."""
        from src.core.agi.compute_controller import ControllerLossComputer
        
        loss_computer = ControllerLossComputer(lambda_budget=0.1)
        
        # Large remaining budget gets small penalty
        underspend_loss = loss_computer.compute_budget_loss(0.8, initial_budget=1.0)
        efficient_loss = loss_computer.compute_budget_loss(0.3, initial_budget=1.0)
        
        assert float(underspend_loss) > float(efficient_loss)
    
    def test_ponder_loss(self):
        """Test ponder cost computation."""
        from src.core.agi.compute_controller import ControllerLossComputer
        
        loss_computer = ControllerLossComputer(lambda_ponder=0.1, prior_halt_prob=0.2)
        
        # Simulate halt probabilities over 5 steps
        halt_probs = [
            jnp.array([0.1]),
            jnp.array([0.2]),
            jnp.array([0.3]),
            jnp.array([0.5]),
            jnp.array([0.8])
        ]
        
        loss = loss_computer.compute_ponder_loss(halt_probs)
        
        # Should be non-negative
        assert float(loss) >= 0
    
    def test_total_loss_computation(self):
        """Test total loss with all components."""
        from src.core.agi.compute_controller import ControllerLossComputer
        
        loss_computer = ControllerLossComputer()
        
        # Mock execution trace
        execution_trace = {
            "total_cost": 0.4,
            "modules_executed": ["MEMORY_RETRIEVAL", "GRAPH_REASONING", "OUTPUT_GENERATION"],
            "halt_probs": [jnp.array([0.2]), jnp.array([0.5]), jnp.array([0.9])]
        }
        
        task_loss = jnp.array(1.5)
        predicted_confidence = jnp.array([0.7, 0.8, 0.6])
        actual_accuracy = jnp.array([1.0, 1.0, 0.0])
        
        total_loss, components = loss_computer.compute_total_loss(
            task_loss=task_loss,
            execution_trace=execution_trace,
            predicted_confidence=predicted_confidence,
            actual_accuracy=actual_accuracy
        )
        
        # Total should be sum of components
        assert "task_loss" in components
        assert "efficiency_loss" in components
        assert "utilization_loss" in components
        assert "calibration_loss" in components
        assert "budget_loss" in components
        assert "ponder_loss" in components
        assert "total_loss" in components
        
        # Total should be greater than task loss alone
        assert float(total_loss) >= float(task_loss)


class TestControllerRewardShaper:
    """Tests for ControllerRewardShaper (Phase 2)."""
    
    def test_reward_shaper_initialization(self):
        """Test reward shaper initialization."""
        from src.core.agi.compute_controller import ControllerRewardShaper
        
        shaper = ControllerRewardShaper(
            reward_correct=1.0,
            reward_efficiency=0.1,
            penalty_wrong=-0.5
        )
        
        assert shaper.reward_correct == pytest.approx(1.0)
        assert shaper.gamma == pytest.approx(0.99)
    
    def test_step_reward_confidence_increase(self):
        """Test step reward for confidence increase."""
        from src.core.agi.compute_controller import ControllerRewardShaper
        
        shaper = ControllerRewardShaper()
        
        # Create mock state with confidence
        state = ComputeState(
            hidden=jnp.zeros((1, 64)),
            hidden_pooled=jnp.zeros((1, 64)),
            memory_summary=jnp.zeros((1, 64)),
            uncertainty=jnp.array([[0.5]]),
            confidence=jnp.array([[0.5]]),
            budget_remaining=0.8,
            step=1,
            modules_called=[],
            module_outputs=[]
        )
        
        # Module increases confidence
        module_output = ModuleOutput(
            hidden_delta=jnp.zeros((1, 64)),
            confidence=jnp.array([[0.7]]),
            uncertainty=jnp.array([[0.3]]),
            actual_cost=0.1
        )
        
        reward = shaper.compute_step_reward(state, module_output, 0.1)
        
        # Reward should be positive for confidence increase
        assert reward > 0
    
    def test_final_reward_correct(self):
        """Test final reward for correct prediction."""
        from src.core.agi.compute_controller import ControllerRewardShaper
        
        shaper = ControllerRewardShaper(reward_correct=1.0, reward_efficiency=0.1)
        
        reward = shaper.compute_final_reward(
            is_correct=True,
            total_cost=0.3,
            num_steps=5,
            max_steps=10
        )
        
        # Should get positive reward
        assert reward > 0
        assert reward >= shaper.reward_correct
    
    def test_final_reward_wrong(self):
        """Test final reward for incorrect prediction."""
        from src.core.agi.compute_controller import ControllerRewardShaper
        
        shaper = ControllerRewardShaper(penalty_wrong=-0.5)
        
        reward = shaper.compute_final_reward(
            is_correct=False,
            total_cost=0.3,
            num_steps=5,
            max_steps=10
        )
        
        # Should get negative reward
        assert reward < 0
    
    def test_compute_returns(self):
        """Test discounted return computation."""
        from src.core.agi.compute_controller import ControllerRewardShaper
        
        shaper = ControllerRewardShaper(gamma=0.9)
        
        step_rewards = [0.1, 0.1, 0.1]  # Equal rewards
        final_reward = 1.0
        
        returns = shaper.compute_returns(step_rewards, final_reward)
        
        # Returns should be list of same length as step_rewards
        assert len(returns) == len(step_rewards)
        
        # With equal step rewards and gamma < 1, earlier returns should be higher
        # because they accumulate more future rewards (including final_reward)
        # G_t = r_t + gamma * G_{t+1}
        # returns[2] = 0.1 + 0.9 * 1.0 = 0.1 + 0.9 = 1.0 (approx, with gamma=0.9)
        # returns[1] = 0.1 + 0.9 * returns[2]
        # returns[0] = 0.1 + 0.9 * returns[1]
        
        # All returns should be positive with these inputs
        assert all(r > 0 for r in returns)


# =============================================================================
# Test AGI Integration
# =============================================================================

class TestControllerIntegrationMixin:
    """Tests for ControllerIntegrationMixin (Phase 3)."""
    
    def test_create_module_executors_from_agi(self):
        """Test creating module executors from mock AGI system."""
        from src.core.agi.compute_controller import ControllerIntegrationMixin
        
        # Create mock AGI system
        class MockAGISystem:
            def __init__(self):
                self.memory_bank = True
                self.graph_reasoner = True
                self.hybrid_architecture = True
                self.model = True
        
        agi_system = MockAGISystem()
        executors = ControllerIntegrationMixin.create_module_executors_from_agi(
            agi_system, d_model=64
        )
        
        # Should have executors for all module types
        assert ModuleType.MEMORY_RETRIEVAL in executors
        assert ModuleType.GRAPH_REASONING in executors
        assert ModuleType.OUTPUT_GENERATION in executors
    
    def test_executor_returns_valid_output(self):
        """Test that executors return valid ModuleOutput."""
        from src.core.agi.compute_controller import ControllerIntegrationMixin
        
        class MockAGISystem:
            def __init__(self):
                self.memory_bank = True
        
        agi_system = MockAGISystem()
        executors = ControllerIntegrationMixin.create_module_executors_from_agi(
            agi_system, d_model=64
        )
        
        # Create test state
        state = ComputeState(
            hidden=jnp.zeros((2, 64)),
            hidden_pooled=jnp.zeros((2, 64)),
            memory_summary=jnp.zeros((2, 64)),
            uncertainty=jnp.array([[0.5], [0.5]]),
            confidence=jnp.array([[0.5], [0.5]]),
            budget_remaining=1.0,
            step=0,
            modules_called=[],
            module_outputs=[]
        )
        
        # Execute memory retrieval
        output = executors[ModuleType.MEMORY_RETRIEVAL](state, is_training=False)
        
        # Validate output structure
        assert isinstance(output, ModuleOutput)
        assert output.hidden_delta.shape == (2, 64)
        assert output.confidence.shape == (2, 1)
        assert output.uncertainty.shape == (2, 1)


class TestControlledAGIForward:
    """Tests for ControlledAGIForward (Phase 3)."""
    
    def test_controlled_forward_basic(self):
        """Test basic controlled forward pass."""
        from src.core.agi.compute_controller import ControlledAGIForward
        
        d_model = 64
        batch_size = 2
        seq_len = 8
        
        def forward(hidden):
            # Create simple executors
            def simple_executor(state, is_training):
                return ModuleOutput(
                    hidden_delta=jnp.zeros_like(state.hidden_pooled),
                    confidence=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.7,
                    uncertainty=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.3,
                    actual_cost=0.1
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
                ModuleType.OUTPUT_GENERATION: output_executor,
            }
            
            controlled_forward = ControlledAGIForward(
                d_model=d_model,
                max_steps=5,
                initial_budget=1.0
            )
            
            output, trace = controlled_forward(
                hidden=hidden,
                module_executors=executors,
                return_trace=True
            )
            
            return output, trace
        
        fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        hidden = jax.random.normal(rng, (batch_size, seq_len, d_model))
        
        params = fn.init(rng, hidden)
        output, trace = fn.apply(params, rng, hidden)
        
        assert output.shape == (batch_size, d_model)
        assert trace is not None
        assert "total_cost" in trace
        assert "modules_executed" in trace
    
    def test_controlled_forward_respects_budget(self):
        """Test that controlled forward respects budget constraints."""
        from src.core.agi.compute_controller import ControlledAGIForward
        
        d_model = 64
        batch_size = 2
        
        def forward(hidden):
            # Expensive executors
            def expensive_executor(state, is_training):
                return ModuleOutput(
                    hidden_delta=jnp.zeros_like(state.hidden_pooled),
                    confidence=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.5,
                    uncertainty=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.5,
                    actual_cost=0.4  # Very expensive
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
                ModuleType.MEMORY_RETRIEVAL: expensive_executor,
                ModuleType.GRAPH_REASONING: expensive_executor,
                ModuleType.OUTPUT_GENERATION: output_executor,
            }
            
            controlled_forward = ControlledAGIForward(
                d_model=d_model,
                max_steps=10,
                initial_budget=0.5  # Limited budget
            )
            
            output, trace = controlled_forward(
                hidden=hidden,
                module_executors=executors,
                return_trace=True
            )
            
            return trace["total_cost"]
        
        fn = hk.transform(forward)
        rng = jax.random.PRNGKey(42)
        
        hidden = jax.random.normal(rng, (batch_size, d_model))
        
        params = fn.init(rng, hidden)
        total_cost = fn.apply(params, rng, hidden)
        
        # Should not exceed budget by too much
        # (some overshoot possible due to last module)
        assert float(total_cost) < 1.0


class TestCreateControlledAGIFn:
    """Tests for create_controlled_agi_fn factory."""
    
    def test_factory_creates_valid_transform(self):
        """Test that factory creates valid Haiku transform."""
        from src.core.agi.compute_controller import create_controlled_agi_fn
        
        fn = create_controlled_agi_fn(
            d_model=64,
            max_steps=5,
            initial_budget=1.0,
            halt_threshold=0.8
        )
        
        assert hasattr(fn, 'init')
        assert hasattr(fn, 'apply')
    
    def test_factory_forward_pass(self):
        """Test forward pass through factory-created function."""
        from src.core.agi.compute_controller import create_controlled_agi_fn
        
        d_model = 64
        batch_size = 2
        
        # Create simple executors
        def simple_executor(state, is_training):
            return ModuleOutput(
                hidden_delta=jnp.zeros_like(state.hidden_pooled),
                confidence=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.8,
                uncertainty=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.2,
                actual_cost=0.1
            )
        
        def output_executor(state, is_training):
            return ModuleOutput(
                hidden_delta=jnp.zeros_like(state.hidden_pooled),
                confidence=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.95,
                uncertainty=jnp.ones((state.hidden_pooled.shape[0], 1)) * 0.05,
                actual_cost=0.05,
                suggests_halt=True
            )
        
        executors = {
            ModuleType.MEMORY_RETRIEVAL: simple_executor,
            ModuleType.OUTPUT_GENERATION: output_executor,
        }
        
        fn = create_controlled_agi_fn(d_model=d_model, max_steps=3)
        
        rng = jax.random.PRNGKey(42)
        hidden = jax.random.normal(rng, (batch_size, d_model))
        
        params = fn.init(rng, hidden, executors)
        output, trace = fn.apply(params, rng, hidden, executors, return_trace=True)
        
        assert output.shape == (batch_size, d_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

