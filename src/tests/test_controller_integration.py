"""
Integration Tests for Compute Controller

Tests the full integration of ComputeController with:
- RTDLMAGISystem (controller-driven forward pass)
- Training loop (controller losses)
- End-to-end training with controller
"""

import pytest
import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict

from src.config.agi_config import AGIConfig
from src.rtdlm import (
    create_rtdlm_agi,
    compute_agi_loss,
)
from src.core.agi.compute_controller import (
    ModuleType,
    ModuleOutput,
    ComputeState,
    ControllerLossComputer,
)


class TestControllerConfigIntegration:
    """Tests for AGIConfig compute controller settings."""
    
    def test_config_has_controller_settings(self):
        """Test that AGIConfig has controller settings."""
        config = AGIConfig()
        
        # Check controller settings exist
        assert hasattr(config, 'use_compute_controller')
        assert hasattr(config, 'controller_max_steps')
        assert hasattr(config, 'controller_initial_budget')
        assert hasattr(config, 'controller_halt_threshold')
        assert hasattr(config, 'controller_strategy')
    
    def test_config_controller_defaults(self):
        """Test default values for controller settings."""
        config = AGIConfig()
        
        assert config.use_compute_controller == False  # Disabled by default
        assert config.controller_max_steps == 10
        assert config.controller_initial_budget == pytest.approx(1.0)
        assert config.controller_halt_threshold == pytest.approx(0.8)
        assert config.controller_strategy == "balanced"
    
    def test_config_enable_controller(self):
        """Test enabling controller via config."""
        config = AGIConfig(use_compute_controller=True)
        
        assert config.use_compute_controller == True
    
    def test_config_controller_validation(self):
        """Test controller config validation."""
        # Valid config should not raise
        config = AGIConfig(
            use_compute_controller=True,
            controller_max_steps=5,
            controller_initial_budget=2.0,
            controller_halt_threshold=0.7,
            controller_strategy="fast"
        )
        assert config.controller_strategy == "fast"
        
        # Invalid strategy should raise
        with pytest.raises(AssertionError):
            AGIConfig(
                use_compute_controller=True,
                controller_strategy="invalid_strategy"
            )
    
    def test_config_print_summary_includes_controller(self, capsys):
        """Test that print_summary includes controller info."""
        config = AGIConfig(use_compute_controller=True)
        config.print_summary()
        
        captured = capsys.readouterr()
        assert "Compute Controller" in captured.out
        assert "Enabled: True" in captured.out


class TestRTDLMWithController:
    """Tests for RTDLMAGISystem with Compute Controller."""
    
    @pytest.fixture
    def config_with_controller(self):
        """Create config with controller enabled."""
        return AGIConfig(
            d_model=64,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            max_seq_length=32,
            moe_experts=2,
            use_compute_controller=True,
            controller_max_steps=3,
            controller_initial_budget=0.5,
            controller_halt_threshold=0.7,
            # Disable expensive features for testing
            multimodal_enabled=False,
            quantum_layers=0,
            consciousness_simulation=False,
            scientific_reasoning=False,
            creative_generation=False,
        )
    
    @pytest.fixture
    def config_without_controller(self):
        """Create config without controller (static mode)."""
        return AGIConfig(
            d_model=64,
            num_heads=4,
            num_layers=2,
            vocab_size=1000,
            max_seq_length=32,
            moe_experts=2,
            use_compute_controller=False,
            # Disable expensive features for testing
            multimodal_enabled=False,
            quantum_layers=0,
            consciousness_simulation=False,
            scientific_reasoning=False,
            creative_generation=False,
        )
    
    def test_model_creates_with_controller(self, config_with_controller):
        """Test model creation with controller enabled."""
        model_fn = create_rtdlm_agi(config_with_controller)
        
        rng = jax.random.PRNGKey(42)
        batch_size = 2
        seq_len = 16
        
        inputs = {"text": jax.random.randint(rng, (batch_size, seq_len), 0, 1000)}
        
        # Model uses transform_with_state
        params, state = model_fn.init(rng, inputs=inputs)
        
        # Should have controller params
        assert params is not None
        assert state is not None
    
    def test_model_forward_with_controller(self, config_with_controller):
        """Test forward pass with controller enabled."""
        model_fn = create_rtdlm_agi(config_with_controller)
        
        rng = jax.random.PRNGKey(42)
        batch_size = 2
        seq_len = 16
        
        inputs = {"text": jax.random.randint(rng, (batch_size, seq_len), 0, 1000)}
        
        params, state = model_fn.init(rng, inputs=inputs)
        output, _ = model_fn.apply(params, state, rng, inputs=inputs)
        
        # Check output structure
        assert "logits" in output
        assert "controller_trace" in output
        assert "confidence" in output
        assert "modules_executed" in output
        assert "total_compute_cost" in output
        
        # Check shapes
        assert output["logits"].shape == (batch_size, seq_len, config_with_controller.vocab_size)
    
    def test_model_forward_without_controller(self, config_without_controller):
        """Test forward pass without controller (static mode)."""
        model_fn = create_rtdlm_agi(config_without_controller)
        
        rng = jax.random.PRNGKey(42)
        batch_size = 2
        seq_len = 16
        
        inputs = {"text": jax.random.randint(rng, (batch_size, seq_len), 0, 1000)}
        
        params, state = model_fn.init(rng, inputs=inputs)
        output, _ = model_fn.apply(params, state, rng, inputs=inputs)
        
        # Check output structure (no controller trace)
        assert "logits" in output
        assert "controller_trace" not in output
        
        # Check shapes
        assert output["logits"].shape == (batch_size, seq_len, config_without_controller.vocab_size)
    
    def test_controller_respects_budget(self, config_with_controller):
        """Test that controller respects budget constraints."""
        model_fn = create_rtdlm_agi(config_with_controller)
        
        rng = jax.random.PRNGKey(42)
        batch_size = 2
        seq_len = 16
        
        inputs = {"text": jax.random.randint(rng, (batch_size, seq_len), 0, 1000)}
        
        params, state = model_fn.init(rng, inputs=inputs)
        output, _ = model_fn.apply(params, state, rng, inputs=inputs)
        
        # Total cost should not greatly exceed initial budget
        total_cost = output.get("total_compute_cost", 0.0)
        initial_budget = config_with_controller.controller_initial_budget
        
        # Allow some overshoot for the final module
        assert total_cost < initial_budget + 0.3
    
    def test_controller_execution_trace(self, config_with_controller):
        """Test that execution trace is populated."""
        model_fn = create_rtdlm_agi(config_with_controller)
        
        rng = jax.random.PRNGKey(42)
        batch_size = 2
        seq_len = 16
        
        inputs = {"text": jax.random.randint(rng, (batch_size, seq_len), 0, 1000)}
        
        params, state = model_fn.init(rng, inputs=inputs)
        output, _ = model_fn.apply(params, state, rng, inputs=inputs)
        
        trace = output.get("controller_trace", {})
        
        # Trace should have expected keys
        assert "steps" in trace
        assert "total_cost" in trace
        assert "modules_executed" in trace
        assert "final_step" in trace
        
        # Should have executed at least one module
        assert len(trace.get("modules_executed", [])) >= 1


class TestComputeAGILossWithController:
    """Tests for compute_agi_loss with controller losses."""
    
    def test_loss_without_controller(self):
        """Test loss computation without controller."""
        logits = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 1000))
        targets = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, 1000)
        
        config = AGIConfig(use_compute_controller=False, vocab_size=1000)
        
        loss = compute_agi_loss(logits, targets, config=config)
        
        assert loss > 0
        assert jnp.isfinite(loss)
    
    def test_loss_with_controller_trace(self):
        """Test loss computation with controller trace."""
        logits = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 1000))
        targets = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, 1000)
        
        config = AGIConfig(
            use_compute_controller=True,
            vocab_size=1000,
            controller_lambda_compute=0.01,
        )
        
        # Mock controller trace
        aux_outputs = {
            "controller_trace": {
                "total_cost": 0.4,
                "modules_executed": ["MEMORY_RETRIEVAL", "GRAPH_REASONING", "OUTPUT_GENERATION"],
                "halt_probs": [jnp.array([0.2]), jnp.array([0.5]), jnp.array([0.9])],
                "final_step": 3,
            },
            "confidence": jnp.array([[0.7], [0.8]]),
            "logits": logits,
        }
        
        loss = compute_agi_loss(logits, targets, aux_outputs=aux_outputs, config=config)
        
        assert loss > 0
        assert jnp.isfinite(loss)
        
        # Check loss components were added
        assert "loss_components" in aux_outputs
        components = aux_outputs["loss_components"]
        
        assert "task_loss" in components
        assert "controller_efficiency_loss" in components
        assert "controller_utilization_loss" in components
    
    def test_loss_components_affect_total(self):
        """Test that controller losses affect total loss."""
        logits = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 1000))
        targets = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, 1000)
        
        # Loss without controller
        config_no_ctrl = AGIConfig(use_compute_controller=False, vocab_size=1000)
        loss_no_ctrl = compute_agi_loss(logits, targets, config=config_no_ctrl)
        
        # Loss with controller (high compute cost should add penalty)
        config_ctrl = AGIConfig(
            use_compute_controller=True,
            vocab_size=1000,
            controller_lambda_compute=0.5,  # High penalty
        )
        
        aux_outputs = {
            "controller_trace": {
                "total_cost": 0.9,  # High cost
                "modules_executed": ["MEMORY_RETRIEVAL"] * 5,  # Many modules
                "halt_probs": [],
                "final_step": 5,
            },
            "confidence": jnp.array([[0.5], [0.5]]),
            "logits": logits,
        }
        
        loss_ctrl = compute_agi_loss(logits, targets, aux_outputs=aux_outputs, config=config_ctrl)
        
        # Controller loss should be higher due to penalties
        assert loss_ctrl > loss_no_ctrl


class TestEndToEndControllerTraining:
    """End-to-end tests for training with controller."""
    
    @pytest.fixture
    def mini_config(self):
        """Create minimal config for fast testing."""
        return AGIConfig(
            d_model=32,
            num_heads=2,
            num_layers=1,
            vocab_size=500,
            max_seq_length=16,
            moe_experts=2,
            use_compute_controller=True,
            controller_max_steps=2,
            controller_initial_budget=0.3,
            # Disable expensive features
            multimodal_enabled=False,
            quantum_layers=0,
            consciousness_simulation=False,
            scientific_reasoning=False,
            creative_generation=False,
        )
    
    def test_single_train_step(self, mini_config):
        """Test a single training step with controller."""
        import optax
        
        model_fn = create_rtdlm_agi(mini_config)
        
        rng = jax.random.PRNGKey(42)
        batch_size = 2
        seq_len = 8
        
        # Create data
        rng, data_rng = jax.random.split(rng)
        inputs = {"text": jax.random.randint(data_rng, (batch_size, seq_len), 0, 500)}
        targets = jax.random.randint(data_rng, (batch_size, seq_len), 0, 500)
        
        # Initialize (model uses transform_with_state)
        rng, init_rng = jax.random.split(rng)
        params, state = model_fn.init(init_rng, inputs=inputs)
        
        # Create optimizer
        optimizer = optax.adam(1e-4)
        opt_state = optimizer.init(params)
        
        # Define loss function
        def loss_fn(params, state, rng_key, inputs, targets):
            output, new_state = model_fn.apply(params, state, rng_key, inputs=inputs, is_training=True)
            loss = compute_agi_loss(
                output["logits"], 
                targets, 
                aux_outputs=output, 
                config=mini_config
            )
            return loss, (output, new_state)
        
        # Compute gradients
        rng, step_rng = jax.random.split(rng)
        (loss, (output, new_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, step_rng, inputs, targets
        )
        
        # Update params
        updates, _ = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Verify
        assert jnp.isfinite(loss)
        assert new_params is not None
        
        # Check output has controller info
        assert "controller_trace" in output
        assert "modules_executed" in output
    
    def test_multiple_train_steps(self, mini_config):
        """Test multiple training steps with controller."""
        import optax
        
        model_fn = create_rtdlm_agi(mini_config)
        
        rng = jax.random.PRNGKey(42)
        batch_size = 2
        seq_len = 8
        num_steps = 3
        
        # Initialize (model uses transform_with_state)
        rng, init_rng, data_rng = jax.random.split(rng, 3)
        inputs = {"text": jax.random.randint(data_rng, (batch_size, seq_len), 0, 500)}
        targets = jax.random.randint(data_rng, (batch_size, seq_len), 0, 500)
        
        params, state = model_fn.init(init_rng, inputs=inputs)
        optimizer = optax.adam(1e-4)
        opt_state = optimizer.init(params)
        
        losses = []
        
        def make_loss_fn(rng_key, current_state):
            def loss_fn(params):
                output, _ = model_fn.apply(params, current_state, rng_key, inputs=inputs, is_training=True)
                return compute_agi_loss(
                    output["logits"], targets, aux_outputs=output, config=mini_config
                )
            return loss_fn
        
        for _ in range(num_steps):
            rng, step_rng = jax.random.split(rng)
            loss_fn = make_loss_fn(step_rng, state)
            
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            losses.append(float(loss))
        
        # All losses should be finite
        assert all(jnp.isfinite(l) for l in losses)
        
        # Loss should generally decrease or stay stable (not explode)
        assert losses[-1] < losses[0] * 10  # Not exploding


class TestControllerModeComparison:
    """Compare controller vs static mode."""
    
    def test_both_modes_produce_valid_output(self):
        """Test that both modes produce valid logits."""
        base_config = {
            "d_model": 64,
            "num_heads": 4,
            "num_layers": 2,
            "vocab_size": 1000,
            "max_seq_length": 32,
            "moe_experts": 2,
            "multimodal_enabled": False,
            "quantum_layers": 0,
            "consciousness_simulation": False,
        }
        
        config_static = AGIConfig(**base_config, use_compute_controller=False)
        config_controller = AGIConfig(**base_config, use_compute_controller=True, controller_max_steps=3)
        
        rng = jax.random.PRNGKey(42)
        batch_size = 2
        seq_len = 16
        inputs = {"text": jax.random.randint(rng, (batch_size, seq_len), 0, 1000)}
        
        # Static mode (uses transform_with_state)
        model_static = create_rtdlm_agi(config_static)
        params_static, state_static = model_static.init(rng, inputs=inputs)
        output_static, _ = model_static.apply(params_static, state_static, rng, inputs=inputs)
        
        # Controller mode (uses transform_with_state)
        model_ctrl = create_rtdlm_agi(config_controller)
        params_ctrl, state_ctrl = model_ctrl.init(rng, inputs=inputs)
        output_ctrl, _ = model_ctrl.apply(params_ctrl, state_ctrl, rng, inputs=inputs)
        
        # Both should have valid logits
        assert output_static["logits"].shape == (batch_size, seq_len, 1000)
        assert output_ctrl["logits"].shape == (batch_size, seq_len, 1000)
        
        # Both should be finite
        assert jnp.all(jnp.isfinite(output_static["logits"]))
        assert jnp.all(jnp.isfinite(output_ctrl["logits"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
