"""
Tests for Gradient Accumulation Utilities.

Tests gradient accumulation, batch splitting, and training step wrapping.
"""

import pytest
import jax
import jax.numpy as jnp
from typing import Dict

from core.gradient_accumulation import (
    BatchGradientAccumulator,
    create_accumulating_train_step,
    split_batch_for_accumulation,
    calculate_effective_batch_size,
    recommend_accumulation_steps,
)


# Mock functions for testing
def mock_loss_fn(outputs: Dict, batch: Dict) -> jnp.ndarray:
    """Simple MSE loss."""
    return jnp.mean((outputs["predictions"] - batch["labels"]) ** 2)


def mock_model_apply_fn(params: Dict, rng: jnp.ndarray, **batch) -> Dict:
    """Simple linear model."""
    x = batch["input"]
    predictions = jnp.dot(x, params["w"]) + params["b"]
    return {"predictions": predictions}


class TestBatchGradientAccumulator:
    """Tests for BatchGradientAccumulator class."""
    
    def test_initialization(self):
        """Test accumulator initialization."""
        accumulator = BatchGradientAccumulator(
            accumulation_steps=4,
            loss_fn=mock_loss_fn,
            model_apply_fn=mock_model_apply_fn,
        )
        
        assert accumulator.accumulation_steps == 4
        assert accumulator.current_step == 0
    
    def test_invalid_accumulation_steps(self):
        """Test that invalid accumulation_steps raises error."""
        with pytest.raises(ValueError, match="accumulation_steps must be >= 1"):
            BatchGradientAccumulator(
                accumulation_steps=0,
                loss_fn=mock_loss_fn,
                model_apply_fn=mock_model_apply_fn,
            )
    
    def test_reset(self):
        """Test accumulator reset."""
        accumulator = BatchGradientAccumulator(
            accumulation_steps=4,
            loss_fn=mock_loss_fn,
            model_apply_fn=mock_model_apply_fn,
        )
        
        # Manually set internal state
        accumulator._current_step = 3
        accumulator._accumulated_loss = 1.5
        
        accumulator.reset()
        
        assert accumulator.current_step == 0
        assert accumulator._accumulated_loss == pytest.approx(0.0)
    
    def test_is_complete_property(self):
        """Test is_complete property."""
        accumulator = BatchGradientAccumulator(
            accumulation_steps=2,
            loss_fn=mock_loss_fn,
            model_apply_fn=mock_model_apply_fn,
        )
        
        assert not accumulator.is_complete
        accumulator._current_step = 1
        assert not accumulator.is_complete
        accumulator._current_step = 2
        assert accumulator.is_complete
    
    def test_accumulate_returns_completion_status(self):
        """Test accumulate returns completion status."""
        accumulator = BatchGradientAccumulator(
            accumulation_steps=2,
            loss_fn=mock_loss_fn,
            model_apply_fn=mock_model_apply_fn,
        )
        
        params = {"w": jnp.ones((4, 2)), "b": jnp.zeros((2,))}
        batch = {"input": jnp.ones((3, 4)), "labels": jnp.zeros((3, 2))}
        rng = jax.random.PRNGKey(0)
        
        # First accumulation - not complete
        done1 = accumulator.accumulate(params, batch, rng)
        assert not done1
        
        # Second accumulation - complete
        done2 = accumulator.accumulate(params, batch, rng)
        assert done2
    
    def test_get_accumulated_loss(self):
        """Test getting accumulated loss."""
        accumulator = BatchGradientAccumulator(
            accumulation_steps=2,
            loss_fn=mock_loss_fn,
            model_apply_fn=mock_model_apply_fn,
        )
        
        params = {"w": jnp.ones((4, 2)), "b": jnp.zeros((2,))}
        batch = {"input": jnp.ones((3, 4)), "labels": jnp.zeros((3, 2))}
        rng = jax.random.PRNGKey(0)
        
        accumulator.accumulate(params, batch, rng)
        accumulator.accumulate(params, batch, rng)
        
        loss = accumulator.get_accumulated_loss()
        assert isinstance(loss, float) or jnp.isscalar(loss)
    
    def test_get_accumulated_grads_before_accumulation_raises(self):
        """Test getting grads before any accumulation raises error."""
        accumulator = BatchGradientAccumulator(
            accumulation_steps=2,
            loss_fn=mock_loss_fn,
            model_apply_fn=mock_model_apply_fn,
        )
        
        with pytest.raises(RuntimeError, match="No gradients accumulated"):
            accumulator.get_accumulated_grads()
    
    def test_full_accumulation_cycle(self):
        """Test complete accumulation cycle."""
        accumulator = BatchGradientAccumulator(
            accumulation_steps=4,
            loss_fn=mock_loss_fn,
            model_apply_fn=mock_model_apply_fn,
        )
        
        params = {"w": jnp.ones((4, 2)), "b": jnp.zeros((2,))}
        rng = jax.random.PRNGKey(42)
        
        for i in range(4):
            batch = {"input": jnp.ones((3, 4)) * (i + 1), "labels": jnp.zeros((3, 2))}
            step_rng = jax.random.fold_in(rng, i)
            done = accumulator.accumulate(params, batch, step_rng)
            
            if i < 3:
                assert not done
            else:
                assert done
        
        # Get averaged gradients
        grads = accumulator.get_accumulated_grads()
        
        assert "w" in grads
        assert "b" in grads
        assert grads["w"].shape == params["w"].shape


class TestCreateAccumulatingTrainStep:
    """Tests for create_accumulating_train_step function."""
    
    def test_creates_callable(self):
        """Test creates a callable train step."""
        import optax
        optimizer = optax.adam(1e-3)
        
        train_step = create_accumulating_train_step(
            model_apply_fn=mock_model_apply_fn,
            loss_fn=mock_loss_fn,
            optimizer=optimizer,
            accumulation_steps=2,
        )
        
        assert callable(train_step)
    
    def test_train_step_updates_params(self):
        """Test train step updates parameters."""
        import optax
        optimizer = optax.adam(1e-3)
        
        train_step = create_accumulating_train_step(
            model_apply_fn=mock_model_apply_fn,
            loss_fn=mock_loss_fn,
            optimizer=optimizer,
            accumulation_steps=2,
        )
        
        params = {"w": jnp.ones((4, 2)), "b": jnp.zeros((2,))}
        opt_state = optimizer.init(params)
        
        micro_batches = [
            {"input": jnp.ones((3, 4)), "labels": jnp.zeros((3, 2))},
            {"input": jnp.ones((3, 4)) * 2, "labels": jnp.ones((3, 2))},
        ]
        rng = jax.random.PRNGKey(0)
        
        new_params, _, _, _ = train_step(
            params, opt_state, micro_batches, rng
        )
        
        # Params should be different after update
        assert not jnp.allclose(new_params["w"], params["w"])
    
    def test_train_step_wrong_num_batches_raises(self):
        """Test train step raises with wrong number of micro-batches."""
        import optax
        optimizer = optax.adam(1e-3)
        
        train_step = create_accumulating_train_step(
            model_apply_fn=mock_model_apply_fn,
            loss_fn=mock_loss_fn,
            optimizer=optimizer,
            accumulation_steps=4,
        )
        
        params = {"w": jnp.ones((4, 2)), "b": jnp.zeros((2,))}
        opt_state = optimizer.init(params)
        
        # Only 2 batches when 4 expected
        micro_batches = [
            {"input": jnp.ones((3, 4)), "labels": jnp.zeros((3, 2))},
            {"input": jnp.ones((3, 4)), "labels": jnp.zeros((3, 2))},
        ]
        rng = jax.random.PRNGKey(0)
        
        with pytest.raises(ValueError, match="Expected 4 micro-batches"):
            train_step(params, opt_state, micro_batches, rng)


class TestSplitBatchForAccumulation:
    """Tests for split_batch_for_accumulation function."""
    
    def test_even_split(self):
        """Test batch splits evenly."""
        batch = {
            "input_ids": jnp.ones((16, 128)),
            "labels": jnp.ones((16,)),
        }
        
        micro_batches = split_batch_for_accumulation(batch, accumulation_steps=4)
        
        assert len(micro_batches) == 4
        assert micro_batches[0]["input_ids"].shape == (4, 128)
        assert micro_batches[0]["labels"].shape == (4,)
    
    def test_uneven_split_raises(self):
        """Test batch that doesn't split evenly raises error."""
        batch = {
            "input_ids": jnp.ones((10, 64)),
        }
        
        with pytest.raises(ValueError, match="not divisible"):
            split_batch_for_accumulation(batch, accumulation_steps=3)
    
    def test_single_accumulation(self):
        """Test with no accumulation (steps=1)."""
        batch = {
            "data": jnp.ones((8, 32)),
        }
        
        micro_batches = split_batch_for_accumulation(batch, accumulation_steps=1)
        
        assert len(micro_batches) == 1
        assert micro_batches[0]["data"].shape == (8, 32)
    
    def test_preserves_all_keys(self):
        """Test all batch keys are preserved in micro batches."""
        batch = {
            "input_ids": jnp.ones((8, 64)),
            "attention_mask": jnp.ones((8, 64)),
            "labels": jnp.ones((8,)),
        }
        
        micro_batches = split_batch_for_accumulation(batch, accumulation_steps=2)
        
        for mb in micro_batches:
            assert "input_ids" in mb
            assert "attention_mask" in mb
            assert "labels" in mb


class TestCalculateEffectiveBatchSize:
    """Tests for calculate_effective_batch_size function."""
    
    def test_basic_calculation(self):
        """Test basic effective batch size calculation."""
        result = calculate_effective_batch_size(
            micro_batch_size=8,
            accumulation_steps=4,
            num_devices=1,
        )
        
        assert result == 32
    
    def test_with_multiple_devices(self):
        """Test effective batch size with multiple devices."""
        result = calculate_effective_batch_size(
            micro_batch_size=8,
            accumulation_steps=4,
            num_devices=8,
        )
        
        assert result == 256
    
    def test_single_step(self):
        """Test with single accumulation step."""
        result = calculate_effective_batch_size(
            micro_batch_size=16,
            accumulation_steps=1,
            num_devices=1,
        )
        
        assert result == 16


class TestRecommendAccumulationSteps:
    """Tests for recommend_accumulation_steps function."""
    
    def test_target_batch_size_achievable(self):
        """Test when target batch size is achievable."""
        accum_steps, micro_batch = recommend_accumulation_steps(
            target_batch_size=64,
            max_micro_batch_size=8,
            num_devices=1,
        )
        
        # Effective batch = micro_batch * accum_steps * devices
        assert accum_steps * micro_batch == 64
    
    def test_with_multiple_devices(self):
        """Test recommendation with multiple devices."""
        accum_steps, micro_batch = recommend_accumulation_steps(
            target_batch_size=64,
            max_micro_batch_size=8,
            num_devices=4,
        )
        
        # Effective batch = micro_batch * accum_steps * 4
        assert accum_steps * micro_batch * 4 == 64
    
    def test_minimum_steps(self):
        """Test minimum accumulation steps is 1."""
        accum_steps, _ = recommend_accumulation_steps(
            target_batch_size=4,
            max_micro_batch_size=16,
            num_devices=1,
        )
        
        assert accum_steps >= 1
    
    def test_large_target(self):
        """Test with large target batch size."""
        accum_steps, micro_batch = recommend_accumulation_steps(
            target_batch_size=1024,
            max_micro_batch_size=8,
            num_devices=1,
        )
        
        # Should achieve close to 1024
        assert accum_steps >= 1
        assert micro_batch <= 8


class TestBatchGradientAccumulatorIntegration:
    """Integration tests for gradient accumulation workflow."""
    
    def test_full_accumulation_cycle(self):
        """Test a complete accumulation cycle."""
        accumulator = BatchGradientAccumulator(
            accumulation_steps=4,
            loss_fn=mock_loss_fn,
            model_apply_fn=mock_model_apply_fn,
        )
        
        params = {"w": jnp.ones((4, 2)), "b": jnp.zeros((2,))}
        rng = jax.random.PRNGKey(0)
        
        # Simulate 4 micro-batches with varying inputs
        for i in range(4):
            batch = {
                "input": jnp.ones((3, 4)) * (i + 1),
                "labels": jnp.zeros((3, 2)),
            }
            step_rng = jax.random.fold_in(rng, i)
            done = accumulator.accumulate(params, batch, step_rng)
            
            if i < 3:
                assert not done
            else:
                assert done
        
        # Get averaged gradients
        grads = accumulator.get_accumulated_grads()
        assert "w" in grads
        assert "b" in grads
    
    def test_multiple_accumulation_cycles(self):
        """Test multiple accumulation cycles."""
        accumulator = BatchGradientAccumulator(
            accumulation_steps=2,
            loss_fn=mock_loss_fn,
            model_apply_fn=mock_model_apply_fn,
        )
        
        params = {"w": jnp.ones((4, 2)), "b": jnp.zeros((2,))}
        batch = {"input": jnp.ones((3, 4)), "labels": jnp.zeros((3, 2))}
        rng = jax.random.PRNGKey(0)
        
        # First cycle
        accumulator.accumulate(params, batch, rng)
        done1 = accumulator.accumulate(params, batch, rng)
        
        assert done1
        loss1 = accumulator.get_accumulated_loss()
        
        # Reset for next cycle
        accumulator.reset()
        
        # Second cycle with different inputs
        batch2 = {"input": jnp.ones((3, 4)) * 2, "labels": jnp.ones((3, 2))}
        accumulator.accumulate(params, batch2, rng)
        done2 = accumulator.accumulate(params, batch2, rng)
        
        assert done2
        loss2 = accumulator.get_accumulated_loss()
        
        # Losses should be different
        assert loss1 != loss2
    
    def test_nested_params_accumulation(self):
        """Test accumulation with nested parameter PyTree."""
        def nested_model_apply(params, rng, **batch):
            x = batch["input"]
            h = jnp.dot(x, params["encoder"]["w1"])
            out = jnp.dot(h, params["decoder"]["w2"])
            return {"predictions": out}
        
        accumulator = BatchGradientAccumulator(
            accumulation_steps=2,
            loss_fn=mock_loss_fn,
            model_apply_fn=nested_model_apply,
        )
        
        params = {
            "encoder": {"w1": jnp.ones((4, 3))},
            "decoder": {"w2": jnp.ones((3, 2))},
        }
        batch = {"input": jnp.ones((2, 4)), "labels": jnp.zeros((2, 2))}
        rng = jax.random.PRNGKey(42)
        
        accumulator.accumulate(params, batch, rng)
        accumulator.accumulate(params, batch, rng)
        
        grads = accumulator.get_accumulated_grads()
        
        # Verify nested structure is preserved
        assert "encoder" in grads
        assert "w1" in grads["encoder"]
        assert "decoder" in grads
        assert "w2" in grads["decoder"]
