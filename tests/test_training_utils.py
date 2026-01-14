"""
Tests for Training Utilities

Tests for:
- Mixed precision training
- Gradient checkpointing
- Distributed training utilities
"""

import pytest
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))


class TestMixedPrecisionPolicy:
    """Tests for mixed precision training"""
    
    def test_default_policy_is_float32(self):
        """Test default policy uses float32"""
        from core.training_utils import MixedPrecisionPolicy
        
        policy = MixedPrecisionPolicy()
        assert policy.param_dtype == jnp.float32
        assert policy.compute_dtype == jnp.float32
        assert policy.output_dtype == jnp.float32
        
    def test_bfloat16_policy(self):
        """Test bfloat16 mixed precision policy"""
        from core.training_utils import MixedPrecisionPolicy
        
        policy = MixedPrecisionPolicy(
            param_dtype="bfloat16",
            compute_dtype="bfloat16",
            output_dtype="float32"
        )
        assert policy.param_dtype == jnp.bfloat16
        assert policy.compute_dtype == jnp.bfloat16
        assert policy.output_dtype == jnp.float32
        
    def test_float16_policy(self):
        """Test float16 mixed precision policy"""
        from core.training_utils import MixedPrecisionPolicy
        
        policy = MixedPrecisionPolicy(
            param_dtype="float16",
            compute_dtype="float16"
        )
        assert policy.param_dtype == jnp.float16
        assert policy.compute_dtype == jnp.float16
        
    def test_cast_to_compute(self):
        """Test casting tensors to compute dtype"""
        from core.training_utils import MixedPrecisionPolicy
        
        policy = MixedPrecisionPolicy(compute_dtype="bfloat16")
        x = jnp.ones((2, 3), dtype=jnp.float32)
        
        x_cast = policy.cast_to_compute(x)
        assert x_cast.dtype == jnp.bfloat16
        
    def test_cast_params(self):
        """Test casting parameter dictionary"""
        from core.training_utils import MixedPrecisionPolicy
        
        policy = MixedPrecisionPolicy(param_dtype="bfloat16")
        params = {
            "layer1": {"w": jnp.ones((10, 10), dtype=jnp.float32)},
            "layer2": {"w": jnp.ones((10, 10), dtype=jnp.float32)},
        }
        
        cast_params = policy.cast_params(params)
        assert cast_params["layer1"]["w"].dtype == jnp.bfloat16
        assert cast_params["layer2"]["w"].dtype == jnp.bfloat16
        
    def test_is_mixed_precision(self):
        """Test mixed precision detection"""
        from core.training_utils import MixedPrecisionPolicy
        
        default = MixedPrecisionPolicy()
        assert not default.is_mixed_precision()
        
        mixed = MixedPrecisionPolicy(compute_dtype="bfloat16")
        assert mixed.is_mixed_precision()
        

class TestGradientCheckpointing:
    """Tests for gradient checkpointing"""
    
    def test_checkpointing_config_default(self):
        """Test default checkpointing config"""
        from core.training_utils import GradientCheckpointingConfig
        
        config = GradientCheckpointingConfig()
        assert not config.enabled
        assert config.checkpoint_every_n_layers == 2
        
    def test_checkpointing_config_enabled(self):
        """Test enabled checkpointing config"""
        from core.training_utils import GradientCheckpointingConfig
        
        config = GradientCheckpointingConfig(
            enabled=True,
            checkpoint_every_n_layers=3
        )
        assert config.enabled
        assert config.should_checkpoint_layer(0)
        assert not config.should_checkpoint_layer(1)
        assert not config.should_checkpoint_layer(2)
        assert config.should_checkpoint_layer(3)
        
    def test_checkpointing_disabled_returns_false(self):
        """Test disabled checkpointing never checkpoints"""
        from core.training_utils import GradientCheckpointingConfig
        
        config = GradientCheckpointingConfig(enabled=False)
        for i in range(10):
            assert not config.should_checkpoint_layer(i)
            
    def test_create_checkpointed_layer(self):
        """Test checkpointed layer creation"""
        from core.training_utils import create_checkpointed_layer
        
        def simple_layer(x):
            return x * 2
        
        checkpointed = create_checkpointed_layer(simple_layer, checkpoint=True)
        normal = create_checkpointed_layer(simple_layer, checkpoint=False)
        
        x = jnp.ones((2, 3))
        assert jnp.allclose(checkpointed(x), normal(x))


class TestDistributedTraining:
    """Tests for distributed training utilities"""
    
    def test_distributed_config_default(self):
        """Test default distributed config"""
        from core.training_utils import DistributedTrainingConfig
        
        config = DistributedTrainingConfig()
        assert not config.enabled
        assert config.gradient_accumulation_steps == 1
        
    def test_effective_batch_multiplier(self):
        """Test effective batch size calculation"""
        from core.training_utils import DistributedTrainingConfig
        
        config = DistributedTrainingConfig(
            enabled=True,
            num_devices=4,
            gradient_accumulation_steps=2
        )
        assert config.effective_batch_size_multiplier == 8
        
    def test_shard_batch(self):
        """Test batch sharding for multiple devices"""
        from core.training_utils import shard_batch_for_devices
        
        batch = {
            "input_ids": jnp.ones((8, 32)),
            "targets": jnp.ones((8, 32)),
        }
        
        sharded = shard_batch_for_devices(batch, num_devices=4)
        
        assert sharded["input_ids"].shape == (4, 2, 32)
        assert sharded["targets"].shape == (4, 2, 32)
        
    def test_unshard_batch(self):
        """Test batch unsharding"""
        from core.training_utils import shard_batch_for_devices, unshard_batch
        
        batch = {"x": jnp.ones((8, 32))}
        sharded = shard_batch_for_devices(batch, num_devices=4)
        unsharded = unshard_batch(sharded)
        
        assert unsharded["x"].shape == (8, 32)
        

class TestGradientAccumulator:
    """Tests for gradient accumulation"""
    
    def test_accumulator_initialization(self):
        """Test accumulator initialization"""
        from core.training_utils import GradientAccumulator
        
        acc = GradientAccumulator(accumulation_steps=4)
        assert acc.accumulation_steps == 4
        assert acc.current_step == 0
        assert acc.accumulated_grads is None
        
    def test_accumulator_single_step(self):
        """Test single accumulation step"""
        from core.training_utils import GradientAccumulator
        
        acc = GradientAccumulator(accumulation_steps=2)
        grads = {"w": jnp.ones((3, 3))}
        
        should_update, avg_grads = acc.accumulate(grads)
        assert not should_update
        assert avg_grads is None
        
    def test_accumulator_full_accumulation(self):
        """Test full gradient accumulation"""
        from core.training_utils import GradientAccumulator
        
        acc = GradientAccumulator(accumulation_steps=2)
        grads1 = {"w": jnp.ones((3, 3)) * 2}
        grads2 = {"w": jnp.ones((3, 3)) * 4}
        
        should_update1, _ = acc.accumulate(grads1)
        assert not should_update1
        
        should_update2, avg_grads = acc.accumulate(grads2)
        assert should_update2
        assert avg_grads is not None
        assert jnp.allclose(avg_grads["w"], jnp.ones((3, 3)) * 3)  # (2+4)/2 = 3
        
    def test_accumulator_reset(self):
        """Test accumulator reset"""
        from core.training_utils import GradientAccumulator
        
        acc = GradientAccumulator(accumulation_steps=2)
        acc.accumulate({"w": jnp.ones((3, 3))})
        
        acc.reset()
        assert acc.current_step == 0
        assert acc.accumulated_grads is None


class TestTrainingOptimizations:
    """Tests for combined training optimizations"""
    
    def test_from_config(self):
        """Test creating optimizations from config"""
        from core.training_utils import TrainingOptimizations
        from config.agi_config import AGIConfig
        
        config = AGIConfig(
            mixed_precision=True,
            precision_dtype="bfloat16",
            gradient_checkpointing=True,
            distributed_training=False
        )
        
        opts = TrainingOptimizations.from_config(config)
        
        assert opts.precision_policy.is_mixed_precision()
        assert opts.checkpoint_config.enabled
        assert not opts.distributed.enabled
        
    def test_summary(self):
        """Test optimization summary"""
        from core.training_utils import TrainingOptimizations
        from config.agi_config import AGIConfig
        
        config = AGIConfig()
        opts = TrainingOptimizations.from_config(config)
        
        summary = opts.summary()
        assert "Training Optimizations" in summary
        assert "Precision" in summary


class TestMemoryUtilities:
    """Tests for memory utilities"""
    
    def test_get_memory_stats(self):
        """Test memory stats retrieval"""
        from core.training_utils import get_memory_stats
        
        stats = get_memory_stats()
        # May be empty dict if devices don't support memory_stats
        assert isinstance(stats, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
