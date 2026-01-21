"""
Distributed Training Tests for RT-DLM.

Tests for validating distributed training functionality:
- ScalableMesh configuration
- Data parallelism
- Tensor parallelism
- Gradient synchronization
- Checkpoint compatibility across device counts

Note: Multi-GPU tests require hardware. Single-device simulation tests
run on any machine to validate logic.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict


class TestScalableMeshConfiguration:
    """Test ScalableMesh configuration logic."""
    
    @pytest.fixture
    def single_device_config(self):
        from config.model_parallel_config import ModelParallelConfig
        return ModelParallelConfig(
            tensor_parallel=False,
            pipeline_parallel=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
    
    @pytest.fixture
    def data_parallel_config(self):
        from config.model_parallel_config import ModelParallelConfig
        return ModelParallelConfig(
            tensor_parallel=False,
            pipeline_parallel=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
    
    @pytest.fixture
    def tensor_parallel_config(self):
        from config.model_parallel_config import ModelParallelConfig
        return ModelParallelConfig(
            tensor_parallel=True,
            pipeline_parallel=False,
            tensor_parallel_size=2,
            pipeline_parallel_size=1
        )
    
    def test_single_device_mesh(self, single_device_config):
        """Test mesh creation on single device."""
        from core.scalable_training import ScalableMesh
        
        mesh = ScalableMesh(single_device_config)
        
        assert mesh.num_devices >= 1
        assert mesh.data_parallel_size >= 1
        assert mesh.tensor_parallel_size == 1
        assert mesh.pipeline_parallel_size == 1
        assert mesh.mesh is not None
    
    def test_mesh_properties(self, single_device_config):
        """Test mesh property accessors."""
        from core.scalable_training import ScalableMesh
        
        mesh = ScalableMesh(single_device_config)
        
        assert isinstance(mesh.has_tensor_parallel, bool)
        assert isinstance(mesh.has_pipeline_parallel, bool)
        assert isinstance(mesh.is_distributed, bool)
    
    def test_mesh_sharding_creation(self, single_device_config):
        """Test creating sharding specs from mesh."""
        from core.scalable_training import ScalableMesh
        from jax.sharding import PartitionSpec as P
        
        mesh = ScalableMesh(single_device_config)
        
        sharding = mesh.get_sharding(P())
        assert sharding is not None
        assert sharding.mesh == mesh.mesh


class TestParamSharding:
    """Test parameter sharding specifications."""
    
    @pytest.fixture
    def mesh(self):
        from config.model_parallel_config import ModelParallelConfig
        from core.scalable_training import ScalableMesh
        
        config = ModelParallelConfig(
            tensor_parallel=True,
            tensor_parallel_size=2
        )
        return ScalableMesh(config)
    
    def test_embedding_sharding(self, mesh):
        """Test embedding parameters get sharded on vocab dimension."""
        from core.scalable_training import get_param_sharding_spec
        
        spec = get_param_sharding_spec(
            "model/embed/embeddings",
            (32000, 384),
            mesh
        )
        
        if mesh.has_tensor_parallel:
            assert spec is not None
    
    def test_attention_sharding(self, mesh):
        """Test attention parameters get sharded on heads dimension."""
        from core.scalable_training import get_param_sharding_spec
        
        spec = get_param_sharding_spec(
            "model/attention/query",
            (384, 384),
            mesh
        )
        
        assert spec is not None
    
    def test_layernorm_not_sharded(self, mesh):
        """Test layer norm parameters are not sharded."""
        from core.scalable_training import get_param_sharding_spec
        from jax.sharding import PartitionSpec as P
        
        spec = get_param_sharding_spec(
            "model/layer_norm/scale",
            (384,),
            mesh
        )
        
        assert spec == P()


class TestMemoryEstimation:
    """Test memory estimation utilities."""
    
    def test_estimate_model_memory(self):
        """Test memory estimation from parameters."""
        from core.scalable_training import estimate_model_memory
        
        params = {
            "layer1": jnp.ones((1000, 384)),
            "layer2": jnp.ones((384, 384)),
        }
        
        estimate = estimate_model_memory(params)
        
        assert "parameters_gb" in estimate
        assert "optimizer_gb" in estimate
        assert "gradients_gb" in estimate
        assert "total_gb" in estimate
        assert "total_params" in estimate
        
        expected_params = 1000 * 384 + 384 * 384
        assert estimate["total_params"] == expected_params
    
    def test_recommend_parallelism_single_device(self):
        """Test parallelism recommendation for small model."""
        from core.scalable_training import recommend_parallelism
        
        rec = recommend_parallelism(
            model_memory_gb=2.0,
            device_memory_gb=16.0,
            num_devices=1
        )
        
        assert rec["strategy"] == "single_device"
        assert rec["data_parallel"] == False
        assert rec["tensor_parallel"] == False
    
    def test_recommend_parallelism_data_parallel(self):
        """Test parallelism recommendation for multi-GPU with small model."""
        from core.scalable_training import recommend_parallelism
        
        rec = recommend_parallelism(
            model_memory_gb=2.0,
            device_memory_gb=16.0,
            num_devices=4
        )
        
        assert rec["strategy"] == "data_parallel"
        assert rec["data_parallel"] == True
        assert rec["tensor_parallel"] == False
    
    def test_recommend_parallelism_tensor_parallel(self):
        """Test parallelism recommendation for large model."""
        from core.scalable_training import recommend_parallelism
        
        rec = recommend_parallelism(
            model_memory_gb=50.0,
            device_memory_gb=16.0,
            num_devices=8
        )
        
        assert rec["tensor_parallel"] == True


class TestDistributedValidation:
    """Test distributed setup validation."""
    
    def test_validate_distributed_setup(self):
        """Test validation function runs without error."""
        from config.model_parallel_config import ModelParallelConfig
        from core.scalable_training import ScalableMesh, validate_distributed_setup
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        
        results = validate_distributed_setup(mesh)
        
        assert "num_devices" in results
        assert "checks" in results
        assert "valid" in results
        assert results["checks"]["devices_visible"] == True
        assert results["checks"]["mesh_valid"] == True


class TestTrainStepCreation:
    """Test train step creation for different parallelism modes."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        import haiku as hk
        
        def forward(x):
            return hk.Linear(64)(x)
        
        return hk.transform(forward)
    
    @pytest.fixture
    def simple_loss(self):
        def loss_fn(outputs, batch):
            targets = batch.get("targets", jnp.zeros_like(outputs))
            return jnp.mean((outputs - targets) ** 2)
        return loss_fn
    
    def test_create_single_device_train_step(self, simple_model, simple_loss):
        """Test train step creation for single device."""
        from config.model_parallel_config import ModelParallelConfig
        from core.scalable_training import ScalableMesh, create_scalable_train_step
        import optax
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        optimizer = optax.adam(1e-3)
        
        train_step = create_scalable_train_step(
            model_apply_fn=simple_model.apply,
            optimizer=optimizer,
            mesh=mesh,
            loss_fn=simple_loss
        )
        
        assert callable(train_step)


class TestDistributedProfiler:
    """Test distributed profiler functionality."""
    
    def test_profiler_creation(self):
        """Test profiler can be created."""
        from config.model_parallel_config import ModelParallelConfig
        from core.scalable_training import ScalableMesh, DistributedProfiler
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        profiler = DistributedProfiler(mesh)
        
        assert profiler.mesh == mesh
        assert "all_reduce" in profiler.timings
    
    def test_profiler_all_reduce_single_device(self):
        """Test all_reduce profiling on single device."""
        from config.model_parallel_config import ModelParallelConfig
        from core.scalable_training import ScalableMesh, DistributedProfiler
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        profiler = DistributedProfiler(mesh)
        
        tensor = jnp.ones((100, 100))
        result = profiler.profile_all_reduce(tensor, num_iterations=3)
        
        assert "latency_ms" in result
        if not mesh.is_distributed:
            assert result["latency_ms"] == 0.0
    
    def test_profiler_summary(self):
        """Test profiler summary generation."""
        from config.model_parallel_config import ModelParallelConfig
        from core.scalable_training import ScalableMesh, DistributedProfiler
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        profiler = DistributedProfiler(mesh)
        
        summary = profiler.get_summary()
        
        assert "avg_all_reduce_ms" in summary
        assert "num_devices" in summary
        assert "is_distributed" in summary


class TestCheckpointCompatibility:
    """Test checkpoint saving/loading across device counts."""
    
    def test_unreplicate_params(self):
        """Test unreplicating parameters from replicated state."""
        from core.scalable_training import unreplicate_params
        
        params = {
            "layer1": jnp.stack([jnp.ones((10, 10)) for _ in range(2)]),
            "layer2": jnp.stack([jnp.ones((10,)) for _ in range(2)]),
        }
        
        unreplicated = unreplicate_params(params)
        
        assert unreplicated["layer1"].shape == (10, 10)
        assert unreplicated["layer2"].shape == (10,)


class TestGradientSynchronization:
    """Test gradient synchronization logic."""
    
    def test_pmean_simulation(self):
        """Test that pmean logic is correct in simulation."""
        grads = jnp.ones((4, 10))
        
        mean_grads = jnp.mean(grads, axis=0, keepdims=True)
        mean_grads = jnp.broadcast_to(mean_grads, grads.shape)
        
        expected = jnp.ones((4, 10))
        assert jnp.allclose(mean_grads, expected)


class TestProfileCollectiveCommunication:
    """Test collective communication profiling."""
    
    def test_profile_single_device(self):
        """Test profiling on single device returns zero overhead."""
        from config.model_parallel_config import ModelParallelConfig
        from core.scalable_training import ScalableMesh, profile_collective_communication
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        
        result = profile_collective_communication(mesh, array_size_bytes=1_000_000)
        
        assert "all_reduce_time_ms" in result
        assert "bandwidth_gbps" in result
        assert "num_devices" in result
        
        if not mesh.is_distributed:
            assert result["all_reduce_time_ms"] < 0.01  # Effectively zero
            assert "message" in result
    
    def test_profile_returns_expected_keys(self):
        """Test profiling returns all expected keys."""
        from config.model_parallel_config import ModelParallelConfig
        from core.scalable_training import ScalableMesh, profile_collective_communication
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        
        result = profile_collective_communication(mesh)
        
        assert "all_reduce_time_ms" in result
        assert "num_devices" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
