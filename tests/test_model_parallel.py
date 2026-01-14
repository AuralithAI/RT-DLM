"""
Tests for Model Parallelism Module

Tests for tensor parallelism, pipeline parallelism, and device mesh management.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from jax.sharding import PartitionSpec as P

from core.model_parallel import (
    ModelParallelConfig,
    DeviceMesh,
    TensorParallelLinear,
    TensorParallelAttention,
    TensorParallelMLP,
    PipelineStage,
    PipelineParallelModel,
    ModelParallelTransformer,
    create_model_parallel_config,
    create_model_parallel_system,
)


class TestModelParallelConfig:
    """Tests for ModelParallelConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ModelParallelConfig()
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert config.tensor_parallel is False
        assert config.pipeline_parallel is False
        assert config.activation_checkpointing is True
    
    def test_tensor_parallel_enabled(self):
        """Test config with tensor parallelism enabled"""
        config = ModelParallelConfig(tensor_parallel=True)
        assert config.tensor_parallel is True
    
    def test_config_mesh_shape(self):
        """Test mesh shape configuration"""
        config = ModelParallelConfig(
            mesh_shape=(1,),
            mesh_axis_names=("data",)
        )
        assert config.mesh_shape == (1,)
        assert config.mesh_axis_names == ("data",)
    
    def test_memory_optimization_flags(self):
        """Test memory optimization flags"""
        config = ModelParallelConfig(
            activation_checkpointing=True,
            offload_to_cpu=True
        )
        assert config.activation_checkpointing is True
        assert config.offload_to_cpu is True
    
    def test_communication_flags(self):
        """Test communication optimization flags"""
        config = ModelParallelConfig(
            async_communication=False,
            gradient_compression=True
        )
        assert config.async_communication is False
        assert config.gradient_compression is True


class TestDeviceMesh:
    """Tests for DeviceMesh"""
    
    def test_mesh_creation(self):
        """Test device mesh creation"""
        config = ModelParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
        mesh = DeviceMesh(config)
        assert mesh.config == config
        assert mesh.num_devices >= 1
    
    def test_mesh_with_tensor_parallel(self):
        """Test mesh with tensor parallelism enabled"""
        config = ModelParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel=True
        )
        mesh = DeviceMesh(config)
        assert mesh.tensor_axis == "tensor"
    
    def test_mesh_without_tensor_parallel(self):
        """Test mesh without tensor parallelism"""
        config = ModelParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel=False
        )
        mesh = DeviceMesh(config)
        assert mesh.tensor_axis is None
    
    def test_get_sharding(self):
        """Test getting sharding from mesh"""
        config = ModelParallelConfig()
        mesh = DeviceMesh(config)
        sharding = mesh.get_sharding(P())
        assert sharding is not None
    
    def test_data_axis(self):
        """Test data axis property"""
        config = ModelParallelConfig()
        mesh = DeviceMesh(config)
        assert mesh.data_axis == "data"
    
    def test_pipeline_axis(self):
        """Test pipeline axis property"""
        config = ModelParallelConfig(pipeline_parallel=True)
        mesh = DeviceMesh(config)
        assert mesh.pipeline_axis == "pipeline"


class TestTensorParallelLinear:
    """Tests for TensorParallelLinear"""
    
    def test_linear_creation(self):
        """Test creating tensor parallel linear layer"""
        config = ModelParallelConfig(tensor_parallel_size=1)
        mesh = DeviceMesh(config)
        
        def forward(x):
            layer = TensorParallelLinear(
                output_size=128,
                mesh=mesh,
                parallel_mode="column",
                name="tp_linear"
            )
            return layer(x)
        
        model = hk.without_apply_rng(hk.transform(forward))
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 4, 64))
        
        params = model.init(rng, x)
        output = model.apply(params, x)
        
        assert output.shape == (2, 4, 128)


class TestTensorParallelAttention:
    """Tests for TensorParallelAttention"""
    
    def test_attention_creation(self):
        """Test creating tensor parallel attention"""
        config = ModelParallelConfig(tensor_parallel_size=1)
        mesh = DeviceMesh(config)
        
        def forward(x):
            attn = TensorParallelAttention(
                d_model=64,
                num_heads=4,
                mesh=mesh,
                name="tp_attn"
            )
            return attn(x)
        
        model = hk.without_apply_rng(hk.transform(forward))
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 8, 64))
        
        params = model.init(rng, x)
        # Verify params created
        assert params is not None


class TestTensorParallelMLP:
    """Tests for TensorParallelMLP"""
    
    def test_mlp_creation(self):
        """Test creating tensor parallel MLP"""
        config = ModelParallelConfig(tensor_parallel_size=1)
        mesh = DeviceMesh(config)
        
        def forward(x):
            mlp = TensorParallelMLP(
                d_model=64,
                d_ff=256,
                mesh=mesh,
                name="tp_mlp"
            )
            return mlp(x)
        
        model = hk.without_apply_rng(hk.transform(forward))
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 4, 64))
        
        params = model.init(rng, x)
        # Verify params created
        assert params is not None


class TestPipelineParallel:
    """Tests for Pipeline Parallelism"""
    
    def test_pipeline_stage_creation(self):
        """Test creating pipeline stage"""
        def forward(x):
            stage = PipelineStage(
                d_model=64,
                num_heads=4,
                d_ff=256,
                num_layers_in_stage=2,
                stage_id=0,
                name="stage_0"
            )
            return stage(x)
        
        model = hk.without_apply_rng(hk.transform(forward))
        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (2, 8, 64))
        
        params = model.init(rng, x)
        output = model.apply(params, x)
        
        assert output.shape == x.shape
    
    def test_pipeline_parallel_model_creation(self):
        """Test creating pipeline parallel model"""
        def forward(input_ids):
            model = PipelineParallelModel(
                d_model=64,
                num_heads=4,
                d_ff=256,
                total_layers=4,
                num_stages=2,
                vocab_size=1000,
                name="pipeline_model"
            )
            return model(input_ids)
        
        transformed = hk.without_apply_rng(hk.transform(forward))
        rng = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(rng, (2, 8), 0, 1000)
        
        params = transformed.init(rng, input_ids)
        output = transformed.apply(params, input_ids)
        
        # Output should have vocab_size as last dim
        assert output.shape == (2, 8, 1000)


class TestModelParallelTransformer:
    """Tests for ModelParallelTransformer"""
    
    def test_transformer_creation(self):
        """Test creating model parallel transformer"""
        mp_config = ModelParallelConfig()
        mesh = DeviceMesh(mp_config)
        
        # Create a simple config for the transformer
        class TransformerConfig:
            vocab_size = 1000
            d_model = 64
            num_heads = 4
            num_layers = 2
        
        def forward(input_ids):
            transformer = ModelParallelTransformer(
                config=TransformerConfig(),
                mesh=mesh,
                name="mp_transformer"
            )
            return transformer(input_ids)
        
        model = hk.without_apply_rng(hk.transform(forward))
        rng = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(rng, (2, 16), 0, 1000)
        
        params = model.init(rng, input_ids)
        # Verify params created successfully
        assert params is not None


class TestModelParallelSystem:
    """Tests for model parallel system creation"""
    
    def test_create_system(self):
        """Test creating model parallel system"""
        class MockConfig:
            model_parallel = False
            num_devices = 1
            gradient_checkpointing = True
        
        mesh, mp_config = create_model_parallel_system(MockConfig())
        
        assert isinstance(mesh, DeviceMesh)
        assert isinstance(mp_config, ModelParallelConfig)
    
    def test_system_config_propagation(self):
        """Test config propagation"""
        class MockConfig:
            model_parallel = True
            num_devices = 1
            gradient_checkpointing = False
        
        mesh, mp_config = create_model_parallel_system(MockConfig())
        
        assert mp_config.activation_checkpointing is False


class TestConfigCreation:
    """Tests for config creation from AGI config"""
    
    def test_create_from_agi_config(self):
        """Test creating config from AGI-style config"""
        class MockAGIConfig:
            model_parallel = True
            num_devices = 1
            gradient_checkpointing = True
        
        config = create_model_parallel_config(MockAGIConfig())
        assert isinstance(config, ModelParallelConfig)


class TestGradientCommunication:
    """Tests for gradient communication patterns"""
    
    def test_all_reduce_simulation(self):
        """Test simulated all-reduce for gradients"""
        grads_device_0 = jnp.array([1.0, 2.0, 3.0])
        grads_device_1 = jnp.array([4.0, 5.0, 6.0])
        
        all_grads = jnp.stack([grads_device_0, grads_device_1])
        reduced = jnp.mean(all_grads, axis=0)
        
        expected = jnp.array([2.5, 3.5, 4.5])
        assert jnp.allclose(reduced, expected)
    
    def test_gradient_accumulation_pattern(self):
        """Test gradient accumulation for pipeline parallelism"""
        microbatch_grads = [
            jnp.array([1.0, 2.0]),
            jnp.array([3.0, 4.0]),
            jnp.array([5.0, 6.0]),
            jnp.array([7.0, 8.0])
        ]
        
        accumulated = jnp.stack(microbatch_grads).sum(axis=0)
        averaged = accumulated / len(microbatch_grads)
        
        expected = jnp.array([4.0, 5.0])
        assert jnp.allclose(averaged, expected)


class TestMemoryEstimation:
    """Tests for memory estimation utilities"""
    
    def test_param_memory_estimate(self):
        """Test parameter memory estimation"""
        num_params = 1_000_000
        bytes_per_param = 4
        
        expected_mb = (num_params * bytes_per_param) / (1024 ** 2)
        
        assert expected_mb == pytest.approx(3.81, rel=0.1)
    
    def test_activation_memory_estimate(self):
        """Test activation memory estimation"""
        batch_size = 32
        seq_len = 512
        d_model = 768
        num_layers = 12
        bytes_per_elem = 4
        
        activation_elements = 2 * batch_size * seq_len * d_model * num_layers
        activation_mb = (activation_elements * bytes_per_elem) / (1024 ** 2)
        
        assert activation_mb > 0
        assert activation_mb < 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
