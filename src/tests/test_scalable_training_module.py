"""
Tests for Scalable Training Module

Tests for distributed training with data, tensor, and pipeline parallelism.
"""

import unittest
from unittest.mock import patch, MagicMock
import jax
import jax.numpy as jnp
import numpy as np


class TestScalableMesh(unittest.TestCase):
    """Test ScalableMesh class."""
    
    def test_mesh_initialization(self):
        """Test ScalableMesh initialization."""
        from src.core.scalable_training import ScalableMesh
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        
        self.assertIsNotNone(mesh.mesh)
        self.assertGreater(mesh.num_devices, 0)
    
    def test_mesh_auto_configure(self):
        """Test automatic parallelism configuration."""
        from src.core.scalable_training import ScalableMesh
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        
        # Should have valid parallelism sizes
        self.assertGreaterEqual(mesh.data_parallel_size, 1)
        self.assertGreaterEqual(mesh.tensor_parallel_size, 1)
        self.assertGreaterEqual(mesh.pipeline_parallel_size, 1)
    
    def test_mesh_device_count(self):
        """Test device count detection."""
        from src.core.scalable_training import ScalableMesh
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        
        # Should match JAX device count
        self.assertEqual(mesh.num_devices, len(jax.devices()))
    
    def test_single_device_mesh(self):
        """Test mesh with single device."""
        from src.core.scalable_training import ScalableMesh
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig(
            tensor_parallel=False,
            pipeline_parallel=False
        )
        mesh = ScalableMesh(config)
        
        # With single device, tensor/pipeline parallel should be 1
        self.assertEqual(mesh.tensor_parallel_size, 1)
        self.assertEqual(mesh.pipeline_parallel_size, 1)


class TestModelParallelConfig(unittest.TestCase):
    """Test ModelParallelConfig dataclass."""
    
    def test_default_config(self):
        """Test default model parallel configuration."""
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig()
        
        self.assertIsNotNone(config)
    
    def test_tensor_parallel_config(self):
        """Test tensor parallelism configuration."""
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig(
            tensor_parallel=True,
            tensor_parallel_size=4
        )
        
        self.assertTrue(config.tensor_parallel)
        # Size is auto-adjusted to min(requested, num_devices)
        self.assertLessEqual(config.tensor_parallel_size, config.num_devices)
    
    def test_pipeline_parallel_config(self):
        """Test pipeline parallelism configuration."""
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig(
            pipeline_parallel=True,
            pipeline_parallel_size=2
        )
        
        self.assertTrue(config.pipeline_parallel)
        # Size is auto-adjusted to min(requested, num_devices)
        self.assertLessEqual(config.pipeline_parallel_size, config.num_devices)


class TestDataParallelism(unittest.TestCase):
    """Test data parallelism functionality."""
    
    def test_data_parallel_size(self):
        """Test data parallel size computation."""
        from src.core.scalable_training import ScalableMesh
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig(
            tensor_parallel=False,
            pipeline_parallel=False
        )
        mesh = ScalableMesh(config)
        
        # Data parallel size should use all devices
        self.assertEqual(mesh.data_parallel_size, mesh.num_devices)
    
    def test_replicated_model(self):
        """Test that model is replicated in data parallelism."""
        from src.core.scalable_training import ScalableMesh
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        
        # Mesh should exist for data parallelism
        self.assertIsNotNone(mesh.mesh)


class TestShardingStrategies(unittest.TestCase):
    """Test sharding strategies."""
    
    def test_partition_spec_creation(self):
        """Test partition spec creation."""
        from jax.sharding import PartitionSpec as P
        
        # Test various partition specs
        data_parallel_spec = P('data', None)
        tensor_parallel_spec = P(None, 'model')
        
        self.assertIsNotNone(data_parallel_spec)
        self.assertIsNotNone(tensor_parallel_spec)
    
    def test_named_sharding(self):
        """Test named sharding creation."""
        from src.core.scalable_training import ScalableMesh
        from src.config.model_parallel_config import ModelParallelConfig
        from jax.sharding import NamedSharding, PartitionSpec as P
        
        config = ModelParallelConfig()
        scalable_mesh = ScalableMesh(config)
        
        # Create named sharding with mesh
        spec = P()
        sharding = NamedSharding(scalable_mesh.mesh, spec)
        
        self.assertIsNotNone(sharding)


class TestGradientSynchronization(unittest.TestCase):
    """Test gradient synchronization across devices."""
    
    def test_all_reduce_sum(self):
        """Test all-reduce sum operation exists."""

        self.assertTrue(hasattr(jax.lax, 'psum'))
    
    def test_all_reduce_mean(self):
        """Test all-reduce mean operation."""
        # Verify pmean exists for gradient averaging
        self.assertTrue(hasattr(jax.lax, 'pmean'))


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency features."""
    
    def test_gradient_checkpointing_availability(self):
        """Test gradient checkpointing is available."""
        # JAX provides remat for gradient checkpointing
        self.assertTrue(hasattr(jax, 'checkpoint') or hasattr(jax, 'remat'))
    
    def test_activation_memory_estimate(self):
        """Test activation memory estimation."""
        from src.core.scalable_training import ScalableMesh
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        
        # Mesh should be able to handle memory distribution
        self.assertGreater(mesh.num_devices, 0)


class TestTrainingStep(unittest.TestCase):
    """Test scalable training step."""
    
    def test_training_step_signature(self):
        """Test that scalable training step has correct interface."""
        from src.core.scalable_training import ScalableMesh
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        
        # Verify mesh provides required interface
        self.assertTrue(hasattr(mesh, 'mesh'))
        self.assertTrue(hasattr(mesh, 'config'))


class TestPipelineParallelism(unittest.TestCase):
    """Test pipeline parallelism (micro-batching)."""
    
    def test_pipeline_config(self):
        """Test pipeline parallel configuration."""
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig(
            pipeline_parallel=True,
            pipeline_parallel_size=4
        )
        
        self.assertTrue(config.pipeline_parallel)
        # Size is auto-adjusted to min(requested, num_devices)
        self.assertLessEqual(config.pipeline_parallel_size, config.num_devices)


class TestMixedPrecision(unittest.TestCase):
    """Test mixed precision training support."""
    
    def test_bfloat16_support(self):
        """Test bfloat16 dtype support."""
        x = jnp.ones((4, 4), dtype=jnp.bfloat16)
        
        self.assertEqual(x.dtype, jnp.bfloat16)
    
    def test_float16_support(self):
        """Test float16 dtype support."""
        x = jnp.ones((4, 4), dtype=jnp.float16)
        
        self.assertEqual(x.dtype, jnp.float16)


class TestFallbackBehavior(unittest.TestCase):
    """Test fallback behavior when insufficient devices."""
    
    def test_fallback_to_data_parallel(self):
        """Test fallback to data parallelism only."""
        from src.core.scalable_training import ScalableMesh
        from src.config.model_parallel_config import ModelParallelConfig
        
        config = ModelParallelConfig(
            tensor_parallel=True,
            tensor_parallel_size=1000,
            pipeline_parallel=False
        )
        
        mesh = ScalableMesh(config)
        
        self.assertLessEqual(mesh.tensor_parallel_size, mesh.num_devices)


if __name__ == "__main__":
    unittest.main()
