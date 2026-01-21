"""
Multi-GPU specific tests for distributed training.

These tests require 2+ GPUs and are automatically skipped if not available.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


@pytest.mark.multi_gpu
class TestDataParallelTraining:
    """Tests requiring multiple GPUs for data parallelism."""
    
    def test_pmap_gradient_sync(self):
        """Test that pmap correctly synchronizes gradients."""
        num_devices = jax.device_count()
        if num_devices < 2:
            pytest.skip(f"Requires 2+ GPUs, found {num_devices}")
        
        @partial(jax.pmap, axis_name="data")
        def sync_grads(grads):
            return lax.pmean(grads, axis_name="data")
        
        grads = jnp.stack([jnp.ones((10,)) * (i + 1) for i in range(num_devices)])
        grads = jax.device_put_sharded(list(grads), jax.devices()[:num_devices])
        
        synced = sync_grads(grads)
        
        expected_mean = (1 + num_devices) / 2
        assert jnp.allclose(synced[0], expected_mean)
    
    def test_replicated_params(self):
        """Test parameter replication across devices."""
        num_devices = jax.device_count()
        if num_devices < 2:
            pytest.skip(f"Requires 2+ GPUs, found {num_devices}")
        
        params = {"w": jnp.ones((100, 100))}
        
        replicated = jax.device_put_replicated(params, jax.devices()[:num_devices])
        
        assert replicated["w"].shape == (num_devices, 100, 100)
        assert jnp.allclose(replicated["w"][0], replicated["w"][1])
    
    def test_data_parallel_forward(self):
        """Test data parallel forward pass."""
        num_devices = jax.device_count()
        if num_devices < 2:
            pytest.skip(f"Requires 2+ GPUs, found {num_devices}")
        
        import haiku as hk
        
        def forward(x):
            return hk.Linear(64)(x)
        
        model = hk.transform(forward)
        
        rng = jax.random.PRNGKey(42)
        sample = jnp.ones((1, 32))
        params = model.init(rng, sample)
        
        replicated_params = jax.device_put_replicated(params, jax.devices()[:num_devices])
        
        batch = jnp.stack([jnp.ones((4, 32)) for _ in range(num_devices)])
        rngs = jax.random.split(rng, num_devices)
        
        @partial(jax.pmap, axis_name="data")
        def parallel_forward(params, x, rng):
            return model.apply(params, rng, x)
        
        outputs = parallel_forward(replicated_params, batch, rngs)
        
        assert outputs.shape == (num_devices, 4, 64)


@pytest.mark.multi_gpu
class TestTensorParallelTraining:
    """Tests requiring multiple GPUs for tensor parallelism."""
    
    def test_sharded_matmul(self):
        """Test sharded matrix multiplication."""
        num_devices = jax.device_count()
        if num_devices < 2:
            pytest.skip(f"Requires 2+ GPUs, found {num_devices}")
        
        from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
        from jax.experimental import mesh_utils
        
        devices = mesh_utils.create_device_mesh((num_devices,))
        mesh = Mesh(devices, ("tensor",))
        
        x = jnp.ones((100, 256))
        w = jnp.ones((256, 512))
        
        w_sharding = NamedSharding(mesh, P(None, "tensor"))
        w_sharded = jax.device_put(w, w_sharding)
        
        @jax.jit
        def matmul(x, w):
            return x @ w
        
        result = matmul(x, w_sharded)
        
        assert result.shape == (100, 512)


@pytest.mark.multi_gpu
class TestCommunicationProfiling:
    """Tests for communication overhead profiling."""
    
    def test_all_reduce_bandwidth(self):
        """Measure all-reduce bandwidth across devices."""
        num_devices = jax.device_count()
        if num_devices < 2:
            pytest.skip(f"Requires 2+ GPUs, found {num_devices}")
        
        from config.model_parallel_config import ModelParallelConfig
        from core.scalable_training import ScalableMesh, DistributedProfiler
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        profiler = DistributedProfiler(mesh)
        
        sizes_mb = [1, 10, 100]
        for size_mb in sizes_mb:
            num_elements = int(size_mb * 1e6 / 4)
            tensor = jnp.ones(num_elements)
            
            result = profiler.profile_all_reduce(tensor, num_iterations=5)
            
            print(f"Size: {size_mb} MB, Latency: {result['latency_ms']:.2f} ms, "
                  f"Bandwidth: {result['bandwidth_gbps']:.2f} GB/s")
            
            assert result["latency_ms"] > 0


@pytest.mark.multi_gpu
@pytest.mark.slow
class TestFullTrainingLoop:
    """End-to-end training tests on multiple GPUs."""
    
    def test_data_parallel_training_step(self):
        """Test complete training step with data parallelism."""
        num_devices = jax.device_count()
        if num_devices < 2:
            pytest.skip(f"Requires 2+ GPUs, found {num_devices}")
        
        import haiku as hk
        import optax
        from config.model_parallel_config import ModelParallelConfig
        from core.scalable_training import ScalableMesh, create_scalable_train_step
        
        def forward(x):
            x = hk.Linear(128)(x)
            x = jax.nn.relu(x)
            x = hk.Linear(64)(x)
            return x
        
        model = hk.transform(forward)
        
        def loss_fn(outputs, batch):
            targets = batch.get("targets", jnp.zeros_like(outputs))
            return jnp.mean((outputs - targets) ** 2)
        
        config = ModelParallelConfig()
        mesh = ScalableMesh(config)
        optimizer = optax.adam(1e-3)
        
        rng = jax.random.PRNGKey(42)
        sample = jnp.ones((1, 32))
        params = model.init(rng, sample)
        opt_state = optimizer.init(params)
        
        replicated_params = jax.device_put_replicated(params, jax.devices()[:num_devices])
        replicated_opt_state = jax.device_put_replicated(opt_state, jax.devices()[:num_devices])
        
        batch = {
            "x": jnp.stack([jnp.ones((8, 32)) for _ in range(num_devices)]),
            "targets": jnp.stack([jnp.zeros((8, 64)) for _ in range(num_devices)])
        }
        rngs = jax.random.split(rng, num_devices)
        
        train_step = create_scalable_train_step(
            model_apply_fn=model.apply,
            optimizer=optimizer,
            mesh=mesh,
            loss_fn=loss_fn
        )
        
        new_params, new_opt_state, loss, _ = train_step(
            replicated_params, replicated_opt_state, batch, rngs
        )
        
        assert loss.shape == (num_devices,)
        assert jnp.allclose(loss[0], loss[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
