"""
Scalable Training for RT-DLM.

Production-ready parallelism supporting:
- Data parallelism: Same model replicated, different data batches
- Tensor parallelism: Sharded model across devices (for large models)
- Combined: Both strategies together (production scale)

Tested strategies: Single device, multi-GPU data parallelism.
Model parallelism requires 8+ GPUs with high-bandwidth interconnect.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import haiku as hk
from typing import Dict, Any, Optional, Tuple, Callable
from functools import partial
import logging
import time

from src.config.model_parallel_config import ModelParallelConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Device Mesh for Scalable Training
# =============================================================================

class ScalableMesh:
    """
    Production-ready device mesh for scalable training.
    
    Automatically configures the optimal parallelism strategy based on:
    - Number of available devices
    - Model size
    - User preferences
    
    Example configurations:
    - 1 GPU: No parallelism (standard training)
    - 4 GPUs: Data parallelism (replicate model, split data)
    - 8+ GPUs: Can combine data + tensor parallelism
    - 1000+ GPUs: Full 3D parallelism (data + tensor + pipeline)
    """
    
    def __init__(self, config: ModelParallelConfig):
        self.config = config
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        
        self._auto_configure()
        
        self.mesh = self._create_mesh()
        
        logger.info("ScalableMesh initialized:")
        logger.info(f"  Devices: {self.num_devices}")
        logger.info(f"  Data parallel size: {self.data_parallel_size}")
        logger.info(f"  Tensor parallel size: {self.tensor_parallel_size}")
        logger.info(f"  Pipeline parallel size: {self.pipeline_parallel_size}")
        
    def _auto_configure(self):
        """Auto-configure parallelism based on device count"""
        if self.config.tensor_parallel:
            self.tensor_parallel_size = min(
                self.config.tensor_parallel_size, 
                self.num_devices
            )
        else:
            self.tensor_parallel_size = 1
            
        if self.config.pipeline_parallel:
            self.pipeline_parallel_size = min(
                self.config.pipeline_parallel_size,
                self.num_devices // self.tensor_parallel_size
            )
        else:
            self.pipeline_parallel_size = 1
            
        self.data_parallel_size = max(
            1,
            self.num_devices // (self.tensor_parallel_size * self.pipeline_parallel_size)
        )
        
    def _create_mesh(self) -> Mesh:
        """Create JAX device mesh"""
        total_needed = (
            self.data_parallel_size * 
            self.tensor_parallel_size * 
            self.pipeline_parallel_size
        )
        
        if total_needed > self.num_devices:
            logger.warning(
                f"Requested {total_needed} devices but only {self.num_devices} available. "
                "Falling back to data parallelism only."
            )
            mesh_shape: Tuple[int, ...] = (self.num_devices,)
            axis_names: Tuple[str, ...] = ("data",)
        elif self.tensor_parallel_size > 1 and self.pipeline_parallel_size > 1:
            mesh_shape = (
                self.data_parallel_size,
                self.tensor_parallel_size,
                self.pipeline_parallel_size
            )
            axis_names = ("data", "tensor", "pipeline")
        elif self.tensor_parallel_size > 1:
            mesh_shape = (self.data_parallel_size, self.tensor_parallel_size)
            axis_names = ("data", "tensor")
        elif self.pipeline_parallel_size > 1:
            mesh_shape = (self.data_parallel_size, self.pipeline_parallel_size)
            axis_names = ("data", "pipeline")
        else:
            mesh_shape = (self.data_parallel_size,)
            axis_names = ("data",)
            
        devices = mesh_utils.create_device_mesh(mesh_shape)
        return Mesh(devices, axis_names)
    
    def get_sharding(self, spec: P) -> NamedSharding:
        """Get named sharding for a partition spec"""
        return NamedSharding(self.mesh, spec)
    
    @property
    def has_tensor_parallel(self) -> bool:
        return self.tensor_parallel_size > 1
    
    @property
    def has_pipeline_parallel(self) -> bool:
        return self.pipeline_parallel_size > 1
    
    @property
    def is_distributed(self) -> bool:
        return self.num_devices > 1


# =============================================================================
# Sharding Specifications for Model Parameters
# =============================================================================

def get_param_sharding_spec(
    param_name: str, 
    param_shape: Tuple[int, ...],
    mesh: ScalableMesh
) -> P:
    """
    Get partition spec for a parameter based on its name and shape.
    
    Sharding strategy:
    - Embedding: Shard vocabulary across tensor parallel axis
    - Attention Q/K/V: Shard heads across tensor parallel axis
    - MLP: Shard hidden dim across tensor parallel axis
    - Layer Norm: Replicate (no sharding)
    - Biases: Replicate (no sharding)
    """
    if not mesh.has_tensor_parallel:
        return P()
    
    name_lower = param_name.lower()
    
    if "embed" in name_lower and len(param_shape) == 2:
        return P(None, "tensor")
    
    if any(x in name_lower for x in ["query", "key", "value", "q_proj", "k_proj", "v_proj"]):
        if len(param_shape) == 2:
            return P(None, "tensor") 
    
    if "output" in name_lower or "o_proj" in name_lower:
        if len(param_shape) == 2:
            return P("tensor", None)
    
    if any(x in name_lower for x in ["fc1", "up_proj", "gate_proj", "w1", "w3"]):
        if len(param_shape) == 2:
            return P(None, "tensor")
    
    if any(x in name_lower for x in ["fc2", "down_proj", "w2"]):
        if len(param_shape) == 2:
            return P("tensor", None)
    
    return P()


def create_sharded_params(
    params: Dict,
    mesh: ScalableMesh
) -> Tuple[Dict, Dict]:
    """
    Create sharding specifications for all parameters.
    
    Returns:
        Tuple of (params, param_shardings)
    """
    def get_spec_for_path(path: Tuple[str, ...], param: jnp.ndarray) -> P:
        param_name = "/".join(str(p) for p in path)
        return get_param_sharding_spec(param_name, param.shape, mesh)
    
    param_specs = jax.tree_util.tree_map_with_path(
        get_spec_for_path,
        params
    )
    
    param_shardings = jax.tree_util.tree_map(
        lambda spec: mesh.get_sharding(spec),
        param_specs
    )
    
    return params, param_shardings


# =============================================================================
# Scalable Training Step
# =============================================================================

def create_scalable_train_step(
    model_apply_fn: Callable,
    optimizer,
    mesh: ScalableMesh,
    loss_fn: Callable,
    param_shardings: Optional[Dict] = None
) -> Callable:
    """
    Create a production-ready training step with automatic parallelism.
    
    Supports:
    - Single device: Standard JIT compilation
    - Data parallelism: pmap with gradient all-reduce
    - Tensor parallelism: shard_map with explicit sharding
    - Combined: Both data and tensor parallelism
    """
    
    def train_step_impl(params, opt_state, batch, rng):
        """Core training step logic."""
        
        def compute_loss(params, batch, rng):
            outputs = model_apply_fn(params, rng, **batch)
            loss = loss_fn(outputs, batch)
            return loss, outputs
        
        (loss, outputs), grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(params, batch, rng)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = jax.tree_util.tree_map(
            lambda p, u: p + u, params, updates
        )
        
        return new_params, new_opt_state, loss, outputs
    
    def train_step_with_sync(params, opt_state, batch, rng):
        """Training step with gradient synchronization for data parallelism."""
        
        def compute_loss(params, batch, rng):
            outputs = model_apply_fn(params, rng, **batch)
            loss = loss_fn(outputs, batch)
            return loss, outputs
        
        (loss, outputs), grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(params, batch, rng)
        
        grads = lax.pmean(grads, axis_name="data")
        loss = lax.pmean(loss, axis_name="data")
        
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = jax.tree_util.tree_map(
            lambda p, u: p + u, params, updates
        )
        
        return new_params, new_opt_state, loss, outputs
    
    if mesh.has_tensor_parallel and param_shardings is not None:
        in_shardings = (
            param_shardings,
            param_shardings,
            mesh.get_sharding(P("data")),
            mesh.get_sharding(P())
        )
        out_shardings = in_shardings
        
        @partial(jax.jit, in_shardings=in_shardings, out_shardings=out_shardings)
        def tensor_parallel_step(params, opt_state, batch, rng):
            return train_step_impl(params, opt_state, batch, rng)
        
        return tensor_parallel_step
        
    elif mesh.is_distributed and not mesh.has_tensor_parallel:
        return jax.pmap(
            train_step_with_sync,
            axis_name="data",
            in_axes=(0, 0, 0, 0),
            out_axes=(0, 0, 0, 0)
        )
    else:
        return jax.jit(train_step_impl)


# =============================================================================
# Factory Functions
# =============================================================================

def create_scalable_mesh(config) -> ScalableMesh:
    """
    Create a scalable mesh from AGI config.
    
    Args:
        config: AGIConfig instance
        
    Returns:
        ScalableMesh configured for the available hardware
    """
    mp_config = ModelParallelConfig.from_agi_config(config)
    return ScalableMesh(mp_config)


def setup_scalable_training(
    model_fn: Callable,
    config,
    sample_batch: Dict
) -> Tuple[ScalableMesh, Dict, Callable]:
    """
    Set up scalable training for the AGI model.
    
    Args:
        model_fn: The Haiku transformed model (from create_rtdlm_agi)
        config: AGIConfig
        sample_batch: Sample batch for initialization
        
    Returns:
        Tuple of (mesh, initial_params, train_step_fn)
    """
    mesh = create_scalable_mesh(config)
    
    rng = jax.random.PRNGKey(42)
    params = model_fn.init(rng, **sample_batch)
    
    if mesh.has_tensor_parallel:
        params, param_shardings = create_sharded_params(params, mesh)
        params = jax.tree_util.tree_map(
            lambda p, s: jax.device_put(p, s),
            params, param_shardings
        )
    elif mesh.is_distributed:
        params = jax.device_put_replicated(params, jax.devices()[:mesh.data_parallel_size])
    
    logger.info(f"Model initialized with {sum(p.size for p in jax.tree_util.tree_leaves(params)):,} parameters")
    
    return mesh, params


def replicate_for_data_parallel(params, num_devices: int):
    """Replicate parameters for data parallel training"""
    return jax.device_put_replicated(params, jax.devices()[:num_devices])


def unreplicate_params(params):
    """Get single copy of parameters from replicated state"""
    return jax.tree_util.tree_map(lambda x: x[0], params)


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_model_memory(params) -> Dict[str, float]:
    """Estimate memory usage of model parameters"""
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    
    # Assume float32 (4 bytes per param)
    param_memory_gb = (total_params * 4) / (1024 ** 3)
    
    # Optimizer states (Adam has 2x for momentum and variance)
    optimizer_memory_gb = param_memory_gb * 2
    
    # Gradients
    gradient_memory_gb = param_memory_gb
    
    # Activations (rough estimate: 2-4x params for typical batch)
    activation_memory_gb = param_memory_gb * 3
    
    total_gb = param_memory_gb + optimizer_memory_gb + gradient_memory_gb + activation_memory_gb
    
    return {
        "parameters_gb": param_memory_gb,
        "optimizer_gb": optimizer_memory_gb,
        "gradients_gb": gradient_memory_gb,
        "activations_gb": activation_memory_gb,
        "total_gb": total_gb,
        "total_params": total_params
    }


def recommend_parallelism(
    model_memory_gb: float,
    device_memory_gb: float = 16.0,
    num_devices: int = None
) -> Dict[str, Any]:
    """
    Recommend parallelism strategy based on model and device memory.
    
    Args:
        model_memory_gb: Total memory needed for model
        device_memory_gb: Memory per device (default 16GB for V100)
        num_devices: Number of available devices
        
    Returns:
        Recommended configuration
    """
    if num_devices is None:
        num_devices = jax.device_count()
    
    # Can model fit on single device?
    single_device_fit = model_memory_gb < device_memory_gb * 0.8
    
    if single_device_fit:
        if num_devices == 1:
            return {
                "strategy": "single_device",
                "data_parallel": False,
                "tensor_parallel": False,
                "recommendation": "Model fits on single device. No parallelism needed."
            }
        else:
            return {
                "strategy": "data_parallel",
                "data_parallel": True,
                "tensor_parallel": False,
                "data_parallel_size": num_devices,
                "recommendation": f"Use data parallelism across {num_devices} devices for faster training."
            }
    else:
        # Model doesn't fit on single device
        min_devices_needed = int(model_memory_gb / (device_memory_gb * 0.7)) + 1
        
        if min_devices_needed > num_devices:
            return {
                "strategy": "insufficient_memory",
                "data_parallel": False,
                "tensor_parallel": True,
                "tensor_parallel_size": num_devices,
                "recommendation": f"Model needs ~{min_devices_needed} devices but only {num_devices} available. "
                                  "Enable gradient checkpointing and reduce batch size."
            }
        else:
            tp_size = min_devices_needed
            dp_size = num_devices // tp_size
            
            return {
                "strategy": "combined",
                "data_parallel": dp_size > 1,
                "tensor_parallel": True,
                "tensor_parallel_size": tp_size,
                "data_parallel_size": dp_size,
                "recommendation": f"Use tensor parallelism ({tp_size} devices) + "
                                  f"data parallelism ({dp_size} replicas)."
            }


# =============================================================================
# Communication Profiling
# =============================================================================

class DistributedProfiler:
    """Profile distributed training communication overhead."""
    
    def __init__(self, mesh: ScalableMesh):
        self.mesh = mesh
        self.timings: Dict[str, list] = {
            "all_reduce": [],
            "all_gather": [],
            "broadcast": [],
            "total_step": [],
            "compute": [],
        }
    
    def profile_all_reduce(self, tensor: jnp.ndarray, num_iterations: int = 10) -> Dict[str, float]:
        """Measure all-reduce latency and bandwidth."""
        if not self.mesh.is_distributed:
            return {"latency_ms": 0.0, "bandwidth_gbps": 0.0, "message": "Single device, no communication"}
        
        tensor_bytes = tensor.size * tensor.dtype.itemsize
        
        jax.block_until_ready(tensor)
        
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            result = lax.psum(tensor, axis_name="data")
            jax.block_until_ready(result)
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times[2:]) / len(times[2:]) if len(times) > 2 else sum(times) / len(times)
        bandwidth = (tensor_bytes * 2 * (self.mesh.data_parallel_size - 1)) / avg_time / 1e9
        
        self.timings["all_reduce"].append(avg_time * 1000)
        
        return {
            "latency_ms": avg_time * 1000,
            "bandwidth_gbps": bandwidth,
            "tensor_size_mb": tensor_bytes / 1e6,
            "num_devices": self.mesh.data_parallel_size
        }
    
    def profile_train_step(
        self,
        train_step_fn: Callable,
        params,
        opt_state,
        batch: Dict,
        rng,
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """Profile complete training step including communication."""
        jax.block_until_ready(params)
        
        times = []
        for i in range(num_iterations):
            rng, step_rng = jax.random.split(rng)
            start = time.perf_counter()
            params, opt_state, loss, _ = train_step_fn(params, opt_state, batch, step_rng)
            jax.block_until_ready(loss)
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times[2:]) / len(times[2:]) if len(times) > 2 else sum(times) / len(times)
        
        self.timings["total_step"].append(avg_time * 1000)
        
        return {
            "avg_step_time_ms": avg_time * 1000,
            "throughput_steps_per_sec": 1.0 / avg_time,
            "num_devices": self.mesh.num_devices,
            "is_distributed": self.mesh.is_distributed
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        def safe_avg(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        return {
            "avg_all_reduce_ms": safe_avg(self.timings["all_reduce"]),
            "avg_step_time_ms": safe_avg(self.timings["total_step"]),
            "num_devices": self.mesh.num_devices,
            "data_parallel_size": self.mesh.data_parallel_size,
            "tensor_parallel_size": self.mesh.tensor_parallel_size,
            "is_distributed": self.mesh.is_distributed
        }


def profile_collective_communication(
    mesh: ScalableMesh,
    array_size_bytes: int = 1_000_000
) -> Dict[str, float]:
    """
    Profile collective communication operations.
    
    Args:
        mesh: ScalableMesh to profile
        array_size_bytes: Size of test array in bytes
        
    Returns:
        Dict with timing and bandwidth results
    """
    num_elements = array_size_bytes // 4  # float32
    test_array = jnp.ones(num_elements, dtype=jnp.float32)
    
    if not mesh.is_distributed:
        return {
            "all_reduce_time_ms": 0.0,
            "bandwidth_gbps": float('inf'),
            "num_devices": 1,
            "message": "Single device - no communication overhead"
        }
    
    profiler = DistributedProfiler(mesh)
    results = profiler.profile_all_reduce(test_array, num_iterations=10)
    
    return {
        "all_reduce_time_ms": results["latency_ms"],
        "bandwidth_gbps": results["bandwidth_gbps"],
        "num_devices": mesh.num_devices,
        "array_size_mb": array_size_bytes / 1e6
    }


def validate_distributed_setup(mesh: ScalableMesh) -> Dict[str, Any]:
    """Validate distributed training setup is working correctly."""
    results = {
        "num_devices": mesh.num_devices,
        "device_types": [str(d) for d in jax.devices()[:4]],
        "mesh_shape": str(mesh.mesh.shape),
        "checks": {}
    }
    
    results["checks"]["devices_visible"] = mesh.num_devices > 0
    
    if mesh.is_distributed:
        try:
            test_tensor = jnp.ones((mesh.data_parallel_size, 100))
            test_tensor = jax.device_put_replicated(jnp.ones(100), jax.devices()[:mesh.data_parallel_size])
            
            @partial(jax.pmap, axis_name="data")
            def test_sync(x):
                return lax.psum(x, axis_name="data")
            
            result = test_sync(test_tensor)
            expected = mesh.data_parallel_size
            results["checks"]["all_reduce_works"] = bool(jnp.allclose(result[0][0], expected))
        except Exception as e:
            results["checks"]["all_reduce_works"] = False
            results["checks"]["all_reduce_error"] = str(e)
    else:
        results["checks"]["all_reduce_works"] = True
    
    results["checks"]["mesh_valid"] = mesh.mesh is not None
    results["valid"] = all(v for k, v in results["checks"].items() if isinstance(v, bool))
    
    return results
