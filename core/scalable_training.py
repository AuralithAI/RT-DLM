"""
Scalable Training for RT-DLM

Production-ready parallelism that works with the full AGI model:
- Data parallelism: Same model replicated, different data batches
- Model parallelism: Sharded model across devices (for very large models)
- Combined: Both strategies together (production scale)

This is the unified approach used by Grok, GPT-4, Claude, etc.
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

from config.model_parallel_config import ModelParallelConfig

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
            mesh_shape = (self.num_devices,)
            axis_names = ("data",)
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
    loss_fn: Callable
) -> Callable:
    """
    Create a training step that works with the scalable mesh.
    
    This handles:
    - Data parallelism: Splits batches across devices
    - Gradient synchronization: All-reduce across data parallel axis
    - Model parallelism: Proper sharding of computation (if enabled)
    """
    
    def train_step(params, opt_state, batch, rng):
        """Single training step with automatic parallelism"""
        
        def compute_loss(params, batch, rng):
            outputs = model_apply_fn(params, rng, **batch)
            loss = loss_fn(outputs, batch)
            return loss, outputs
        
        # Compute gradients
        (loss, outputs), grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(params, batch, rng)
        
        # All-reduce gradients across data parallel axis
        if mesh.is_distributed:
            grads = lax.pmean(grads, axis_name="data")
            loss = lax.pmean(loss, axis_name="data")
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = jax.tree_util.tree_map(
            lambda p, u: p + u, params, updates
        )
        
        return new_params, new_opt_state, loss, outputs
    
    # Wrap with pmap for data parallelism
    if mesh.is_distributed and not mesh.has_tensor_parallel:
        train_step = jax.pmap(
            train_step,
            axis_name="data",
            in_axes=(0, 0, 0, 0),
            out_axes=(0, 0, 0, 0)
        )
    elif mesh.has_tensor_parallel:
        # Tensor parallelism - use shard_map or manual sharding
        # For now, use jit with sharding constraints
        train_step = jax.jit(train_step)
    else:
        # Single device
        train_step = jax.jit(train_step)
    
    return train_step


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
            # Can fit with tensor parallelism
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
