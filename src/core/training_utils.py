"""
Training Utilities for RT-DLM

Performance optimization utilities including:
- Mixed precision training (bfloat16/float16)
- Gradient checkpointing for memory efficiency
- Distributed training support (data/model parallelism)
"""

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint as jax_checkpoint  # Gradient checkpointing
import haiku as hk
import optax
from typing import Dict, Any, Optional, Callable, Tuple, List
from functools import partial
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Mixed Precision Training
# =============================================================================

class MixedPrecisionPolicy:
    """
    Mixed precision training policy for JAX/Haiku.
    
    Supports:
    - float32 (full precision)
    - bfloat16 (brain floating point - preferred for TPUs)
    - float16 (half precision - preferred for GPUs with Tensor Cores)
    """
    
    DTYPE_MAP = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }
    
    def __init__(self, 
                 param_dtype: str = "float32",
                 compute_dtype: str = "float32",
                 output_dtype: str = "float32"):
        """
        Initialize mixed precision policy.
        
        Args:
            param_dtype: Dtype for model parameters (storage)
            compute_dtype: Dtype for forward/backward computations
            output_dtype: Dtype for outputs (usually float32 for loss)
        """
        self.param_dtype = self.DTYPE_MAP.get(param_dtype, jnp.float32)
        self.compute_dtype = self.DTYPE_MAP.get(compute_dtype, jnp.float32)
        self.output_dtype = self.DTYPE_MAP.get(output_dtype, jnp.float32)
        
        self._param_dtype_str = param_dtype
        self._compute_dtype_str = compute_dtype
        
    def cast_to_compute(self, x: jnp.ndarray) -> jnp.ndarray:
        """Cast tensor to compute dtype"""
        return x.astype(self.compute_dtype)
    
    def cast_to_param(self, x: jnp.ndarray) -> jnp.ndarray:
        """Cast tensor to parameter dtype"""
        return x.astype(self.param_dtype)
    
    def cast_to_output(self, x: jnp.ndarray) -> jnp.ndarray:
        """Cast tensor to output dtype (float32 for loss stability)"""
        return x.astype(self.output_dtype)
    
    def cast_params(self, params: Dict) -> Dict:
        """Cast all parameters to param dtype"""
        return jax.tree_util.tree_map(
            lambda x: x.astype(self.param_dtype) if hasattr(x, 'astype') else x,
            params
        )
    
    def cast_inputs(self, inputs: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Cast input tensors to compute dtype"""
        return jax.tree_util.tree_map(
            lambda x: x.astype(self.compute_dtype) if hasattr(x, 'astype') and x.dtype in [jnp.float32, jnp.float16, jnp.bfloat16] else x,
            inputs
        )
    
    def is_mixed_precision(self) -> bool:
        """Check if using mixed precision"""
        return self.compute_dtype != jnp.float32 or self.param_dtype != jnp.float32
    
    def __repr__(self):
        return f"MixedPrecisionPolicy(param={self._param_dtype_str}, compute={self._compute_dtype_str})"


def create_mixed_precision_policy(config) -> MixedPrecisionPolicy:
    """Create mixed precision policy from config"""
    if config.mixed_precision:
        return MixedPrecisionPolicy(
            param_dtype=config.precision_dtype,
            compute_dtype=config.compute_dtype,
            output_dtype="float32"  # Always use float32 for loss
        )
    return MixedPrecisionPolicy()  # Default full precision


def apply_loss_scaling(loss: jnp.ndarray, scale: float = 65536.0) -> jnp.ndarray:
    """Apply loss scaling for mixed precision training to prevent underflow"""
    return loss * scale


def unscale_gradients(grads: Dict, scale: float = 65536.0) -> Dict:
    """Unscale gradients after backward pass"""
    return jax.tree_util.tree_map(lambda g: g / scale, grads)


# =============================================================================
# Gradient Checkpointing (Activation Checkpointing)
# =============================================================================

def checkpoint_layer(fn: Callable, *args, **kwargs) -> Any:
    """
    Apply gradient checkpointing to a layer function.
    
    This recomputes activations during backward pass instead of storing them,
    trading compute for memory.
    
    Args:
        fn: Layer function to checkpoint
        *args: Positional arguments for fn
        **kwargs: Keyword arguments for fn
        
    Returns:
        Output of fn with checkpointing applied
    """
    return jax_checkpoint(fn)(*args, **kwargs)


def create_checkpointed_layer(layer_fn: Callable, checkpoint: bool = True) -> Callable:
    """
    Wrap a layer function with optional gradient checkpointing.
    
    Args:
        layer_fn: Original layer function
        checkpoint: Whether to apply checkpointing
        
    Returns:
        Checkpointed or original layer function
    """
    if checkpoint:
        return jax_checkpoint(layer_fn)
    return layer_fn


class GradientCheckpointingConfig:
    """Configuration for gradient checkpointing"""
    
    def __init__(self, 
                 enabled: bool = False,
                 checkpoint_every_n_layers: int = 2,
                 checkpoint_attention: bool = True,
                 checkpoint_ffn: bool = True):
        self.enabled = enabled
        self.checkpoint_every_n_layers = checkpoint_every_n_layers
        self.checkpoint_attention = checkpoint_attention
        self.checkpoint_ffn = checkpoint_ffn
        
    def should_checkpoint_layer(self, layer_idx: int) -> bool:
        """Determine if a specific layer should be checkpointed"""
        if not self.enabled:
            return False
        return layer_idx % self.checkpoint_every_n_layers == 0


def checkpointed_transformer_block(
    layer_fn: Callable,
    layer_idx: int,
    checkpoint_config: GradientCheckpointingConfig,
    x: jnp.ndarray,
    *args,
    **kwargs
) -> jnp.ndarray:
    """
    Apply transformer block with optional gradient checkpointing.
    
    Args:
        layer_fn: Transformer layer function
        layer_idx: Index of the layer
        checkpoint_config: Checkpointing configuration
        x: Input tensor
        
    Returns:
        Output tensor
    """
    if checkpoint_config.should_checkpoint_layer(layer_idx):
        return jax_checkpoint(layer_fn)(x, *args, **kwargs)
    return layer_fn(x, *args, **kwargs)


# =============================================================================
# Distributed Training Support
# =============================================================================

class DistributedTrainingConfig:
    """Configuration for distributed training"""
    
    def __init__(self,
                 enabled: bool = False,
                 num_devices: int = 1,
                 data_parallel: bool = True,
                 model_parallel: bool = False,
                 gradient_accumulation_steps: int = 1):
        self.enabled = enabled
        self.num_devices = num_devices
        self.data_parallel = data_parallel
        self.model_parallel = model_parallel
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Auto-detect devices if not specified
        if self.enabled and self.num_devices == 1:
            self.num_devices = jax.device_count()
            
    @property
    def effective_batch_size_multiplier(self) -> int:
        """Get effective batch size multiplier from parallelism + accumulation"""
        return self.num_devices * self.gradient_accumulation_steps


def create_distributed_config(config) -> DistributedTrainingConfig:
    """Create distributed training config from AGI config"""
    return DistributedTrainingConfig(
        enabled=config.distributed_training,
        num_devices=config.num_devices,
        data_parallel=config.data_parallel,
        model_parallel=config.model_parallel,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )


def shard_batch_for_devices(batch: Dict[str, jnp.ndarray], 
                            num_devices: int) -> Dict[str, jnp.ndarray]:
    """
    Shard a batch across multiple devices for data parallelism.
    
    Args:
        batch: Dictionary of batch tensors [batch_size, ...]
        num_devices: Number of devices to shard across
        
    Returns:
        Sharded batch [num_devices, batch_per_device, ...]
    """
    def shard_array(x):
        if x is None:
            return None
        batch_size = x.shape[0]
        if batch_size % num_devices != 0:
            # Pad to make divisible
            pad_size = num_devices - (batch_size % num_devices)
            pad_shape = (pad_size,) + x.shape[1:]
            x = jnp.concatenate([x, jnp.zeros(pad_shape, dtype=x.dtype)], axis=0)
        
        return x.reshape(num_devices, -1, *x.shape[1:])
    
    return jax.tree_util.tree_map(shard_array, batch)


def unshard_batch(sharded_batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """
    Unshard a batch from multiple devices back to single batch.
    
    Args:
        sharded_batch: Sharded batch [num_devices, batch_per_device, ...]
        
    Returns:
        Unsharded batch [batch_size, ...]
    """
    def unshard_array(x):
        if x is None:
            return None
        return x.reshape(-1, *x.shape[2:])
    
    return jax.tree_util.tree_map(unshard_array, sharded_batch)


def create_pmap_train_step(train_step_fn: Callable,
                           axis_name: str = "devices") -> Callable:
    """
    Create a pmapped training step for data parallelism.
    
    Args:
        train_step_fn: Single-device training step function
        axis_name: Axis name for gradient aggregation
        
    Returns:
        Parallelized training step
    """
    @partial(jax.pmap, axis_name=axis_name)
    def pmap_train_step(params, opt_state, batch, rng):
        # Run training step on each device
        new_params, new_opt_state, loss, outputs = train_step_fn(
            params, opt_state, batch, rng
        )
        
        # Average gradients across devices (implicitly done in optimizer)
        # Average loss across devices
        loss = jax.lax.pmean(loss, axis_name=axis_name)
        
        return new_params, new_opt_state, loss, outputs
    
    return pmap_train_step


def replicate_params(params: Dict, num_devices: int) -> Dict:
    """Replicate parameters across devices for pmap"""
    return jax.device_put_replicated(params, jax.devices()[:num_devices])


def unreplicate_params(replicated_params: Dict) -> Dict:
    """Get single copy of parameters from replicated params"""
    return jax.tree_util.tree_map(lambda x: x[0], replicated_params)


# =============================================================================
# Gradient Accumulation
# =============================================================================

class GradientAccumulator:
    """
    Gradient accumulation for effective larger batch sizes.
    
    Useful when GPU memory is limited but larger effective batch sizes
    are needed for stable training.
    """
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_grads = None
        
    def accumulate(self, grads: Dict) -> Tuple[bool, Optional[Dict]]:
        """
        Accumulate gradients over multiple steps.
        
        Args:
            grads: Gradients from current batch
            
        Returns:
            Tuple of (should_update, averaged_grads_or_none)
        """
        if self.accumulated_grads is None:
            self.accumulated_grads = grads
        else:
            self.accumulated_grads = jax.tree_util.tree_map(
                lambda a, b: a + b,
                self.accumulated_grads,
                grads
            )
        
        self.current_step += 1
        
        if self.current_step >= self.accumulation_steps:
            # Average gradients and return
            averaged_grads = jax.tree_util.tree_map(
                lambda g: g / self.accumulation_steps,
                self.accumulated_grads
            )
            self.reset()
            return True, averaged_grads
        
        return False, None
    
    def reset(self):
        """Reset accumulator state"""
        self.current_step = 0
        self.accumulated_grads = None


def create_accumulated_train_step(
    base_train_step: Callable,
    accumulation_steps: int
) -> Callable:
    """
    Create training step with gradient accumulation.
    
    Args:
        base_train_step: Base training step that returns (params, opt_state, loss, grads)
        accumulation_steps: Number of steps to accumulate
        
    Returns:
        Training step with accumulation
    """
    def accumulated_step(params, opt_state, batches: List[Dict], rng, optimizer):
        """
        Run accumulated training step over multiple batches.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            batches: List of batches to accumulate over
            rng: Random key
            optimizer: Optax optimizer
            
        Returns:
            Updated params, opt_state, averaged loss
        """
        total_loss = 0.0
        accumulated_grads = None
        
        for i, batch in enumerate(batches[:accumulation_steps]):
            rng, step_rng = jax.random.split(rng)
            
            # Get gradients for this batch (capture batch and step_rng in closure properly)
            def loss_fn(p, b=batch, r=step_rng):
                return base_train_step(p, opt_state, b, r)[2]
            
            loss, grads = jax.value_and_grad(loss_fn)(params)
            
            total_loss += loss
            
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = jax.tree_util.tree_map(
                    jnp.add, accumulated_grads, grads
                )
        
        # Average gradients
        averaged_grads = jax.tree_util.tree_map(
            lambda g: g / accumulation_steps,
            accumulated_grads
        )
        
        # Apply averaged gradients using optimizer
        updates, new_opt_state = optimizer.update(averaged_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        avg_loss = total_loss / accumulation_steps
        
        return new_params, new_opt_state, avg_loss, None
    
    return accumulated_step


# =============================================================================
# Memory Optimization Utilities
# =============================================================================

def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics from JAX devices"""
    stats = {}
    for i, device in enumerate(jax.devices()):
        try:
            mem_stats = device.memory_stats()
            if mem_stats:
                stats[f"device_{i}"] = {
                    "bytes_in_use": mem_stats.get("bytes_in_use", 0),
                    "bytes_limit": mem_stats.get("bytes_limit", 0),
                    "peak_bytes_in_use": mem_stats.get("peak_bytes_in_use", 0),
                }
        except Exception:
            pass  # Not all devices support memory_stats
    return stats


def log_memory_usage(prefix: str = ""):
    """Log current memory usage"""
    stats = get_memory_stats()
    for device_name, device_stats in stats.items():
        bytes_used = device_stats.get("bytes_in_use", 0)
        bytes_limit = device_stats.get("bytes_limit", 0)
        if bytes_limit > 0:
            usage_pct = 100 * bytes_used / bytes_limit
            logger.info(f"{prefix}{device_name}: {bytes_used / 1e9:.2f}GB / {bytes_limit / 1e9:.2f}GB ({usage_pct:.1f}%)")


def optimize_for_memory(config) -> None:
    """Apply memory optimizations based on config"""
    # Enable memory-efficient attention if needed
    if config.gradient_checkpointing:
        logger.info("Gradient checkpointing enabled - trading compute for memory")
    
    if config.mixed_precision:
        logger.info(f"Mixed precision enabled - using {config.precision_dtype} for weights")
        
    # Set JAX memory preallocation (prevent OOM)
    import os
    if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# =============================================================================
# Training Utilities Factory
# =============================================================================

class TrainingOptimizations:
    """
    Combined training optimizations container.
    
    Usage:
        optimizations = TrainingOptimizations.from_config(config)
        policy = optimizations.precision_policy
        
        if optimizations.distributed.enabled:
            train_step = optimizations.create_distributed_train_step(base_step)
    """
    
    def __init__(self,
                 precision_policy: MixedPrecisionPolicy,
                 checkpoint_config: GradientCheckpointingConfig,
                 distributed_config: DistributedTrainingConfig):
        self.precision_policy = precision_policy
        self.checkpoint_config = checkpoint_config
        self.distributed = distributed_config
        
    @classmethod
    def from_config(cls, config) -> "TrainingOptimizations":
        """Create training optimizations from AGI config"""
        precision_policy = create_mixed_precision_policy(config)
        
        checkpoint_config = GradientCheckpointingConfig(
            enabled=config.gradient_checkpointing,
            checkpoint_every_n_layers=config.checkpoint_every_n_layers,
        )
        
        distributed_config = create_distributed_config(config)
        
        return cls(precision_policy, checkpoint_config, distributed_config)
    
    def prepare_batch(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Prepare batch with precision casting and optional sharding"""
        # Cast to compute dtype
        batch = self.precision_policy.cast_inputs(batch)
        
        # Shard for distributed training
        if self.distributed.enabled and self.distributed.data_parallel:
            batch = shard_batch_for_devices(batch, self.distributed.num_devices)
            
        return batch
    
    def prepare_params(self, params: Dict) -> Dict:
        """Prepare parameters with precision casting and replication"""
        # Cast to param dtype
        params = self.precision_policy.cast_params(params)
        
        # Replicate for distributed training
        if self.distributed.enabled and self.distributed.data_parallel:
            params = replicate_params(params, self.distributed.num_devices)
            
        return params
    
    def create_train_step(self, base_step: Callable) -> Callable:
        """Create optimized training step with all features"""
        step = base_step
        
        # Add distributed training
        if self.distributed.enabled and self.distributed.data_parallel:
            step = create_pmap_train_step(step)
            logger.info(f"Created pmap training step for {self.distributed.num_devices} devices")
            
        return step
    
    def summary(self) -> str:
        """Get summary of active optimizations"""
        lines = ["Training Optimizations:"]
        lines.append(f"  - Precision: {self.precision_policy}")
        lines.append(f"  - Gradient checkpointing: {self.checkpoint_config.enabled}")
        lines.append(f"  - Distributed training: {self.distributed.enabled}")
        if self.distributed.enabled:
            lines.append(f"    - Devices: {self.distributed.num_devices}")
            lines.append(f"    - Data parallel: {self.distributed.data_parallel}")
            lines.append(f"    - Model parallel: {self.distributed.model_parallel}")
            lines.append(f"    - Gradient accumulation: {self.distributed.gradient_accumulation_steps}")
        return "\n".join(lines)
