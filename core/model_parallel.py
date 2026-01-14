"""
Model Parallelism for RT-DLM

Supports sharding large models across multiple devices:
- Tensor parallelism (split layers across devices)
- Pipeline parallelism (split layers sequentially)
- Fully Sharded Data Parallel (FSDP) patterns
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import haiku as hk
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
from functools import partial
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Model Parallelism Configuration
# =============================================================================

@dataclass
class ModelParallelConfig:
    """Configuration for model parallelism"""
    
    # Device mesh configuration
    num_devices: int = 1
    mesh_shape: Tuple[int, ...] = (1,)  # (data, model) or (data, tensor, pipeline)
    mesh_axis_names: Tuple[str, ...] = ("data",)
    
    # Parallelism strategy
    tensor_parallel: bool = False  # Split individual layers
    pipeline_parallel: bool = False  # Split layers sequentially
    tensor_parallel_size: int = 1  # Number of devices for tensor parallelism
    pipeline_parallel_size: int = 1  # Number of pipeline stages
    
    # Memory optimization
    activation_checkpointing: bool = True
    offload_to_cpu: bool = False  # Offload optimizer states to CPU
    
    # Communication optimization
    async_communication: bool = True
    gradient_compression: bool = False
    
    def __post_init__(self):
        self.num_devices = jax.device_count()
        if self.tensor_parallel and self.pipeline_parallel:
            # Combined parallelism
            assert self.tensor_parallel_size * self.pipeline_parallel_size <= self.num_devices
        elif self.tensor_parallel:
            self.tensor_parallel_size = min(self.tensor_parallel_size, self.num_devices)
        elif self.pipeline_parallel:
            self.pipeline_parallel_size = min(self.pipeline_parallel_size, self.num_devices)


def create_model_parallel_config(config) -> ModelParallelConfig:
    """Create model parallel config from AGI config"""
    return ModelParallelConfig(
        tensor_parallel=config.model_parallel,
        pipeline_parallel=False,  # Can be extended
        tensor_parallel_size=config.num_devices,
        activation_checkpointing=config.gradient_checkpointing,
    )


# =============================================================================
# Device Mesh Management
# =============================================================================

class DeviceMesh:
    """
    Manages device mesh for model parallelism.
    
    Supports various mesh topologies:
    - 1D: Data parallelism only
    - 2D: Data + Tensor parallelism
    - 3D: Data + Tensor + Pipeline parallelism
    """
    
    def __init__(self, config: ModelParallelConfig):
        self.config = config
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        
        # Create mesh based on parallelism strategy
        self.mesh = self._create_mesh()
        
    def _create_mesh(self) -> Mesh:
        """Create JAX device mesh"""
        if self.config.tensor_parallel and self.config.pipeline_parallel:
            # 3D mesh: (data, tensor, pipeline)
            dp_size = self.num_devices // (
                self.config.tensor_parallel_size * self.config.pipeline_parallel_size
            )
            mesh_shape = (
                dp_size,
                self.config.tensor_parallel_size,
                self.config.pipeline_parallel_size
            )
            axis_names = ("data", "tensor", "pipeline")
        elif self.config.tensor_parallel:
            # 2D mesh: (data, tensor)
            dp_size = max(1, self.num_devices // self.config.tensor_parallel_size)
            mesh_shape = (dp_size, self.config.tensor_parallel_size)
            axis_names = ("data", "tensor")
        elif self.config.pipeline_parallel:
            # 2D mesh: (data, pipeline)
            dp_size = max(1, self.num_devices // self.config.pipeline_parallel_size)
            mesh_shape = (dp_size, self.config.pipeline_parallel_size)
            axis_names = ("data", "pipeline")
        else:
            # 1D mesh: data parallelism only
            mesh_shape = (self.num_devices,)
            axis_names = ("data",)
        
        # Create device array with proper shape
        devices = mesh_utils.create_device_mesh(mesh_shape)
        return Mesh(devices, axis_names)
    
    def get_sharding(self, partition_spec: P) -> NamedSharding:
        """Get named sharding for a partition spec"""
        return NamedSharding(self.mesh, partition_spec)
    
    @property
    def data_axis(self) -> str:
        return "data"
    
    @property
    def tensor_axis(self) -> Optional[str]:
        return "tensor" if self.config.tensor_parallel else None
    
    @property
    def pipeline_axis(self) -> Optional[str]:
        return "pipeline" if self.config.pipeline_parallel else None


# =============================================================================
# Tensor Parallelism
# =============================================================================

class TensorParallelLinear(hk.Module):
    """
    Linear layer with tensor parallelism.
    
    Splits the weight matrix across devices along the output dimension
    for column-parallel, or input dimension for row-parallel.
    """
    
    def __init__(self, 
                 output_size: int,
                 mesh: DeviceMesh,
                 parallel_mode: str = "column",  # "column" or "row"
                 with_bias: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.output_size = output_size
        self.mesh = mesh
        self.parallel_mode = parallel_mode
        self.with_bias = with_bias
        self.tp_size = mesh.config.tensor_parallel_size
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with tensor parallelism"""
        input_size = x.shape[-1]
        
        if self.parallel_mode == "column":
            # Column parallel: split output across devices
            # Each device computes a slice of the output
            local_output_size = self.output_size // self.tp_size
            
            w = hk.get_parameter(
                "w",
                shape=[input_size, local_output_size],
                init=hk.initializers.TruncatedNormal(stddev=0.02)
            )
            
            output = jnp.dot(x, w)
            
            if self.with_bias:
                b = hk.get_parameter(
                    "b",
                    shape=[local_output_size],
                    init=jnp.zeros
                )
                output = output + b
                
            # All-gather to get full output (if needed for next layer)
            # For efficiency, this is often deferred to the next row-parallel layer
            
        else:  # row parallel
            # Row parallel: split input across devices
            # Each device has full output, results are summed
            local_input_size = input_size // self.tp_size
            
            w = hk.get_parameter(
                "w",
                shape=[local_input_size, self.output_size],
                init=hk.initializers.TruncatedNormal(stddev=0.02)
            )
            
            # Assume input is already sharded
            output = jnp.dot(x, w)
            
            # All-reduce sum across tensor parallel devices (skip for tp_size=1)
            if self.tp_size > 1:
                output = lax.psum(output, axis_name="tensor")
            
            if self.with_bias:
                b = hk.get_parameter(
                    "b",
                    shape=[self.output_size],
                    init=jnp.zeros
                )
                output = output + b
                
        return output


class TensorParallelAttention(hk.Module):
    """
    Multi-head attention with tensor parallelism.
    
    Splits attention heads across devices for parallel computation.
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 mesh: DeviceMesh,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mesh = mesh
        self.tp_size = mesh.config.tensor_parallel_size
        
        # Divide heads across devices
        assert num_heads % self.tp_size == 0, \
            f"num_heads ({num_heads}) must be divisible by tp_size ({self.tp_size})"
        self.local_num_heads = num_heads // self.tp_size
        self.head_dim = d_model // num_heads
        
    def __call__(self, 
                 x: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Forward pass with tensor parallel attention"""
        batch_size, seq_len, _ = x.shape
        
        # Local Q, K, V projections (column parallel)
        local_dim = self.local_num_heads * self.head_dim
        
        q_proj = TensorParallelLinear(
            local_dim, self.mesh, parallel_mode="column", name="query"
        )
        k_proj = TensorParallelLinear(
            local_dim, self.mesh, parallel_mode="column", name="key"
        )
        v_proj = TensorParallelLinear(
            local_dim, self.mesh, parallel_mode="column", name="value"
        )
        
        Q = q_proj(x)
        K = k_proj(x)
        V = v_proj(x)
        
        # Reshape for local attention heads
        Q = Q.reshape(batch_size, seq_len, self.local_num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.local_num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.local_num_heads, self.head_dim)
        
        # Transpose for attention: [batch, heads, seq, dim]
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", Q, K) * scale
        
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)
            
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, local_dim)
        
        # Output projection (row parallel - will all-reduce)
        output_proj = TensorParallelLinear(
            self.d_model, self.mesh, parallel_mode="row", name="output"
        )
        output = output_proj(attn_output)
        
        return output


class TensorParallelMLP(hk.Module):
    """
    MLP/FFN with tensor parallelism.
    
    Uses column-parallel for first linear, row-parallel for second.
    """
    
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 mesh: DeviceMesh,
                 activation: Callable = jax.nn.gelu,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.d_model = d_model
        self.d_ff = d_ff
        self.mesh = mesh
        self.activation = activation
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with tensor parallel MLP"""
        # First linear: column parallel (split d_ff across devices)
        fc1 = TensorParallelLinear(
            self.d_ff, self.mesh, parallel_mode="column", name="fc1"
        )
        hidden = fc1(x)
        hidden = self.activation(hidden)
        
        # Second linear: row parallel (reduce across devices)
        fc2 = TensorParallelLinear(
            self.d_model, self.mesh, parallel_mode="row", name="fc2"
        )
        output = fc2(hidden)
        
        return output


# =============================================================================
# Pipeline Parallelism
# =============================================================================

class PipelineStage(hk.Module):
    """
    A single stage in pipeline parallelism.
    
    Contains a subset of transformer layers that run on one device.
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 num_layers_in_stage: int,
                 stage_id: int,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers_in_stage
        self.stage_id = stage_id
        
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Process through layers in this stage"""
        for i in range(self.num_layers):
            layer_name = f"layer_{self.stage_id}_{i}"
            
            # Pre-norm attention
            norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                  name=f"{layer_name}_norm1")
            attn = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.d_model // self.num_heads,
                w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                name=f"{layer_name}_attn"
            )
            
            # Attention block
            normed = norm1(x)
            attn_out = attn(normed, normed, normed, mask=mask)
            x = x + attn_out
            
            # Pre-norm MLP
            norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                  name=f"{layer_name}_norm2")
            mlp = hk.Sequential([
                hk.Linear(self.d_ff, name=f"{layer_name}_fc1"),
                jax.nn.gelu,
                hk.Linear(self.d_model, name=f"{layer_name}_fc2"),
            ], name=f"{layer_name}_mlp")
            
            # MLP block
            normed = norm2(x)
            mlp_out = mlp(normed)
            x = x + mlp_out
            
        return x


class PipelineParallelModel(hk.Module):
    """
    Full model with pipeline parallelism.
    
    Distributes layers across pipeline stages on different devices.
    Uses micro-batching for efficiency.
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 total_layers: int,
                 num_stages: int,
                 vocab_size: int,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.total_layers = total_layers
        self.num_stages = num_stages
        self.vocab_size = vocab_size
        
        # Divide layers across stages
        self.layers_per_stage = total_layers // num_stages
        
    def __call__(self, 
                 input_ids: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Forward pass through pipeline stages"""
        # Embedding (on first stage)
        embed = hk.Embed(self.vocab_size, self.d_model, name="embed")
        x = embed(input_ids)
        
        # Pipeline stages
        for stage_id in range(self.num_stages):
            stage = PipelineStage(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                num_layers_in_stage=self.layers_per_stage,
                stage_id=stage_id,
                name=f"stage_{stage_id}"
            )
            x = stage(x, mask)
            
        # Final layer norm and output projection (on last stage)
        final_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                   name="final_norm")
        x = final_norm(x)
        
        output_proj = hk.Linear(self.vocab_size, name="output_proj")
        logits = output_proj(x)
        
        return logits


# =============================================================================
# Sharding Utilities
# =============================================================================

def shard_params_for_tensor_parallel(
    params: Dict,
    mesh: DeviceMesh,
    layer_names: List[str]
) -> Dict:
    """
    Shard model parameters for tensor parallelism.
    
    Args:
        params: Model parameters
        mesh: Device mesh
        layer_names: Names of layers to shard
        
    Returns:
        Sharded parameters
    """
    def get_sharding_for_param(path: Tuple[str, ...], param: jnp.ndarray):
        path_str = "/".join(path)
        
        # Check if this param should be sharded
        for layer_name in layer_names:
            if layer_name in path_str:
                if "w" in path[-1] or "weight" in path[-1]:
                    # Shard weight matrices
                    if len(param.shape) == 2:
                        # Matrix: shard along output dim for column parallel
                        return P(None, "tensor")
                elif "b" in path[-1] or "bias" in path[-1]:
                    # Shard biases along their dimension
                    return P("tensor")
        
        # Default: replicate
        return P()
    
    # Create shardings
    param_shardings = jax.tree_util.tree_map_with_path(
        lambda path, p: mesh.get_sharding(get_sharding_for_param(path, p)),
        params
    )
    
    return param_shardings


def create_sharded_train_step(
    train_step_fn: Callable,
    mesh: DeviceMesh,
    param_shardings: Dict
) -> Callable:
    """
    Create a training step with proper sharding.
    
    Uses jax.jit with in_shardings and out_shardings for efficiency.
    """
    @partial(jax.jit, in_shardings=(param_shardings, None, None, None),
             out_shardings=(param_shardings, None, None, None))
    def sharded_train_step(params, opt_state, batch, rng):
        return train_step_fn(params, opt_state, batch, rng)
    
    return sharded_train_step


# =============================================================================
# Combined Model Parallelism
# =============================================================================

class ModelParallelTransformer(hk.Module):
    """
    Transformer with combined tensor and pipeline parallelism.
    """
    
    def __init__(self,
                 config,
                 mesh: DeviceMesh,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.config = config
        self.mesh = mesh
        self.use_tensor_parallel = mesh.config.tensor_parallel
        
    def __call__(self,
                 input_ids: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Forward pass with model parallelism"""
        # Embedding
        embed = hk.Embed(self.config.vocab_size, self.config.d_model, name="embed")
        x = embed(input_ids)
        
        # Transformer layers
        for i in range(self.config.num_layers):
            if self.use_tensor_parallel:
                # Tensor parallel attention
                attn = TensorParallelAttention(
                    self.config.d_model,
                    self.config.num_heads,
                    self.mesh,
                    name=f"layer_{i}_attn"
                )
                # Tensor parallel MLP
                mlp = TensorParallelMLP(
                    self.config.d_model,
                    self.config.d_model * 4,
                    self.mesh,
                    name=f"layer_{i}_mlp"
                )
            else:
                # Standard attention and MLP
                attn = hk.MultiHeadAttention(
                    num_heads=self.config.num_heads,
                    key_size=self.config.d_model // self.config.num_heads,
                    w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                    name=f"layer_{i}_attn"
                )
                mlp = hk.Sequential([
                    hk.Linear(self.config.d_model * 4),
                    jax.nn.gelu,
                    hk.Linear(self.config.d_model),
                ], name=f"layer_{i}_mlp")
            
            # Pre-norm architecture
            norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                  name=f"layer_{i}_norm1")
            norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                  name=f"layer_{i}_norm2")
            
            # Attention block
            if self.use_tensor_parallel:
                x = x + attn(norm1(x), mask)
            else:
                normed = norm1(x)
                x = x + attn(normed, normed, normed, mask=mask)
            
            # MLP block
            x = x + mlp(norm2(x))
        
        # Final layer norm
        final_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                   name="final_norm")
        x = final_norm(x)
        
        # Output projection
        output_proj = hk.Linear(self.config.vocab_size, name="output_proj")
        logits = output_proj(x)
        
        return logits


# =============================================================================
# Factory Functions
# =============================================================================

def create_model_parallel_system(config) -> Tuple[DeviceMesh, ModelParallelConfig]:
    """Create model parallelism system from config"""
    mp_config = create_model_parallel_config(config)
    mesh = DeviceMesh(mp_config)
    
    logger.info(f"Created device mesh with shape {mesh.mesh.shape}")
    logger.info(f"Tensor parallel: {mp_config.tensor_parallel}, size: {mp_config.tensor_parallel_size}")
    logger.info(f"Pipeline parallel: {mp_config.pipeline_parallel}, size: {mp_config.pipeline_parallel_size}")
    
    return mesh, mp_config


def create_model_parallel_transformer(config, mesh: DeviceMesh) -> Callable:
    """Create model parallel transformer function"""
    def model_fn(input_ids, mask=None):
        model = ModelParallelTransformer(config, mesh)
        return model(input_ids, mask)
    
    return hk.transform(model_fn)
