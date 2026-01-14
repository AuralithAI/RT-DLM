"""
Model Parallelism Configuration for RT-DLM

Configuration for scaling models across multiple devices using
tensor parallelism and pipeline parallelism.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import jax


@dataclass
class ModelParallelConfig:
    """
    Configuration for model parallelism.
    
    Supports:
    - Tensor parallelism: Split individual layers across devices
    - Pipeline parallelism: Split layers sequentially across devices
    - Combined parallelism: Both tensor and pipeline together
    """
    
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
        """Initialize device count and validate settings"""
        self.num_devices = jax.device_count()
        if self.tensor_parallel and self.pipeline_parallel:
            # Combined parallelism
            assert self.tensor_parallel_size * self.pipeline_parallel_size <= self.num_devices
        elif self.tensor_parallel:
            self.tensor_parallel_size = min(self.tensor_parallel_size, self.num_devices)
        elif self.pipeline_parallel:
            self.pipeline_parallel_size = min(self.pipeline_parallel_size, self.num_devices)
    
    @classmethod
    def from_agi_config(cls, agi_config) -> "ModelParallelConfig":
        """Create model parallel config from AGI config"""
        return cls(
            tensor_parallel=agi_config.model_parallel,
            pipeline_parallel=False,  # Can be extended via agi_config
            tensor_parallel_size=agi_config.num_devices,
            activation_checkpointing=agi_config.gradient_checkpointing,
        )
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "num_devices": self.num_devices,
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "activation_checkpointing": self.activation_checkpointing,
            "offload_to_cpu": self.offload_to_cpu,
            "async_communication": self.async_communication,
            "gradient_compression": self.gradient_compression,
        }
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 50)
        print("Model Parallelism Configuration")
        print("=" * 50)
        print(f"  Devices available: {self.num_devices}")
        print(f"  Tensor parallel: {self.tensor_parallel} (size: {self.tensor_parallel_size})")
        print(f"  Pipeline parallel: {self.pipeline_parallel} (size: {self.pipeline_parallel_size})")
        print(f"  Activation checkpointing: {self.activation_checkpointing}")
        print(f"  CPU offloading: {self.offload_to_cpu}")
        print("=" * 50)
