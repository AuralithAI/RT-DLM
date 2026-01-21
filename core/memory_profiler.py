"""
Memory Profiler for RT-DLM Training.

Provides utilities for:
- Tracking peak memory usage during training
- Estimating memory requirements for model presets
- Recommending batch sizes for different GPU configurations
"""

import jax
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""
    timestamp: float
    step: int
    phase: str  # "forward", "backward", "optimizer", "idle"
    bytes_in_use: int
    peak_bytes: int
    device_id: int = 0


@dataclass 
class MemoryProfile:
    """Memory profile for a training run."""
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    model_params_bytes: int = 0
    optimizer_state_bytes: int = 0
    
    @property
    def peak_memory_gb(self) -> float:
        if not self.snapshots:
            return 0.0
        return max(s.peak_bytes for s in self.snapshots) / (1024 ** 3)
    
    @property
    def average_memory_gb(self) -> float:
        if not self.snapshots:
            return 0.0
        return sum(s.bytes_in_use for s in self.snapshots) / len(self.snapshots) / (1024 ** 3)
    
    def summary(self) -> Dict[str, Any]:
        return {
            "peak_memory_gb": self.peak_memory_gb,
            "average_memory_gb": self.average_memory_gb,
            "model_params_gb": self.model_params_bytes / (1024 ** 3),
            "optimizer_state_gb": self.optimizer_state_bytes / (1024 ** 3),
            "num_snapshots": len(self.snapshots),
        }


class MemoryProfiler:
    """
    Memory profiler for training loops.
    
    Usage:
        profiler = MemoryProfiler()
        
        for step, batch in enumerate(dataloader):
            profiler.snapshot(step, "forward")
            loss, grads = compute_loss_and_grads(params, batch)
            
            profiler.snapshot(step, "backward")
            params = update_params(params, grads)
            
            profiler.snapshot(step, "optimizer")
        
        print(profiler.summary())
    """
    
    def __init__(self, enabled: bool = True, log_every_n_steps: int = 100):
        self.enabled = enabled
        self.log_every_n_steps = log_every_n_steps
        self.profile = MemoryProfile()
        self._start_time = time.time()
    
    def set_model_size(self, params) -> None:
        """Record model parameter size."""
        if not self.enabled:
            return
        total_bytes = sum(
            p.size * p.dtype.itemsize 
            for p in jax.tree_util.tree_leaves(params)
        )
        self.profile.model_params_bytes = total_bytes
        logger.info(f"Model parameters: {total_bytes / (1024**3):.2f} GB")
    
    def set_optimizer_size(self, opt_state) -> None:
        """Record optimizer state size."""
        if not self.enabled:
            return
        total_bytes = sum(
            p.size * p.dtype.itemsize 
            for p in jax.tree_util.tree_leaves(opt_state)
            if hasattr(p, 'size')
        )
        self.profile.optimizer_state_bytes = total_bytes
        logger.info(f"Optimizer state: {total_bytes / (1024**3):.2f} GB")
    
    def snapshot(self, step: int, phase: str = "unknown") -> Optional[MemorySnapshot]:
        """Take a memory snapshot."""
        if not self.enabled:
            return None
        
        try:
            device = jax.devices()[0]
            mem_stats = device.memory_stats()
            if mem_stats:
                snapshot = MemorySnapshot(
                    timestamp=time.time() - self._start_time,
                    step=step,
                    phase=phase,
                    bytes_in_use=mem_stats.get("bytes_in_use", 0),
                    peak_bytes=mem_stats.get("peak_bytes_in_use", 0),
                    device_id=0,
                )
                self.profile.snapshots.append(snapshot)
                
                if step > 0 and step % self.log_every_n_steps == 0:
                    self._log_snapshot(snapshot)
                
                return snapshot
        except Exception as e:
            logger.debug(f"Memory stats not available: {e}")
        
        return None
    
    def _log_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Log a memory snapshot."""
        logger.info(
            f"Step {snapshot.step} [{snapshot.phase}]: "
            f"{snapshot.bytes_in_use / (1024**3):.2f} GB in use, "
            f"{snapshot.peak_bytes / (1024**3):.2f} GB peak"
        )
    
    def summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        return self.profile.summary()
    
    def reset(self) -> None:
        """Reset profiler state."""
        self.profile = MemoryProfile()
        self._start_time = time.time()


# =============================================================================
# Memory Estimation Functions
# =============================================================================

def estimate_memory_for_preset(preset: str) -> Dict[str, Any]:
    """
    Estimate memory requirements for a model preset.
    
    Args:
        preset: One of "tiny", "small", "base", "large", "xlarge", "xxlarge"
        
    Returns:
        Dictionary with memory estimates
    """
    presets = {
        "tiny": {"d_model": 256, "layers": 4, "heads": 4, "params_m": 10},
        "small": {"d_model": 512, "layers": 6, "heads": 8, "params_m": 50},
        "base": {"d_model": 768, "layers": 12, "heads": 12, "params_m": 125},
        "large": {"d_model": 1024, "layers": 24, "heads": 16, "params_m": 350},
        "xlarge": {"d_model": 2048, "layers": 32, "heads": 32, "params_m": 1300},
        "xxlarge": {"d_model": 4096, "layers": 48, "heads": 64, "params_m": 7000},
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")
    
    config = presets[preset]
    params_m = config["params_m"]
    
    # Memory estimates (in GB)
    # FP32: 4 bytes per param, FP16/BF16: 2 bytes per param
    params_fp32 = params_m * 4 / 1024  # GB
    params_fp16 = params_m * 2 / 1024  # GB
    
    # Optimizer (Adam): 2x params for momentum + variance (often stays FP32)
    optimizer_fp32 = params_fp32 * 2
    
    # Gradients: 1x params
    gradients_fp32 = params_fp32
    gradients_fp16 = params_fp16
    
    # Activations: Highly dependent on batch size and sequence length
    # Rough estimate: 2-4x params for batch_size=1, seq_len=1024
    activations_base = params_fp32 * 2
    
    # Total for training (without gradient checkpointing)
    total_fp32 = params_fp32 + optimizer_fp32 + gradients_fp32 + activations_base
    total_fp16 = params_fp16 + optimizer_fp32 + gradients_fp16 + activations_base * 0.5
    
    # With gradient checkpointing: ~30% reduction in activations
    total_fp16_checkpointed = params_fp16 + optimizer_fp32 + gradients_fp16 + activations_base * 0.35
    
    return {
        "preset": preset,
        "parameters_millions": params_m,
        "d_model": config["d_model"],
        "layers": config["layers"],
        "heads": config["heads"],
        "memory": {
            "params_fp32_gb": round(params_fp32, 2),
            "params_fp16_gb": round(params_fp16, 2),
            "training_fp32_gb": round(total_fp32, 2),
            "training_fp16_gb": round(total_fp16, 2),
            "training_fp16_checkpointed_gb": round(total_fp16_checkpointed, 2),
        },
        "recommended_gpu": _recommend_gpu(total_fp16_checkpointed),
        "recommended_batch_size": _recommend_batch_size(preset, total_fp16_checkpointed),
    }


def _recommend_gpu(memory_gb: float) -> str:
    """Recommend GPU based on memory requirements."""
    if memory_gb <= 8:
        return "RTX 3070/4070 (8GB) or better"
    elif memory_gb <= 16:
        return "RTX 3090/4080 (16GB) or V100 (16GB)"
    elif memory_gb <= 24:
        return "RTX 4090 (24GB) or A10 (24GB)"
    elif memory_gb <= 40:
        return "A100 (40GB)"
    elif memory_gb <= 80:
        return "A100 (80GB) or H100 (80GB)"
    else:
        return "Multi-GPU required (tensor parallelism)"


def _recommend_batch_size(preset: str, base_memory_gb: float) -> Dict[str, int]:
    """Recommend batch sizes for different GPU memory sizes."""
    # Base batch sizes for each preset (at fp16 with checkpointing)
    base_batch = {
        "tiny": 64,
        "small": 32,
        "base": 16,
        "large": 8,
        "xlarge": 4,
        "xxlarge": 1,
    }.get(preset, 8)
    
    # Scale based on available GPU memory
    gpu_configs = {
        "RTX_4090_24GB": 24,
        "A100_40GB": 40,
        "A100_80GB": 80,
        "H100_80GB": 80,
    }
    
    recommendations = {}
    for gpu_name, gpu_memory in gpu_configs.items():
        # Leave 20% headroom for system use
        available = gpu_memory * 0.8
        memory_ratio = available / max(base_memory_gb, 1)
        recommended = max(1, int(base_batch * min(memory_ratio, 4)))
        recommendations[gpu_name] = recommended
    
    return recommendations


def estimate_batch_memory(
    num_params: int,
    batch_size: int,
    seq_length: int,
    d_model: int,
    num_layers: int,
    dtype_bytes: int = 2,
    gradient_checkpointing: bool = True,
) -> Dict[str, float]:
    """
    Estimate memory for a specific batch configuration.
    
    Args:
        num_params: Number of model parameters
        batch_size: Batch size
        seq_length: Sequence length
        d_model: Model dimension
        num_layers: Number of transformer layers
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)
        gradient_checkpointing: Whether gradient checkpointing is enabled
        
    Returns:
        Memory estimates in GB
    """
    # Parameters
    params_bytes = num_params * dtype_bytes
    params_gb = params_bytes / (1024 ** 3)
    
    # Optimizer states (Adam: 2x params, usually fp32)
    optimizer_gb = (num_params * 4 * 2) / (1024 ** 3)
    
    # Gradients
    gradients_gb = params_gb
    
    # Activations per layer
    # Main activations: batch * seq * d_model
    # Attention: batch * heads * seq * seq (for QK^T)
    activation_per_layer = (
        batch_size * seq_length * d_model * dtype_bytes +  # Hidden states
        batch_size * seq_length * seq_length * dtype_bytes * 2  # Attention (Q@K, softmax)
    ) / (1024 ** 3)
    
    if gradient_checkpointing:
        # Only store activations for checkpointed layers (every sqrt(n) layers)
        import math
        checkpoint_layers = int(math.sqrt(num_layers)) + 1
        activations_gb = activation_per_layer * checkpoint_layers
    else:
        activations_gb = activation_per_layer * num_layers
    
    total_gb = params_gb + optimizer_gb + gradients_gb + activations_gb
    
    return {
        "params_gb": round(params_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "gradients_gb": round(gradients_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "total_gb": round(total_gb, 2),
    }


def get_all_preset_memory_requirements() -> Dict[str, Dict]:
    """Get memory requirements for all presets."""
    presets = ["tiny", "small", "base", "large", "xlarge", "xxlarge"]
    return {preset: estimate_memory_for_preset(preset) for preset in presets}


def print_memory_requirements_table() -> str:
    """Print formatted table of memory requirements."""
    requirements = get_all_preset_memory_requirements()
    
    lines = [
        "=" * 80,
        "RT-DLM Memory Requirements by Preset",
        "=" * 80,
        "",
        f"{'Preset':<10} {'Params':<10} {'FP32 Train':<12} {'FP16 Train':<12} {'FP16+Ckpt':<12} {'Recommended GPU':<25}",
        "-" * 80,
    ]
    
    for preset, req in requirements.items():
        mem = req["memory"]
        lines.append(
            f"{preset:<10} {req['parameters_millions']:<10}M "
            f"{mem['training_fp32_gb']:<12.1f} {mem['training_fp16_gb']:<12.1f} "
            f"{mem['training_fp16_checkpointed_gb']:<12.1f} {req['recommended_gpu']:<25}"
        )
    
    lines.extend([
        "-" * 80,
        "",
        "Notes:",
        "- FP32 Train: Full precision training memory",
        "- FP16 Train: Mixed precision training memory",  
        "- FP16+Ckpt: Mixed precision with gradient checkpointing",
        "- Actual memory may vary based on batch size and sequence length",
        "",
    ])
    
    return "\n".join(lines)
