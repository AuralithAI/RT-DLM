"""
RT-DLM Model Quantization Utilities

Provides post-training quantization (PTQ) for model compression.
Supports INT8 and INT4 quantization for inference optimization.
"""

from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
from pathlib import Path
import logging

import jax
import jax.numpy as jnp
import numpy as np

try:
    from safetensors.numpy import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    # Quantization precision
    precision: str = "int8"  # "int8", "int4", "fp16"
    
    # Symmetric vs asymmetric quantization
    symmetric: bool = True
    
    # Per-channel vs per-tensor quantization
    per_channel: bool = True
    
    # Layers to exclude from quantization (keep in fp32)
    exclude_layers: List[str] = None
    
    # Calibration settings
    num_calibration_samples: int = 512
    calibration_method: str = "minmax"  # "minmax", "percentile", "mse"
    percentile: float = 99.99
    
    def __post_init__(self):
        if self.exclude_layers is None:
            self.exclude_layers = ["embedding", "layer_norm", "final_proj"]


@dataclass
class QuantizationResult:
    """Result of model quantization."""
    
    quantized_params: Dict[str, Any]
    scales: Dict[str, jnp.ndarray]
    zero_points: Dict[str, jnp.ndarray]
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float


def compute_scale_zero_point(
    tensor: jnp.ndarray,
    bits: int = 8,
    symmetric: bool = True,
    per_channel: bool = True,
    axis: int = -1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute quantization scale and zero point for a tensor.
    
    Args:
        tensor: Input tensor to quantize
        bits: Number of bits for quantization (8 or 4)
        symmetric: Use symmetric quantization
        per_channel: Quantize per channel vs per tensor
        axis: Channel axis for per-channel quantization
    
    Returns:
        Tuple of (scale, zero_point)
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    
    if per_channel and tensor.ndim > 1:
        # Compute min/max per channel
        reduce_axes = tuple(i for i in range(tensor.ndim) if i != axis)
        t_min = jnp.min(tensor, axis=reduce_axes, keepdims=True)
        t_max = jnp.max(tensor, axis=reduce_axes, keepdims=True)
    else:
        t_min = jnp.min(tensor)
        t_max = jnp.max(tensor)
    
    if symmetric:
        # Symmetric quantization: zero_point = 0
        abs_max = jnp.maximum(jnp.abs(t_min), jnp.abs(t_max))
        scale = abs_max / qmax
        zero_point = jnp.zeros_like(scale, dtype=jnp.int32)
    else:
        # Asymmetric quantization
        scale = (t_max - t_min) / (qmax - qmin)
        zero_point = jnp.round(qmin - t_min / scale).astype(jnp.int32)
    
    # Avoid division by zero
    scale = jnp.where(scale == 0, 1.0, scale)
    
    return scale, zero_point


def quantize_tensor(
    tensor: jnp.ndarray,
    scale: jnp.ndarray,
    zero_point: jnp.ndarray,
    bits: int = 8,
) -> jnp.ndarray:
    """
    Quantize a tensor using the given scale and zero point.
    
    Args:
        tensor: Input tensor
        scale: Quantization scale
        zero_point: Quantization zero point
        bits: Number of bits
    
    Returns:
        Quantized tensor
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    
    quantized = jnp.round(tensor / scale) + zero_point
    quantized = jnp.clip(quantized, qmin, qmax)
    
    if bits == 8:
        return quantized.astype(jnp.int8)
    elif bits == 4:
        # Pack two 4-bit values into one int8
        return quantized.astype(jnp.int8)
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


def dequantize_tensor(
    quantized: jnp.ndarray,
    scale: jnp.ndarray,
    zero_point: jnp.ndarray,
) -> jnp.ndarray:
    """
    Dequantize a tensor back to floating point.
    
    Args:
        quantized: Quantized tensor
        scale: Quantization scale
        zero_point: Quantization zero point
    
    Returns:
        Dequantized tensor
    """
    return (quantized.astype(jnp.float32) - zero_point) * scale


def should_quantize_layer(name: str, config: QuantizationConfig) -> bool:
    """Check if a layer should be quantized based on config."""
    for exclude_pattern in config.exclude_layers:
        if exclude_pattern.lower() in name.lower():
            return False
    return True


def get_tensor_size_mb(tensor: jnp.ndarray) -> float:
    """Calculate tensor size in megabytes."""
    return tensor.size * tensor.dtype.itemsize / (1024 * 1024)


class ModelQuantizer:
    """
    Post-training quantization for RT-DLM models.
    
    Usage:
        quantizer = ModelQuantizer(config=QuantizationConfig(precision="int8"))
        result = quantizer.quantize(params, calibration_data)
        quantizer.save(result, "model_int8.safetensors")
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize the quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()
        self._bits = 8 if self.config.precision == "int8" else 4
    
    def calibrate(
        self,
        params: Dict[str, Any],
        calibration_fn: Optional[Callable] = None,
        calibration_data: Optional[jnp.ndarray] = None,
    ) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Calibrate quantization parameters using calibration data.
        
        For static quantization, this runs forward passes to collect
        activation statistics.
        
        Args:
            params: Model parameters
            calibration_fn: Function to run calibration forward passes
            calibration_data: Calibration dataset
        
        Returns:
            Dictionary mapping param names to (scale, zero_point)
        """
        scales_zp = {}
        
        def process_params(params: Dict, prefix: str = ""):
            for key, value in params.items():
                full_name = f"{prefix}/{key}" if prefix else key
                
                if isinstance(value, dict):
                    process_params(value, full_name)
                elif isinstance(value, (jnp.ndarray, np.ndarray)):
                    if should_quantize_layer(full_name, self.config):
                        scale, zp = compute_scale_zero_point(
                            value,
                            bits=self._bits,
                            symmetric=self.config.symmetric,
                            per_channel=self.config.per_channel,
                        )
                        scales_zp[full_name] = (scale, zp)
                        logger.debug(f"Calibrated {full_name}: scale={scale.shape}")
        
        process_params(params)
        return scales_zp
    
    def quantize(
        self,
        params: Dict[str, Any],
        calibration_fn: Optional[Callable] = None,
        calibration_data: Optional[jnp.ndarray] = None,
    ) -> QuantizationResult:
        """
        Quantize model parameters.
        
        Args:
            params: Model parameters to quantize
            calibration_fn: Optional calibration function
            calibration_data: Optional calibration data
        
        Returns:
            QuantizationResult with quantized params and metadata
        """
        logger.info(f"Starting {self.config.precision} quantization...")
        
        # Calibrate to get scales and zero points
        scales_zp = self.calibrate(params, calibration_fn, calibration_data)
        
        quantized_params = {}
        scales = {}
        zero_points = {}
        original_size = 0.0
        quantized_size = 0.0
        
        def process_params(params: Dict, prefix: str = ""):
            nonlocal original_size, quantized_size
            
            result = {}
            for key, value in params.items():
                full_name = f"{prefix}/{key}" if prefix else key
                
                if isinstance(value, dict):
                    result[key] = process_params(value, full_name)
                elif isinstance(value, (jnp.ndarray, np.ndarray)):
                    value = jnp.array(value)
                    original_size += get_tensor_size_mb(value)
                    
                    if full_name in scales_zp:
                        scale, zp = scales_zp[full_name]
                        q_tensor = quantize_tensor(value, scale, zp, self._bits)
                        
                        result[key] = q_tensor
                        scales[full_name] = scale
                        zero_points[full_name] = zp
                        quantized_size += get_tensor_size_mb(q_tensor)
                        
                        logger.debug(
                            f"Quantized {full_name}: "
                            f"{value.shape} ({value.dtype}) -> {q_tensor.shape} ({q_tensor.dtype})"
                        )
                    else:
                        # Keep in original precision
                        result[key] = value
                        quantized_size += get_tensor_size_mb(value)
                else:
                    result[key] = value
            
            return result
        
        quantized_params = process_params(params)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        
        logger.info(
            f"Quantization complete: "
            f"{original_size:.2f}MB -> {quantized_size:.2f}MB "
            f"({compression_ratio:.2f}x compression)"
        )
        
        return QuantizationResult(
            quantized_params=quantized_params,
            scales=scales,
            zero_points=zero_points,
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=compression_ratio,
        )
    
    def save(
        self,
        result: QuantizationResult,
        output_path: str,
    ):
        """
        Save quantized model to SafeTensors format.
        
        Args:
            result: Quantization result
            output_path: Output file path
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors not installed")
        
        output_path = Path(output_path)
        
        # Flatten params for saving
        flat_tensors = {}
        
        def flatten(d: Dict, prefix: str = ""):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flatten(v, key)
                elif isinstance(v, (jnp.ndarray, np.ndarray)):
                    flat_tensors[key] = np.array(v)
        
        flatten(result.quantized_params, "params")
        
        # Add scales and zero points
        for name, scale in result.scales.items():
            flat_tensors[f"scales.{name.replace('/', '.')}"] = np.array(scale)
        
        for name, zp in result.zero_points.items():
            flat_tensors[f"zero_points.{name.replace('/', '.')}"] = np.array(zp)
        
        # Save metadata
        metadata = {
            "precision": self.config.precision,
            "symmetric": str(self.config.symmetric),
            "per_channel": str(self.config.per_channel),
            "original_size_mb": str(result.original_size_mb),
            "quantized_size_mb": str(result.quantized_size_mb),
            "compression_ratio": str(result.compression_ratio),
        }
        
        save_file(flat_tensors, output_path, metadata=metadata)
        logger.info(f"Saved quantized model to {output_path}")
    
    @staticmethod
    def load(
        path: str,
    ) -> Tuple[Dict[str, Any], Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], Dict[str, str]]:
        """
        Load quantized model from SafeTensors format.
        
        Args:
            path: Path to quantized model file
        
        Returns:
            Tuple of (params, scales, zero_points, metadata)
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors not installed")
        
        from safetensors import safe_open
        with safe_open(path, framework="numpy") as f:
            metadata = f.metadata()
            
            params: Dict[str, Any] = {}
            scales: Dict[str, Any] = {}
            zero_points: Dict[str, Any] = {}
            
            for key in f.keys():
                tensor = jnp.array(f.get_tensor(key))
                
                if key.startswith("params."):
                    # Reconstruct nested dict
                    parts = key[7:].split(".")
                    current = params
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = tensor
                elif key.startswith("scales."):
                    scales[key[7:].replace(".", "/")] = tensor
                elif key.startswith("zero_points."):
                    zero_points[key[12:].replace(".", "/")] = tensor
        
        return params, scales, zero_points, metadata


def quantize_model_int8(
    params: Dict[str, Any],
    calibration_data: Optional[jnp.ndarray] = None,
) -> QuantizationResult:
    """
    Convenience function for INT8 quantization.
    
    Args:
        params: Model parameters
        calibration_data: Optional calibration data
    
    Returns:
        Quantization result
    """
    config = QuantizationConfig(precision="int8")
    quantizer = ModelQuantizer(config)
    return quantizer.quantize(params, calibration_data=calibration_data)


def quantize_model_int4(
    params: Dict[str, Any],
    calibration_data: Optional[jnp.ndarray] = None,
) -> QuantizationResult:
    """
    Convenience function for INT4 quantization.
    
    Args:
        params: Model parameters
        calibration_data: Optional calibration data
    
    Returns:
        Quantization result
    """
    config = QuantizationConfig(precision="int4")
    quantizer = ModelQuantizer(config)
    return quantizer.quantize(params, calibration_data=calibration_data)
