"""
Tests for Quantization Module

Tests for model quantization utilities including INT8 and INT4
quantization, calibration, and dequantization.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np


class TestQuantizationConfig(unittest.TestCase):
    """Test QuantizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default quantization configuration."""
        from src.core.quantization.quantize import QuantizationConfig
        
        config = QuantizationConfig()
        
        self.assertEqual(config.precision, "int8")
        self.assertTrue(config.symmetric)
        self.assertTrue(config.per_channel)
        self.assertEqual(config.num_calibration_samples, 512)
    
    def test_custom_config(self):
        """Test custom quantization configuration."""
        from src.core.quantization.quantize import QuantizationConfig
        
        config = QuantizationConfig(
            precision="int4",
            symmetric=False,
            per_channel=False,
            exclude_layers=["embedding", "output"]
        )
        
        self.assertEqual(config.precision, "int4")
        self.assertFalse(config.symmetric)
        self.assertIn("embedding", config.exclude_layers)
    
    def test_config_post_init(self):
        """Test config post-initialization defaults."""
        from src.core.quantization.quantize import QuantizationConfig
        
        config = QuantizationConfig()
        
        # Default exclude_layers should be set
        self.assertIsNotNone(config.exclude_layers)
        self.assertIn("embedding", config.exclude_layers)


class TestQuantizationResult(unittest.TestCase):
    """Test QuantizationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating quantization result."""
        from src.core.quantization.quantize import QuantizationResult
        
        result = QuantizationResult(
            quantized_params={"layer": jnp.zeros((10, 10), dtype=jnp.int8)},
            scales={"layer": jnp.ones(10)},
            zero_points={"layer": jnp.zeros(10, dtype=jnp.int32)},
            original_size_mb=100.0,
            quantized_size_mb=25.0,
            compression_ratio=4.0
        )
        
        self.assertEqual(result.original_size_mb, 100.0)
        self.assertEqual(result.quantized_size_mb, 25.0)
        self.assertEqual(result.compression_ratio, 4.0)


class TestScaleZeroPointComputation(unittest.TestCase):
    """Test scale and zero point computation."""
    
    def test_compute_scale_zero_point_symmetric(self):
        """Test symmetric quantization scale computation."""
        from src.core.quantization.quantize import compute_scale_zero_point
        
        tensor = jnp.array([[1.0, 2.0, -3.0, 4.0]])
        
        scale, zero_point = compute_scale_zero_point(
            tensor, bits=8, symmetric=True, per_channel=False
        )
        
        # Symmetric quantization has zero_point = 0
        self.assertTrue(jnp.all(zero_point == 0))
        # Scale should be positive
        self.assertTrue(jnp.all(scale > 0))
    
    def test_compute_scale_zero_point_asymmetric(self):
        """Test asymmetric quantization scale computation."""
        from src.core.quantization.quantize import compute_scale_zero_point
        
        tensor = jnp.array([[0.0, 1.0, 2.0, 3.0]])
        
        scale, zero_point = compute_scale_zero_point(
            tensor, bits=8, symmetric=False, per_channel=False
        )
        
        # Scale should be positive
        self.assertTrue(jnp.all(scale > 0))
    
    def test_compute_scale_per_channel(self):
        """Test per-channel quantization."""
        from src.core.quantization.quantize import compute_scale_zero_point
        
        tensor = jnp.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0]
        ])
        
        scale, zero_point = compute_scale_zero_point(
            tensor, bits=8, symmetric=True, per_channel=True, axis=-1
        )
        
        # Should have different scales for different channels
        self.assertEqual(scale.shape[-1], 1)  # Keepdims
    
    def test_int4_quantization(self):
        """Test 4-bit quantization range."""
        from src.core.quantization.quantize import compute_scale_zero_point
        
        tensor = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        
        scale, zero_point = compute_scale_zero_point(
            tensor, bits=4, symmetric=True, per_channel=False
        )
        
        # Scale should accommodate 4-bit range (-8 to 7)
        self.assertTrue(jnp.all(scale > 0))


class TestQuantizeTensor(unittest.TestCase):
    """Test tensor quantization function."""
    
    def test_quantize_tensor_int8(self):
        """Test INT8 tensor quantization."""
        from src.core.quantization.quantize import (
            compute_scale_zero_point, quantize_tensor
        )
        
        tensor = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        scale, zero_point = compute_scale_zero_point(
            tensor, bits=8, symmetric=True
        )
        
        quantized = quantize_tensor(tensor, scale, zero_point, bits=8)
        
        self.assertEqual(quantized.dtype, jnp.int8)
        # Values should be in INT8 range
        self.assertTrue(jnp.all(quantized >= -128))
        self.assertTrue(jnp.all(quantized <= 127))
    
    def test_quantize_tensor_int4(self):
        """Test INT4 tensor quantization."""
        from src.core.quantization.quantize import (
            compute_scale_zero_point, quantize_tensor
        )
        
        tensor = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        scale, zero_point = compute_scale_zero_point(
            tensor, bits=4, symmetric=True
        )
        
        quantized = quantize_tensor(tensor, scale, zero_point, bits=4)
        
        # INT4 packed in INT8
        self.assertEqual(quantized.dtype, jnp.int8)
        # Values should be in INT4 range
        self.assertTrue(jnp.all(quantized >= -8))
        self.assertTrue(jnp.all(quantized <= 7))


class TestDequantizeTensor(unittest.TestCase):
    """Test tensor dequantization function."""
    
    def test_dequantize_tensor(self):
        """Test tensor dequantization."""
        from src.core.quantization.quantize import (
            compute_scale_zero_point, quantize_tensor, dequantize_tensor
        )
        
        original = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        scale, zero_point = compute_scale_zero_point(
            original, bits=8, symmetric=True
        )
        
        quantized = quantize_tensor(original, scale, zero_point, bits=8)
        dequantized = dequantize_tensor(quantized, scale, zero_point)
        
        # Should be close to original
        self.assertTrue(jnp.allclose(original, dequantized, atol=0.1))
    
    def test_quantize_dequantize_roundtrip(self):
        """Test quantization roundtrip preserves values approximately."""
        from src.core.quantization.quantize import (
            compute_scale_zero_point, quantize_tensor, dequantize_tensor
        )
        
        rng = jax.random.PRNGKey(42)
        original = jax.random.normal(rng, (10, 10))
        
        scale, zero_point = compute_scale_zero_point(original, bits=8)
        quantized = quantize_tensor(original, scale, zero_point, bits=8)
        dequantized = dequantize_tensor(quantized, scale, zero_point)
        
        # Quantization error should be small
        error = jnp.abs(original - dequantized)
        max_error = jnp.max(error)
        
        # Error should be bounded by scale/2 approximately
        self.assertTrue(max_error < 1.0)


class TestQuantizationIntegration(unittest.TestCase):
    """Integration tests for quantization pipeline."""
    
    def test_quantize_model_params(self):
        """Test quantizing model parameters."""
        from src.core.quantization.quantize import (
            QuantizationConfig, compute_scale_zero_point, quantize_tensor
        )
        
        config = QuantizationConfig(precision="int8")
        
        # Simulate model params
        params = {
            "layer1": {
                "weight": jax.random.normal(jax.random.PRNGKey(0), (64, 64)),
                "bias": jax.random.normal(jax.random.PRNGKey(1), (64,))
            },
            "layer2": {
                "weight": jax.random.normal(jax.random.PRNGKey(2), (32, 64)),
                "bias": jax.random.normal(jax.random.PRNGKey(3), (32,))
            }
        }
        
        quantized_params = {}
        scales = {}
        
        def quantize_dict(d, prefix=""):
            result = {}
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    result[key] = quantize_dict(value, full_key)
                elif isinstance(value, jnp.ndarray):
                    scale, zp = compute_scale_zero_point(value, bits=8)
                    result[key] = quantize_tensor(value, scale, zp, bits=8)
                    scales[full_key] = scale
            return result
        
        quantized_params = quantize_dict(params)
        
        # Verify quantized types
        self.assertEqual(quantized_params["layer1"]["weight"].dtype, jnp.int8)
        self.assertEqual(quantized_params["layer2"]["weight"].dtype, jnp.int8)
    
    def test_compression_ratio(self):
        """Test that quantization achieves expected compression."""
        from src.core.quantization.quantize import (
            compute_scale_zero_point, quantize_tensor
        )
        
        # FP32 tensor
        original = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))
        original_size = original.size * 4  # 4 bytes per float32
        
        # Quantize to INT8
        scale, zp = compute_scale_zero_point(original, bits=8)
        quantized = quantize_tensor(original, scale, zp, bits=8)
        quantized_size = quantized.size * 1  # 1 byte per int8
        
        compression_ratio = original_size / quantized_size
        
        # Should achieve ~4x compression
        self.assertGreater(compression_ratio, 3.5)


if __name__ == "__main__":
    unittest.main()
