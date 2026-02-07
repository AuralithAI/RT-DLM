"""
RT-DLM ONNX Exporter

Exports JAX models to ONNX format for cross-platform inference.
Uses jax2tf and tf2onnx for conversion.
"""

from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export."""
    
    # ONNX opset version
    opset_version: int = 15
    
    # Input shapes for tracing
    batch_size: int = 1
    sequence_length: int = 512
    
    # Optimization options
    optimize: bool = True
    fold_constants: bool = True
    
    # Output settings
    use_external_data: bool = False  # For large models
    external_data_threshold: int = 1024  # KB
    
    # Validation
    validate_output: bool = True
    validation_tolerance: float = 1e-5


class ONNXExporter:
    """
    Export RT-DLM models to ONNX format.
    
    Usage:
        exporter = ONNXExporter(model_fn, params, config)
        exporter.export("model.onnx")
        
        # Validate
        exporter.validate("model.onnx", sample_input)
    """
    
    def __init__(
        self,
        model_fn: Callable,
        params: Dict[str, Any],
        config: Optional[ONNXExportConfig] = None,
    ):
        """
        Initialize the ONNX exporter.
        
        Args:
            model_fn: JAX model function (takes params and input, returns output)
            params: Model parameters
            config: Export configuration
        """
        self.model_fn = model_fn
        self.params = params
        self.config = config or ONNXExportConfig()
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check that required packages are installed."""
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for ONNX export. "
                "Install with: pip install tensorflow"
            )
        
        try:
            from jax.experimental import jax2tf
            self.jax2tf = jax2tf
        except ImportError:
            raise ImportError(
                "jax2tf is required for ONNX export. "
                "This should be included with JAX."
            )
        
        try:
            import tf2onnx
            self.tf2onnx = tf2onnx
        except ImportError:
            raise ImportError(
                "tf2onnx is required for ONNX export. "
                "Install with: pip install tf2onnx"
            )
    
    def _create_tf_function(self) -> Tuple[Any, List[Any]]:
        """
        Convert JAX function to TensorFlow function.
        
        Returns:
            Tuple of (tf_function, input_specs)
        """
        # Create input signature
        input_shape = (
            self.config.batch_size,
            self.config.sequence_length,
        )
        
        # Convert JAX function to TF
        @self.tf.function
        def tf_model(input_ids):
            # Use jax2tf to convert
            jax_fn = lambda x: self.model_fn(self.params, x)
            tf_fn = self.jax2tf.convert(
                jax_fn,
                polymorphic_shapes=["(b, s)"],
            )
            return tf_fn(input_ids)
        
        # Create input spec
        input_spec = [
            self.tf.TensorSpec(
                shape=(None, None),  # Dynamic batch and sequence
                dtype=self.tf.int32,
                name="input_ids",
            )
        ]
        
        return tf_model, input_spec
    
    def export(
        self,
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Output file path
            input_names: Names for input tensors
            output_names: Names for output tensors
        
        Returns:
            Path to exported ONNX model
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting model to ONNX (opset {self.config.opset_version})...")
        
        # Convert to TensorFlow
        tf_model, input_spec = self._create_tf_function()
        
        # Get concrete function
        sample_input = self.tf.zeros(
            (self.config.batch_size, self.config.sequence_length),
            dtype=self.tf.int32,
        )
        concrete_func = tf_model.get_concrete_function(sample_input)
        
        # Convert to ONNX
        input_names = input_names or ["input_ids"]
        output_names = output_names or ["logits"]
        
        onnx_model, _ = self.tf2onnx.convert.from_function(
            concrete_func,
            input_signature=input_spec,
            opset=self.config.opset_version,
            output_path=str(output_path),
            inputs_as_nchw=None,
        )
        
        logger.info(f"Exported ONNX model to {output_path}")
        
        # Optimize if requested
        if self.config.optimize:
            self._optimize_onnx(output_path)
        
        # Validate if requested
        if self.config.validate_output:
            self._validate_export(output_path)
        
        return str(output_path)
    
    def _optimize_onnx(self, model_path: Path):
        """Apply ONNX optimizations."""
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            
            logger.info("Optimizing ONNX model...")
            
            # Load model
            model = onnx.load(str(model_path))
            
            # Basic optimizations
            from onnx import optimizer as onnx_optimizer
            passes = [
                "eliminate_identity",
                "eliminate_nop_transpose",
                "fuse_consecutive_transposes",
                "fuse_transpose_into_gemm",
            ]
            
            if self.config.fold_constants:
                passes.append("fuse_add_bias_into_conv")
            
            optimized_model = onnx_optimizer.optimize(model, passes)
            
            # Save optimized model
            onnx.save(optimized_model, str(model_path))
            logger.info("ONNX optimization complete")
            
        except ImportError:
            logger.warning("ONNX optimization skipped (onnx/onnxruntime not installed)")
    
    def _validate_export(self, model_path: Path):
        """Validate exported ONNX model against JAX model."""
        try:
            import onnxruntime as ort
            
            logger.info("Validating ONNX export...")
            
            # Create sample input
            rng = jax.random.PRNGKey(0)
            sample_input = jax.random.randint(
                rng,
                (self.config.batch_size, self.config.sequence_length),
                0,
                1000,
            )
            
            # Get JAX output
            jax_output = self.model_fn(self.params, sample_input)
            
            # Get ONNX output
            session = ort.InferenceSession(str(model_path))
            onnx_input = {"input_ids": np.array(sample_input, dtype=np.int32)}
            onnx_output = session.run(None, onnx_input)[0]
            
            # Compare outputs
            jax_out_np = np.array(jax_output)
            max_diff = np.max(np.abs(jax_out_np - onnx_output))
            
            if max_diff < self.config.validation_tolerance:
                logger.info(f"✅ Validation passed (max diff: {max_diff:.2e})")
            else:
                logger.warning(
                    f"⚠️ Validation warning: max diff {max_diff:.2e} "
                    f"exceeds tolerance {self.config.validation_tolerance:.2e}"
                )
            
        except ImportError:
            logger.warning("Validation skipped (onnxruntime not installed)")
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
    
    @staticmethod
    def verify_onnx(model_path: str) -> bool:
        """
        Verify that an ONNX model is valid.
        
        Args:
            model_path: Path to ONNX model
        
        Returns:
            True if model is valid
        """
        try:
            import onnx
            
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            logger.info(f"✅ ONNX model {model_path} is valid")
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX model validation failed: {e}")
            return False
    
    @staticmethod
    def get_model_info(model_path: str) -> Dict[str, Any]:
        """
        Get information about an ONNX model.
        
        Args:
            model_path: Path to ONNX model
        
        Returns:
            Dictionary with model information
        """
        import onnx
        
        model = onnx.load(model_path)
        
        # Get input/output info
        inputs = []
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            inputs.append({
                "name": inp.name,
                "shape": shape,
                "dtype": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
            })
        
        outputs = []
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            outputs.append({
                "name": out.name,
                "shape": shape,
                "dtype": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
            })
        
        return {
            "opset_version": model.opset_import[0].version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "inputs": inputs,
            "outputs": outputs,
            "num_nodes": len(model.graph.node),
        }


def export_to_onnx(
    model_fn: Callable,
    params: Dict[str, Any],
    output_path: str,
    batch_size: int = 1,
    sequence_length: int = 512,
    opset_version: int = 15,
) -> str:
    """
    Convenience function to export a JAX model to ONNX.
    
    Args:
        model_fn: JAX model function
        params: Model parameters
        output_path: Output file path
        batch_size: Batch size for tracing
        sequence_length: Sequence length for tracing
        opset_version: ONNX opset version
    
    Returns:
        Path to exported model
    """
    config = ONNXExportConfig(
        opset_version=opset_version,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    exporter = ONNXExporter(model_fn, params, config)
    return exporter.export(output_path)
