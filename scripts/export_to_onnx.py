#!/usr/bin/env python
"""
RT-DLM ONNX Export Script

Exports trained JAX models to ONNX format for cross-platform inference.

Usage:
    python scripts/export_to_onnx.py \
        --checkpoint checkpoints/model.safetensors \
        --config config/agi_config.py \
        --output models/model.onnx \
        --opset 15
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export RT-DLM model to ONNX format"
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for ONNX model (.onnx)",
    )
    
    # Model configuration
    parser.add_argument(
        "--preset",
        type=str,
        choices=["tiny", "small", "medium", "large"],
        default="medium",
        help="Model preset (default: medium)",
    )
    
    # Export options
    parser.add_argument(
        "--opset",
        type=int,
        default=15,
        help="ONNX opset version (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for tracing (default: 1)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Sequence length for tracing (default: 512)",
    )
    
    # Optimization options
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Apply ONNX optimizations (default: True)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_false",
        dest="optimize",
        help="Skip ONNX optimizations",
    )
    
    # Validation
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate exported model (default: True)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Skip validation",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Validation tolerance (default: 1e-5)",
    )
    
    # Verbose output
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check dependencies
    try:
        import tensorflow as tf
        from jax.experimental import jax2tf
        import tf2onnx
        logger.info(f"TensorFlow version: {tf.__version__}")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install tensorflow tf2onnx")
        sys.exit(1)
    
    # Load model configuration
    from src.config.agi_config import AGIConfig
    
    if args.preset == "tiny":
        config = AGIConfig.tiny()
    elif args.preset == "small":
        config = AGIConfig.small()
    elif args.preset == "medium":
        config = AGIConfig.medium()
    elif args.preset == "large":
        config = AGIConfig.large()
    else:
        config = AGIConfig()
    
    logger.info(f"Model config: {config.hidden_dim}d, {config.num_layers}L")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    try:
        from safetensors.numpy import load_file
        import jax.numpy as jnp
        
        flat_params = load_file(str(checkpoint_path))
        
        # Reconstruct nested params
        params = {}
        for key, value in flat_params.items():
            parts = key.split(".")
            current = params
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = jnp.array(value)
        
        logger.info(f"Loaded {len(flat_params)} parameter tensors")
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)
    
    # Create model function
    logger.info("Creating model function...")
    
    from src.rtdlm import create_rtdlm_agi
    import haiku as hk
    
    def model_fn(params, input_ids):
        """Model forward function for export."""
        def forward(x):
            model = create_rtdlm_agi(config)
            return model(x, is_training=False)
        
        forward_fn = hk.transform(forward)
        rng = jax.random.PRNGKey(0)
        return forward_fn.apply(params, rng, input_ids)
    
    # Import JAX here to avoid issues
    import jax
    
    # Export to ONNX
    from src.core.export import ONNXExporter, ONNXExportConfig
    
    export_config = ONNXExportConfig(
        opset_version=args.opset,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        optimize=args.optimize,
        validate_output=args.validate,
        validation_tolerance=args.tolerance,
    )
    
    logger.info(f"Exporting to ONNX (opset {args.opset})...")
    
    try:
        exporter = ONNXExporter(model_fn, params, export_config)
        output_file = exporter.export(str(output_path))
        
        logger.info("=" * 60)
        logger.info("Export Results:")
        logger.info(f"  Output file: {output_file}")
        logger.info(f"  File size:   {output_path.stat().st_size / (1024 * 1024):.2f} MB")
        logger.info("=" * 60)
        
        # Print model info
        if args.verbose:
            info = ONNXExporter.get_model_info(str(output_path))
            logger.info("Model Info:")
            logger.info(f"  Opset:     {info['opset_version']}")
            logger.info(f"  Nodes:     {info['num_nodes']}")
            logger.info(f"  Inputs:    {info['inputs']}")
            logger.info(f"  Outputs:   {info['outputs']}")
        
        logger.info("✅ ONNX export complete!")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
