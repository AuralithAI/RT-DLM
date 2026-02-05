#!/usr/bin/env python
"""
RT-DLM Model Quantization Script

Quantizes trained models to INT8 or INT4 for inference optimization.

Usage:
    python scripts/quantize_model.py \
        --checkpoint checkpoints/model.safetensors \
        --output checkpoints/model_int8.safetensors \
        --precision int8
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.quantization import (
    QuantizationConfig,
    ModelQuantizer,
    quantize_model_int8,
    quantize_model_int4,
)
from src.core.checkpoint_manager import CheckpointManager
from src.config.agi_config import AGIConfig

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize RT-DLM model for inference optimization"
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
        help="Output path for quantized model",
    )
    
    # Quantization options
    parser.add_argument(
        "--precision",
        type=str,
        choices=["int8", "int4", "fp16"],
        default="int8",
        help="Quantization precision (default: int8)",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=True,
        help="Use symmetric quantization (default: True)",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        default=True,
        help="Use per-channel quantization (default: True)",
    )
    
    # Calibration options
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Path to calibration data (.safetensors)",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)",
    )
    
    # Layers to exclude
    parser.add_argument(
        "--exclude-layers",
        type=str,
        nargs="+",
        default=["embedding", "layer_norm", "final_proj"],
        help="Layer patterns to exclude from quantization",
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
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_mgr = CheckpointManager(checkpoint_path.parent)
    
    # Load params (assumes SafeTensors format)
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
    
    # Load calibration data if provided
    calibration_data = None
    if args.calibration_data:
        logger.info(f"Loading calibration data from {args.calibration_data}...")
        try:
            cal_data = load_file(args.calibration_data)
            calibration_data = jnp.array(list(cal_data.values())[0])
            logger.info(f"Loaded calibration data: {calibration_data.shape}")
        except Exception as e:
            logger.warning(f"Failed to load calibration data: {e}")
    
    # Create quantization config
    config = QuantizationConfig(
        precision=args.precision,
        symmetric=args.symmetric,
        per_channel=args.per_channel,
        exclude_layers=args.exclude_layers,
        num_calibration_samples=args.num_calibration_samples,
    )
    
    logger.info(f"Quantization config: {config}")
    
    # Quantize model
    logger.info(f"Starting {args.precision} quantization...")
    quantizer = ModelQuantizer(config)
    result = quantizer.quantize(params, calibration_data=calibration_data)
    
    # Print results
    logger.info("=" * 60)
    logger.info("Quantization Results:")
    logger.info(f"  Original size:   {result.original_size_mb:.2f} MB")
    logger.info(f"  Quantized size:  {result.quantized_size_mb:.2f} MB")
    logger.info(f"  Compression:     {result.compression_ratio:.2f}x")
    logger.info("=" * 60)
    
    # Save quantized model
    logger.info(f"Saving quantized model to {output_path}...")
    quantizer.save(result, str(output_path))
    
    logger.info("✅ Quantization complete!")
    
    # Print file sizes
    original_size = checkpoint_path.stat().st_size / (1024 * 1024)
    quantized_size = output_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"File sizes:")
    logger.info(f"  Original:   {original_size:.2f} MB")
    logger.info(f"  Quantized:  {quantized_size:.2f} MB")


if __name__ == "__main__":
    main()
