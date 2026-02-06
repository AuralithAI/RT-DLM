#!/usr/bin/env python
"""
RT-DLM Model Validation Script

Validates model initialization and training readiness.

Usage:
    python scripts/validate_model.py --preset tiny
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
)
logger = logging.getLogger(__name__)


def validate_imports():
    """Validate all required imports."""
    logger.info("Validating imports...")
    
    try:
        import jax
        import jax.numpy as jnp
        logger.info(f"  ✅ JAX {jax.__version__}")
    except ImportError as e:
        logger.error(f"  ❌ JAX: {e}")
        return False
    
    try:
        import haiku as hk
        logger.info(f"  ✅ Haiku {hk.__version__}")
    except ImportError as e:
        logger.error(f"  ❌ Haiku: {e}")
        return False
    
    try:
        import optax
        logger.info(f"  ✅ Optax {optax.__version__}")
    except ImportError as e:
        logger.error(f"  ❌ Optax: {e}")
        return False
    
    try:
        from safetensors import safe_open
        logger.info("  ✅ SafeTensors")
    except ImportError as e:
        logger.error(f"  ❌ SafeTensors: {e}")
        return False
    
    return True


def validate_devices():
    """Validate JAX devices."""
    import jax
    
    logger.info("Validating devices...")
    devices = jax.devices()
    
    for device in devices:
        device_type = str(device.platform).upper()
        logger.info(f"  ✅ {device_type}: {device}")
    
    has_gpu = any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devices)
    
    if has_gpu:
        logger.info("  🚀 GPU acceleration available")
    else:
        logger.warning("  ⚠️ Running on CPU only")
    
    return True


def validate_model_init(preset: str):
    """Validate model initialization."""
    import jax
    import jax.numpy as jnp
    import haiku as hk
    
    from src.config.agi_config import AGIConfig
    from src.rtdlm import create_rtdlm_agi
    
    logger.info(f"Validating model initialization (preset: {preset})...")
    
    # Get config
    if preset == "tiny":
        config = AGIConfig.tiny()
    elif preset == "small":
        config = AGIConfig.small()
    elif preset == "medium":
        config = AGIConfig.medium()
    elif preset == "large":
        config = AGIConfig.large()
    else:
        config = AGIConfig()
    
    logger.info(f"  Config: {config.hidden_dim}d, {config.num_layers}L, {config.num_heads}H")
    
    # Initialize model
    def forward(x):
        model = create_rtdlm_agi(config)
        return model(x, is_training=False)
    
    forward_fn = hk.transform(forward)
    
    # Create sample input
    batch_size = 2
    seq_length = 32
    rng = jax.random.PRNGKey(42)
    
    sample_input = jax.random.randint(
        rng, (batch_size, seq_length), 0, config.vocab_size
    )
    
    # Initialize parameters
    start_time = time.time()
    params = forward_fn.init(rng, sample_input)
    init_time = time.time() - start_time
    
    logger.info(f"  ✅ Model initialized in {init_time:.2f}s")
    
    # Count parameters
    def count_params(params):
        return sum(p.size for p in jax.tree_util.tree_leaves(params))
    
    num_params = count_params(params)
    logger.info(f"  ✅ Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Forward pass
    start_time = time.time()
    output = forward_fn.apply(params, rng, sample_input)
    forward_time = time.time() - start_time
    
    logger.info(f"  ✅ Forward pass in {forward_time:.3f}s")
    logger.info(f"  ✅ Output shape: {output.shape}")
    
    # Check for NaN
    has_nan = jnp.any(jnp.isnan(output))
    if has_nan:
        logger.error("  ❌ Output contains NaN values!")
        return False
    
    logger.info("  ✅ No NaN values in output")
    
    return True


def validate_training_step(preset: str):
    """Validate a training step."""
    import jax
    import jax.numpy as jnp
    import haiku as hk
    import optax
    
    from config.agi_config import AGIConfig
    from rtdlm import create_rtdlm_agi, create_agi_optimizer, compute_agi_loss
    
    logger.info(f"Validating training step...")
    
    # Get config
    if preset == "tiny":
        config = AGIConfig.tiny()
    else:
        config = AGIConfig.tiny()  # Use tiny for training validation
    
    # Initialize model
    def forward(x):
        model = create_rtdlm_agi(config)
        return model(x, is_training=True)
    
    forward_fn = hk.transform(forward)
    
    # Create sample data
    batch_size = 4
    seq_length = 32
    rng = jax.random.PRNGKey(42)
    
    input_ids = jax.random.randint(
        rng, (batch_size, seq_length), 0, config.vocab_size
    )
    targets = jax.random.randint(
        rng, (batch_size, seq_length), 0, config.vocab_size
    )
    
    # Initialize
    params = forward_fn.init(rng, input_ids)
    optimizer = create_agi_optimizer(config)
    opt_state = optimizer.init(params)
    
    # Define loss function
    def loss_fn(params, rng, inputs, targets):
        logits = forward_fn.apply(params, rng, inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, targets
        ).mean()
        return loss
    
    # Compute gradients
    start_time = time.time()
    loss, grads = jax.value_and_grad(loss_fn)(params, rng, input_ids, targets)
    grad_time = time.time() - start_time
    
    logger.info(f"  ✅ Loss computed: {loss:.4f}")
    logger.info(f"  ✅ Gradients computed in {grad_time:.3f}s")
    
    # Check gradient norm
    grad_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)
    ))
    logger.info(f"  ✅ Gradient norm: {grad_norm:.4f}")
    
    # Check for NaN gradients
    has_nan_grad = any(
        jnp.any(jnp.isnan(g)) for g in jax.tree_util.tree_leaves(grads)
    )
    if has_nan_grad:
        logger.error("  ❌ Gradients contain NaN values!")
        return False
    
    logger.info("  ✅ No NaN values in gradients")
    
    # Apply update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    # Verify params changed
    old_flat = jax.tree_util.tree_leaves(params)
    new_flat = jax.tree_util.tree_leaves(new_params)
    
    changed = any(
        not jnp.allclose(o, n) for o, n in zip(old_flat, new_flat)
    )
    
    if changed:
        logger.info("  ✅ Parameters updated successfully")
    else:
        logger.warning("  ⚠️ Parameters did not change")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate RT-DLM model and training readiness"
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        choices=["tiny", "small", "medium", "large"],
        default="tiny",
        help="Model preset to validate (default: tiny)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training step validation",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("RT-DLM Model Validation")
    logger.info("=" * 60)
    
    all_passed = True
    
    # Validate imports
    if not validate_imports():
        all_passed = False
    
    # Validate devices
    if not validate_devices():
        all_passed = False
    
    # Validate model initialization
    if not validate_model_init(args.preset):
        all_passed = False
    
    # Validate training step
    if not args.skip_training:
        if not validate_training_step(args.preset):
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("✅ All validations passed!")
        logger.info("Model is ready for training.")
    else:
        logger.error("❌ Some validations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
