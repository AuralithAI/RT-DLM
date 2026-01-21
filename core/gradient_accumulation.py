"""
Gradient Accumulation for RT-DLM Training.

Enables training with effective batch sizes larger than GPU memory allows
by accumulating gradients over multiple micro-batches.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable, Tuple, Optional
from functools import partial
import logging

logger = logging.getLogger(__name__)


class BatchGradientAccumulator:
    """
    Gradient accumulator for large effective batch sizes.
    
    When GPU memory is limited, this allows training with larger effective
    batch sizes by:
    1. Splitting the batch into micro-batches
    2. Computing gradients for each micro-batch
    3. Accumulating (averaging) gradients
    4. Applying the optimizer update once
    
    Example:
        accumulator = GradientAccumulator(
            accumulation_steps=4,
            loss_fn=loss_fn,
            model_apply_fn=model.apply,
        )
        
        # Effective batch = micro_batch_size * accumulation_steps
        for micro_batch in micro_batches:
            done = accumulator.accumulate(params, micro_batch, rng)
            if done:
                grads = accumulator.get_accumulated_grads()
                params = apply_updates(params, grads)
                accumulator.reset()
    """
    
    def __init__(
        self,
        accumulation_steps: int,
        loss_fn: Callable,
        model_apply_fn: Callable,
    ):
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        
        self.accumulation_steps = accumulation_steps
        self.loss_fn = loss_fn
        self.model_apply_fn = model_apply_fn
        
        self._accumulated_grads = None
        self._accumulated_loss = 0.0
        self._current_step = 0
        
    def reset(self) -> None:
        """Reset accumulated gradients and loss."""
        self._accumulated_grads = None
        self._accumulated_loss = 0.0
        self._current_step = 0
    
    def accumulate(
        self,
        params: Dict,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray,
    ) -> bool:
        """
        Accumulate gradients from a micro-batch.
        
        Args:
            params: Model parameters
            batch: Micro-batch of data
            rng: Random key
            
        Returns:
            True if accumulation is complete (ready for optimizer step)
        """
        # Compute loss and gradients for this micro-batch
        loss, grads = self._compute_grads(params, batch, rng)
        
        # Accumulate
        if self._accumulated_grads is None:
            self._accumulated_grads = grads
        else:
            self._accumulated_grads = jax.tree_util.tree_map(
                lambda a, g: a + g,
                self._accumulated_grads,
                grads
            )
        
        self._accumulated_loss += loss
        self._current_step += 1
        
        return self._current_step >= self.accumulation_steps
    
    def _compute_grads(
        self,
        params: Dict,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray,
    ) -> Tuple[float, Dict]:
        """Compute loss and gradients for a single micro-batch."""
        def loss_wrapper(params):
            outputs = self.model_apply_fn(params, rng, **batch)
            return self.loss_fn(outputs, batch)
        
        loss, grads = jax.value_and_grad(loss_wrapper)(params)
        return loss, grads
    
    def get_accumulated_grads(self) -> Dict:
        """Get averaged accumulated gradients."""
        if self._accumulated_grads is None:
            raise RuntimeError("No gradients accumulated yet")
        
        # Average the gradients
        return jax.tree_util.tree_map(
            lambda g: g / self.accumulation_steps,
            self._accumulated_grads
        )
    
    def get_accumulated_loss(self) -> float:
        """Get averaged accumulated loss."""
        return self._accumulated_loss / max(self._current_step, 1)
    
    @property
    def is_complete(self) -> bool:
        """Check if accumulation is complete."""
        return self._current_step >= self.accumulation_steps
    
    @property
    def current_step(self) -> int:
        """Current accumulation step."""
        return self._current_step


def create_accumulating_train_step(
    model_apply_fn: Callable,
    loss_fn: Callable,
    optimizer,
    accumulation_steps: int = 1,
) -> Callable:
    """
    Create a training step function with gradient accumulation.
    
    Args:
        model_apply_fn: Model forward function
        loss_fn: Loss function (outputs, batch) -> scalar
        optimizer: Optax optimizer
        accumulation_steps: Number of micro-batches to accumulate
        
    Returns:
        Training step function
    """
    
    @jax.jit
    def compute_grads(params, batch, rng):
        """Compute loss and gradients for a micro-batch."""
        def loss_wrapper(params):
            outputs = model_apply_fn(params, rng, **batch)
            return loss_fn(outputs, batch)
        
        loss, grads = jax.value_and_grad(loss_wrapper)(params)
        return loss, grads
    
    @jax.jit
    def apply_updates(params, opt_state, grads):
        """Apply optimizer updates."""
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = jax.tree_util.tree_map(
            lambda p, u: p + u, params, updates
        )
        return new_params, new_opt_state
    
    def train_step(
        params: Dict,
        opt_state: Any,
        micro_batches: list,
        rng: jnp.ndarray,
    ) -> Tuple[Dict, Any, float, Dict]:
        """
        Perform one training step with gradient accumulation.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            micro_batches: List of micro-batches (length = accumulation_steps)
            rng: Random key
            
        Returns:
            Tuple of (new_params, new_opt_state, avg_loss, metrics)
        """
        if len(micro_batches) != accumulation_steps:
            raise ValueError(
                f"Expected {accumulation_steps} micro-batches, got {len(micro_batches)}"
            )
        
        # Accumulate gradients
        accumulated_grads = None
        total_loss = 0.0
        
        for i, batch in enumerate(micro_batches):
            step_rng = jax.random.fold_in(rng, i)
            loss, grads = compute_grads(params, batch, step_rng)
            
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = jax.tree_util.tree_map(
                    lambda a, g: a + g,
                    accumulated_grads,
                    grads
                )
            total_loss += loss
        
        # Average gradients
        avg_grads = jax.tree_util.tree_map(
            lambda g: g / accumulation_steps,
            accumulated_grads
        )
        avg_loss = total_loss / accumulation_steps
        
        # Apply updates
        new_params, new_opt_state = apply_updates(params, opt_state, avg_grads)
        
        metrics = {
            "loss": avg_loss,
            "accumulation_steps": accumulation_steps,
        }
        
        return new_params, new_opt_state, avg_loss, metrics
    
    return train_step


def split_batch_for_accumulation(
    batch: Dict[str, jnp.ndarray],
    accumulation_steps: int,
) -> list:
    """
    Split a batch into micro-batches for gradient accumulation.
    
    Args:
        batch: Full batch dictionary
        accumulation_steps: Number of micro-batches to create
        
    Returns:
        List of micro-batch dictionaries
    """
    # Get batch size from first array
    first_key = next(iter(batch.keys()))
    batch_size = batch[first_key].shape[0]
    
    if batch_size % accumulation_steps != 0:
        raise ValueError(
            f"Batch size {batch_size} not divisible by accumulation_steps {accumulation_steps}"
        )
    
    micro_batch_size = batch_size // accumulation_steps
    micro_batches = []
    
    for i in range(accumulation_steps):
        start = i * micro_batch_size
        end = start + micro_batch_size
        micro_batch = {
            key: value[start:end] for key, value in batch.items()
        }
        micro_batches.append(micro_batch)
    
    return micro_batches


def calculate_effective_batch_size(
    micro_batch_size: int,
    accumulation_steps: int,
    num_devices: int = 1,
) -> int:
    """Calculate effective batch size from training configuration."""
    return micro_batch_size * accumulation_steps * num_devices


def recommend_accumulation_steps(
    target_batch_size: int,
    max_micro_batch_size: int,
    num_devices: int = 1,
) -> Tuple[int, int]:
    """
    Recommend accumulation steps to achieve target batch size.
    
    Args:
        target_batch_size: Desired effective batch size
        max_micro_batch_size: Maximum batch size that fits in memory
        num_devices: Number of devices for data parallelism
        
    Returns:
        Tuple of (accumulation_steps, actual_micro_batch_size)
    """
    # Calculate minimum accumulation steps needed
    min_accum = target_batch_size // (max_micro_batch_size * num_devices)
    min_accum = max(1, min_accum)
    
    # Find micro batch size that divides evenly
    actual_micro_batch = target_batch_size // (min_accum * num_devices)
    
    # Adjust if needed
    while actual_micro_batch > max_micro_batch_size:
        min_accum += 1
        actual_micro_batch = target_batch_size // (min_accum * num_devices)
    
    return min_accum, actual_micro_batch
