"""
RT-DLM Evaluation Module

Production-grade evaluation metrics for language model training:
- Perplexity computation (token-level and sequence-level)
- Gradient norm monitoring
- Structured metric logging
- Validation loop utilities
- Training stability indicators

This module focuses on metrics needed during TRAINING, not inference.

Usage:
    from core.training import EvaluationMetrics, MetricLogger, ValidationRunner
    
    # Initialize
    evaluator = EvaluationMetrics()
    logger = MetricLogger(log_dir="./logs", experiment_name="rt-dlm-run-1")
    
    # During training
    metrics = evaluator.compute_batch_metrics(logits, targets, loss)
    logger.log_step(step, metrics)
    
    # Validation
    val_runner = ValidationRunner(model_fn, evaluator)
    val_metrics = val_runner.run_validation(params, val_dataset)
"""

import jax
import jax.numpy as jnp
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Callable, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Core Metrics Data Structures
# =============================================================================

@dataclass
class BatchMetrics:
    """Metrics computed for a single batch."""
    loss: float
    perplexity: float
    token_accuracy: float
    num_tokens: int
    sequence_length: int
    
    # Optional detailed metrics
    top1_accuracy: Optional[float] = None
    top5_accuracy: Optional[float] = None
    entropy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class GradientMetrics:
    """Metrics about gradient health during training."""
    global_norm: float
    max_norm: float
    min_norm: float
    mean_norm: float
    
    # Per-layer stats (optional)
    layer_norms: Optional[Dict[str, float]] = None
    
    # Health indicators
    has_nan: bool = False
    has_inf: bool = False
    is_exploding: bool = False  # norm > threshold
    is_vanishing: bool = False  # norm < threshold
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'global_norm': self.global_norm,
            'max_norm': self.max_norm,
            'min_norm': self.min_norm,
            'mean_norm': self.mean_norm,
            'has_nan': self.has_nan,
            'has_inf': self.has_inf,
            'is_exploding': self.is_exploding,
            'is_vanishing': self.is_vanishing,
        }
        if self.layer_norms:
            result['layer_norms'] = self.layer_norms
        return result


@dataclass
class TrainingStepMetrics:
    """Complete metrics for a training step."""
    step: int
    timestamp: float
    
    # Core metrics
    batch_metrics: BatchMetrics
    learning_rate: float
    
    # Optional
    gradient_metrics: Optional[GradientMetrics] = None
    throughput_tokens_per_sec: Optional[float] = None
    memory_used_gb: Optional[float] = None
    
    # Auxiliary losses (if any)
    auxiliary_losses: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'step': self.step,
            'timestamp': self.timestamp,
            'loss': self.batch_metrics.loss,
            'perplexity': self.batch_metrics.perplexity,
            'token_accuracy': self.batch_metrics.token_accuracy,
            'learning_rate': self.learning_rate,
        }
        
        if self.gradient_metrics:
            result['gradient_norm'] = self.gradient_metrics.global_norm
            result['gradient_health'] = {
                'has_nan': self.gradient_metrics.has_nan,
                'has_inf': self.gradient_metrics.has_inf,
            }
        
        if self.throughput_tokens_per_sec:
            result['throughput_tokens_per_sec'] = self.throughput_tokens_per_sec
            
        if self.auxiliary_losses:
            result['auxiliary_losses'] = self.auxiliary_losses
            
        return result


@dataclass  
class ValidationMetrics:
    """Aggregated metrics from validation run."""
    total_loss: float
    perplexity: float
    token_accuracy: float
    
    num_batches: int
    num_tokens: int
    total_time_sec: float
    
    # Per-batch statistics
    loss_std: float = 0.0
    perplexity_std: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Evaluation Metrics Calculator
# =============================================================================

class EvaluationMetrics:
    """
    Core evaluation metrics computation for language model training.
    
    All computations are JAX-compatible and can be JIT-compiled.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ):
        """
        Initialize evaluation metrics calculator.
        
        Args:
            vocab_size: Size of vocabulary (for perplexity bounds)
            pad_token_id: Token ID used for padding
            ignore_index: Index to ignore in loss computation
        """
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        
        logger.info(f"EvaluationMetrics initialized: vocab_size={vocab_size}")
    
    def compute_perplexity(
        self,
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Compute perplexity from logits and targets.
        
        Perplexity = exp(cross_entropy_loss)
        
        Lower perplexity = better model (1.0 is perfect)
        
        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            targets: Target token IDs [batch, seq_len]
            mask: Optional mask for valid tokens [batch, seq_len]
            
        Returns:
            Tuple of (perplexity, cross_entropy_loss)
        """
        # Reshape for cross entropy
        batch_size, seq_len, vocab_size = logits.shape
        
        # Compute per-token cross entropy
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # Gather log probs for target tokens
        # targets: [batch, seq_len] -> one-hot would be [batch, seq_len, vocab]
        target_log_probs = jnp.take_along_axis(
            log_probs, 
            targets[:, :, None], 
            axis=-1
        ).squeeze(-1)  # [batch, seq_len]
        
        # Apply mask if provided
        if mask is None:
            # Create mask that ignores padding
            mask = (targets != self.pad_token_id).astype(jnp.float32)
        
        # Masked mean of negative log probs
        masked_nll = -target_log_probs * mask
        total_nll = jnp.sum(masked_nll)
        num_tokens = jnp.sum(mask) + 1e-8  # Avoid division by zero
        
        avg_nll = total_nll / num_tokens
        perplexity = jnp.exp(avg_nll)
        
        # Clip perplexity to avoid inf
        perplexity = jnp.clip(perplexity, 1.0, 1e6)
        
        return float(perplexity), float(avg_nll)
    
    def compute_token_accuracy(
        self,
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        top_k: int = 1,
    ) -> float:
        """
        Compute top-k token prediction accuracy.
        
        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            targets: Target token IDs [batch, seq_len]
            mask: Optional mask for valid tokens
            top_k: Compute top-k accuracy (1 for standard accuracy)
            
        Returns:
            Accuracy as float between 0 and 1
        """
        if mask is None:
            mask = (targets != self.pad_token_id).astype(jnp.float32)
        
        if top_k == 1:
            predictions = jnp.argmax(logits, axis=-1)
            correct = (predictions == targets).astype(jnp.float32)
        else:
            # Top-k accuracy
            top_k_preds = jnp.argsort(logits, axis=-1)[:, :, -top_k:]
            correct = jnp.any(
                top_k_preds == targets[:, :, None], 
                axis=-1
            ).astype(jnp.float32)
        
        masked_correct = correct * mask
        accuracy = jnp.sum(masked_correct) / (jnp.sum(mask) + 1e-8)
        
        return float(accuracy)
    
    def compute_entropy(
        self,
        logits: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> float:
        """
        Compute average entropy of predictions.
        
        High entropy = uncertain predictions
        Low entropy = confident predictions
        
        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            mask: Optional mask for valid positions
            
        Returns:
            Average entropy in nats
        """
        probs = jax.nn.softmax(logits, axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # Entropy = -sum(p * log(p))
        entropy = -jnp.sum(probs * log_probs, axis=-1)  # [batch, seq_len]
        
        if mask is not None:
            entropy = entropy * mask
            avg_entropy = jnp.sum(entropy) / (jnp.sum(mask) + 1e-8)
        else:
            avg_entropy = jnp.mean(entropy)
        
        return float(avg_entropy)
    
    def compute_batch_metrics(
        self,
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        loss: float,
        mask: Optional[jnp.ndarray] = None,
        compute_entropy: bool = False,
    ) -> BatchMetrics:
        """
        Compute all batch metrics in one call.
        
        Args:
            logits: Model output [batch, seq_len, vocab_size]
            targets: Target tokens [batch, seq_len]
            loss: Pre-computed loss value
            mask: Optional validity mask
            compute_entropy: Whether to compute entropy (slower)
            
        Returns:
            BatchMetrics dataclass with all computed metrics
        """
        perplexity, _ = self.compute_perplexity(logits, targets, mask)
        token_accuracy = self.compute_token_accuracy(logits, targets, mask)
        top5_accuracy = self.compute_token_accuracy(logits, targets, mask, top_k=5)
        
        batch_size, seq_len = targets.shape
        if mask is not None:
            num_tokens = int(jnp.sum(mask))
        else:
            num_tokens = batch_size * seq_len
        
        entropy = None
        if compute_entropy:
            entropy = self.compute_entropy(logits, mask)
        
        return BatchMetrics(
            loss=float(loss),
            perplexity=perplexity,
            token_accuracy=token_accuracy,
            num_tokens=num_tokens,
            sequence_length=seq_len,
            top1_accuracy=token_accuracy,
            top5_accuracy=top5_accuracy,
            entropy=entropy,
        )


# =============================================================================
# Gradient Monitoring
# =============================================================================

class GradientMonitor:
    """
    Monitor gradient health during training.
    
    Tracks:
    - Global gradient norm
    - Per-layer norms
    - NaN/Inf detection
    - Exploding/vanishing gradient detection
    """
    
    def __init__(
        self,
        exploding_threshold: float = 100.0,
        vanishing_threshold: float = 1e-7,
        track_per_layer: bool = False,
    ):
        """
        Initialize gradient monitor.
        
        Args:
            exploding_threshold: Norm above this indicates exploding gradients
            vanishing_threshold: Norm below this indicates vanishing gradients
            track_per_layer: Whether to track per-layer statistics
        """
        self.exploding_threshold = exploding_threshold
        self.vanishing_threshold = vanishing_threshold
        self.track_per_layer = track_per_layer
        
        # History for trend analysis
        self.norm_history: List[float] = []
        self.max_history_size = 1000
        
        logger.info(
            f"GradientMonitor initialized: "
            f"exploding_threshold={exploding_threshold}, "
            f"vanishing_threshold={vanishing_threshold}"
        )
    
    def compute_gradient_metrics(
        self,
        grads: Any,  # PyTree of gradients
    ) -> GradientMetrics:
        """
        Compute comprehensive gradient metrics.
        
        Args:
            grads: PyTree of gradients (from jax.grad)
            
        Returns:
            GradientMetrics with all computed statistics
        """
        # Flatten all gradients
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        
        # Filter out None gradients
        flat_grads = [g for g in flat_grads if g is not None]
        
        if not flat_grads:
            return GradientMetrics(
                global_norm=0.0,
                max_norm=0.0,
                min_norm=0.0,
                mean_norm=0.0,
                has_nan=False,
                has_inf=False,
            )
        
        # Compute per-array norms
        norms = [float(jnp.linalg.norm(g.flatten())) for g in flat_grads]
        
        # Global norm (L2 of all gradients)
        global_norm = float(jnp.sqrt(sum(n**2 for n in norms)))
        
        # Check for NaN/Inf
        has_nan = any(bool(jnp.any(jnp.isnan(g))) for g in flat_grads)
        has_inf = any(bool(jnp.any(jnp.isinf(g))) for g in flat_grads)
        
        # Health checks
        is_exploding = global_norm > self.exploding_threshold
        is_vanishing = global_norm < self.vanishing_threshold and global_norm > 0
        
        # Update history
        self.norm_history.append(global_norm)
        if len(self.norm_history) > self.max_history_size:
            self.norm_history.pop(0)
        
        # Per-layer norms (if tracking)
        layer_norms = None
        if self.track_per_layer:
            layer_norms = self._compute_layer_norms(grads)
        
        return GradientMetrics(
            global_norm=global_norm,
            max_norm=max(norms) if norms else 0.0,
            min_norm=min(norms) if norms else 0.0,
            mean_norm=sum(norms) / len(norms) if norms else 0.0,
            layer_norms=layer_norms,
            has_nan=has_nan,
            has_inf=has_inf,
            is_exploding=is_exploding,
            is_vanishing=is_vanishing,
        )
    
    def _compute_layer_norms(self, grads: Any) -> Dict[str, float]:
        """Compute norms for each named layer."""
        layer_norms = {}
        
        def extract_norms(path: str, leaf):
            if leaf is not None and hasattr(leaf, 'shape'):
                norm = float(jnp.linalg.norm(leaf.flatten()))
                layer_norms[path] = norm
        
        def traverse(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    traverse(v, f"{path}/{k}" if path else k)
            elif hasattr(obj, '_fields'):  # NamedTuple
                for k in obj._fields:
                    traverse(getattr(obj, k), f"{path}/{k}" if path else k)
            else:
                extract_norms(path, obj)
        
        traverse(grads)
        return layer_norms
    
    def get_trend(self, window: int = 100) -> Dict[str, float]:
        """
        Analyze gradient norm trend over recent steps.
        
        Returns:
            Dictionary with trend statistics
        """
        if len(self.norm_history) < 2:
            return {'trend': 0.0, 'volatility': 0.0}
        
        recent = self.norm_history[-window:]
        
        # Simple linear trend
        x = jnp.arange(len(recent))
        y = jnp.array(recent)
        
        # Linear regression slope
        x_mean = jnp.mean(x)
        y_mean = jnp.mean(y)
        slope = jnp.sum((x - x_mean) * (y - y_mean)) / (jnp.sum((x - x_mean)**2) + 1e-8)
        
        # Volatility (standard deviation)
        volatility = float(jnp.std(y))
        
        return {
            'trend': float(slope),
            'volatility': volatility,
            'recent_mean': float(y_mean),
            'recent_max': float(jnp.max(y)),
            'recent_min': float(jnp.min(y)),
        }


# =============================================================================
# Structured Metric Logging
# =============================================================================

class MetricLogger:
    """
    Structured logging for training metrics.
    
    Features:
    - JSON-lines format for easy parsing
    - Automatic timestamps
    - Configurable logging frequency
    - Console and file output
    - TensorBoard-compatible format (optional)
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        log_every_n_steps: int = 10,
        console_log_every_n_steps: int = 100,
        enable_tensorboard: bool = False,
    ):
        """
        Initialize metric logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name for this experiment run
            log_every_n_steps: File logging frequency
            console_log_every_n_steps: Console output frequency
            enable_tensorboard: Whether to write TensorBoard format
        """
        self.log_dir = Path(log_dir)
        self.log_every_n_steps = log_every_n_steps
        self.console_log_every_n_steps = console_log_every_n_steps
        self.enable_tensorboard = enable_tensorboard
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"rtdlm_{timestamp}"
        self.experiment_name = experiment_name
        
        # Create log directory
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.metrics_file = self.experiment_dir / "metrics.jsonl"
        self.validation_file = self.experiment_dir / "validation.jsonl"
        self.config_file = self.experiment_dir / "config.json"
        
        # In-memory buffers for aggregation
        self.step_metrics: List[Dict[str, Any]] = []
        self.validation_metrics: List[Dict[str, Any]] = []
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        logger.info(f"MetricLogger initialized: {self.experiment_dir}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_with_meta = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': config,
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_with_meta, f, indent=2, default=str)
        
        logger.info(f"Configuration logged to {self.config_file}")
    
    def log_step(
        self,
        step: int,
        metrics: TrainingStepMetrics,
        force: bool = False,
    ):
        """
        Log metrics for a training step.
        
        Args:
            step: Current training step
            metrics: TrainingStepMetrics object
            force: Force logging regardless of frequency
        """
        # Convert to dict
        metrics_dict = metrics.to_dict()
        metrics_dict['wall_time'] = time.time() - self.start_time
        
        # File logging
        if force or step % self.log_every_n_steps == 0:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics_dict, default=str) + '\n')
        
        # Console logging
        if force or step % self.console_log_every_n_steps == 0:
            elapsed = time.time() - self.last_log_time
            self.last_log_time = time.time()
            
            # Calculate steps per second
            steps_per_sec = self.console_log_every_n_steps / elapsed if elapsed > 0 else 0
            
            grad_info = ""
            if metrics.gradient_metrics:
                grad_info = f" | grad_norm: {metrics.gradient_metrics.global_norm:.4f}"
                if metrics.gradient_metrics.has_nan:
                    grad_info += " [NaN!]"
                if metrics.gradient_metrics.is_exploding:
                    grad_info += " [EXPLODING!]"
            
            throughput_info = ""
            if metrics.throughput_tokens_per_sec:
                throughput_info = f" | {metrics.throughput_tokens_per_sec:.0f} tok/s"
            
            logger.info(
                f"Step {step:>7d} | "
                f"loss: {metrics.batch_metrics.loss:.4f} | "
                f"ppl: {metrics.batch_metrics.perplexity:.2f} | "
                f"acc: {metrics.batch_metrics.token_accuracy:.4f} | "
                f"lr: {metrics.learning_rate:.2e}"
                f"{grad_info}{throughput_info} | "
                f"{steps_per_sec:.1f} steps/s"
            )
        
        # Store in memory
        self.step_metrics.append(metrics_dict)
        
        # Prevent memory growth
        if len(self.step_metrics) > 10000:
            self.step_metrics = self.step_metrics[-5000:]
    
    def log_validation(
        self,
        step: int,
        metrics: ValidationMetrics,
    ):
        """Log validation metrics."""
        metrics_dict = metrics.to_dict()
        metrics_dict['step'] = step
        metrics_dict['wall_time'] = time.time() - self.start_time
        metrics_dict['timestamp'] = datetime.now().isoformat()
        
        # File logging
        with open(self.validation_file, 'a') as f:
            f.write(json.dumps(metrics_dict, default=str) + '\n')
        
        # Console logging
        logger.info(
            f"\n{'='*60}\n"
            f"VALIDATION at step {step}\n"
            f"  Loss:       {metrics.total_loss:.4f} (±{metrics.loss_std:.4f})\n"
            f"  Perplexity: {metrics.perplexity:.2f} (±{metrics.perplexity_std:.2f})\n"
            f"  Accuracy:   {metrics.token_accuracy:.4f}\n"
            f"  Tokens:     {metrics.num_tokens:,}\n"
            f"  Time:       {metrics.total_time_sec:.1f}s\n"
            f"{'='*60}\n"
        )
        
        self.validation_metrics.append(metrics_dict)
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics seen so far."""
        if not self.step_metrics:
            return {}
        
        losses = [m['loss'] for m in self.step_metrics]
        perplexities = [m['perplexity'] for m in self.step_metrics]
        accuracies = [m['token_accuracy'] for m in self.step_metrics]
        
        return {
            'best_loss': min(losses),
            'best_perplexity': min(perplexities),
            'best_accuracy': max(accuracies),
            'best_loss_step': self.step_metrics[losses.index(min(losses))]['step'],
        }
    
    def summary(self) -> str:
        """Generate training summary."""
        if not self.step_metrics:
            return "No metrics logged yet."
        
        total_time = time.time() - self.start_time
        total_steps = len(self.step_metrics)
        best = self.get_best_metrics()
        
        summary_lines = [
            f"\n{'='*60}",
            f"TRAINING SUMMARY: {self.experiment_name}",
            f"{'='*60}",
            f"Total steps:      {total_steps:,}",
            f"Total time:       {total_time/3600:.2f} hours",
            f"Avg steps/sec:    {total_steps/total_time:.2f}",
            f"",
            f"Best metrics:",
            f"  Loss:       {best.get('best_loss', 'N/A'):.4f} (step {best.get('best_loss_step', 'N/A')})",
            f"  Perplexity: {best.get('best_perplexity', 'N/A'):.2f}",
            f"  Accuracy:   {best.get('best_accuracy', 'N/A'):.4f}",
            f"",
            f"Logs saved to: {self.experiment_dir}",
            f"{'='*60}\n",
        ]
        
        return '\n'.join(summary_lines)


# =============================================================================
# Validation Runner
# =============================================================================

class ValidationRunner:
    """
    Run validation loop and aggregate metrics.
    
    Handles:
    - Batched validation
    - Metric aggregation
    - Progress reporting
    """
    
    def __init__(
        self,
        model_apply_fn: Callable,
        evaluator: EvaluationMetrics,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize validation runner.
        
        Args:
            model_apply_fn: Function to apply model: (params, inputs) -> logits
            evaluator: EvaluationMetrics instance
            batch_size: Override batch size for validation
        """
        self.model_apply_fn = model_apply_fn
        self.evaluator = evaluator
        self.batch_size = batch_size
        
        logger.info("ValidationRunner initialized")
    
    def run_validation(
        self,
        params: Any,
        val_data: List[Tuple[jnp.ndarray, jnp.ndarray]],
        max_batches: Optional[int] = None,
    ) -> ValidationMetrics:
        """
        Run validation on dataset.
        
        Args:
            params: Model parameters
            val_data: List of (inputs, targets) batches
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Aggregated ValidationMetrics
        """
        start_time = time.time()
        
        all_losses = []
        all_perplexities = []
        all_accuracies = []
        total_tokens = 0
        
        num_batches = len(val_data)
        if max_batches:
            num_batches = min(num_batches, max_batches)
        
        for i, (inputs, targets) in enumerate(val_data[:num_batches]):
            # Forward pass
            logits = self.model_apply_fn(params, inputs)
            
            # Handle tuple outputs
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # Compute metrics
            batch_metrics = self.evaluator.compute_batch_metrics(
                logits, targets, loss=0.0  # We'll compute loss from perplexity
            )
            
            all_losses.append(batch_metrics.loss if batch_metrics.loss > 0 else -jnp.log(batch_metrics.perplexity))
            all_perplexities.append(batch_metrics.perplexity)
            all_accuracies.append(batch_metrics.token_accuracy)
            total_tokens += batch_metrics.num_tokens
            
            # Progress
            if (i + 1) % 10 == 0:
                logger.info(f"Validation progress: {i+1}/{num_batches} batches")
        
        total_time = time.time() - start_time
        
        # Aggregate
        return ValidationMetrics(
            total_loss=float(jnp.mean(jnp.array(all_losses))),
            perplexity=float(jnp.mean(jnp.array(all_perplexities))),
            token_accuracy=float(jnp.mean(jnp.array(all_accuracies))),
            num_batches=num_batches,
            num_tokens=total_tokens,
            total_time_sec=total_time,
            loss_std=float(jnp.std(jnp.array(all_losses))),
            perplexity_std=float(jnp.std(jnp.array(all_perplexities))),
        )


# =============================================================================
# Training Loop Integration Helper
# =============================================================================

class TrainingEvaluator:
    """
    High-level integration for evaluation in training loops.
    
    Combines all evaluation components into a simple interface.
    
    Usage:
        evaluator = TrainingEvaluator(
            vocab_size=50257,
            log_dir="./logs",
            experiment_name="my-run"
        )
        
        # In training loop
        for step, batch in enumerate(dataloader):
            loss, logits, grads = train_step(...)
            
            evaluator.on_train_step(
                step=step,
                loss=loss,
                logits=logits,
                targets=batch['targets'],
                grads=grads,
                learning_rate=current_lr,
            )
        
        # Validation
        evaluator.run_validation(params, val_data)
        
        # End
        print(evaluator.summary())
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        log_every_n_steps: int = 10,
        console_log_every_n_steps: int = 100,
        validate_every_n_steps: int = 1000,
        track_gradients: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize training evaluator.
        
        Args:
            vocab_size: Vocabulary size
            log_dir: Directory for logs
            experiment_name: Experiment name
            log_every_n_steps: File log frequency
            console_log_every_n_steps: Console log frequency
            validate_every_n_steps: Validation frequency
            track_gradients: Whether to track gradient health
            config: Optional config dict to log
        """
        self.metrics = EvaluationMetrics(vocab_size=vocab_size)
        self.gradient_monitor = GradientMonitor() if track_gradients else None
        self.logger = MetricLogger(
            log_dir=log_dir,
            experiment_name=experiment_name,
            log_every_n_steps=log_every_n_steps,
            console_log_every_n_steps=console_log_every_n_steps,
        )
        
        self.validate_every_n_steps = validate_every_n_steps
        self.track_gradients = track_gradients
        
        # Log config
        if config:
            self.logger.log_config(config)
        
        # Timing for throughput
        self.last_step_time = time.time()
        self.last_step_tokens = 0
        
        logger.info(
            f"TrainingEvaluator initialized: "
            f"log_every={log_every_n_steps}, "
            f"console_every={console_log_every_n_steps}, "
            f"validate_every={validate_every_n_steps}"
        )
    
    def on_train_step(
        self,
        step: int,
        loss: float,
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        learning_rate: float,
        grads: Optional[Any] = None,
        auxiliary_losses: Optional[Dict[str, float]] = None,
        mask: Optional[jnp.ndarray] = None,
    ):
        """
        Process a training step and log metrics.
        
        Call this after each training step.
        
        Args:
            step: Current step number
            loss: Computed loss value
            logits: Model output logits
            targets: Target tokens
            learning_rate: Current learning rate
            grads: Optional gradients for monitoring
            auxiliary_losses: Optional dict of auxiliary losses
            mask: Optional mask for valid tokens
        """
        # Compute batch metrics
        batch_metrics = self.metrics.compute_batch_metrics(
            logits, targets, loss, mask
        )
        
        # Compute gradient metrics if tracking
        gradient_metrics = None
        if self.track_gradients and grads is not None:
            gradient_metrics = self.gradient_monitor.compute_gradient_metrics(grads)
        
        # Compute throughput
        current_time = time.time()
        elapsed = current_time - self.last_step_time
        tokens_this_step = batch_metrics.num_tokens
        throughput = tokens_this_step / elapsed if elapsed > 0 else 0
        
        self.last_step_time = current_time
        
        # Create step metrics
        step_metrics = TrainingStepMetrics(
            step=step,
            timestamp=current_time,
            batch_metrics=batch_metrics,
            learning_rate=learning_rate,
            gradient_metrics=gradient_metrics,
            throughput_tokens_per_sec=throughput,
            auxiliary_losses=auxiliary_losses,
        )
        
        # Log
        self.logger.log_step(step, step_metrics)
        
        return step_metrics
    
    def should_validate(self, step: int) -> bool:
        """Check if validation should run at this step."""
        return step > 0 and step % self.validate_every_n_steps == 0
    
    def run_validation(
        self,
        model_apply_fn: Callable,
        params: Any,
        val_data: List[Tuple[jnp.ndarray, jnp.ndarray]],
        step: int,
        max_batches: Optional[int] = None,
    ) -> ValidationMetrics:
        """
        Run validation and log results.
        
        Args:
            model_apply_fn: Model forward function
            params: Model parameters
            val_data: Validation dataset
            step: Current training step
            max_batches: Max batches to evaluate
            
        Returns:
            ValidationMetrics
        """
        runner = ValidationRunner(model_apply_fn, self.metrics)
        val_metrics = runner.run_validation(params, val_data, max_batches)
        self.logger.log_validation(step, val_metrics)
        return val_metrics
    
    def summary(self) -> str:
        """Get training summary."""
        return self.logger.summary()
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics seen during training."""
        return self.logger.get_best_metrics()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data structures
    'BatchMetrics',
    'GradientMetrics',
    'TrainingStepMetrics',
    'ValidationMetrics',
    
    # Core components
    'EvaluationMetrics',
    'GradientMonitor',
    'MetricLogger',
    'ValidationRunner',
    
    # High-level integration
    'TrainingEvaluator',
]
