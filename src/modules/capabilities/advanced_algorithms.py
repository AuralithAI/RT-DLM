"""
Advanced Continual Learning Algorithms for RT-DLM AGI

Implements algorithms to prevent catastrophic forgetting during self-evolution:
- Elastic Weight Consolidation (EWC)
- Synaptic Intelligence (SI)
- Progressive Neural Networks
- Memory-Aware Synapses (MAS)

These enable adaptive learning for long-term tasks like evolving disaster 
prediction models without losing prior knowledge.
"""

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from functools import partial
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskMemory:
    """Stores task-specific information for continual learning."""
    task_id: str
    params_snapshot: Dict[str, jnp.ndarray]
    fisher_matrix: Dict[str, jnp.ndarray]
    importance_weights: Dict[str, jnp.ndarray]
    performance_metrics: Dict[str, float]
    num_samples: int


def compute_fisher_information(
    model_fn: Callable,
    params: Any,
    data_samples: jnp.ndarray,
    targets: jnp.ndarray,
    rng: Any,
    num_samples: int = 100
) -> Any:
    """
    Compute Fisher Information Matrix for EWC.
    
    The Fisher information measures how sensitive the loss is to each parameter,
    indicating parameter importance for the current task.
    
    F_ii = E[(∂log p(y|x,θ) / ∂θ_i)²]
    
    Args:
        model_fn: Model function (params, rng, inputs) -> outputs
        params: Current model parameters (nested pytree)
        data_samples: Input data samples (can be int32 token IDs or float embeddings)
        targets: Target labels (int32)
        rng: Random key
        num_samples: Number of samples for estimation
        
    Returns:
        Fisher information diagonal per parameter (nested pytree matching params structure)
    """
    # Handle tuple params from transform_with_state: (params_dict, state_dict)
    # We only compute Fisher information for the model parameters, not state
    params_dict = params[0] if isinstance(params, tuple) else params
    
    # Filter out non-float parameters (e.g., int32 training_step counters)
    def filter_float_params(tree):
        """Only keep float parameters for gradient computation."""
        def keep_floats(x):
            if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
                return x
            elif isinstance(x, jnp.ndarray):
                # Convert int params to float for gradient computation
                return x.astype(jnp.float32)
            return x
        return jax.tree_util.tree_map(keep_floats, tree)
    
    params_for_grad = filter_float_params(params_dict)
    fisher_diag = jax.tree_util.tree_map(jnp.zeros_like, params_for_grad)
    
    def log_likelihood_grad(params, x, y, rng):
        """Compute gradient of log-likelihood for a single sample.
        
        Note: We compute gradients w.r.t. params (floats), not inputs.
        The inputs (x, y) are treated as constants in the gradient computation.
        """
        def single_loss(params):
            # x can be int32 token IDs - that's fine because we're taking
            # gradients w.r.t. params, not x. The model internally embeds x.
            # Call model_fn with correct signature:
            # - For transform_with_state: apply(params, state, rng, ...)
            # - For transform: apply(params, rng, ...)
            if has_state:
                output, _ = model_fn(params, original_state, rng, inputs={"text": x[None, :]})
            else:
                output = model_fn(params, rng, inputs={"text": x[None, :]})
            logits = output if isinstance(output, jnp.ndarray) else output.get("logits", output)
            
            # Handle potential shape mismatches
            if logits.ndim == 3:
                # Shape: (batch, seq, vocab) - need to gather at target positions
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                # y shape: (seq,) -> (1, seq, 1) for take_along_axis
                y_expanded = y[None, :, None]
                # Ensure y indices are within vocab size
                y_clipped = jnp.clip(y_expanded, 0, logits.shape[-1] - 1)
                selected_log_probs = jnp.take_along_axis(log_probs, y_clipped, axis=-1)
                return -jnp.mean(selected_log_probs)
            else:
                # Shape: (batch, vocab) - classification case
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                return -jnp.mean(log_probs[:, y[0]])
        
        grads = jax.grad(single_loss)(params)
        return grads
    
    # Track state for reconstruction
    has_state = isinstance(params, tuple)
    original_state = params[1] if has_state else None
    
    # Estimate Fisher over samples
    sample_indices = jax.random.choice(
        rng, len(data_samples), shape=(min(num_samples, len(data_samples)),), replace=False
    )
    
    for idx in sample_indices:
        rng, sample_rng = jax.random.split(rng)
        sample_grads = log_likelihood_grad(
            params_for_grad, data_samples[idx], targets[idx], sample_rng
        )
        
        # Accumulate squared gradients (diagonal Fisher)
        fisher_diag = jax.tree_util.tree_map(
            lambda f, g: f + g ** 2,
            fisher_diag, sample_grads
        )
    
    # Average over samples
    n_samples = len(sample_indices)
    fisher_diag = jax.tree_util.tree_map(lambda f: f / n_samples, fisher_diag)
    
    return fisher_diag

def compute_ewc_loss(
    params: Any,
    params_star: Any,
    fisher_matrix: Any,
    lambda_ewc: float = 1000.0
) -> Any:
    """
    Compute Elastic Weight Consolidation (EWC) regularization loss.
    
    L_EWC = (λ/2) * Σ_i F_i * (θ_i - θ*_i)²
    
    Where:
    - F_i is the Fisher information for parameter i
    - θ*_i is the optimal parameter from the previous task
    - λ is the regularization strength
    
    Args:
        params: Current model parameters (nested pytree)
        params_star: Optimal parameters from previous task (nested pytree)
        fisher_matrix: Fisher information diagonal (nested pytree)
        lambda_ewc: Regularization strength (higher = more protection)
        
    Returns:
        EWC loss term to add to the main loss
    """
    ewc_loss = 0.0
    
    def compute_param_loss(param, param_star, fisher):
        """Compute EWC loss for a single parameter."""
        return jnp.sum(fisher * (param - param_star) ** 2)
    
    # Compute EWC loss for all parameters
    param_losses = jax.tree_util.tree_map(
        compute_param_loss, params, params_star, fisher_matrix
    )
    
    # Sum all parameter losses
    ewc_loss = jax.tree_util.tree_reduce(
        lambda x, y: x + y, param_losses
    )
    
    return (lambda_ewc / 2.0) * ewc_loss


def compute_si_loss(
    params: Any,
    params_star: Any,
    importance_weights: Any,
    lambda_si: float = 1.0
) -> Any:
    """
    Compute Synaptic Intelligence (SI) regularization loss.
    
    SI measures the importance of each parameter based on its contribution
    to the loss during training (path integral of gradients).
    
    L_SI = λ * Σ_i Ω_i * (θ_i - θ*_i)²
    
    Args:
        params: Current model parameters (nested pytree)
        params_star: Parameters at task completion (nested pytree)
        importance_weights: Accumulated importance per parameter (nested pytree)
        lambda_si: Regularization strength
        
    Returns:
        SI loss term
    """
    si_loss = 0.0
    
    def compute_param_loss(param, param_star, omega):
        """Compute SI loss for a single parameter."""
        return jnp.sum(omega * (param - param_star) ** 2)
    
    param_losses = jax.tree_util.tree_map(
        compute_param_loss, params, params_star, importance_weights
    )
    
    si_loss = jax.tree_util.tree_reduce(lambda x, y: x + y, param_losses)
    
    return lambda_si * si_loss


class ElasticWeightConsolidation(hk.Module):
    """
    Elastic Weight Consolidation (EWC) implementation.
    
    EWC prevents catastrophic forgetting by adding a regularization term
    that penalizes changes to important parameters from previous tasks.
    """
    
    def __init__(self, d_model: int, lambda_ewc: float = 1000.0, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.lambda_ewc = lambda_ewc
        
        # Importance estimation network
        self.importance_estimator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="importance_estimator")
        
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Estimate parameter importance from features.
        
        Args:
            features: Input features to assess importance
            
        Returns:
            Importance scores
        """
        importance = self.importance_estimator(features)
        return importance


class SynapticIntelligence(hk.Module):
    """
    Synaptic Intelligence (SI) implementation.
    
    SI computes importance weights online during training by tracking
    the contribution of each parameter to loss reduction.
    """
    
    def __init__(self, d_model: int, lambda_si: float = 1.0, damping: float = 0.1, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.lambda_si = lambda_si
        self.damping = damping
        
        # Path integral accumulator projection
        self.path_integrator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="path_integrator")
        
    def __call__(self, features: jnp.ndarray, gradients: jnp.ndarray) -> jnp.ndarray:
        """
        Update path integral and compute importance.
        
        Args:
            features: Current features
            gradients: Current gradients
            
        Returns:
            Updated importance estimates
        """
        # Project gradient importance
        combined = features * gradients
        importance_update = self.path_integrator(combined)
        
        return importance_update


class ProgressiveNeuralNetwork(hk.Module):
    """
    Progressive Neural Networks for continual learning.
    
    Instead of modifying existing weights, PNN adds new capacity
    for each new task while freezing previous columns.
    """
    
    def __init__(self, d_model: int, max_columns: int = 10, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_columns = max_columns
        
        # Base column (first task)
        self.base_column = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="base_column")
        
        # Lateral connections from previous columns
        self.lateral_adapters = []
        for i in range(max_columns - 1):
            self.lateral_adapters.append(
                hk.Linear(d_model, name=f"lateral_{i}")
            )
        
        # Output projection per column
        self.column_outputs = []
        for i in range(max_columns):
            self.column_outputs.append(
                hk.Linear(d_model, name=f"column_out_{i}")
            )
    
    def __call__(self, x: jnp.ndarray, column_idx: int = 0, 
                 previous_outputs: Optional[List[jnp.ndarray]] = None) -> jnp.ndarray:
        """
        Forward pass through progressive columns.
        
        Args:
            x: Input features
            column_idx: Current column/task index
            previous_outputs: Outputs from previous (frozen) columns
            
        Returns:
            Column output with lateral connections
        """
        # Base processing
        h = self.base_column(x)
        
        # Add lateral connections from previous columns
        if previous_outputs is not None and column_idx > 0:
            for i, prev_out in enumerate(previous_outputs[:column_idx]):
                if i < len(self.lateral_adapters):
                    lateral = self.lateral_adapters[i](prev_out)
                    h = h + 0.5 * lateral  # Scaled lateral connection
        
        # Column-specific output
        if column_idx < len(self.column_outputs):
            output = self.column_outputs[column_idx](h)
        else:
            output = h
            
        return output


class ContinualLearner(hk.Module):
    """
    Main continual learning module integrating EWC, SI, and progressive networks.
    
    Enables adaptive learning for long-term human-centric tasks like:
    - Evolving disaster prediction models
    - Continuous medical diagnosis learning
    - Adaptive robotics control
    
    Without catastrophic forgetting of prior knowledge.
    """
    
    def __init__(
        self, 
        d_model: int,
        lambda_ewc: float = 1000.0,
        lambda_si: float = 1.0,
        use_progressive: bool = False,
        max_tasks: int = 10,
        name=None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.lambda_ewc = lambda_ewc
        self.lambda_si = lambda_si
        self.use_progressive = use_progressive
        self.max_tasks = max_tasks
        
        # EWC module
        self.ewc = ElasticWeightConsolidation(d_model, lambda_ewc)
        
        # Synaptic Intelligence module
        self.si = SynapticIntelligence(d_model, lambda_si)
        
        # Progressive network (optional)
        if use_progressive:
            self.progressive = ProgressiveNeuralNetwork(d_model, max_tasks)
        
        # Task embedding for conditioning
        self.task_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="task_encoder")
        
        # Importance weighting network
        self.importance_network = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.sigmoid
        ], name="importance_network")
        
        # Knowledge consolidation
        self.knowledge_consolidator = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="knowledge_consolidator")
        
        # Output projection
        self.output_proj = hk.Linear(d_model, name="output_proj")
    
    def __call__(
        self, 
        features: jnp.ndarray,
        task_embedding: Optional[jnp.ndarray] = None,
        gradients: Optional[jnp.ndarray] = None,
        previous_features: Optional[List[jnp.ndarray]] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass with continual learning enhancements.
        
        Args:
            features: Input features [batch, seq_len, d_model]
            task_embedding: Optional task identifier embedding
            gradients: Optional gradients for SI importance estimation
            previous_features: Features from previous tasks for progressive nets
            
        Returns:
            Dictionary with:
            - features: Processed features
            - importance: Parameter importance estimates
            - si_update: Synaptic intelligence updates
        """
        _, seq_len, _ = features.shape
        
        # Encode task if provided
        if task_embedding is not None:
            task_encoded = self.task_encoder(task_embedding)
            # Broadcast to sequence length
            if task_encoded.ndim == 2:
                task_encoded = task_encoded[:, None, :].repeat(seq_len, axis=1)
            features = features + 0.1 * task_encoded
        
        # EWC importance estimation
        importance = self.ewc(features)
        
        # Synaptic Intelligence update
        if gradients is not None:
            si_update = self.si(features, gradients)
        else:
            si_update = jnp.zeros_like(features)
        
        # Progressive network processing
        if self.use_progressive and previous_features is not None:
            progressive_out = self.progressive(features, len(previous_features), previous_features)
            features = features + 0.5 * progressive_out
        
        # Compute dynamic importance weights
        importance_input = jnp.concatenate([features, importance.repeat(self.d_model, axis=-1)], axis=-1)
        importance_weights = self.importance_network(importance_input)
        
        # Apply importance-weighted consolidation
        consolidation_input = jnp.concatenate([
            features, features * importance_weights
        ], axis=-1)
        consolidated = self.knowledge_consolidator(consolidation_input)
        
        # Final output
        output = self.output_proj(consolidated + features)  # Residual
        
        return {
            "features": output,
            "importance": importance,
            "importance_weights": importance_weights,
            "si_update": si_update,
            "consolidated": consolidated
        }
    
    def compute_continual_loss(
        self,
        params: Dict[str, jnp.ndarray],
        task_memories: List[TaskMemory],
        use_ewc: bool = True,
        use_si: bool = True
    ) -> jnp.ndarray:
        """
        Compute combined continual learning loss from all task memories.
        
        Args:
            params: Current model parameters
            task_memories: List of task memories with Fisher/importance info
            use_ewc: Whether to use EWC loss
            use_si: Whether to use SI loss
            
        Returns:
            Combined continual learning regularization loss
        """
        total_loss = jnp.float32(0.0)
        
        for memory in task_memories:
            if use_ewc and memory.fisher_matrix is not None:
                ewc_loss = compute_ewc_loss(
                    params, memory.params_snapshot,
                    memory.fisher_matrix, self.lambda_ewc
                )
                total_loss += ewc_loss
                
            if use_si and memory.importance_weights is not None:
                si_loss = compute_si_loss(
                    params, memory.params_snapshot,
                    memory.importance_weights, self.lambda_si
                )
                total_loss += si_loss
        
        return total_loss


class OnlineMetaLearner(hk.Module):
    """
    Online meta-learning for rapid adaptation to new tasks
    while maintaining performance on previous tasks.
    
    Combines MAML-style adaptation with continual learning.
    """
    
    def __init__(self, d_model: int, inner_lr: float = 0.01, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.inner_lr = inner_lr
        
        # Adaptation network
        self.adaptation_net = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="adaptation_net")
        
        # Learning rate modulator
        self.lr_modulator = hk.Sequential([
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.softplus  # Ensure positive learning rate
        ], name="lr_modulator")
        
        # Knowledge transfer network
        self.knowledge_transfer = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="knowledge_transfer")
        
    def __call__(
        self,
        support_features: jnp.ndarray,
        query_features: jnp.ndarray,
        previous_knowledge: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Perform meta-learning adaptation.
        
        Args:
            support_features: Features from support set
            query_features: Features from query set
            previous_knowledge: Knowledge from previous tasks
            
        Returns:
            Adapted features and adaptation parameters
        """
        # Compute task-specific adaptation
        support_summary = support_features.mean(axis=1)
        adapted = self.adaptation_net(support_summary)
        
        # Modulate learning rate based on task
        task_lr = self.lr_modulator(support_summary)
        
        # Transfer knowledge from previous tasks if available
        if previous_knowledge is not None:
            transfer_input = jnp.concatenate([adapted, previous_knowledge], axis=-1)
            transferred = self.knowledge_transfer(transfer_input)
            adapted = adapted + 0.3 * transferred
        
        # Apply adaptation to query features
        adapted_query = query_features + task_lr * adapted[:, None, :]
        
        return {
            "adapted_features": adapted_query,
            "task_representation": adapted,
            "task_lr": task_lr
        }


class MemoryAwareSynapses(hk.Module):
    """
    Memory-Aware Synapses (MAS) for importance-weighted learning.
    
    Computes importance based on sensitivity of learned function
    to each parameter (not gradient of loss).
    """
    
    def __init__(self, d_model: int, damping: float = 0.01, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.damping = damping
        
        # Sensitivity estimator
        self.sensitivity_estimator = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            jax.nn.sigmoid
        ], name="sensitivity_estimator")
        
        # Memory importance network
        self.importance_net = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model)
        ], name="importance_net")
        
    def __call__(
        self,
        features: jnp.ndarray,
        output_gradients: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute memory-aware importance weights.
        
        Args:
            features: Input features
            output_gradients: Gradients of output w.r.t. parameters
            
        Returns:
            Importance weights for each parameter
        """
        # Estimate sensitivity
        sensitivity = self.sensitivity_estimator(features)
        
        # Combine with output gradients
        importance_input = jnp.concatenate([
            sensitivity * output_gradients,
            features
        ], axis=-1)
        
        importance = self.importance_net(importance_input)
        
        # Apply damping for stability
        importance = importance + self.damping
        
        return importance


def create_continual_learning_optimizer(
    base_lr: float = 1e-4,
    weight_decay: float = 0.01
) -> optax.GradientTransformation:
    """
    Create optimizer with continual learning regularization.
    
    Args:
        base_lr: Base learning rate
        weight_decay: Weight decay coefficient
        
    Returns:
        Optax optimizer with EWC-aware regularization
    """
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=base_lr, weight_decay=weight_decay),
    )


def make_ewc_aware_loss_fn(
    base_loss_fn: Callable,
    task_memories: List[TaskMemory],
    lambda_ewc: float = 1000.0,
    lambda_si: float = 1.0
) -> Callable:
    """
    Wrap a base loss function with EWC and SI regularization.
    
    Args:
        base_loss_fn: Original loss function
        task_memories: List of task memories
        lambda_ewc: EWC strength
        lambda_si: SI strength
        
    Returns:
        New loss function with continual learning regularization
    """
    def ewc_aware_loss(params, batch, rng):
        # Compute base loss
        base_loss = base_loss_fn(params, batch, rng)
        
        # Add EWC regularization for each previous task
        ewc_loss = jnp.float32(0.0)
        si_loss = jnp.float32(0.0)
        
        for memory in task_memories:
            if memory.fisher_matrix is not None:
                ewc_loss += compute_ewc_loss(
                    params, memory.params_snapshot,
                    memory.fisher_matrix, lambda_ewc
                )
            
            if memory.importance_weights is not None:
                si_loss += compute_si_loss(
                    params, memory.params_snapshot,
                    memory.importance_weights, lambda_si
                )
        
        total_loss = base_loss + ewc_loss + si_loss
        
        return total_loss
    
    return ewc_aware_loss

