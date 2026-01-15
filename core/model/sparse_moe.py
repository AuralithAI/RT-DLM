import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict

def keep_top_k(scores, k):
    """
    Select the top-k values and set others to -inf to enforce sparsity.
    """
    top_k_values, top_k_indices = jax.lax.top_k(scores, k)
    mask = scores >= jnp.min(top_k_values, axis=-1, keepdims=True)
    return jnp.where(mask, scores, -1e9), top_k_indices


def apply_router_jitter(logits: jnp.ndarray, rng_key, 
                        jitter_noise: float = 0.1, is_training: bool = True) -> jnp.ndarray:
    """
    Apply router jitter for better expert utilization (Switch Transformer technique).
    
    During training, adds multiplicative noise to routing logits to encourage
    exploration of different experts and prevent expert collapse.
    
    Args:
        logits: Router logits [batch, seq_len, num_experts]
        rng_key: JAX random key
        jitter_noise: Standard deviation of multiplicative noise
        is_training: Whether in training mode
        
    Returns:
        Jittered logits
    """
    if not is_training or jitter_noise <= 0:
        return logits
    
    # Multiplicative jitter (more effective than additive)
    noise = jax.random.uniform(rng_key, logits.shape, minval=1.0 - jitter_noise, maxval=1.0 + jitter_noise)
    return logits * noise


def compute_capacity_factor_loss(
    routing_probs: jnp.ndarray, 
    num_experts: int,
    capacity_factor: float = 1.25,
) -> jnp.ndarray:
    """
    Compute capacity factor loss to prevent expert overflow.
    
    Ensures no expert receives more than capacity_factor * (tokens / num_experts) tokens.
    This is critical for efficient batched computation.
    
    Args:
        routing_probs: Routing probabilities [batch, seq_len, num_experts]
        num_experts: Number of experts
        capacity_factor: Maximum capacity multiplier (>1 allows overflow buffer)
        
    Returns:
        Capacity overflow penalty loss
    """
    batch_size, seq_len, _ = routing_probs.shape
    
    # Count tokens routed to each expert
    expert_loads = jnp.sum(routing_probs, axis=(0, 1))  # [num_experts]
    
    # Target load per expert
    target_load = batch_size * seq_len / num_experts
    
    # Penalize experts that exceed capacity
    overflow = jnp.maximum(0, expert_loads - capacity_factor * target_load)
    overflow_loss = jnp.sum(overflow ** 2) / (batch_size * seq_len)
    
    return overflow_loss


def load_balancing_loss(gating_logits, num_experts):
    """
    Compute auxiliary loss to ensure all experts get utilized.
    """
    routing_probs = jax.nn.softmax(gating_logits, axis=-1)
    expert_usage = jnp.sum(routing_probs, axis=(0, 1))  
    expert_importance = expert_usage / (jnp.sum(expert_usage) + 1e-10)
    mean_importance = jnp.mean(expert_importance)
    std_importance = jnp.std(expert_importance)
    coeff_variation = std_importance / (mean_importance + 1e-10)
    return coeff_variation

def expert_specialization_loss(expert_outputs, expert_indices, num_experts):
    """
    Encourage expert specialization by maximizing inter-expert diversity.
    """
    # Calculate expert-specific representations
    expert_representations = []
    for expert_id in range(num_experts):
        expert_mask = (expert_indices == expert_id)
        if jnp.sum(expert_mask) > 0:
            expert_repr = jnp.mean(expert_outputs[expert_mask], axis=0)
        else:
            expert_repr = jnp.zeros_like(expert_outputs[0])
        expert_representations.append(expert_repr)
    
    expert_stack = jnp.stack(expert_representations)
    
    # Calculate pairwise similarities and minimize them
    similarities = jnp.dot(expert_stack, expert_stack.T)
    # Remove diagonal (self-similarity)
    mask = 1.0 - jnp.eye(num_experts)
    similarity_loss = jnp.sum(similarities * mask) / (num_experts * (num_experts - 1))
    
    return similarity_loss

def dynamic_load_balancing_loss(gating_logits, expert_usage_history, num_experts, alpha=0.1):
    """
    Advanced load balancing that considers historical usage patterns.
    """
    # Current routing probabilities
    routing_probs = jax.nn.softmax(gating_logits, axis=-1)
    current_usage = jnp.sum(routing_probs, axis=(0, 1))
    
    # Combine with historical usage
    target_usage = jnp.ones(num_experts) / num_experts
    historical_weight = 0.9
    combined_usage = historical_weight * expert_usage_history + (1 - historical_weight) * current_usage
    
    # Calculate KL divergence from uniform distribution
    usage_distribution = combined_usage / jnp.sum(combined_usage)
    kl_loss = jnp.sum(usage_distribution * jnp.log(usage_distribution * num_experts + 1e-8))
    
    # Add entropy regularization to encourage exploration
    entropy = -jnp.sum(usage_distribution * jnp.log(usage_distribution + 1e-8))
    entropy_bonus = alpha * entropy
    
    return kl_loss - entropy_bonus


class AdaptiveGatingNetwork(hk.Module):
    """Advanced gating network with context-aware routing."""
    
    def __init__(self, d_model: int, num_experts: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Multi-layer gating network
        self.gate_network = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(d_model // 2),
            jax.nn.silu,
            hk.Linear(num_experts)
        ])
        
        # Context encoder for routing decisions
        self.context_encoder = hk.Sequential([
            hk.Linear(d_model // 4),
            jax.nn.silu,
            hk.Linear(num_experts)
        ])
        
        # Expert affinity predictor
        self.affinity_predictor = hk.Linear(num_experts)
    
    def __call__(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute gating scores with context awareness.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            context: Optional context information
            
        Returns:
            Gating logits [batch_size, seq_len, num_experts]
        """
        # Base gating scores
        gate_logits = self.gate_network(x)
        
        # Add context-aware routing if context is provided
        if context is not None:
            context_scores = self.context_encoder(context)
            if context_scores.ndim == 2:  # Broadcast to sequence dimension
                context_scores = jnp.expand_dims(context_scores, 1)
            gate_logits = gate_logits + 0.5 * context_scores
        
        # Add input-dependent affinity bias
        input_summary = jnp.mean(x, axis=1)  # [batch_size, d_model]
        affinity_bias = self.affinity_predictor(input_summary)
        affinity_bias = jnp.expand_dims(affinity_bias, 1)  # Add sequence dimension
        
        return gate_logits + 0.3 * affinity_bias

class SparseMoE(hk.Module):
    """
    Sparse Mixture of Experts (MoE) with Top-K Gating, Load Balancing, and Expert Specialization.
    Enhanced with dynamic routing, context awareness, and advanced optimization techniques.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, expert_capacity: int, 
                 specialization_weight: float = 0.1, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity
        self.specialization_weight = specialization_weight
        
        # Enhanced experts with different architectures for specialization
        self.experts = []
        for i in range(num_experts):
            if i % 3 == 0:  # Language understanding experts
                expert = hk.Sequential([
                    hk.Linear(d_model * 4, w_init=hk.initializers.VarianceScaling(1.0)),
                    jax.nn.gelu,
                    hk.Linear(d_model, w_init=hk.initializers.VarianceScaling(1.0))
                ], name=f"language_expert_{i}")
            elif i % 3 == 1:  # Reasoning experts
                expert = hk.Sequential([
                    hk.Linear(d_model * 2, w_init=hk.initializers.VarianceScaling(1.0)),
                    jax.nn.silu,
                    hk.Linear(d_model * 2, w_init=hk.initializers.VarianceScaling(1.0)),
                    jax.nn.silu,
                    hk.Linear(d_model, w_init=hk.initializers.VarianceScaling(1.0))
                ], name=f"reasoning_expert_{i}")
            else:  # Creative experts
                expert = hk.Sequential([
                    hk.Linear(d_model * 3, w_init=hk.initializers.VarianceScaling(1.0)),
                    lambda x: jax.nn.silu(x) * jax.nn.sigmoid(x),  # Swish activation
                    hk.Linear(d_model, w_init=hk.initializers.VarianceScaling(1.0))
                ], name=f"creative_expert_{i}")
            self.experts.append(expert)
        
        # Advanced gating network
        self.gate = AdaptiveGatingNetwork(d_model, num_experts)
        
        # Expert usage tracking with momentum
        self.expert_usage = hk.get_state("expert_usage", [num_experts], 
                                       dtype=jnp.float32, init=jnp.zeros)

    def apply_spiking_attention(self, x, spike_threshold, epsilon):
        """
        Enhanced spiking attention with adaptive thresholding.
        """
        if spike_threshold is None or epsilon is None or not 0 <= spike_threshold <= 1:
            return x
            
        # Calculate importance scores
        importance_scores = jnp.linalg.norm(x, axis=-1, keepdims=True)
        
        # Adaptive threshold based on input distribution
        adaptive_threshold = jnp.percentile(importance_scores, 
                                          (1.0 - spike_threshold) * 100)
        
        # Create spiking mask
        spiking_mask = importance_scores > adaptive_threshold
        spiked_x = jnp.where(spiking_mask, x, 0.0)
        
        # Normalize with temperature scaling
        temperature = 1.0 + 0.1 * jnp.std(importance_scores)
        normalization = jnp.sum(jnp.abs(spiked_x), axis=-1, keepdims=True) + epsilon
        
        return spiked_x / (normalization * temperature)
    
    def update_expert_usage(self, top_k_indices):
        """
        Update usage statistics for experts based on their selection frequency.
        """
        expert_usage_update = jnp.bincount(top_k_indices.flatten(), minlength=self.num_experts, length=self.num_experts) / self.num_experts
        current_usage = hk.get_state("expert_usage", [self.num_experts], dtype=jnp.float32, init=jnp.zeros)
        new_usage = current_usage + expert_usage_update
        hk.set_state("expert_usage", new_usage)

    def prune_experts_and_neurons(self, expert_threshold=0.01, neuron_threshold=0.01):
        """
        Prune experts with usage below a threshold, with weight interpolation for better performance.
        Returns a new model with pruned experts or updates in-place.
        """
        usage = self.expert_usage
        active_experts = usage > expert_threshold
        if jnp.sum(active_experts) < 1:
            raise ValueError("Cannot prune all experts.")

        new_num_experts = int(jnp.sum(active_experts))
        if new_num_experts == self.num_experts:
            return self

        new_model = SparseMoE(
            d_model=self.d_model,
            num_experts=new_num_experts,
            top_k=self.top_k,
            expert_capacity=self.expert_capacity,
            specialization_weight=self.specialization_weight,
            name=self.name
        )
        active_indices = jnp.where(active_experts)[0]

        new_experts = []
        for i, idx in enumerate(active_indices):
            expert = self.experts[idx]
            expert_ffn_usage = jnp.mean(jnp.abs(expert.layers[0].w), axis=0) 
            active_neurons = expert_ffn_usage > neuron_threshold
            new_intermediate_size = int(jnp.sum(active_neurons))
            if new_intermediate_size < 1:
                new_intermediate_size = self.d_model * 2 

            new_expert = hk.Sequential([
                hk.Linear(new_intermediate_size, w_init=hk.initializers.VarianceScaling(1.0)),
                jax.nn.silu,
                hk.Linear(self.d_model, w_init=hk.initializers.VarianceScaling(1.0))
            ])
            new_expert.layers[0].w = jnp.take(expert.layers[0].w, jnp.where(active_neurons)[0], axis=1)
            new_expert.layers[0].b = jnp.take(expert.layers[0].b, jnp.where(active_neurons)[0])
            new_expert.layers[2].w = expert.layers[2].w
            new_expert.layers[2].b = expert.layers[2].b
            new_experts.append(new_expert)
        new_model.experts = new_experts
        new_gate_w = jnp.take(self.gate.w, active_indices, axis=1)
        new_gate_b = jnp.take(self.gate.b, active_indices)
        new_model.gate.w = new_gate_w
        new_model.gate.b = new_gate_b

        return new_model

    def __call__(self, x, context: Optional[jnp.ndarray] = None, spike_threshold=0.1, epsilon=1e-8):
        """
        Enhanced forward pass with improved routing and specialization.
        """
        x = jnp.asarray(x, dtype=jnp.float32)
        batch_size, seq_len, _ = x.shape

        # Apply enhanced spiking attention
        x = self.apply_spiking_attention(x, spike_threshold, epsilon)
        
        # Get gating scores with context awareness
        gating_logits = self.gate(x, context)
        
        # Dynamic noise injection based on expert usage diversity
        usage_entropy = -jnp.sum(self.expert_usage * jnp.log(self.expert_usage + 1e-8))
        noise_scale = jnp.maximum(0.01 * (2.0 - usage_entropy), 1e-4)
        gating_logits += jax.random.normal(hk.next_rng_key(), gating_logits.shape) * noise_scale
        
        # Temperature scaling based on training progress
        training_step = hk.get_state("training_step", [], dtype=jnp.int32, init=jnp.zeros)
        temperature = 1.0 + 0.5 * jnp.exp(-training_step / 1000.0)  # Decay temperature
        gating_logits = gating_logits / temperature
        
        # Top-k selection with improved balancing
        top_k_values, top_k_indices = jax.lax.top_k(gating_logits, self.top_k)
        mask = gating_logits >= jnp.min(top_k_values, axis=-1, keepdims=True)
        gating_logits = jnp.where(mask, gating_logits, -1e9)
        gating_probs = jax.nn.softmax(gating_logits, axis=-1)

        # Compute expert outputs efficiently
        expert_outputs = jnp.stack([expert(x) for expert in self.experts], axis=0)

        # Advanced routing with capacity constraints
        batch_indices = jnp.arange(batch_size)[:, None, None]
        seq_indices = jnp.arange(seq_len)[None, :, None]
        
        selected_expert_outputs = expert_outputs[top_k_indices, batch_indices, seq_indices]
        top_k_gating_probs = jnp.take_along_axis(gating_probs, top_k_indices, axis=-1)
        
        # Weighted combination with residual connection
        final_output = jnp.einsum('bskd,bsk->bsd', selected_expert_outputs, top_k_gating_probs)
        final_output = final_output + 0.1 * x  # Residual connection
        
        # Update expert usage with momentum
        current_usage = jnp.sum(gating_probs, axis=(0, 1)) / (batch_size * seq_len)
        momentum = 0.9
        new_usage = momentum * self.expert_usage + (1 - momentum) * current_usage
        hk.set_state("expert_usage", new_usage)
        hk.set_state("training_step", training_step + 1)
        
        # Calculate comprehensive losses
        load_balance_loss = dynamic_load_balancing_loss(gating_logits, self.expert_usage, 
                                                      self.num_experts)
        specialization_loss = expert_specialization_loss(
            expert_outputs.reshape(-1, expert_outputs.shape[-1]),
            top_k_indices.flatten(),
            self.num_experts
        )
        
        total_aux_loss = load_balance_loss + self.specialization_weight * specialization_loss
        
        # Detailed metrics for monitoring
        metrics = {
            "expert_usage": self.expert_usage,
            "load_balance_loss": load_balance_loss,
            "specialization_loss": specialization_loss,
            "usage_entropy": usage_entropy,
            "temperature": temperature,
            "noise_scale": noise_scale,
            "gating_max": jnp.max(gating_probs),
            "gating_min": jnp.min(gating_probs)
        }
        
        return final_output, top_k_indices, total_aux_loss, metrics
