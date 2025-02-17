import haiku as hk
import jax
import jax.numpy as jnp

def keep_top_k(scores, k):
    """
    Select the top-k values and set others to -inf to enforce sparsity.
    """
    top_k_values, top_k_indices = jax.lax.top_k(scores, k)
    mask = scores >= jnp.min(top_k_values, axis=-1, keepdims=True)
    return jnp.where(mask, scores, -1e9), top_k_indices

def load_balancing_loss(gating_logits, num_experts):
    """
    Compute auxiliary loss to ensure all experts get utilized.
    """
    routing_probs = jax.nn.softmax(gating_logits, axis=-1)
    expert_usage = jnp.sum(routing_probs, axis=(0, 1))  # Sum over batch and sequence length
    expert_importance = expert_usage / jnp.sum(expert_usage)

    # Compute Coefficient of Variation (CV) for expert importance distribution
    mean_importance = jnp.mean(expert_importance)
    std_importance = jnp.std(expert_importance)
    coeff_variation = std_importance / (mean_importance + 1e-10)

    return coeff_variation

class SparseMoE(hk.Module):
    """
    Sparse Mixture of Experts (MoE) with Top-K Gating and Load Balancing.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, expert_capacity: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity

        # Initialize experts as Sequential (Silu) feedforward layers
        self.experts = [hk.Sequential([
            hk.Linear(d_model * 2),  
            jax.nn.silu,
            hk.Linear(d_model)  
        ], name=f"expert_{i}") for i in range(num_experts)]
        
        # Gating network
        self.gate = hk.Linear(num_experts, name="gate")

    def __call__(self, x):
        """
        Forward pass for Sparse MoE.
        x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        x = jnp.asarray(x, dtype=jnp.float32)
        batch_size, seq_len, d_model = x.shape

        # Compute gating scores
        gating_logits = self.gate(x)  
        gating_logits += jax.random.normal(hk.next_rng_key(), gating_logits.shape) * 1e-2  # Gaussian noise
        gating_logits, top_k_indices = keep_top_k(gating_logits, self.top_k)
        gating_probs = jax.nn.softmax(gating_logits, axis=-1)  

        # Compute expert outputs
        expert_outputs = jnp.stack([expert(x) for expert in self.experts], axis=0)  

        batch_indices = jnp.arange(batch_size)[:, None, None]  
        seq_indices = jnp.arange(seq_len)[None, :, None] 

        # Gather selected expert outputs using advanced indexing
        selected_expert_outputs = expert_outputs[top_k_indices, batch_indices, seq_indices]
        
        # Weighted sum of expert outputs using gate values
        top_k_gating_probs = jnp.take_along_axis(gating_probs, top_k_indices, axis=-1)  
        final_output = jnp.einsum('bskd,bsk->bsd', selected_expert_outputs, top_k_gating_probs)

        # Compute load balancing loss
        aux_loss = load_balancing_loss(gating_logits, self.num_experts)
        return final_output, top_k_indices, aux_loss
