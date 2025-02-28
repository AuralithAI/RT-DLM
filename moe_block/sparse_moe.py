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
    expert_usage = jnp.sum(routing_probs, axis=(0, 1))  
    expert_importance = expert_usage / (jnp.sum(expert_usage) + 1e-10)
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
        self.experts = [hk.Sequential([
            hk.Linear(d_model * 2),  
            jax.nn.silu,
            hk.Linear(d_model)  
        ], name=f"expert_{i}") for i in range(num_experts)]
        self.gate = hk.Linear(num_experts, name="gate")
        # Initialize expert_usage as a state with correct shape
        self.expert_usage = hk.get_state("expert_usage", [num_experts], dtype=jnp.float32, init=lambda shape: jnp.zeros(shape))

    def apply_spiking_attention(self, x, spike_threshold, epsilon):
        """
        Apply Spiking Attention to input tensor.
        """
        if spike_threshold is None or epsilon is None or not 0 <= spike_threshold <= 1:
            return x
        scores = jnp.mean(x, axis=-1, keepdims=True)
        spiking_mask = scores > spike_threshold
        spiked_x = jnp.where(spiking_mask, x, 0.0)
        return spiked_x / (jnp.sum(spiked_x, axis=-1, keepdims=True) + epsilon)
    
    def update_expert_usage(self, top_k_indices):
        """
        Update usage statistics for experts based on their selection frequency.
        """
        expert_usage_update = jnp.bincount(top_k_indices.flatten(), minlength=self.num_experts, length=self.num_experts) / self.num_experts
        current_usage = hk.get_state("expert_usage", [], dtype=jnp.float32, init=lambda shape: jnp.zeros(shape))
        new_usage = current_usage + expert_usage_update
        hk.set_state("expert_usage", new_usage)
        return new_usage

    def prune_experts(self, threshold=0.01):
        """
        Prune experts with usage below a threshold, with weight interpolation for better performance.
        Returns a new model with pruned experts or updates in-place.
        """
        usage = self.expert_usage
        active_experts = usage > threshold
        if jnp.sum(active_experts) < 1:
            raise ValueError("Cannot prune all experts; at least one must remain active.")
        
        new_num_experts = int(jnp.sum(active_experts))
        if new_num_experts == self.num_experts:
            return self  # No pruning needed

        new_model = SparseMoE(
            d_model=self.d_model,
            num_experts=new_num_experts,
            top_k=self.top_k,
            expert_capacity=self.expert_capacity,
            name=self.name
        )
        new_model.expert_usage = hk.get_state("expert_usage", [new_num_experts], dtype=jnp.float32, init=lambda shape: jnp.zeros(shape))
        active_indices = jnp.where(active_experts)[0]

        # Initialize new experts and gate with interpolated weights
        new_model.experts = []
        for i in range(new_num_experts):
            idx = active_indices[i]
            new_expert = hk.Sequential([
                hk.Linear(self.d_model * 2, w_init=hk.initializers.VarianceScaling(1.0)),
                jax.nn.silu,
                hk.Linear(self.d_model, w_init=hk.initializers.VarianceScaling(1.0))
            ], name=f"expert_{i}")
            # Copy weights from the active expert (simplified interpolation)
            new_expert.layers[0].w = self.experts[idx].layers[0].w
            new_expert.layers[0].b = self.experts[idx].layers[0].b
            new_expert.layers[2].w = self.experts[idx].layers[2].w
            new_expert.layers[2].b = self.experts[idx].layers[2].b
            new_model.experts.append(new_expert)

        # Update gate with interpolated weights
        new_model.gate = hk.Linear(new_num_experts, name="gate")
        new_gate_w = jnp.zeros((self.d_model, new_num_experts))
        new_gate_b = jnp.zeros(new_num_experts)
        for i, idx in enumerate(active_indices):
            new_gate_w[:, i] = self.gate.w[:, idx]
            new_gate_b[i] = self.gate.b[idx]
        new_model.gate.w = new_gate_w
        new_model.gate.b = new_gate_b

        return new_model

    def __call__(self, x, spike_threshold=0.1, epsilon=1e-8):
        """
        Forward pass for Sparse MoE.
        x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        x = jnp.asarray(x, dtype=jnp.float32)
        batch_size, seq_len, d_model = x.shape

        x = self.apply_spiking_attention(x, spike_threshold, epsilon)
        gating_logits = self.gate(x)  
        gating_logits += jax.random.normal(hk.next_rng_key(), gating_logits.shape) * 1e-2  
        gating_logits, top_k_indices = keep_top_k(gating_logits, self.top_k)
        gating_probs = jax.nn.softmax(gating_logits, axis=-1)

        expert_outputs = jnp.stack([expert(x) for expert in self.experts], axis=0)  

        batch_indices = jnp.arange(batch_size)[:, None, None]  
        seq_indices = jnp.arange(seq_len)[None, :, None] 

        selected_expert_outputs = expert_outputs[top_k_indices, batch_indices, seq_indices]

        top_k_gating_probs = jnp.take_along_axis(gating_probs, top_k_indices, axis=-1)  
        final_output = jnp.einsum('bskd,bsk->bsd', selected_expert_outputs, top_k_gating_probs)
        self.update_expert_usage(top_k_indices)
        aux_loss = load_balancing_loss(gating_logits, self.num_experts)
        return final_output, top_k_indices, aux_loss