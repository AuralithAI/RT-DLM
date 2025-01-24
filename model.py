import haiku as hk
import jax.numpy as jnp
import jax
from config import RTDLMConfig

"""
    EmbeddingLayer class is used to create embeddings for token and positional embeddings.
"""
class EmbeddingLayer():
    """
       Constructor for EmbeddingLayer   
    """
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int):
        super().__init__()
        self.token_embedding = hk.Embed(vocab_size, d_model)
        self.position_embedding = hk.Embed(max_seq_length, d_model)

    def __call__(self, token_ids: jnp.ndarray, seq_length: int):
        """
        Combine token and positional embeddings.
        Parameters:
            token_ids: jnp.ndarray - Token IDs
            seq_length: int - Sequence length
        Returns:
            jnp.ndarray: Combined embeddings
        """
        position_ids = jnp.arange(seq_length)[None, :] 
        token_embeds = self.token_embedding(token_ids)  
        position_embeds = self.position_embedding(position_ids)  
        return token_embeds + position_embeds  
    

"""
    SelfAttention class is used to create self-attention mechanism using hk.multihead_attention.
"""
class SelfAttention(hk.Module):
    """
       Constructor for SelfAttention 
       By-Default:
            MultiHeadAttention is used with key_size = d_model // num_heads
            But it needs a w_init parameter to be passed.
            So we are passing it as hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"):
                * scale = 1.0 : scale factor of weights.
                * mode = "fan_avg" : Mode of computation. (Balances initialization for both forward and backward passes)
                * distribution = "uniform" : Distribution of weights is uniform
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = hk.MultiHeadAttention(
            num_heads=num_heads, 
            key_size=d_model // num_heads,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        )

    def __call__(self, x: jnp.ndarray):
        """
        Apply self-attention mechanism.
        Parameters:
            x: jnp.ndarray - Input tensor
        Returns:
            jnp.ndarray: Output tensor
            Here, assumption is that query, key and value [Q,K,V] are same. (May change later..)
        """
        return self.attention(x, x, x)
    

"""
    Create a Transformer model using EmbeddingLayer and SelfAttention.
"""
class TransformerBlock(hk.Module):
    """
       Constructor for TransformerBlock:
        Here we are using a 2 Layer MLP (Multi-Layer Preceptron) with 4*d_model and d_model units. (May change later)
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = SelfAttention(d_model, num_heads)
        self.layer_norm_1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.feedforward = hk.nets.MLP([d_model * 4, d_model])  
        self.layer_norm_2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x: jnp.ndarray):
        """
        Forward pass for Transformer Block.
        """
        attention_output = self.attention(x)
        x = self.addAndNormalize(x, attention_output, self.layer_norm_1)
        feedforward_output = self.feedforward(x)
        x = self.addAndNormalize(x, feedforward_output, self.layer_norm_2)

        return x
    
    def addAndNormalize(self, x: jnp.ndarray, output: jnp.ndarray, layer_norm: hk.LayerNorm) -> jnp.ndarray:
        """
        Add and normalize the output using the provided LayerNorm.
        Parameters:
            x: jnp.ndarray - Input tensor
            output: jnp.ndarray - Output tensor to add
            layer_norm: hk.LayerNorm - LayerNorm module to apply normalization
        Returns:
            jnp.ndarray: Output tensor after addition and normalization
        """
        return layer_norm(x + output)
    

"""
    MixtureOfExperts class is used to create a Mixture of Experts mechanism.
    This mechanism selects top-k experts and combines their outputs. (Good for scalability)
"""
class MixtureOfExperts(hk.Module):
    """
       Constructor for MixtureOfExperts:
        
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.experts = [hk.nets.MLP([d_model * 4, d_model]) for _ in range(num_experts)]
        self.gating = hk.Linear(num_experts)
        self.top_k = top_k

    def __call__(self, x: jnp.ndarray):
        """
        MoE mechanism: Select top-k experts and combine their outputs.
        """
        gate_scores = jax.nn.softmax(self.gating(x), axis=-1)
        top_k_indices = jax.lax.top_k(gate_scores, self.top_k)[1]
        outputs = []

        for b in range(x.shape[0]):
            expert_outputs = [
                self.experts[idx](x[b:b+1]) for idx in top_k_indices[b]
            ]
            outputs.append(jnp.sum(jnp.stack(expert_outputs), axis=0))

        return jnp.concatenate(outputs, axis=0)

