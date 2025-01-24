import haiku as hk
import jax.numpy as jnp
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