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
class SelfAttention():
    def __init__(self):
        super().__init__()