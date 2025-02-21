import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from self_attention.model_module_self_attention import SelfAttentionModel
from transformer_block.model_transformer_module import TransformerModel
from moe_block.sparse_moe import SparseMoE
from memory_bank import MemoryBank

class TMSModel(hk.Module):
    """
    Transformer + MoE + Self-Attention (TMS) Model
    """
    def __init__(self, d_model: int, num_heads: int, num_layers: int, vocab_size: int, max_seq_length: int, 
                 moe_experts: int, moe_top_k: int, memory_size: int, retrieval_k: int, name=None):
        super().__init__(name=name)

        # Token embedding and positional encoding
        self.embedding = hk.Embed(vocab_size, d_model)
        self.position_enc = hk.Embed(max_seq_length, d_model)

        # Self-Attention Model
        self.self_attention = SelfAttentionModel(d_model, num_heads, vocab_size, max_seq_length)

        # Transformer Model
        self.transformer = TransformerModel(d_model, num_heads, num_layers, vocab_size, max_seq_length)

        # Sparse Mixture of Experts (MoE)
        self.moe = SparseMoE(d_model, moe_experts, moe_top_k, expert_capacity=3)

        # Memory Bank (FAISS-based)
        self.memory = MemoryBank(memory_size, d_model, retrieval_k)

        # Projection Layer for Retrieved Memory
        self.memory_projection = hk.Linear(d_model)
        self.memory_to_logits = hk.Linear(vocab_size)

        # Final normalization and projection
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.proj = hk.Linear(vocab_size)

    def memory_projection_forward(self, retrieved_memory):
        """Applies the memory projection layer to retrieved memory embeddings."""
        assert retrieved_memory.ndim == 3, f"Expected shape (batch, seq, d_model), got {retrieved_memory.shape}"
        return self.memory_projection(retrieved_memory)

    def __call__(self, inputs, rng=None, return_attention=False, retrieved_memory=None):
        inputs = jnp.asarray(inputs, dtype=jnp.int32)

        # Embedding + Positional Encoding
        x = self.embedding(inputs) + self.position_enc(jnp.arange(inputs.shape[1], dtype=jnp.int32))

        # Apply retrieved memory embeddings to input
        if retrieved_memory is None:
            dummy_memory = jnp.zeros((inputs.shape[0], 1, self.embedding.embed_dim), dtype=jnp.float32)
            _ = self.memory_projection(dummy_memory)
            _ = self.memory_to_logits(dummy_memory)
            memory_logits = None  
        else:
            retrieved_memory = jnp.expand_dims(retrieved_memory, axis=1)  
            retrieved_memory = jnp.repeat(retrieved_memory, x.shape[1], axis=1)  
            retrieved_memory = self.memory_projection(retrieved_memory)  
            x += retrieved_memory  
            memory_logits = self.memory_to_logits(retrieved_memory)  

        # Apply Self-Attention
        x, attn_weights_self = self.self_attention(inputs, return_attention=True)

        # Apply Transformer Model
        x, attn_weights_transformer = self.transformer(x, rng, return_attention=True)

        # Apply Sparse MoE
        x, top_k_expert_indices, aux_loss = self.moe(x)

        # Final Normalization and Projection
        x = self.norm(x)
        logits = self.proj(x)

        if memory_logits is not None:
            logits += memory_logits

        if return_attention:
             return logits, (attn_weights_self, attn_weights_transformer), top_k_expert_indices, aux_loss
        return logits