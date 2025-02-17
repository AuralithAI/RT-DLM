import haiku as hk
import jax
import jax.numpy as jnp
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from self_attention.model_module_self_attention import SelfAttentionModel
from transformer_block.model_transformer_module import TransformerModel
from moe_block.sparse_moe import SparseMoE

class TMSModel(hk.Module):
    """
    Transformer + MoE + Self-Attention (TMS) Model
    """
    def __init__(self, d_model: int, num_heads: int, num_layers: int, vocab_size: int, max_seq_length: int, 
                 moe_experts: int, moe_top_k: int, name=None):
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

        # Final normalization and projection
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.proj = hk.Linear(vocab_size)

    def __call__(self, inputs, rng=None, return_attention=False):
        inputs = jnp.asarray(inputs, dtype=jnp.int32)
        
        # Embedding + Positional Encoding
        x = self.embedding(inputs) + self.position_enc(jnp.arange(inputs.shape[1], dtype=jnp.int32))

        # Apply Self-Attention
        x, attn_weights_self = self.self_attention(inputs, return_attention=True)

        # Apply Transformer Model
        x, attn_weights_transformer = self.transformer(x, rng, return_attention=True)

        # Apply Sparse MoE
        x, top_k_expert_indices, aux_loss = self.moe(x)

        # Final Normalization and Projection
        x = self.norm(x)
        logits = self.proj(x)

        if return_attention:
            return logits, jnp.concatenate([attn_weights_self, attn_weights_transformer]), top_k_expert_indices, aux_loss
        return logits