import haiku as hk
import jax.numpy as jnp
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from self_attention.model_module_self_attention import SelfAttentionModel
from transformer_block.model_transformer_module import TransformerModel

# Combined Transformer + Self-Attention Model
class TransformerSelfAttentionModel(hk.Module):
    """
    Combines Self-Attention Model and Transformer Model into a single architecture.
    """
    def __init__(self, d_model: int, num_heads: int, num_layers: int, vocab_size: int, max_seq_length: int, name=None):
        super().__init__(name=name)
        self.embedding = hk.Embed(vocab_size, d_model)
        self.position_enc = hk.Embed(max_seq_length, d_model)

        # Self-Attention Model
        self.self_attention = SelfAttentionModel(d_model, num_heads, vocab_size, max_seq_length)

        # Transformer Model
        self.transformer = TransformerModel(d_model, num_heads, num_layers, vocab_size, max_seq_length)

        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.proj = hk.Linear(vocab_size)

    def __call__(self, inputs, rng=None, return_attention=False):
        inputs = jnp.asarray(inputs, dtype=jnp.int32)  # Ensure integer inputs
        x = self.embedding(inputs) + self.position_enc(jnp.arange(inputs.shape[1], dtype=jnp.int32))

        # Apply Self-Attention
        x, attn_weights_self = self.self_attention(inputs, return_attention=True)
        attn_weights_self = jnp.expand_dims(attn_weights_self, axis=0)
        #print(f"[INFO] - Transformer Module ==> X Shape: {x.shape} | Self Attn Weights Shape: {attn_weights_self.shape}")

        # Apply Transformer Model
        logits, attn_weights_transformer = self.transformer(x, rng, return_attention=True)
        #print(f"[INFO] - Transformer Module ==> Logits Shape: {logits.shape} | Transformer Attn Weights Shape: {attn_weights_transformer.shape}")

        if return_attention:
            return logits, jnp.concatenate([attn_weights_self, attn_weights_transformer])
        return logits