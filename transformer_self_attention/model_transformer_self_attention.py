import haiku as hk
import jax
import jax.numpy as jnp
from transformer_block.model_transformer_module import TransformerBlock
from self_attention.model_module_self_attention import SelfAttentionModel

class TransformerSelfAttentionModel(hk.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, vocab_size: int, max_seq_length: int, name=None):
        super().__init__(name=name)
        self.embedding = hk.Embed(vocab_size, d_model)
        self.position_enc = hk.Embed(max_seq_length, d_model)
        
        # Add self-attention as the first processing layer
        self.self_attention = SelfAttentionModel(d_model, num_heads, vocab_size, max_seq_length)

        # Transformer layers
        self.layers = [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.proj = hk.Linear(vocab_size)

    def __call__(self, inputs, rng=None, return_attention=False):
        mask = (inputs != 0).astype(jnp.float32)[:, None, None, :]
        seq_length = inputs.shape[1]

        # Embedding and positional encoding
        x = self.embedding(inputs) + self.position_enc(jnp.arange(seq_length))

        # Apply self-attention layer
        x, attn_weights_self = self.self_attention(inputs, return_attention=True)

        attention_maps = [attn_weights_self]
        for layer in self.layers:
            layer_rng, rng = jax.random.split(rng) if rng is not None else (None, None)
            if return_attention:
                x, attn_weights = layer(x, mask, rng=layer_rng, return_attention=True)
                attention_maps.append(attn_weights)
            else:
                x = layer(x, mask, rng=layer_rng)

        x = self.norm(x)
        logits = self.proj(x)

        if return_attention:
            return logits, jnp.array(attention_maps)
        return logits
