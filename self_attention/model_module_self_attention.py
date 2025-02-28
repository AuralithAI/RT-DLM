import haiku as hk
import jax
import os
import sys
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class SelfAttentionModel(hk.Module):
    def __init__(self, d_model: int, num_heads: int, vocab_size: int, max_seq_length: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.embedding = hk.Embed(vocab_size=vocab_size, embed_dim=d_model, name="token_embedding")
        self.attention = hk.MultiHeadAttention(
            num_heads=num_heads,
            key_size=d_model // num_heads,
            model_size=d_model,
            w_init=hk.initializers.VarianceScaling(1.0),
        )
        self.norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.ffn = hk.Sequential([
            hk.Linear(d_model * 4),  
            jax.nn.silu,
            hk.Linear(d_model),  
        ])
        self.proj = hk.Linear(vocab_size)

    def apply_spiking_attention(self, scores, spike_threshold, epsilon):
        """
        Apply Spiking Attention by thresholding attention scores to activate only important tokens.
        """
        if spike_threshold is None or epsilon is None:
            return scores
        spiking_mask = scores > spike_threshold
        spiked_scores = jnp.where(spiking_mask, scores, 0.0)
        return spiked_scores / (jnp.sum(spiked_scores, axis=-1, keepdims=True) + epsilon) 

    def __call__(self, inputs, return_attention=False, spike_threshold=0.1, epsilon=1e-8):
        """
        Forward pass with Spiking Attention, accepting spike_threshold and epsilon as arguments.
        """
        inputs = jnp.asarray(inputs, dtype=jnp.int32, copy=True)  
        mask = (inputs != 0).astype(jnp.float32)[:, None, None, :]
        x = self.embedding(inputs) * jnp.sqrt(self.d_model)
        x = self.norm1(x)
        attn_out = self.attention(query=x, key=x, value=x, mask=mask)
        attention_weights = jax.nn.softmax(attn_out, axis=-1) if return_attention else None
        spiked_attentions = self.apply_spiking_attention(attn_out, spike_threshold, epsilon)
        x = x + spiked_attentions  
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out  
        logits = self.proj(x)

        if return_attention:
            return logits, attention_weights 
        return logits