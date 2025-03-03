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
        self.head_usage = hk.get_state("head_usage", [num_heads], dtype=jnp.float32, init=lambda shape, dtype: jnp.zeros(shape, dtype=dtype))

    def apply_spiking_attention(self, scores, spike_threshold, epsilon):
        """
        Apply Spiking Attention by thresholding attention scores to activate only important tokens.
        """
        if spike_threshold is None or epsilon is None or not 0 <= spike_threshold <= 1:
            return scores
        spiking_mask = scores > spike_threshold
        spiked_scores = jnp.where(spiking_mask, scores, 0.0)
        return spiked_scores / (jnp.sum(spiked_scores, axis=-1, keepdims=True) + epsilon)

    def update_head_usage(self, attn_weights):
        """
        Update usage statistics for attention heads based on their weights.
        Ensure the shape produces a scalar per head to match num_heads (12).
        """
        if attn_weights.ndim == 3:
            batch_size, seq_length, _ = attn_weights.shape
            head_dim = self.d_model // self.num_heads
            head_usage_update = jnp.zeros((self.num_heads,))
            for head in range(self.num_heads):
                head_start = head * head_dim
                head_end = head_start + head_dim
                head_weights = attn_weights[:, :, head_start:head_end]
                head_usage = jnp.mean(jnp.abs(head_weights))
                head_usage_update = head_usage_update.at[head].add(head_usage)
        elif attn_weights.ndim == 4:
            head_usage_update = jnp.mean(jnp.abs(attn_weights), axis=(0, 2, 3))
        else:
            raise ValueError(f"Unexpected attn_weights shape: {attn_weights.shape}. Expected (batch_size, seq_length, d_model) or (batch_size, num_heads, seq_length, d_model // num_heads)")

        current_usage = hk.get_state("head_usage", [], dtype=jnp.float32, init=lambda shape, dtype: jnp.zeros(shape, dtype=dtype))
        new_usage = current_usage + head_usage_update
        hk.set_state("head_usage", new_usage)
        return new_usage

    def prune_heads(self, threshold=0.01):
        """
        Prune attention heads with usage below a threshold.
        Returns a new model with pruned heads or updates in-place.
        """
        usage = self.head_usage
        active_heads = usage > threshold
        if jnp.sum(active_heads) < 1:
            raise ValueError("Cannot prune all heads; at least one must remain active.")
        
        new_num_heads = int(jnp.sum(active_heads))
        if new_num_heads == self.num_heads:
            return self

        new_model = SelfAttentionModel(
            d_model=self.d_model,
            num_heads=new_num_heads,
            vocab_size=self.embedding.vocab_size,
            max_seq_length=self.max_seq_length,
            name=self.name
        )
        new_model.head_usage = hk.get_state("head_usage", [new_num_heads], dtype=jnp.float32, init=lambda shape, dtype: jnp.zeros(shape, dtype=dtype))

        active_indices = jnp.where(active_heads)[0]
        new_model.attention = hk.MultiHeadAttention(
            num_heads=new_num_heads,
            key_size=self.d_model // new_num_heads,
            model_size=self.d_model,
            w_init=hk.initializers.VarianceScaling(1.0),
        )
        new_qkv = jnp.zeros_like(new_model.attention.w_qkv)
        new_o = jnp.zeros_like(new_model.attention.w_o)
        for i, idx in enumerate(active_indices):
            new_qkv[:, i] = self.attention.w_qkv[:, idx]
            new_o[i] = self.attention.w_o[idx]
        new_model.attention.w_qkv = new_qkv
        new_model.attention.w_o = new_o

        return new_model 

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
        if return_attention:
            self.update_head_usage(attention_weights)
        x = x + spiked_attentions  
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out  
        logits = self.proj(x)

        if return_attention:
            return logits, attention_weights 
        return logits