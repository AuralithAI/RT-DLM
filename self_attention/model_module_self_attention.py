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
            hk.Linear(d_model * 4, w_init=hk.initializers.VarianceScaling(1.0)),
            jax.nn.silu,
            hk.Linear(d_model, w_init=hk.initializers.VarianceScaling(1.0)),
        ])
        self.proj = hk.Linear(vocab_size)
        self.head_usage = hk.get_state("head_usage", [num_heads], dtype=jnp.float32, init=jnp.zeros)
        self.ffn_usage = hk.get_state("ffn_usage", [d_model], dtype=jnp.float32, init=jnp.zeros)

    def apply_spiking_attention(self, scores, spike_threshold, epsilon):
        """
        Apply Spiking Attention by thresholding attention scores to activate only important tokens.
        """
        if spike_threshold is None or epsilon is None or not 0 <= spike_threshold <= 1:
            return scores
        spiking_mask = scores > spike_threshold
        spiked_scores = jnp.where(spiking_mask, scores, 0.0)
        return spiked_scores / (jnp.sum(spiked_scores, axis=-1, keepdims=True) + epsilon)

    def update_usage(self, attn_weights, ffn_out):
        """
        Update usage statistics for attention heads and FFN neurons.
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
                head_usage_update = head_usage_update.at[head].set(head_usage)
        elif attn_weights.ndim == 4:
            head_usage_update = jnp.mean(jnp.abs(attn_weights), axis=(0, 2, 3))
        else:
            raise ValueError(f"Unexpected attn_weights shape: {attn_weights.shape}")
        ffn_usage_update = jnp.mean(jnp.abs(ffn_out), axis=(0, 1))
        current_head_usage = hk.get_state("head_usage", [self.num_heads], dtype=jnp.float32, init=jnp.zeros)
        current_ffn_usage = hk.get_state("ffn_usage", [self.d_model], dtype=jnp.float32, init=jnp.zeros)
        hk.set_state("head_usage", current_head_usage + head_usage_update)
        hk.set_state("ffn_usage", current_ffn_usage + ffn_usage_update)

    def prune_heads_and_ffn(self, head_threshold=0.01, ffn_threshold=0.01):
        """
        Prune attention heads and FFN neurons with usage below thresholds.
        Returns a new model with pruned components.
        """
        usage_heads = hk.get_state("head_usage", [self.num_heads], dtype=jnp.float32, init=jnp.zeros)
        usage_ffn = hk.get_state("ffn_usage", [self.d_model], dtype=jnp.float32, init=jnp.zeros)
        active_heads = usage_heads > head_threshold
        active_ffn = usage_ffn > ffn_threshold

        if jnp.sum(active_heads) < 1:
            raise ValueError("Cannot prune all heads; at least one must remain.")
        if jnp.sum(active_ffn) < 1:
            raise ValueError("Cannot prune all FFN neurons; at least one must remain.")

        new_num_heads = int(jnp.sum(active_heads))
        new_d_model = int(jnp.sum(active_ffn))
        if new_num_heads == self.num_heads and new_d_model == self.d_model:
            return self

        new_model = SelfAttentionModel(
            d_model=new_d_model,
            num_heads=new_num_heads,
            vocab_size=self.embedding.vocab_size,
            max_seq_length=self.max_seq_length,
            name=self.name
        )

        active_head_indices = jnp.where(active_heads)[0]
        new_qkv = jnp.take(self.attention.w_qkv, active_head_indices, axis=1)
        new_o = jnp.take(self.attention.w_o, active_head_indices, axis=0)
        new_model.attention.w_qkv = new_qkv
        new_model.attention.w_o = new_o

        active_ffn_indices = jnp.where(active_ffn)[0]
        intermediate_size = new_d_model * 4
        new_ffn_in = jnp.take(self.ffn.layers[0].w, active_ffn_indices, axis=0)[:, :intermediate_size]
        new_ffn_out = jnp.take(self.ffn.layers[2].w, active_ffn_indices, axis=0)
        new_model.ffn.layers[0].w = new_ffn_in
        new_model.ffn.layers[2].w = new_ffn_out

        new_model.proj = hk.Linear(self.embedding.vocab_size, w_init=hk.initializers.VarianceScaling(1.0))
        return new_model

    def __call__(self, inputs, return_attention=False, spike_threshold=0.1, epsilon=1e-8):
        """
        Forward pass with Spiking Attention and usage tracking.
        """
        inputs = jnp.asarray(inputs, dtype=jnp.int32)
        mask = (inputs != 0).astype(jnp.float32)[:, None, None, :]
        x = self.embedding(inputs) * jnp.sqrt(self.d_model)
        x = self.norm1(x)
        attn_out = self.attention(query=x, key=x, value=x, mask=mask)
        spiked_attentions = self.apply_spiking_attention(attn_out, spike_threshold, epsilon)
        x = x + spiked_attentions
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        logits = self.proj(x)

        if return_attention:
            attention_weights = jax.nn.softmax(attn_out, axis=-1)
            self.update_usage(attention_weights, ffn_out)
            return logits, attention_weights
        return logits