import haiku as hk
import jax
import jax.numpy as jnp

class TransformerBlock(hk.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.2, name=None):
        super().__init__(name=name)
        self.mha = hk.MultiHeadAttention(
            num_heads=num_heads,
            key_size=d_model // num_heads,
            model_size=d_model,
            w_init=hk.initializers.VarianceScaling(1.0)
        )
        self.norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.ffn = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(d_model)
        ])
        self.dropout_rate = dropout_rate

    def apply_spiking_attention(self, scores, spike_threshold, epsilon):
        """
        Apply Spiking Attention by thresholding attention scores.
        """
        if spike_threshold is None or epsilon is None:
            return scores
        spiking_mask = scores > spike_threshold
        spiked_scores = jnp.where(spiking_mask, scores, 0.0)
        return spiked_scores / (jnp.sum(spiked_scores, axis=-1, keepdims=True) + epsilon)

    def __call__(self, x, rng=None, return_attention=False, spike_threshold=0.1, epsilon=1e-8):
        attn_out = self.mha(query=x, key=x, value=x)
        spiked_attentions = self.apply_spiking_attention(attn_out, spike_threshold, epsilon)
        attention_weights = jax.nn.softmax(attn_out, axis=-1) if return_attention else None
        x = self.norm1(x + spiked_attentions)
        ffn_out = self.ffn(x)
        if rng is not None:
            ffn_out = hk.dropout(rng, self.dropout_rate, ffn_out)
        x = self.norm2(x + ffn_out)
        if return_attention:
            return x, attention_weights
        return x
    
class TransformerModel(hk.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, vocab_size: int, max_seq_length: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.embedding = hk.Embed(vocab_size, d_model, name="token_embedding")
        self.position_enc = hk.Embed(max_seq_length, d_model, name="position_embedding")
        self.layers = [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.proj = hk.Linear(d_model)

    def __call__(self, inputs, rng=None, return_attention=False, spike_threshold=0.1, epsilon=1e-8):
        inputs = jnp.asarray(inputs, dtype=jnp.int32, copy=True)
        embed_out = self.embedding(inputs)
        pos_enc = self.position_enc(jnp.arange(inputs.shape[1], dtype=jnp.int32))
        pos_enc = jnp.expand_dims(pos_enc, axis=0)  
        pos_enc = jnp.tile(pos_enc, (inputs.shape[0], 1, 1))
        embed_out = embed_out[:, :, 0, :] 
        x = embed_out + pos_enc
        attention_maps = []
        for layer in self.layers:
            layer_rng, rng = jax.random.split(rng) if rng is not None else (None, None)
            if return_attention:
                x, attn_weights = layer(x, rng=layer_rng, return_attention=True, spike_threshold=spike_threshold, epsilon=epsilon)
                attention_maps.append(attn_weights)
            else:
                x = layer(x, rng=layer_rng, spike_threshold=spike_threshold, epsilon=epsilon)
        x = self.norm(x)
        logits = self.proj(x)

        if return_attention:
            return logits, jnp.array(attention_maps)
        return logits