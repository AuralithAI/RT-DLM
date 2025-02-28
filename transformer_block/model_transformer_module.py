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
        self.head_usage = hk.get_parameter("head_usage", [num_heads], dtype=jnp.float32, init=jnp.zeros)
        self.ffn_usage = hk.get_parameter("ffn_usage", [d_model], dtype=jnp.float32, init=jnp.zeros)

    def apply_spiking_attention(self, scores, spike_threshold, epsilon):
        """
        Apply Spiking Attention by thresholding attention scores.
        """
        if spike_threshold is None or epsilon is None:
            return scores
        spiking_mask = scores > spike_threshold
        spiked_scores = jnp.where(spiking_mask, scores, 0.0)
        return spiked_scores / (jnp.sum(spiked_scores, axis=-1, keepdims=True) + epsilon)
    
    def update_usage(self, attn_weights, ffn_out):
        """
        Update usage statistics for attention heads and FFN neurons.
        """
        head_usage_update = jnp.mean(attn_weights, axis=(0, 1))  # Average usage across batch and sequence
        ffn_usage_update = jnp.mean(jnp.abs(ffn_out), axis=(0, 1))  # Average activation magnitude across batch and sequence
        self.head_usage = hk.get_state("head_usage", [], dtype=jnp.float32, init=lambda x: self.head_usage) + head_usage_update
        self.ffn_usage = hk.get_state("ffn_usage", [], dtype=jnp.float32, init=lambda x: self.ffn_usage) + ffn_usage_update
        return self.head_usage, self.ffn_usage

    def prune_components(self, head_threshold=0.01, ffn_threshold=0.01):
        """
        Prune underutilized attention heads and FFN neurons, with weight interpolation for better performance.
        Returns a new model with pruned components or updates in-place.
        """
        active_heads = self.head_usage > head_threshold
        if jnp.sum(active_heads) < 1:
            raise ValueError("Cannot prune all attention heads; at least one must remain active.")
        
        active_ffn = self.ffn_usage > ffn_threshold
        if jnp.sum(active_ffn) < 1:
            raise ValueError("Cannot prune all FFN neurons; at least one must remain active.")

        new_num_heads = int(jnp.sum(active_heads))
        new_d_model = int(jnp.sum(active_ffn))
        if new_num_heads == self.mha.num_heads and new_d_model == self.d_model:
            return self

        # Create a new TransformerBlock with pruned components
        new_model = TransformerBlock(
            d_model=new_d_model,
            num_heads=new_num_heads,
            dropout_rate=self.dropout_rate,
            name=self.name
        )
        new_model.head_usage = hk.get_parameter("head_usage", [new_num_heads], dtype=jnp.float32, init=jnp.zeros)
        new_model.ffn_usage = hk.get_parameter("ffn_usage", [new_d_model], dtype=jnp.float32, init=jnp.zeros)

        # Update attention weights with interpolation
        active_head_indices = jnp.where(active_heads)[0]
        new_model.mha = hk.MultiHeadAttention(
            num_heads=new_num_heads,
            key_size=new_d_model // new_num_heads,
            model_size=new_d_model,
            w_init=hk.initializers.VarianceScaling(1.0),
        )
        new_qkv = jnp.zeros_like(new_model.mha.w_qkv)
        new_o = jnp.zeros_like(new_model.mha.w_o)
        for i, idx in enumerate(active_head_indices):
            new_qkv[:, i] = self.mha.w_qkv[:, idx]
            new_o[i] = self.mha.w_o[idx]
        new_model.mha.w_qkv = new_qkv
        new_model.mha.w_o = new_o

        # Update FFN weights with interpolation
        active_ffn_indices = jnp.where(active_ffn)[0]
        new_model.ffn = hk.Sequential([
            hk.Linear(new_d_model * 2, w_init=hk.initializers.VarianceScaling(1.0)),
            jax.nn.silu,
            hk.Linear(new_d_model, w_init=hk.initializers.VarianceScaling(1.0))
        ])
        new_ffn_in = jnp.zeros((self.d_model, new_d_model * 2))
        new_ffn_out = jnp.zeros((new_d_model, self.d_model))
        for i, idx in enumerate(active_ffn_indices):
            new_ffn_in[:, i] = self.ffn.layers[0].w[:, idx]
            new_ffn_out[i] = self.ffn.layers[2].w[idx]
        new_model.ffn.layers[0].w = new_ffn_in
        new_model.ffn.layers[2].w = new_ffn_out
        return new_model

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

    def prune_layers(self, threshold=0.01):
        """
        Prune underutilized components across all Transformer blocks.
        Returns a new model with pruned layers or updates in-place.
        """
        new_layers = []
        for layer in self.layers:
            new_layer = layer.prune_components(head_threshold=threshold, ffn_threshold=threshold)
            new_layers.append(new_layer)
        
        new_model = TransformerModel(
            d_model=new_layers[0].d_model,
            num_heads=new_layers[0].mha.num_heads,
            num_layers=len(new_layers),
            vocab_size=self.embedding.vocab_size,
            max_seq_length=self.position_enc.vocab_size,
            name=self.name
        )
        new_model.layers = new_layers
        return new_model

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