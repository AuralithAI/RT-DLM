import haiku as hk
import jax
import jax.numpy as jnp
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import shared spiking/pruning utilities from reusable components
from core.components.reusable_components import SpikingMechanism, PruningManager


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
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn = hk.Sequential([
            hk.Linear(d_model * 2, w_init=hk.initializers.VarianceScaling(1.0)),
            jax.nn.silu,
            hk.Linear(d_model, w_init=hk.initializers.VarianceScaling(1.0))
        ])
        self.dropout_rate = dropout_rate
        self.head_usage = hk.get_state("head_usage", [num_heads], dtype=jnp.float32, init=jnp.zeros)
        self.ffn_usage = hk.get_state("ffn_usage", [d_model], dtype=jnp.float32, init=jnp.zeros)
        
        # Shared spiking mechanism from reusable_components
        self._spiking = SpikingMechanism(spike_threshold=0.1, epsilon=1e-8)
        
        # Shared pruning manager for usage tracking
        self._pruning_manager = PruningManager(
            num_heads=num_heads,
            d_model=d_model,
            head_threshold=0.01,
            ffn_threshold=0.01
        )

    def apply_spiking_attention(self, scores, spike_threshold, epsilon):
        """
        Apply Spiking Attention by thresholding attention scores.
        
        Delegates to shared SpikingMechanism from core.components.reusable_components.
        """
        # Validate parameters (maintains backward compatibility)
        if spike_threshold is None or epsilon is None or not 0 <= spike_threshold <= 1:
            return scores
        
        # Update shared mechanism with current parameters
        self._spiking.spike_threshold = spike_threshold
        self._spiking.epsilon = epsilon
        
        return self._spiking.apply(scores)
    
    def update_usage(self, attn_weights, ffn_out):
        """
        Update usage statistics for attention heads and FFN neurons.
        
        Delegates usage computation to shared PruningManager from 
        core.components.reusable_components for consistency.
        """
        # Use shared PruningManager for head usage computation
        head_usage_update = self._pruning_manager.compute_head_usage(attn_weights)
        
        # Use shared PruningManager for FFN usage computation
        ffn_usage_update = self._pruning_manager.compute_ffn_usage(ffn_out)

        # Accumulate to Haiku state
        current_head_usage = hk.get_state("head_usage", [self.num_heads], dtype=jnp.float32, init=jnp.zeros)
        current_ffn_usage = hk.get_state("ffn_usage", [self.d_model], dtype=jnp.float32, init=jnp.zeros)
        hk.set_state("head_usage", current_head_usage + head_usage_update)
        hk.set_state("ffn_usage", current_ffn_usage + ffn_usage_update)

    def prune_components(self, head_threshold=0.01, ffn_threshold=0.01):
        """
        Prune underutilized attention heads and FFN neurons, with weight interpolation for better performance.
        Returns a new model with pruned components or updates in-place.
        
        Uses shared PruningManager from core.components.reusable_components
        for mask generation while keeping model-specific instantiation.
        """
        # Get usage statistics from Haiku state
        usage_heads = hk.get_state("head_usage", [self.num_heads], dtype=jnp.float32, init=jnp.zeros)
        usage_ffn = hk.get_state("ffn_usage", [self.d_model], dtype=jnp.float32, init=jnp.zeros)
        
        # Update pruning manager thresholds and get masks
        self._pruning_manager.head_threshold = head_threshold
        self._pruning_manager.ffn_threshold = ffn_threshold
        active_heads, active_ffn = self._pruning_manager.get_pruning_mask(usage_heads, usage_ffn)

        new_num_heads = int(jnp.sum(active_heads))
        new_d_model = int(jnp.sum(active_ffn))
        if new_num_heads == self.num_heads and new_d_model == self.d_model:
            return self

        new_model = TransformerBlock(
            d_model=new_d_model,
            num_heads=new_num_heads,
            dropout_rate=self.dropout_rate,
            name=self.name
        )
        active_head_indices = jnp.where(active_heads)[0]
        active_ffn_indices = jnp.where(active_ffn)[0]

        new_qkv = jnp.take(self.mha.w_qkv, active_head_indices, axis=1)
        new_o = jnp.take(self.mha.w_o, active_head_indices, axis=0)
        new_model.mha.w_qkv = new_qkv
        new_model.mha.w_o = new_o

        intermediate_size = new_d_model * 2
        new_ffn = hk.Sequential([
            hk.Linear(intermediate_size, w_init=hk.initializers.VarianceScaling(1.0)),
            jax.nn.silu,
            hk.Linear(new_d_model, w_init=hk.initializers.VarianceScaling(1.0))
        ])
        new_ffn_in = jnp.take(self.ffn.layers[0].w, active_ffn_indices, axis=0)[:, :intermediate_size]
        new_ffn_out = jnp.take(self.ffn.layers[2].w, active_ffn_indices, axis=0)
        new_model.ffn.layers[0].w = new_ffn_in
        new_model.ffn.layers[2].w = new_ffn_out
        return new_model

    def __call__(self, x, rng=None, return_attention=False, spike_threshold=0.1, epsilon=1e-8):
        attn_out = self.mha(query=x, key=x, value=x)
        spiked_attentions = self.apply_spiking_attention(attn_out, spike_threshold, epsilon)
        x = self.norm1(x + spiked_attentions)
        ffn_out = self.ffn(x)
        if rng is not None:
            ffn_out = hk.dropout(rng, self.dropout_rate, ffn_out)
        x = self.norm2(x + ffn_out)
        if return_attention:
            attention_weights = jax.nn.softmax(attn_out, axis=-1)
            self.update_usage(attention_weights, ffn_out)
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
        new_layers = [layer.prune_components(head_threshold=threshold, ffn_threshold=threshold) for layer in self.layers]
        new_model = TransformerModel(
            d_model=new_layers[0].d_model,
            num_heads=new_layers[0].num_heads,
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