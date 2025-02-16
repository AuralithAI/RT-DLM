import haiku as hk
import jax
import jax.numpy as jnp

# Self-Attention Model
class SelfAttentionModel(hk.Module):
    def __init__(self, d_model: int, num_heads: int, vocab_size: int, max_seq_length: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.embedding = hk.Embed(vocab_size=vocab_size, embed_dim=d_model)
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
            jax.nn.relu,
            hk.Linear(d_model),
        ])
        self.proj = hk.Linear(vocab_size)

    def __call__(self, inputs, return_attention=False):
        inputs = jnp.asarray(inputs, dtype=jnp.int32)  # Ensure integer inputs
        x = self.embedding(inputs) * jnp.sqrt(self.d_model)
        x = self.norm1(x)
        attn_out = self.attention(query=x, key=x, value=x)
        attention_weights = jax.nn.softmax(attn_out, axis=-1)
        x = x + attn_out
        x = self.norm2(x)
        x = x + self.ffn(x)
        logits = self.proj(x)

        if return_attention:
            return logits, attention_weights
        return logits

# Transformer Block
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
            jax.nn.relu,
            hk.Linear(d_model)
        ])
        self.dropout_rate = dropout_rate

    def __call__(self, x, rng=None, return_attention=False):
        attn_out = self.mha(query=x, key=x, value=x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        if rng is not None:
            ffn_out = hk.dropout(rng, self.dropout_rate, ffn_out)
        
        x = self.norm2(x + ffn_out)

        if return_attention:
            return x, attn_out  
        return x

# Transformer Model
class TransformerModel(hk.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, vocab_size: int, max_seq_length: int, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.embedding = hk.Embed(vocab_size, d_model, lookup_style="index")
        self.position_enc = hk.Embed(max_seq_length, d_model)
        self.layers = [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.proj = hk.Linear(vocab_size)

    def __call__(self, inputs, rng=None, return_attention=False):
        inputs = jnp.asarray(inputs, dtype=jnp.int32) 
        embed_out = self.embedding(inputs) 
        
        pos_enc = self.position_enc(jnp.arange(inputs.shape[1], dtype=jnp.int32))
        pos_enc = jnp.expand_dims(pos_enc, axis=0)  
        pos_enc = jnp.broadcast_to(pos_enc, (inputs.shape[0], inputs.shape[1], self.d_model))  
        print(f"[INFO] - Transformer Model ==> Embedding Shape: {embed_out.shape} | Positional Encoding Shape: {pos_enc.shape}")
        x = embed_out + pos_enc

        attention_maps = []
        for layer in self.layers:
            layer_rng, rng = jax.random.split(rng) if rng is not None else (None, None)
            if return_attention:
                x, attn_weights = layer(x, rng=layer_rng, return_attention=True)
                attention_maps.append(attn_weights)
            else:
                x = layer(x, rng=layer_rng)

        x = self.norm(x)
        logits = self.proj(x)

        if return_attention:
            return logits, jnp.array(attention_maps)
        return logits

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

        # Apply Transformer Model
        logits, attn_weights_transformer = self.transformer(x, rng, return_attention=True)

        if return_attention:
            return logits, jnp.array([attn_weights_self, attn_weights_transformer])
        return logits