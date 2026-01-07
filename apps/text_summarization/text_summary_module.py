import haiku as hk
import jax.numpy as jnp
import jax

"""
    GPU or CPU device selection. (Based on CUDA availability.)
"""
if jax.devices("gpu"):
    print(f"Using GPU: {jax.devices('gpu')[0]}")
    device = jax.devices("gpu")[0]
else:
    print("No GPU found. Falling back to CPU.")
    device = jax.devices("cpu")[0]

def to_device(x):
    return jax.device_put(jnp.asarray(x, dtype=jnp.float16), device)

class TransformerBlock(hk.Module):
    """
    Transformer block consisting of multi-head attention, layer normalization, and a feed-forward network.
    """
    def __init__(self, d_model: int, num_heads: int, name=None):
        super().__init__(name=name)
        self.attention = hk.MultiHeadAttention(
            num_heads=num_heads, key_size=d_model // num_heads, w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        )
        self.norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.ffn = hk.nets.MLP([d_model * 4, d_model])
    
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the Transformer block.
        - Applies multi-head attention.
        - Adds residual connection and normalizes.
        - Applies a feed-forward network.
        - Adds residual connection and normalizes again.
        """
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return to_device(self.norm2(x + ffn_out))

class TextSummarizationModel(hk.Module):
    """
    Transformer-based text summarization model.
    - Uses an embedding layer for token representation.
    - Adds positional encoding.
    - Passes input through multiple Transformer blocks.
    - Outputs a probability distribution over the vocabulary.
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_length: int):
        super().__init__()
        self.embedding = hk.Embed(vocab_size, d_model)  # Token embeddings
        self.positional_encoding = hk.Embed(max_seq_length, d_model)  # Positional encoding
        self.transformer_blocks = [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]  # Stacked Transformer layers
        self.output_layer = hk.Linear(vocab_size)  # Final output layer
    
    def __call__(self, inputs: jnp.ndarray):
        """
        Forward pass of the text summarization model.
        - Embeds input tokens.
        - Adds positional encoding.
        - Passes input through Transformer blocks.
        - Generates logits for vocabulary tokens.
        """
        seq_length = inputs.shape[1]
        token_embeddings = self.embedding(inputs)
        position_embeddings = self.positional_encoding(jnp.arange(seq_length)[None, :])
        x = token_embeddings + position_embeddings
        
        for block in self.transformer_blocks:
            x = block(x)
        
        logits = self.output_layer(x)
        return to_device(jax.nn.softmax(logits, axis=-1))