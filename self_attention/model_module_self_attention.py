import haiku as hk
import jax
import os
import sys
import jax.numpy as jnp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_config import TrainConfig

config = TrainConfig()
class SelfAttentionModel(hk.Module):
    def __init__(self, d_model: int, num_heads: int, vocab_size: int, max_seq_length: int, name=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
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

    def __call__(self, inputs):
        mask = (inputs != 0).astype(jnp.float32)[:, None, None, :]
        x = self.embedding(inputs) * jnp.sqrt(self.d_model)
        x = self.norm1(x)
        attn_out = self.attention(query=x, key=x, value=x, mask=mask)
        x = x + attn_out 
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out  
        logits = self.proj(x)
        return logits

def forward_fn(inputs):
    model = SelfAttentionModel(d_model=config.d_model, num_heads=config.num_heads, vocab_size=config.vocab_size, max_seq_length=config.max_seq_length)
    return model(inputs)

model = hk.transform(forward_fn)
