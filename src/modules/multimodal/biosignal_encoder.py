import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict, Any


class _ChannelAttention(hk.Module):
    def __init__(self, d_model: int, num_heads: int = 4, name=None):
        super().__init__(name=name)
        self.attn = hk.MultiHeadAttention(
            num_heads=num_heads, key_size=d_model // num_heads,
            w_init=hk.initializers.TruncatedNormal(0.02), name="ch_attn",
        )
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.norm(x + self.attn(x, x, x))


class BiosignalEncoder(hk.Module):
    """EEG / biosignal encoder with channel + temporal attention.
    Input: [B, T, C_channels]. Output: [B, T_out, d_model].
    """

    def __init__(
        self,
        d_model: int,
        max_channels: int = 256,
        downsample: int = 4,
        num_heads: int = 4,
        name=None,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_channels = max_channels
        self.downsample = downsample
        self.channel_proj = hk.Linear(d_model, name="ch_proj")
        self.temporal_conv = hk.Conv1D(
            output_channels=d_model, kernel_shape=7, stride=downsample,
            padding="SAME", name="t_conv",
        )
        self.channel_attn = _ChannelAttention(d_model, num_heads=num_heads, name="ch_attn_block")
        self.temporal_attn = hk.MultiHeadAttention(
            num_heads=num_heads, key_size=d_model // num_heads,
            w_init=hk.initializers.TruncatedNormal(0.02), name="t_attn",
        )
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, signals: jnp.ndarray) -> Dict[str, Any]:
        _, t, c = signals.shape
        per_step = self.channel_proj(signals)
        per_step = self.channel_attn(per_step)
        x = self.temporal_conv(per_step)
        x = self.norm(x + self.temporal_attn(x, x, x))
        return {"features": x, "channels": c, "time": t}
