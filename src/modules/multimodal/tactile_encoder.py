import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict


class TactileEncoder(hk.Module):
    """Tactile array encoder. Input [B, T, S_sensors]. Output [B, d_model] global state."""

    def __init__(
        self,
        d_model: int,
        max_sensors: int = 1024,
        group_size: int = 16,
        name=None,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_sensors = max_sensors
        self.group_size = group_size
        self.sensor_proj = hk.Linear(d_model, name="sensor_proj")
        self.group_mlp = hk.Sequential([hk.Linear(d_model), jax.nn.silu, hk.Linear(d_model)], name="group_mlp")
        self.temporal_conv = hk.Conv1D(output_channels=d_model, kernel_shape=5, padding="SAME", name="t_conv")
        self.norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, signals: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        b, t, s = signals.shape
        gs = max(1, min(self.group_size, s))
        pad = (gs - s % gs) % gs
        if pad:
            signals = jnp.concatenate([signals, jnp.zeros((b, t, pad))], axis=-1)
            s = s + pad
        groups = signals.reshape(b, t, s // gs, gs)
        per_group = self.sensor_proj(groups)
        per_group = self.group_mlp(per_group)
        spatial = jnp.max(per_group, axis=2)
        temporal = self.temporal_conv(spatial)
        temporal = self.norm(temporal)
        global_repr = jnp.mean(temporal, axis=1)
        return {"global": global_repr, "temporal": temporal}
