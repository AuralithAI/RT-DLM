import haiku as hk
import jax.numpy as jnp
from typing import Dict


class ActionEncoder(hk.Module):
    """Robot action encoder: discretizes joint angles + gripper + ee pose, embeds as tokens.

    Default schema (per-step): 7 joint angles + 1 gripper + 7 ee pose (xyz+quat) = 15 dims.
    Configurable via num_axes. Returns sequence representation per timestep.
    """

    def __init__(
        self,
        d_model: int,
        num_axes: int = 15,
        num_bins: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        name=None,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_axes = num_axes
        self.num_bins = num_bins
        self.num_heads = num_heads
        self.embed = hk.Embed(num_bins, d_model, name="action_embed")
        self.fuse = hk.Linear(d_model, name="axis_fuse")
        self.layers = []
        self.norms = []
        for i in range(num_layers):
            self.layers.append(
                hk.MultiHeadAttention(
                    num_heads=num_heads,
                    key_size=d_model // num_heads,
                    w_init=hk.initializers.TruncatedNormal(0.02),
                    name=f"action_attn_{i}",
                )
            )
            self.norms.append(hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"action_ln_{i}"))
        self.predictor = hk.Linear(num_axes * num_bins, name="action_pred")

    def _discretize(self, actions: jnp.ndarray) -> jnp.ndarray:
        """actions [B,T,A] in [-1,1] (or normalized) -> int bins."""
        norm = jnp.clip((actions + 1.0) / 2.0, 0.0, 1.0)
        return jnp.clip((norm * self.num_bins).astype(jnp.int32), 0, self.num_bins - 1)

    def __call__(self, actions: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """actions [B,T,num_axes] continuous in normalized [-1,1]."""
        b, t, a = actions.shape
        assert a == self.num_axes, f"expected {self.num_axes} axes, got {a}"
        bins = self._discretize(actions)
        embedded = self.embed(bins)
        axis_pos = hk.get_parameter(
            "axis_pos", [self.num_axes, self.d_model], init=hk.initializers.TruncatedNormal(0.02)
        )
        embedded = embedded + axis_pos[None, None, :, :]
        merged = self.fuse(embedded.mean(axis=2))
        for layer, norm in zip(self.layers, self.norms):
            merged = norm(merged + layer(merged, merged, merged))
        logits = self.predictor(merged).reshape(b, t, self.num_axes, self.num_bins)
        return {"features": merged, "action_logits": logits, "action_bins": bins}
