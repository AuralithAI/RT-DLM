import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict


def spectral_convergence(target: jnp.ndarray, predicted: jnp.ndarray) -> jnp.ndarray:
    num = jnp.linalg.norm(target - predicted, axis=(-2, -1))
    den = jnp.linalg.norm(target, axis=(-2, -1)) + 1e-6
    return jnp.mean(num / den)


def mel_l1_loss(target: jnp.ndarray, predicted: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(target - predicted))


class SpectrogramDecoder(hk.Module):
    """Mirrored audio decoder: hidden states -> mel spectrogram via upsampling Conv1Ds."""

    def __init__(
        self,
        d_model: int,
        n_mels: int = 128,
        upsample_factors=(2, 2, 2, 2),
        name=None,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_mels = n_mels
        self.upsample_factors = upsample_factors
        self.input_proj = hk.Linear(d_model, name="dec_input_proj")
        self.up_blocks = []
        ch = d_model
        for i, f in enumerate(upsample_factors):
            ch = max(ch // 2, n_mels)
            self.up_blocks.append((f, ch, i))
        self.out_proj = hk.Conv1D(n_mels, kernel_shape=3, padding="SAME", name="dec_out")

    def __call__(self, hidden: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """hidden [B,T,d_model] -> mel [B, T*prod(upsample), n_mels]."""
        x = self.input_proj(hidden)
        for f, ch, i in self.up_blocks:
            new_t = x.shape[1] * f
            x = jax.image.resize(x, (x.shape[0], new_t, x.shape[-1]), method="linear")
            x = hk.Conv1D(ch, kernel_shape=3, padding="SAME", name=f"up_conv_{i}")(x)
            x = jax.nn.silu(x)
        mel = self.out_proj(x)
        return {"mel": mel}
