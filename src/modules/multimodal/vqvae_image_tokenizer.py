import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict


class _VectorQuantizer(hk.Module):
    """VQ layer with EMA-free straight-through gradient (training-friendly)."""

    def __init__(self, num_codes: int, code_dim: int, commitment_cost: float = 0.25, name=None):
        super().__init__(name=name)
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost

    def __call__(self, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        codebook = hk.get_parameter(
            "codebook",
            [self.num_codes, self.code_dim],
            init=hk.initializers.TruncatedNormal(0.02),
        )
        flat = z.reshape(-1, self.code_dim)
        d = jnp.sum(flat**2, axis=-1, keepdims=True) - 2.0 * flat @ codebook.T + jnp.sum(codebook**2, axis=-1)[None, :]
        idx = jnp.argmin(d, axis=-1)
        quantized = codebook[idx].reshape(z.shape)
        codebook_loss = jnp.mean((jax.lax.stop_gradient(z) - quantized) ** 2)
        commit_loss = jnp.mean((z - jax.lax.stop_gradient(quantized)) ** 2)
        loss = codebook_loss + self.commitment_cost * commit_loss
        quantized_st = z + jax.lax.stop_gradient(quantized - z)
        return {
            "quantized": quantized_st,
            "indices": idx.reshape(z.shape[:-1]),
            "loss": loss,
        }


class VQVAEImageTokenizer(hk.Module):
    """Discrete image tokenizer (8192-code default, 16x spatial compression)."""

    def __init__(
        self,
        num_codes: int = 8192,
        code_dim: int = 256,
        downsample_factor: int = 16,
        name=None,
    ):
        super().__init__(name=name)
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.downsample_factor = downsample_factor
        self.encoder = hk.Sequential(
            [
                hk.Conv2D(64, kernel_shape=4, stride=2, padding="SAME"),
                jax.nn.silu,
                hk.Conv2D(128, kernel_shape=4, stride=2, padding="SAME"),
                jax.nn.silu,
                hk.Conv2D(256, kernel_shape=4, stride=2, padding="SAME"),
                jax.nn.silu,
                hk.Conv2D(code_dim, kernel_shape=4, stride=2, padding="SAME"),
            ],
            name="vq_encoder",
        )
        self.quantizer = _VectorQuantizer(num_codes, code_dim, name="quantizer")
        self.decoder = hk.Sequential(
            [
                hk.Conv2DTranspose(256, kernel_shape=4, stride=2, padding="SAME"),
                jax.nn.silu,
                hk.Conv2DTranspose(128, kernel_shape=4, stride=2, padding="SAME"),
                jax.nn.silu,
                hk.Conv2DTranspose(64, kernel_shape=4, stride=2, padding="SAME"),
                jax.nn.silu,
                hk.Conv2DTranspose(3, kernel_shape=4, stride=2, padding="SAME"),
            ],
            name="vq_decoder",
        )

    def encode(self, images: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        z = self.encoder(images)
        out = self.quantizer(z)
        out["latent"] = z
        return out

    def decode(self, quantized: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(quantized)

    def __call__(self, images: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        enc = self.encode(images)
        recon = self.decode(enc["quantized"])
        recon_loss = jnp.mean((recon - images) ** 2)
        return {
            "reconstruction": recon,
            "indices": enc["indices"],
            "vq_loss": enc["loss"],
            "recon_loss": recon_loss,
            "total_loss": recon_loss + enc["loss"],
        }
