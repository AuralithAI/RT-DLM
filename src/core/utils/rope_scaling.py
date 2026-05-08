"""RoPE positional scaling utilities: NTK-aware scaling and YaRN."""

import jax.numpy as jnp


def _default_inv_freq(dim: int, base: float = 10000.0) -> jnp.ndarray:
    """Standard RoPE inverse frequencies for half of `dim`."""
    half = dim // 2
    return base ** (-jnp.arange(0, half, dtype=jnp.float32) / half)


def ntk_aware_scale(
    dim: int,
    base: float = 10000.0,
    scale_factor: float = 1.0,
) -> jnp.ndarray:
    """NTK-aware base scaling: shifts the RoPE base to absorb a longer context."""
    if scale_factor <= 1.0:
        return _default_inv_freq(dim, base)
    new_base = base * (scale_factor ** (dim / max(dim - 2, 1)))
    return _default_inv_freq(dim, new_base)


def _ramp_mask(half_dim: int, low: float, high: float) -> jnp.ndarray:
    """Smooth blend mask in [0, 1] over rotary index range [low, high]."""
    idx = jnp.arange(half_dim, dtype=jnp.float32)
    raw = (idx - low) / max(high - low, 1e-3)
    return jnp.clip(raw, 0.0, 1.0)


def yarn_scale_factors(
    dim: int,
    base: float = 10000.0,
    original_length: int = 4096,
    new_length: int = 131072,
    alpha: float = 1.0,
    beta: float = 32.0,
) -> jnp.ndarray:
    """YaRN inverse-frequency vector blending PI and NTK-aware extrapolation."""
    half = dim // 2
    if new_length <= original_length:
        return _default_inv_freq(dim, base)

    scale = float(new_length) / float(original_length)
    inv_freq_extra = _default_inv_freq(dim, base)
    inv_freq_inter = inv_freq_extra / scale

    wavelens = 2.0 * jnp.pi / inv_freq_extra
    low_idx = float(half) * jnp.log(original_length / (beta * 2.0 * jnp.pi)) / jnp.log(base)
    high_idx = float(half) * jnp.log(original_length / (alpha * 2.0 * jnp.pi)) / jnp.log(base)
    low = float(jnp.maximum(low_idx, 0.0))
    high = float(jnp.minimum(high_idx, half - 1))
    mask = _ramp_mask(half, low, high)
    inv_freq = inv_freq_inter * (1.0 - mask) + inv_freq_extra * mask

    _ = wavelens  # silence unused-warning; retained for potential debug
    return inv_freq


def yarn_attention_scale(scale_factor: float, mscale: float = 0.1) -> float:
    """Logit temperature suggested by YaRN to compensate post-extension softmax."""
    if scale_factor <= 1.0:
        return 1.0
    return float(1.0 + mscale * jnp.log(scale_factor))


def apply_rope_with_inv_freq(
    x: jnp.ndarray, positions: jnp.ndarray, inv_freq: jnp.ndarray
) -> jnp.ndarray:
    """Apply RoPE to `x` of shape [..., seq, dim] using precomputed inverse frequencies."""
    *_, seq, dim = x.shape
    half = dim // 2
    pos = positions.astype(jnp.float32)
    angles = pos[..., None] * inv_freq[None, :half]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    x1, x2 = x[..., :half], x[..., half : 2 * half]
    rotated = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    if dim > 2 * half:
        rotated = jnp.concatenate([rotated, x[..., 2 * half :]], axis=-1)
    _ = seq
    return rotated
