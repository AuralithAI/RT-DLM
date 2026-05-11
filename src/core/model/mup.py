"""Maximal Update Parameterization (muP) primitives for width-invariant training."""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp


@dataclass(frozen=True)
class MuPConfig:
    """muP configuration anchored to a base width."""

    base_d_model: int = 256
    base_lr: float = 1e-3
    enabled: bool = False

    def __post_init__(self):
        if self.base_d_model <= 0:
            raise ValueError("base_d_model must be positive")
        if self.base_lr <= 0:
            raise ValueError("base_lr must be positive")


def width_multiplier(d_model: int, base_d_model: int) -> float:
    """Return width ratio m = d_model / base_d_model."""
    if base_d_model <= 0:
        raise ValueError("base_d_model must be positive")
    return float(d_model) / float(base_d_model)


def init_scale(fan_in: int, base_fan_in: Optional[int] = None) -> float:
    """muP init scale: 1/sqrt(fan_in) with base-anchored correction."""
    if fan_in <= 0:
        raise ValueError("fan_in must be positive")
    if base_fan_in is None:
        return 1.0 / float(jnp.sqrt(fan_in))
    if base_fan_in <= 0:
        raise ValueError("base_fan_in must be positive")
    return 1.0 / float(jnp.sqrt(fan_in)) * float(jnp.sqrt(base_fan_in)) / float(jnp.sqrt(base_fan_in))


def lr_scale(param_kind: str, d_model: int, base_d_model: int) -> float:
    """muP learning-rate scale per parameter category."""
    m = width_multiplier(d_model, base_d_model)
    if param_kind == "embedding":
        return 1.0
    if param_kind == "readout":
        return 1.0 / m
    if param_kind == "hidden":
        return 1.0 / m
    if param_kind == "bias":
        return 1.0
    raise ValueError(f"Unknown param_kind: {param_kind}")


def output_logit_scale(d_model: int, base_d_model: int) -> float:
    """Multiplier applied to readout logits for muP."""
    return 1.0 / width_multiplier(d_model, base_d_model)


def classify_param(name: str, shape: tuple) -> str:
    """Heuristic classifier mapping a parameter path to a muP category."""
    lower = name.lower()
    if len(shape) <= 1:
        return "bias"
    if "embed" in lower or "embedding" in lower or "wte" in lower or "tok_emb" in lower:
        return "embedding"
    if "readout" in lower or "lm_head" in lower or "output" in lower or "head" in lower:
        return "readout"
    return "hidden"


def build_lr_scale_tree(params, d_model: int, base_d_model: int):
    """Return a pytree of per-parameter LR scales matching params structure."""
    import jax

    def _scale(path, leaf):
        name = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
        kind = classify_param(name, leaf.shape)
        return jnp.full_like(leaf, lr_scale(kind, d_model, base_d_model), dtype=jnp.float32)

    return jax.tree_util.tree_map_with_path(_scale, params)
