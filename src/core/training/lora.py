"""LoRA low-rank weight-delta adapters for parameter-efficient fine-tuning."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp


@dataclass
class LoRAConfig:
    """Hyperparameters for a LoRA adapter bank."""

    rank: int = 64
    alpha: float = 128.0
    dropout: float = 0.0
    target_modules: Sequence[str] = field(
        default_factory=lambda: ("query", "key", "value", "output", "ffn_up", "ffn_down")
    )
    init_scale: float = 0.02


class LoRALinear(hk.Module):
    """Linear layer with optional frozen base weight + trainable low-rank delta."""

    def __init__(
        self,
        out_features: int,
        rank: int,
        alpha: float = 128.0,
        dropout: float = 0.0,
        use_bias: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.use_bias = use_bias
        self.scaling = alpha / max(rank, 1)

    def __call__(self, x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """Apply base linear plus scaled low-rank update."""
        in_features = x.shape[-1]
        w = hk.get_parameter(
            "w",
            shape=[in_features, self.out_features],
            init=hk.initializers.TruncatedNormal(0.02),
        )
        base = jnp.dot(x, w)
        if self.use_bias:
            b = hk.get_parameter("b", shape=[self.out_features], init=jnp.zeros)
            base = base + b

        a = hk.get_parameter(
            "lora_A",
            shape=[in_features, self.rank],
            init=hk.initializers.TruncatedNormal(0.02),
        )
        b_lora = hk.get_parameter("lora_B", shape=[self.rank, self.out_features], init=jnp.zeros)
        delta = x
        if is_training and self.dropout > 0.0:
            delta = hk.dropout(hk.next_rng_key(), self.dropout, delta)
        delta = jnp.dot(jnp.dot(delta, a), b_lora) * self.scaling
        return base + delta


class LoRAAdapter(hk.Module):
    """Standalone low-rank adapter (no base linear) — overlay on a frozen layer."""

    def __init__(self, in_features: int, out_features: int, config: LoRAConfig, name: Optional[str] = None):
        super().__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.scaling = config.alpha / max(config.rank, 1)

    def __call__(self, x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """Return low-rank delta only; caller adds to frozen output."""
        a = hk.get_parameter(
            "lora_A",
            shape=[self.in_features, self.config.rank],
            init=hk.initializers.TruncatedNormal(self.config.init_scale),
        )
        b = hk.get_parameter(
            "lora_B",
            shape=[self.config.rank, self.out_features],
            init=jnp.zeros,
        )
        h = x
        if is_training and self.config.dropout > 0.0:
            h = hk.dropout(hk.next_rng_key(), self.config.dropout, h)
        return jnp.dot(jnp.dot(h, a), b) * self.scaling


def is_lora_param(path: str) -> bool:
    """Return True for parameter paths that belong to LoRA adapters."""
    return "lora_A" in path or "lora_B" in path


def _flatten_params(params: Any) -> Dict[str, jnp.ndarray]:
    """Flatten nested param tree into path -> array map."""
    out: Dict[str, jnp.ndarray] = {}

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}/{k}" if prefix else k)
        else:
            out[prefix] = node

    walk(params, "")
    return out


def _unflatten_params(flat: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
    """Inverse of `_flatten_params` using '/' as the path separator."""
    out: Dict[str, Any] = {}
    for path, value in flat.items():
        parts = path.split("/")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def split_lora_params(params: Any) -> Tuple[Any, Any]:
    """Partition params into (frozen_base, trainable_lora) trees."""
    flat = _flatten_params(params)
    base_flat = {p: v for p, v in flat.items() if not is_lora_param(p)}
    lora_flat = {p: v for p, v in flat.items() if is_lora_param(p)}
    return _unflatten_params(base_flat), _unflatten_params(lora_flat)


def freeze_non_lora(params: Any) -> Any:
    """Return a tree of stop_gradient values for non-LoRA params."""

    def walk(node: Any, prefix: str) -> Any:
        if isinstance(node, dict):
            return {k: walk(v, f"{prefix}/{k}" if prefix else k) for k, v in node.items()}
        return node if is_lora_param(prefix) else jax.lax.stop_gradient(node)

    return walk(params, "")


def lora_param_filter() -> Callable[[str, str, jnp.ndarray], bool]:
    """Return a Haiku-compatible filter selecting only LoRA tensors."""

    def predicate(module_name: str, name: str, value: jnp.ndarray) -> bool:
        _ = module_name, value
        return name.startswith("lora_")

    return predicate


def merge_lora_into_base(params: Any) -> Any:
    """Fold lora_A @ lora_B * scaling into the base `w` parameter (in-place merge)."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            keys = set(node.keys())
            if {"w", "lora_A", "lora_B"}.issubset(keys):
                merged = dict(node)
                a = node["lora_A"]
                b = node["lora_B"]
                rank = a.shape[-1]
                scaling = node.get("_scaling", 128.0 / max(rank, 1))
                merged["w"] = node["w"] + jnp.dot(a, b) * scaling
                merged.pop("lora_A", None)
                merged.pop("lora_B", None)
                merged.pop("_scaling", None)
                return merged
            return {k: walk(v) for k, v in node.items()}
        return node

    return walk(params)


def count_trainable_lora(params: Any) -> int:
    """Count parameters in LoRA tensors only."""
    flat: List[int] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}/{k}" if prefix else k)
        else:
            if is_lora_param(prefix):
                flat.append(int(node.size))

    walk(params, "")
    return sum(flat)
