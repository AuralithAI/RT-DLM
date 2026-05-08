"""Model merging utilities: SLERP, TIES, weighted-average."""

from typing import Any, Dict, List, Sequence

import jax.numpy as jnp


def _flatten(tree: Any) -> Dict[str, jnp.ndarray]:
    """Flatten a nested Haiku param tree into name -> array map."""
    out: Dict[str, jnp.ndarray] = {}

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}/{k}" if prefix else k)
        else:
            out[prefix] = node

    walk(tree, "")
    return out


def _unflatten_like(reference: Any, flat: Dict[str, jnp.ndarray]) -> Any:
    """Rebuild nested tree using `reference` structure with values from `flat`."""

    def build(node: Any, prefix: str) -> Any:
        if isinstance(node, dict):
            return {k: build(v, f"{prefix}/{k}" if prefix else k) for k, v in node.items()}
        return flat[prefix]

    return build(reference, "")


def weighted_average(params_list: Sequence[Any], weights: Sequence[float]) -> Any:
    """Linear weighted average of multiple parameter trees."""
    assert len(params_list) == len(weights) and len(params_list) > 0
    total = float(sum(weights))
    norm = [w / total for w in weights]
    flats = [_flatten(p) for p in params_list]
    keys = list(flats[0].keys())
    merged: Dict[str, Any] = {k: sum(norm[i] * flats[i][k] for i in range(len(flats))) for k in keys}
    return _unflatten_like(params_list[0], merged)


def slerp_arrays(a: jnp.ndarray, b: jnp.ndarray, t: float, eps: float = 1e-7) -> jnp.ndarray:
    """Spherical linear interpolation between two flat tensors."""
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    a_norm = jnp.linalg.norm(a_flat) + eps
    b_norm = jnp.linalg.norm(b_flat) + eps
    cos = jnp.clip(jnp.dot(a_flat, b_flat) / (a_norm * b_norm), -1.0, 1.0)
    omega = jnp.arccos(cos)
    sin_omega = jnp.sin(omega) + eps
    if float(jnp.abs(sin_omega)) < 1e-4:
        return ((1.0 - t) * a + t * b).reshape(a.shape)
    w_a = jnp.sin((1.0 - t) * omega) / sin_omega
    w_b = jnp.sin(t * omega) / sin_omega
    return (w_a * a_flat + w_b * b_flat).reshape(a.shape)


def slerp(params_a: Any, params_b: Any, t: float = 0.5) -> Any:
    """SLERP between two parameter trees; falls back to LERP for tiny vectors."""
    flat_a = _flatten(params_a)
    flat_b = _flatten(params_b)
    merged = {k: slerp_arrays(flat_a[k], flat_b[k], t) for k in flat_a}
    return _unflatten_like(params_a, merged)


def _trim_to_density(delta: jnp.ndarray, density: float) -> jnp.ndarray:
    """Keep top-`density` fraction of magnitudes; zero the rest."""
    if density >= 1.0:
        return delta
    flat = jnp.abs(delta).reshape(-1)
    k = max(1, int(flat.shape[0] * density))
    threshold = jnp.sort(flat)[-k]
    mask = (jnp.abs(delta) >= threshold).astype(delta.dtype)
    return delta * mask


def ties_merge(
    base_params: Any,
    candidate_params: Sequence[Any],
    weights: Sequence[float],
    density: float = 0.2,
) -> Any:
    """TIES merge: trim deltas, resolve sign conflicts by weighted majority, average."""
    assert len(candidate_params) == len(weights)
    flat_base = _flatten(base_params)
    flat_candidates = [_flatten(p) for p in candidate_params]
    merged: Dict[str, jnp.ndarray] = {}
    for k, base_v in flat_base.items():
        deltas = [flat_candidates[i][k] - base_v for i in range(len(flat_candidates))]
        trimmed = [_trim_to_density(d, density) for d in deltas]
        signs = jnp.sign(sum(weights[i] * trimmed[i] for i in range(len(trimmed))))
        accepted: List[jnp.ndarray] = []
        accepted_w: List[jnp.ndarray] = []
        for i, d in enumerate(trimmed):
            agree = (jnp.sign(d) == signs).astype(d.dtype)
            accepted.append(d * agree)
            accepted_w.append(agree * weights[i])
        denom = sum(accepted_w) + 1e-8
        delta = sum(accepted) / denom
        merged[k] = base_v + delta
    return _unflatten_like(base_params, merged)


def merge_checkpoints(
    base_params: Any,
    candidates: Dict[str, Any],
    weights: Dict[str, float],
    method: str = "ties",
    density: float = 0.2,
) -> Any:
    """High-level dispatcher for {weighted, slerp, ties} merging strategies."""
    names = list(candidates.keys())
    cand_list = [candidates[n] for n in names]
    w_list = [weights[n] for n in names]
    if method == "weighted":
        return weighted_average(cand_list, w_list)
    if method == "slerp":
        if len(cand_list) != 2:
            raise ValueError("slerp expects exactly 2 candidates")
        t = w_list[1] / (w_list[0] + w_list[1])
        return slerp(cand_list[0], cand_list[1], t)
    if method == "ties":
        return ties_merge(base_params, cand_list, w_list, density=density)
    raise ValueError(f"unknown merge method: {method}")
