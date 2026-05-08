import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict


def _knn_indices(points: jnp.ndarray, k: int) -> jnp.ndarray:
    """points [B,N,3] -> neighbor indices [B,N,k]."""
    sq = jnp.sum(points ** 2, axis=-1, keepdims=True)
    pairwise = sq + jnp.swapaxes(sq, -1, -2) - 2.0 * jnp.einsum("bnd,bmd->bnm", points, points)
    pairwise = -pairwise
    return jax.lax.top_k(pairwise, k)[1]


def _gather_neighbors(features: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """features [B,N,F], idx [B,N,k] -> [B,N,k,F]."""
    b = idx.shape[0]
    batch_idx = jnp.arange(b)[:, None, None]
    return features[batch_idx, idx]


class _SetAbstraction(hk.Module):
    """PointNet++ set abstraction: sample, group via kNN, MLP, max-pool."""

    def __init__(self, num_samples: int, k: int, hidden: int, out_dim: int, name=None):
        super().__init__(name=name)
        self.num_samples = num_samples
        self.k = k
        self.mlp = hk.Sequential(
            [hk.Linear(hidden), jax.nn.silu, hk.Linear(hidden), jax.nn.silu, hk.Linear(out_dim)],
            name="sa_mlp",
        )

    def __call__(self, points: jnp.ndarray, features: jnp.ndarray):
        n = points.shape[1]
        m = min(self.num_samples, n)
        stride = max(n // m, 1)
        idx = jnp.arange(m) * stride
        sampled_points = points[:, idx, :]
        knn_idx = _knn_indices(sampled_points, min(self.k, n))
        nbr_points = _gather_neighbors(points, knn_idx)
        nbr_feats = _gather_neighbors(features, knn_idx)
        rel = nbr_points - sampled_points[:, :, None, :]
        grouped = jnp.concatenate([rel, nbr_feats], axis=-1)
        encoded = self.mlp(grouped)
        pooled = jnp.max(encoded, axis=2)
        return sampled_points, pooled


class PointCloudEncoder(hk.Module):
    """PointNet++-style encoder: progressive abstraction over point clouds."""

    def __init__(self, d_model: int, k: int = 16, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.k = k
        self.sa1 = _SetAbstraction(num_samples=512, k=k, hidden=128, out_dim=128, name="sa1")
        self.sa2 = _SetAbstraction(num_samples=128, k=k, hidden=256, out_dim=256, name="sa2")
        self.sa3 = _SetAbstraction(num_samples=32, k=k, hidden=d_model, out_dim=d_model, name="sa3")
        self.global_proj = hk.Linear(d_model, name="global_proj")
        self.local_proj = hk.Linear(d_model, name="local_proj")

    def __call__(self, points: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """points [B,N,3] (or [B,N,3+F]). Returns global + local representations."""
        coords = points[..., :3]
        feats = points if points.shape[-1] > 3 else jnp.zeros_like(coords)
        p, f = self.sa1(coords, feats)
        p, f = self.sa2(p, f)
        p, f = self.sa3(p, f)
        local = self.local_proj(f)
        glob = self.global_proj(jnp.max(f, axis=1))
        return {"global": glob, "local": local, "centers": p}
