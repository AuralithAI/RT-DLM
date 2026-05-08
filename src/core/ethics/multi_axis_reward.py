"""Multi-axis preference reward model with Bradley-Terry training and calibration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class RewardAxis(str, Enum):
    """Six independent reward axes scored per response."""

    HELPFUL = "helpfulness"
    HARMLESS = "harmlessness"
    HONEST = "honesty"
    FACTUAL = "factuality"
    CODE = "code_correctness"
    MATH = "math_correctness"


DEFAULT_AXIS_WEIGHTS: Dict[str, float] = {
    RewardAxis.HELPFUL.value: 0.30,
    RewardAxis.HARMLESS.value: 0.20,
    RewardAxis.HONEST.value: 0.15,
    RewardAxis.FACTUAL.value: 0.15,
    RewardAxis.CODE.value: 0.10,
    RewardAxis.MATH.value: 0.10,
}


@dataclass
class RewardModelConfig:
    """Reward model architecture + training hyperparameters."""

    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 8
    axes: Tuple[str, ...] = tuple(a.value for a in RewardAxis)
    axis_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_AXIS_WEIGHTS))
    dropout: float = 0.1


class MultiAxisRewardModel(hk.Module):
    """Six-axis Bradley-Terry preference scorer."""

    def __init__(self, config: RewardModelConfig, name: Optional[str] = None):
        super().__init__(name=name)
        self.config = config
        self.attn_layers = [
            hk.MultiHeadAttention(
                num_heads=config.num_heads,
                key_size=config.d_model // config.num_heads,
                w_init=hk.initializers.TruncatedNormal(0.02),
                name=f"rm_attn_{i}",
            )
            for i in range(config.num_layers)
        ]
        self.norm_layers = [
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"rm_ln_{i}")
            for i in range(config.num_layers)
        ]
        self.heads = {axis: hk.Linear(1, name=f"head_{axis}") for axis in config.axes}

    def __call__(self, hidden: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
        """Produce a scalar reward per axis from a hidden-state sequence."""
        x = hidden
        for attn, norm in zip(self.attn_layers, self.norm_layers):
            x = norm(x + attn(x, x, x))
        if attention_mask is not None:
            mask = attention_mask[..., None]
            pooled = (x * mask).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1.0)
        else:
            pooled = x.mean(axis=1)
        return {axis: head(pooled).squeeze(-1) for axis, head in self.heads.items()}


def aggregate_reward(axis_rewards: Dict[str, jnp.ndarray], weights: Optional[Dict[str, float]] = None) -> jnp.ndarray:
    """Weighted sum of per-axis rewards into a single scalar per example."""
    w = weights or DEFAULT_AXIS_WEIGHTS
    return sum(w.get(k, 0.0) * v for k, v in axis_rewards.items())


def bradley_terry_loss(chosen_reward: jnp.ndarray, rejected_reward: jnp.ndarray) -> jnp.ndarray:
    """Pairwise preference loss: -log sigmoid(chosen - rejected)."""
    return -jnp.mean(jax.nn.log_sigmoid(chosen_reward - rejected_reward))


def per_axis_bradley_terry_loss(
    chosen_axes: Dict[str, jnp.ndarray],
    rejected_axes: Dict[str, jnp.ndarray],
    axis_labels: Optional[Dict[str, jnp.ndarray]] = None,
) -> Dict[str, jnp.ndarray]:
    """Per-axis BT loss; if `axis_labels[a]` is provided, mask to labeled examples only."""
    out: Dict[str, jnp.ndarray] = {}
    for axis in chosen_axes:
        diff = chosen_axes[axis] - rejected_axes[axis]
        per = -jax.nn.log_sigmoid(diff)
        if axis_labels is not None and axis in axis_labels:
            mask = axis_labels[axis]
            denom = jnp.maximum(mask.sum(), 1.0)
            out[axis] = (per * mask).sum() / denom
        else:
            out[axis] = jnp.mean(per)
    return out


def total_reward_loss(
    per_axis_losses: Dict[str, jnp.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> jnp.ndarray:
    """Weighted sum of per-axis BT losses."""
    w = weights or DEFAULT_AXIS_WEIGHTS
    return sum(w.get(k, 0.0) * v for k, v in per_axis_losses.items())


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Compute ECE over equally spaced confidence bins."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(confidences), 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() > 0:
            acc = correct[in_bin].mean()
            conf = confidences[in_bin].mean()
            ece += float(in_bin.sum()) / n * abs(acc - conf)
    return float(ece)


def temperature_search(
    chosen_rewards: np.ndarray,
    rejected_rewards: np.ndarray,
    temperatures: Tuple[float, ...] = (0.5, 0.7, 1.0, 1.2, 1.5),
) -> Tuple[float, float]:
    """Sweep temperatures and return (best_T, best_ECE) by BT win-probability calibration."""
    best_t = 1.0
    best_ece = float("inf")
    for t in temperatures:
        diff = (chosen_rewards - rejected_rewards) / max(t, 1e-3)
        prob_chosen = 1.0 / (1.0 + np.exp(-diff))
        # Each pair has a known "chosen wins" label = 1.
        labels = np.ones_like(prob_chosen)
        ece = expected_calibration_error(prob_chosen, labels)
        if ece < best_ece:
            best_ece = ece
            best_t = t
    return best_t, best_ece


def reliability_diagram(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> List[Tuple[float, float, int]]:
    """Return (mean_confidence, accuracy, count) per bin for plotting."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: List[Tuple[float, float, int]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        n = int(in_bin.sum())
        if n == 0:
            out.append((0.5 * (lo + hi), 0.0, 0))
        else:
            out.append((float(confidences[in_bin].mean()), float(correct[in_bin].mean()), n))
    return out
