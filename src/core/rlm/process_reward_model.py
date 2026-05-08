"""Process Reward Model (PRM): step-level correctness scoring for reasoning traces."""

from dataclasses import dataclass
from typing import Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp


@dataclass
class PRMConfig:
    """Hyperparameters for ProcessRewardModel."""

    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 4
    step_token_id: int = 0
    outcome_weight: float = 0.5
    step_weight: float = 0.5


class ProcessRewardModel(hk.Module):
    """Per-step correctness scorer over tokenized reasoning traces."""

    def __init__(self, config: PRMConfig, name: Optional[str] = None):
        super().__init__(name=name)
        self.config = config
        self.layers = []
        self.norms = []
        for i in range(config.num_layers):
            self.layers.append(
                hk.MultiHeadAttention(
                    num_heads=config.num_heads,
                    key_size=config.d_model // config.num_heads,
                    w_init=hk.initializers.TruncatedNormal(0.02),
                    name=f"prm_attn_{i}",
                )
            )
            self.norms.append(hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"prm_ln_{i}"))
        self.step_head = hk.Linear(1, name="step_correct_head")
        self.outcome_head = hk.Linear(1, name="outcome_head")

    def __call__(self, hidden: jnp.ndarray, step_mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return (step_logits [B,T], outcome_logits [B])."""
        x = hidden
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x, x, x))
        step_logits = self.step_head(x).squeeze(-1)
        mask_3d = step_mask[..., None]
        masked = jnp.where(mask_3d > 0, x, 0.0)
        denom = jnp.maximum(step_mask.sum(axis=1, keepdims=True), 1.0)
        pooled = masked.sum(axis=1) / denom
        outcome_logits = self.outcome_head(pooled).squeeze(-1)
        return step_logits, outcome_logits


def step_correctness_loss(step_logits: jnp.ndarray, step_labels: jnp.ndarray, step_mask: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy over steps, masked to active positions."""
    log_p = jax.nn.log_sigmoid(step_logits)
    log_1mp = jax.nn.log_sigmoid(-step_logits)
    per = -(step_labels * log_p + (1.0 - step_labels) * log_1mp)
    weight = jnp.maximum(step_mask.sum(), 1.0)
    return (per * step_mask).sum() / weight


def outcome_loss(outcome_logits: jnp.ndarray, outcome_labels: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy on the trajectory-level outcome."""
    log_p = jax.nn.log_sigmoid(outcome_logits)
    log_1mp = jax.nn.log_sigmoid(-outcome_logits)
    return -jnp.mean(outcome_labels * log_p + (1.0 - outcome_labels) * log_1mp)


def prm_total_loss(
    step_logits: jnp.ndarray,
    outcome_logits: jnp.ndarray,
    step_labels: jnp.ndarray,
    outcome_labels: jnp.ndarray,
    step_mask: jnp.ndarray,
    config: PRMConfig,
) -> jnp.ndarray:
    """Weighted sum of step + outcome losses per `config`."""
    s = step_correctness_loss(step_logits, step_labels, step_mask)
    o = outcome_loss(outcome_logits, outcome_labels)
    return config.step_weight * s + config.outcome_weight * o


def combined_outcome_step_reward(
    outcome_reward: jnp.ndarray,
    step_rewards: jnp.ndarray,
    step_mask: jnp.ndarray,
    alpha: float = 0.5,
) -> jnp.ndarray:
    """Blend trajectory outcome reward with mean masked step reward."""
    denom = jnp.maximum(step_mask.sum(axis=-1), 1.0)
    mean_step = (step_rewards * step_mask).sum(axis=-1) / denom
    return alpha * outcome_reward + (1.0 - alpha) * mean_step
