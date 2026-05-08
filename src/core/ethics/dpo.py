"""Direct Preference Optimization (DPO) loss for offline alignment."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class DPOConfig:
    """DPO hyperparameters."""
    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_free: bool = False


def _sequence_logprob(
    logits: jnp.ndarray, labels: jnp.ndarray, mask: Optional[jnp.ndarray]
) -> jnp.ndarray:
    """Sum of token log-probabilities along the sequence, respecting `mask`."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    per_tok = jnp.sum(one_hot * log_probs, axis=-1)
    if mask is not None:
        per_tok = per_tok * mask
    return per_tok.sum(axis=-1)


def dpo_loss(
    policy_chosen_logits: jnp.ndarray,
    policy_rejected_logits: jnp.ndarray,
    chosen_labels: jnp.ndarray,
    rejected_labels: jnp.ndarray,
    ref_chosen_logits: Optional[jnp.ndarray] = None,
    ref_rejected_logits: Optional[jnp.ndarray] = None,
    chosen_mask: Optional[jnp.ndarray] = None,
    rejected_mask: Optional[jnp.ndarray] = None,
    config: Optional[DPOConfig] = None,
) -> Dict[str, jnp.ndarray]:
    """Compute DPO loss + diagnostic margins."""
    cfg = config or DPOConfig()
    pi_c = _sequence_logprob(policy_chosen_logits, chosen_labels, chosen_mask)
    pi_r = _sequence_logprob(policy_rejected_logits, rejected_labels, rejected_mask)

    if cfg.reference_free or ref_chosen_logits is None or ref_rejected_logits is None:
        ref_c = jnp.zeros_like(pi_c)
        ref_r = jnp.zeros_like(pi_r)
    else:
        ref_c = jax.lax.stop_gradient(
            _sequence_logprob(ref_chosen_logits, chosen_labels, chosen_mask)
        )
        ref_r = jax.lax.stop_gradient(
            _sequence_logprob(ref_rejected_logits, rejected_labels, rejected_mask)
        )

    logits = cfg.beta * ((pi_c - ref_c) - (pi_r - ref_r))
    if cfg.label_smoothing > 0.0:
        loss = -(
            (1.0 - cfg.label_smoothing) * jax.nn.log_sigmoid(logits)
            + cfg.label_smoothing * jax.nn.log_sigmoid(-logits)
        )
    else:
        loss = -jax.nn.log_sigmoid(logits)

    return {
        "loss": jnp.mean(loss),
        "margin": jnp.mean(logits),
        "policy_chosen_logp": jnp.mean(pi_c),
        "policy_rejected_logp": jnp.mean(pi_r),
        "ref_chosen_logp": jnp.mean(ref_c),
        "ref_rejected_logp": jnp.mean(ref_r),
        "implicit_reward_chosen": jnp.mean(cfg.beta * (pi_c - ref_c)),
        "implicit_reward_rejected": jnp.mean(cfg.beta * (pi_r - ref_r)),
    }


def ipo_loss(
    policy_chosen_logits: jnp.ndarray,
    policy_rejected_logits: jnp.ndarray,
    chosen_labels: jnp.ndarray,
    rejected_labels: jnp.ndarray,
    ref_chosen_logits: jnp.ndarray,
    ref_rejected_logits: jnp.ndarray,
    chosen_mask: Optional[jnp.ndarray] = None,
    rejected_mask: Optional[jnp.ndarray] = None,
    beta: float = 0.1,
) -> jnp.ndarray:
    """IPO regression variant: (margin - 1/(2β))^2 — more stable than DPO."""
    pi_c = _sequence_logprob(policy_chosen_logits, chosen_labels, chosen_mask)
    pi_r = _sequence_logprob(policy_rejected_logits, rejected_labels, rejected_mask)
    ref_c = jax.lax.stop_gradient(_sequence_logprob(ref_chosen_logits, chosen_labels, chosen_mask))
    ref_r = jax.lax.stop_gradient(_sequence_logprob(ref_rejected_logits, rejected_labels, rejected_mask))
    margin = (pi_c - ref_c) - (pi_r - ref_r)
    target = 1.0 / (2.0 * max(beta, 1e-3))
    return jnp.mean((margin - target) ** 2)


def dpo_accuracy(
    policy_chosen_logp: jnp.ndarray, policy_rejected_logp: jnp.ndarray
) -> jnp.ndarray:
    """Fraction of pairs where chosen logp exceeds rejected logp."""
    return jnp.mean((policy_chosen_logp > policy_rejected_logp).astype(jnp.float32))
