"""Abstention training: 'I don't know' loss + uncertainty-thresholded routing."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class AbstentionConfig:
    """Abstention training hyperparameters."""
    abstain_token_id: int = 0
    confidence_threshold: float = 0.6
    abstain_loss_weight: float = 0.1
    overconfidence_penalty: float = 0.05


def predictive_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    """Token-level predictive entropy of softmax distribution."""
    log_p = jax.nn.log_softmax(logits, axis=-1)
    p = jnp.exp(log_p)
    return -jnp.sum(p * log_p, axis=-1)


def confidence_score(logits: jnp.ndarray) -> jnp.ndarray:
    """Max softmax probability per position — proxy for confidence."""
    return jnp.max(jax.nn.softmax(logits, axis=-1), axis=-1)


def should_abstain(logits: jnp.ndarray, threshold: float = 0.6) -> jnp.ndarray:
    """Boolean mask: True where the model should emit an abstain token."""
    return confidence_score(logits) < threshold


def abstention_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    abstain_label_mask: jnp.ndarray,
    config: Optional[AbstentionConfig] = None,
) -> Dict[str, jnp.ndarray]:
    """Cross-entropy + abstention term: rewards predicting abstain when label says so."""
    cfg = config or AbstentionConfig()
    log_p = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    ce = -jnp.sum(one_hot * log_p, axis=-1)

    log_p_abstain = log_p[..., cfg.abstain_token_id]
    abstain_term = -log_p_abstain * abstain_label_mask
    weight = jnp.maximum(abstain_label_mask.sum(), 1.0)

    confident = confidence_score(logits)
    over_pen = jnp.maximum(confident - 0.95, 0.0) * abstain_label_mask
    over_term = (over_pen.sum() / weight) * cfg.overconfidence_penalty

    total = ce.mean() + cfg.abstain_loss_weight * (abstain_term.sum() / weight) + over_term
    return {
        "ce": ce.mean(),
        "abstain": abstain_term.sum() / weight,
        "overconfidence": over_term,
        "total": total,
    }


def uncertainty_routing_decision(
    confidence: float, threshold: float = 0.6
) -> str:
    """Return 'verify' for low-confidence predictions, 'commit' otherwise."""
    return "verify" if confidence < threshold else "commit"


def expected_calibration_loss(
    confidence: jnp.ndarray, correct: jnp.ndarray, n_bins: int = 10
) -> jnp.ndarray:
    """Differentiable surrogate for ECE (binned squared confidence-vs-accuracy gap)."""
    edges = jnp.linspace(0.0, 1.0, n_bins + 1)
    total = jnp.zeros(())
    n_total = jnp.maximum(confidence.size, 1)
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = ((confidence > lo) & (confidence <= hi)).astype(jnp.float32)
        denom = jnp.maximum(mask.sum(), 1.0)
        bin_conf = (confidence * mask).sum() / denom
        bin_acc = (correct.astype(jnp.float32) * mask).sum() / denom
        weight = mask.sum() / n_total
        total = total + weight * (bin_conf - bin_acc) ** 2
    return total


def synthesize_abstention_examples(
    factual_qas: List[Tuple[str, str]],
    unknown_qs: List[str],
    abstain_response: str = "I don't know.",
) -> List[Dict[str, str]]:
    """Mix factual QA with unknown-question abstention pairs."""
    out: List[Dict[str, str]] = []
    for q, a in factual_qas:
        out.append({"prompt": q, "response": a, "abstain": "0"})
    for q in unknown_qs:
        out.append({"prompt": q, "response": abstain_response, "abstain": "1"})
    return out


def ensemble_uncertainty(logits_list: List[jnp.ndarray]) -> jnp.ndarray:
    """Predictive variance across an ensemble of forward passes."""
    probs = jnp.stack([jax.nn.softmax(l, axis=-1) for l in logits_list], axis=0)
    return jnp.var(probs, axis=0).sum(axis=-1)
