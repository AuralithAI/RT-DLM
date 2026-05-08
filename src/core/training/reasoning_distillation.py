"""Reasoning-trace knowledge distillation: structural CoT alignment with a teacher."""

from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp


@dataclass
class TraceDistillConfig:
    """Hyperparameters for reasoning-trace distillation."""
    cosine_weight: float = 1.0
    structure_weight: float = 0.3
    logit_weight: float = 0.0
    temperature: float = 2.0


def _normalize(x: jnp.ndarray, axis: int = -1, eps: float = 1e-6) -> jnp.ndarray:
    """L2-normalize along an axis."""
    n = jnp.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n


def cosine_trace_loss(
    student_nodes: jnp.ndarray, teacher_steps: jnp.ndarray, mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """1 - cos_sim between paired student graph nodes and teacher CoT step embeddings."""
    s = _normalize(student_nodes)
    t = _normalize(teacher_steps)
    if s.shape[1] != t.shape[1]:
        target_len = min(s.shape[1], t.shape[1])
        s = s[:, :target_len]
        t = t[:, :target_len]
        if mask is not None:
            mask = mask[:, :target_len]
    sim = jnp.sum(s * t, axis=-1)
    per = 1.0 - sim
    if mask is None:
        return jnp.mean(per)
    weight = jnp.maximum(mask.sum(), 1.0)
    return (per * mask).sum() / weight


def structural_alignment_loss(
    student_adjacency: jnp.ndarray, teacher_adjacency: jnp.ndarray
) -> jnp.ndarray:
    """MSE between student reasoning-graph adjacency and teacher chain-structure matrix."""
    n = min(student_adjacency.shape[-1], teacher_adjacency.shape[-1])
    s = student_adjacency[..., :n, :n]
    t = teacher_adjacency[..., :n, :n]
    return jnp.mean((s - t) ** 2)


def logit_distillation_loss(
    student_logits: jnp.ndarray, teacher_logits: jnp.ndarray, temperature: float = 2.0
) -> jnp.ndarray:
    """Soft-target KL divergence between student and teacher token distributions."""
    s = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    t = jax.nn.softmax(teacher_logits / temperature, axis=-1)
    kl = jnp.sum(t * (jnp.log(t + 1e-9) - s), axis=-1)
    return jnp.mean(kl) * (temperature ** 2)


def chain_to_adjacency(num_steps: int) -> jnp.ndarray:
    """Build a 0/1 adjacency matrix for a linear reasoning chain of length `num_steps`."""
    a = jnp.zeros((num_steps, num_steps), dtype=jnp.float32)
    if num_steps > 1:
        idx = jnp.arange(num_steps - 1)
        a = a.at[idx, idx + 1].set(1.0)
    return a


def reasoning_trace_distillation_loss(
    student_nodes: jnp.ndarray,
    teacher_steps: jnp.ndarray,
    student_adjacency: Optional[jnp.ndarray] = None,
    teacher_adjacency: Optional[jnp.ndarray] = None,
    student_logits: Optional[jnp.ndarray] = None,
    teacher_logits: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    config: Optional[TraceDistillConfig] = None,
) -> Dict[str, jnp.ndarray]:
    """Combined cosine + structural + logit distillation loss."""
    cfg = config or TraceDistillConfig()
    out: Dict[str, jnp.ndarray] = {}
    out["cosine"] = cosine_trace_loss(student_nodes, teacher_steps, mask)
    total = cfg.cosine_weight * out["cosine"]

    if student_adjacency is not None and teacher_adjacency is not None:
        out["structural"] = structural_alignment_loss(student_adjacency, teacher_adjacency)
        total = total + cfg.structure_weight * out["structural"]

    if student_logits is not None and teacher_logits is not None and cfg.logit_weight > 0.0:
        out["logit"] = logit_distillation_loss(
            student_logits, teacher_logits, cfg.temperature
        )
        total = total + cfg.logit_weight * out["logit"]

    out["total"] = total
    return out


def project_teacher_to_student_dim(
    teacher_steps: jnp.ndarray, target_dim: int
) -> jnp.ndarray:
    """Linear projection (truncate or zero-pad) when teacher embedding dim differs."""
    src_dim = teacher_steps.shape[-1]
    if src_dim == target_dim:
        return teacher_steps
    if src_dim > target_dim:
        return teacher_steps[..., :target_dim]
    pad = jnp.zeros((*teacher_steps.shape[:-1], target_dim - src_dim))
    return jnp.concatenate([teacher_steps, pad], axis=-1)
