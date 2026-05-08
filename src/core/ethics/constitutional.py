"""Constitutional AI: rule-based self-critique with revision-imitation loss."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class ConstitutionalRule:
    """A single constitutional rule."""
    rule_id: str
    category: str
    text: str
    severity: float = 1.0


@dataclass
class ConstitutionalRuleset:
    """Collection of rules grouped by harm category."""
    rules: List[ConstitutionalRule] = field(default_factory=list)

    def add(self, rule_id: str, category: str, text: str, severity: float = 1.0) -> None:
        """Append a rule."""
        self.rules.append(ConstitutionalRule(rule_id, category, text, severity))

    def for_category(self, category: str) -> List[ConstitutionalRule]:
        """Return rules in `category`."""
        return [r for r in self.rules if r.category == category]

    def all_categories(self) -> List[str]:
        """Distinct categories present."""
        seen: List[str] = []
        for r in self.rules:
            if r.category not in seen:
                seen.append(r.category)
        return seen


def default_ruleset() -> ConstitutionalRuleset:
    """Compact 12-category default constitution suitable for automated self-critique."""
    rs = ConstitutionalRuleset()
    cats = [
        ("cbrn", "Refuse instructions for chemical/biological/radiological/nuclear harm."),
        ("hate", "Reject hateful, demeaning, or dehumanizing content toward protected groups."),
        ("privacy", "Do not expose private personal information without consent."),
        ("deception", "Avoid intentionally deceiving the user or generating disinformation."),
        ("manipulation", "Refuse covert persuasion or psychological manipulation tactics."),
        ("illegal", "Decline assistance with illegal activity in the user's jurisdiction."),
        ("child_safety", "Treat child safety as paramount; refuse all CSAM-adjacent requests."),
        ("medical", "Avoid medical misinformation; recommend qualified professionals."),
        ("financial", "Refuse fraudulent or predatory financial schemes."),
        ("harassment", "Decline to assist targeted harassment."),
        ("copyright", "Avoid bulk reproduction of copyrighted text without fair-use basis."),
        ("surveillance", "Refuse mass surveillance enabling guidance."),
    ]
    for i, (cat, txt) in enumerate(cats):
        rs.add(f"R{i:03d}", cat, txt)
    return rs


def _ce(logits: jnp.ndarray, labels: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Masked token cross-entropy for [B,T,V] logits with int labels [B,T]."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    per_tok = -jnp.sum(one_hot * log_probs, axis=-1)
    if mask is None:
        return jnp.mean(per_tok)
    weight = jnp.maximum(mask.sum(), 1.0)
    return (per_tok * mask).sum() / weight


def revision_imitation_loss(
    revised_logits: jnp.ndarray,
    revised_labels: jnp.ndarray,
    revised_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Train the model to emit constitutionally revised tokens directly."""
    return _ce(revised_logits, revised_labels, revised_mask)


def critique_consistency_loss(
    original_logits: jnp.ndarray,
    revised_logits: jnp.ndarray,
    violation_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Pull revised distribution toward independence from original on violating tokens."""
    p = jax.nn.softmax(revised_logits, axis=-1)
    q = jax.nn.softmax(original_logits, axis=-1)
    kl = jnp.sum(p * (jnp.log(p + 1e-9) - jnp.log(q + 1e-9)), axis=-1)
    weight = jnp.maximum(violation_mask.sum(), 1.0)
    return -(kl * violation_mask).sum() / weight


def constitutional_self_critique_loss(
    revised_logits: jnp.ndarray,
    revised_labels: jnp.ndarray,
    original_logits: Optional[jnp.ndarray] = None,
    violation_mask: Optional[jnp.ndarray] = None,
    revised_mask: Optional[jnp.ndarray] = None,
    consistency_weight: float = 0.1,
) -> Dict[str, jnp.ndarray]:
    """Combined loss: imitate revised + push away from original on violations."""
    out: Dict[str, jnp.ndarray] = {}
    out["imitation"] = revision_imitation_loss(revised_logits, revised_labels, revised_mask)
    if original_logits is not None and violation_mask is not None:
        out["consistency"] = critique_consistency_loss(
            original_logits, revised_logits, violation_mask
        )
        out["total"] = out["imitation"] + consistency_weight * out["consistency"]
    else:
        out["total"] = out["imitation"]
    return out


CritiqueFn = Callable[[str, ConstitutionalRule], Tuple[bool, str]]


def run_self_critique(
    prompt: str,
    response: str,
    ruleset: ConstitutionalRuleset,
    critique_fn: CritiqueFn,
    revise_fn: Callable[[str, str, List[str]], str],
) -> Dict[str, object]:
    """Apply each rule via `critique_fn`; if any violates, call `revise_fn` to repair."""
    violations: List[str] = []
    explanations: List[str] = []
    for rule in ruleset.rules:
        violated, reason = critique_fn(response, rule)
        if violated:
            violations.append(rule.rule_id)
            explanations.append(reason)
    revised = response
    if violations:
        revised = revise_fn(prompt, response, violations)
    _ = explanations
    return {
        "violations": violations,
        "explanations": explanations,
        "original": response,
        "revised": revised,
    }
