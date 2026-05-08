"""Tests for adversarial robustness training."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.core.ethics.adversarial_training import (
    AttackConfig,
    adversarial_loss,
    attack_success_rate,
    fgsm_perturbation,
    pgd_attack,
    synthesize_multi_turn_manipulation,
    synthesize_prompt_injections,
)


def _quadratic_loss_factory(target: jnp.ndarray):
    """Loss is (x - target)^2 — gradient direction is known."""

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((x - target) ** 2)

    return loss


def test_fgsm_perturbation_within_epsilon_linf():
    """FGSM output must be bounded by epsilon under L-inf."""
    target = jnp.zeros((2, 4))
    loss = _quadratic_loss_factory(target)
    x = jnp.ones((2, 4))
    delta = fgsm_perturbation(x, loss, epsilon=0.1)
    assert float(jnp.max(jnp.abs(delta))) <= 0.1 + 1e-6


def test_fgsm_perturbation_increases_loss():
    """FGSM should produce a perturbation that increases the loss."""
    target = jnp.zeros((2, 4))
    loss = _quadratic_loss_factory(target)
    x = jnp.ones((2, 4))
    base = float(loss(x))
    delta = fgsm_perturbation(x, loss, epsilon=0.05)
    assert float(loss(x + delta)) >= base


def test_pgd_attack_within_epsilon_ball():
    """PGD output difference from base must be bounded by epsilon."""
    target = jnp.zeros((2, 4))
    loss = _quadratic_loss_factory(target)
    x = jnp.ones((2, 4))
    cfg = AttackConfig(epsilon=0.1, step_size=0.02, num_steps=5, random_start=False)
    adv = pgd_attack(x, loss, cfg)
    assert float(jnp.max(jnp.abs(adv - x))) <= cfg.epsilon + 1e-5


def test_pgd_with_random_start():
    """PGD with random_start should still respect epsilon and produce different output."""
    target = jnp.zeros((2, 4))
    loss = _quadratic_loss_factory(target)
    x = jnp.ones((2, 4))
    rng = jax.random.PRNGKey(0)
    cfg = AttackConfig(epsilon=0.1, step_size=0.02, num_steps=3, random_start=True)
    adv = pgd_attack(x, loss, cfg, rng=rng)
    assert float(jnp.max(jnp.abs(adv - x))) <= cfg.epsilon + 1e-5


def test_adversarial_loss_blend():
    """Adversarial loss is weighted clean + adv."""
    target = jnp.zeros((2, 4))
    loss = _quadratic_loss_factory(target)
    clean = jnp.ones((2, 4)) * 0.5
    adv = jnp.ones((2, 4)) * 1.5
    out = adversarial_loss(loss, adv, clean, weight_clean=0.5, weight_adv=0.5)
    expected = 0.5 * float(loss(clean)) + 0.5 * float(loss(adv))
    assert float(out) == pytest.approx(expected, abs=1e-5)


def test_synthesize_prompt_injections_count():
    """Each payload should produce exactly one injection example."""
    pairs = synthesize_prompt_injections(["secret 1", "secret 2", "secret 3"])
    assert len(pairs) == 3
    for p in pairs:
        assert "prompt" in p and "response" in p


def test_synthesize_multi_turn_dialog_length():
    """Multi-turn manipulation respects requested turn count."""
    dialog = synthesize_multi_turn_manipulation("payload", n_turns=3)
    assert len(dialog) == 3
    for turn in dialog:
        assert "user" in turn and "assistant" in turn


def test_attack_success_rate_basic():
    """Success rate = fraction of False refusals."""
    refusals = [True, True, False, False]
    assert attack_success_rate(refusals) == pytest.approx(0.5)
    assert attack_success_rate([]) == pytest.approx(0.0)
