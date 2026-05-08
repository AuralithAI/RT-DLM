"""Tests for GRPO RewardComputer process-reward extension."""

import jax.numpy as jnp
import pytest

from src.train_controller_grpo import RewardComputer, Trajectory
from src.core.agi.compute_controller import ModuleType


def _traj(budget: float = 0.4, modules=None, value: float = 0.9) -> Trajectory:
    """Build a trajectory stub for reward tests."""
    return Trajectory(
        hidden_states=jnp.zeros((4,)),
        log_prob=0.0,
        reward=0.0,
        modules_called=modules or [ModuleType.MEMORY_RETRIEVAL],
        steps_taken=1,
        budget_used=budget,
        value_estimate=value,
    )


def test_reward_outcome_only_when_no_step_rewards():
    """Without step rewards, computer returns the original outcome score."""
    rc = RewardComputer()
    answer = jnp.array([1.0, 0.0])
    majority = jnp.array([1.0, 0.0])
    r = rc.compute_reward(_traj(), majority, answer)
    assert r > 0.0


def test_process_reward_blends_when_provided():
    """With step rewards, blended result should differ from pure outcome."""
    rc = RewardComputer(process_reward_weight=0.5, outcome_reward_weight=0.5)
    answer = jnp.array([1.0, 0.0])
    majority = jnp.array([1.0, 0.0])
    r_outcome = rc.compute_reward(_traj(), majority, answer)
    r_blend = rc.compute_reward(_traj(), majority, answer, step_rewards=[0.0, 0.0, 0.0])
    assert r_blend != r_outcome
    assert r_blend == pytest.approx(0.5 * r_outcome, abs=1e-6)


def test_zero_step_rewards_halve_outcome():
    """All-zero step rewards with 50/50 blend should halve outcome."""
    rc = RewardComputer(process_reward_weight=0.5, outcome_reward_weight=0.5)
    answer = jnp.array([1.0, 0.0])
    majority = jnp.array([1.0, 0.0])
    r_pure = rc.compute_reward(_traj(), majority, answer)
    r_blend = rc.compute_reward(_traj(), majority, answer, step_rewards=[0.0])
    assert abs(r_blend - 0.5 * r_pure) < 1e-6


def test_high_step_rewards_boost_total():
    """High step rewards must increase total reward over outcome alone."""
    rc = RewardComputer(process_reward_weight=0.5, outcome_reward_weight=0.5)
    answer = jnp.array([1.0, 0.0])
    majority = jnp.array([1.0, 0.0])
    r_pure = rc.compute_reward(_traj(), majority, answer)
    r_blend = rc.compute_reward(_traj(), majority, answer, step_rewards=[1.0, 1.0])
    assert r_blend > 0.5 * r_pure
